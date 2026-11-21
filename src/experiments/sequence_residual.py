"""p57: 전체 예보일의 bounded residual TCN. 집계만 저장/출력한다."""
import argparse
import gc
import json
import os
from pathlib import Path
import traceback
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR, PRED_DIR
from ..data import load_generation, load_sites, load_smp
from ..diagnose_season import run_id
from ..features import build, station_table, nbrc_table
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from .sequence_audit import HISTORY, day_slots
from ..solar import solar
from ..splits import fold
from ..tune_v3 import END, WINDOWS, evaluation_context, save
from ..v6_error_budget import load_reference
from ..weather_teacher import CFG

PATH = EXP_DIR/"p57_sequence_residual"/"metrics.json"
SPEC = dict(rule="14.47", channels=287, width=32, dilations=[1, 2, 4, 8],
            bound_kt=.20, lr=.001, weight_decay=.0001, max_epochs=40,
            patience=5, batch_size=64, seeds=[2026, 2027, 2028])


def stable_average(values, weights):
    """동일 가중평균을 BLAS matmul 없이 계산한다."""
    if np.isinf(values).any() or not np.isfinite(weights).all():
        raise ValueError("이웃 평균 입력 유한성 위반")
    return np.einsum("ij,kj->ik", values, weights, optimize=False)


def sequence_fold_features(data, sites, cutoff):
    out = data.copy(deep=False)
    for provider in ("wa", "wb"):
        table = nbrc_table(provider, cutoff+pd.Timedelta(hours=1), weighted_average=stable_average)
        keys = pd.MultiIndex.from_arrays([data.site_id.map(sites[f"{provider}_station"]), data.ts])
        for column in table:
            out[column] = table[column].reindex(keys).to_numpy()
    return out


def make_model():
    import torch.nn as nn
    layers = [nn.Conv1d(SPEC["channels"], 32, 1), nn.GELU()]
    for dilation in SPEC["dilations"]:
        layers += [nn.Conv1d(32, 32, 3, padding=dilation, dilation=dilation), nn.GELU()]
    head = nn.Conv1d(32, 1, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head, nn.Tanh())


class Block:
    """주간 숫자행+공유GFS만 보유. 정답과 기준예측은 입력과 분리."""
    def __init__(self, panel, weather, columns, gfs, sites):
        p = panel.copy()
        day, slots = day_slots(p.ts)
        keys = pd.MultiIndex.from_arrays([p.site_id, day], names=["site_id", "day"])
        codes, self.keys = pd.factorize(keys, sort=False)
        self.keys = self.keys.set_names(["site_id", "day"])
        self.n = len(self.keys)
        self._masks = {}
        self.panel = p
        self.codes, self.slots = codes, slots
        self.cap = self.keys.get_level_values("site_id").map(sites.capacity_kw).to_numpy(dtype="float32")
        if not np.isfinite(self.cap).all() or (self.cap <= 0).any():
            raise ValueError("용량 조건 위반")
        self.target = np.zeros((self.n, 24), dtype="float32")
        self.observed = np.zeros((self.n, 24), dtype=bool)
        self.base = np.zeros((self.n, 24), dtype="float32")
        self.target[codes, slots] = p.energy.to_numpy(dtype="float32")
        self.observed[codes, slots] = True
        self.base[codes, slots] = p.forecast_kwh.to_numpy(dtype="float32")
        w = weather.copy()
        wd, ws = day_slots(w.ts)
        wi = self.keys.get_indexer(pd.MultiIndex.from_arrays([w.site_id, wd]))
        keep = wi >= 0
        w, wi, ws = w.loc[keep], wi[keep], ws[keep]
        if w.duplicated(["site_id", "ts"]).any():
            raise ValueError("주간 특징 키 중복")
        self.row = np.full((self.n, 24), -1, dtype="int32")
        self.row[wi, ws] = np.arange(len(w))
        self.scale = np.zeros((self.n, 24), dtype="float32")
        self.scale[wi, ws] = T.clear_sky(w.cosz)*w.cap.to_numpy()
        self.values = w[columns].to_numpy(dtype="float32")
        self.gfs = gfs
        self.gindex = gfs.index.get_indexer(w.ts)
        self.numeric_columns = list(columns)+list(gfs.columns)
        # 관측 여부와 무관하게 태양위치상 주간 슬롯이 모두 준비돼야 한다.
        all_ts = pd.DatetimeIndex(np.repeat(self.keys.get_level_values("day"), 24))
        all_ts += pd.to_timedelta(np.tile(np.arange(1, 25), self.n), unit="h")
        ids = np.repeat(self.keys.get_level_values("site_id"), 24)
        elev, _, _ = solar(all_ts, sites.lat.reindex(ids).to_numpy(), sites.lon.reindex(ids).to_numpy())
        if not np.array_equal(self.row >= 0, (elev > 0).reshape(self.n, 24)):
            raise ValueError("관측mask와 독립적인 전체 주간 입력 미복원")
        if not np.isfinite(self.target[self.observed]).all():
            raise ValueError("유효 정답 유한성 위반")

    def numeric(self, rows):
        other = np.full((len(rows), 60), np.nan, dtype="float32")
        indices = self.gindex[rows]
        good = indices >= 0
        other[good] = self.gfs.to_numpy()[indices[good]]
        return np.concatenate([self.values[rows], other], axis=1)

    def inputs(self, ids, normalizer):
        row = self.row[ids]
        good = row >= 0
        out = np.zeros((len(ids), 24, SPEC["channels"]), dtype="float32")
        numeric = self.numeric(row[good])
        valid = np.isfinite(numeric)
        normalized = (numeric-normalizer[0])/normalizer[1]
        out[:, :, :143][good] = np.where(valid, normalized, 0)
        out[:, :, 143:286][good] = valid
        out[:, :, -1] = good
        return out.transpose(0, 2, 1).copy()

    def mask_before(self, cutoff):
        if cutoff.value not in self._masks:
            ts = self.keys.get_level_values("day").asi8[:, None]+np.arange(1, 25)*3600*10**9
            self._masks[cutoff.value] = self.observed & (ts <= cutoff.value)
        return self._masks[cutoff.value]


def fit_normalizer(blocks):
    count = np.zeros(143, dtype="int64")
    total, square = np.zeros(143), np.zeros(143)
    for b in blocks:
        for start in range(0, len(b.values), 32768):
            x = b.numeric(np.arange(start, min(start+32768, len(b.values)))).astype("float64")
            valid = np.isfinite(x)
            z = np.where(valid, x, 0)
            count += valid.sum(axis=0)
            total += z.sum(axis=0)
            square += (z*z).sum(axis=0)
    mean = total/np.maximum(count, 1)
    std = np.sqrt(np.maximum(square/np.maximum(count, 1)-mean*mean, 0))
    std[std < 1e-6] = 1
    return mean.astype("float32"), std.astype("float32")


def tensor_batch(block, ids, norm, device, cutoff=None):
    import torch
    x = torch.from_numpy(block.inputs(ids, norm)).to(device)
    def tensor(v):
        return torch.from_numpy(v[ids]).to(device)
    mask = block.observed if cutoff is None else block.mask_before(cutoff)
    return x, tensor(block.base), tensor(block.scale), tensor(block.target), tensor(mask), tensor(block.cap)


def corrected(model, batch):
    import torch
    x, base, scale, _, _, _ = batch
    return torch.clamp(base+SPEC["bound_kt"]*model(x).squeeze(1)*scale, min=0)


def evaluate_loss(model, block, norm, device):
    import torch
    numerator, denominator = 0., 0.
    model.eval()
    with torch.no_grad():
        for start in range(0, block.n, 64):
            batch = tensor_batch(block, np.arange(start, min(start+64, block.n)), norm, device)
            pred = corrected(model, batch)
            mask = batch[4]
            numerator += ((pred-batch[3]).abs()*mask).sum().item()
            denominator += (batch[5][:, None]*mask).sum().item()
    return numerator/denominator


def train_one(blocks, norm, seed, device, validation=None, epochs=None, cutoff=None):
    import torch
    torch.manual_seed(seed)
    model = make_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SPEC["lr"], weight_decay=SPEC["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_epoch, state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, (epochs or SPEC["max_epochs"])+1):
        model.train()
        # 블록/발전소일 순서는 seed로 고정하고 마감 이후 정답은 mask로 제외한다.
        order = rng.permutation(len(blocks))
        numerator, denominator = 0., 0.
        for k in order:
            b = blocks[k]
            ids = rng.permutation(b.n)
            for start in range(0, b.n, 64):
                batch = tensor_batch(b, ids[start:start+64], norm, device, cutoff)
                mask = batch[4]
                denom = (batch[5][:, None]*mask).sum()
                if denom.item() == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                pred = corrected(model, batch)
                num = ((pred-batch[3]).abs()*mask).sum()
                loss = num/denom
                if not torch.isfinite(loss):
                    raise ValueError("학습손실 유한성 위반")
                loss.backward()
                optimizer.step()
                numerator += num.item()
                denominator += denom.item()
        record = dict(epoch=epoch, train_loss=numerator/denominator)
        if validation is not None:
            value = evaluate_loss(model, validation, norm, device)
            record["validation_loss"] = value
            if value < best:
                best, best_epoch, stale = value, epoch, 0
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        print(json.dumps(dict(stage="early_stop" if validation is not None else "refit", seed=seed, **record)), flush=True)
        if validation is not None and stale >= SPEC["patience"]:
            break
    if validation is not None:
        model.load_state_dict(state)
    return model, best_epoch if validation is not None else epochs, history


def predict(model, block, norm, device):
    import torch
    out = np.zeros((block.n, 24), dtype="float32")
    model.eval()
    with torch.no_grad():
        for start in range(0, block.n, 64):
            ids = np.arange(start, min(start+64, block.n))
            out[ids] = corrected(model, tensor_batch(block, ids, norm, device)).cpu().numpy()
    return out[block.codes, block.slots]


def prepare(build_blocks=True):
    print("전체 예보일 입력 복원: 실적행/좌표/ID 출력 없음", flush=True)
    sites = load_sites().set_index("site_id")
    gen = load_generation()
    gen = gen.loc[gen.ts <= END, ["site_id", "ts", "energy"]]
    names = pq.read_schema(CACHE_DIR/"train_features.parquet").names
    features = [c for c in T.select_features(pd.DataFrame(columns=names), CFG["feats"]) if c != "site"]
    d = pd.read_parquet(CACHE_DIR/"train_features.parquet", columns=["site_id", "ts", *features])
    d = d.loc[d.ts <= END]
    if len(features) != 83:
        raise ValueError("기본 특징 조건 위반")
    # 각 관측일을 24슬롯으로 확장하되 정답을 입력 생성 함수에 전달하지 않는다.
    day, _ = day_slots(gen.ts)
    sizes = gen.assign(day=day).groupby(["site_id", "day"]).size()
    partial = sizes[sizes != 24].index.to_frame(index=False)
    grid = partial.merge(pd.DataFrame(dict(slot=np.arange(24))), how="cross")
    grid["ts"] = grid.day+pd.to_timedelta(grid.slot+1, unit="h")
    elev, _, _ = solar(pd.DatetimeIndex(grid.ts), grid.site_id.map(sites.lat).to_numpy(), grid.site_id.map(sites.lon).to_numpy())
    existing = pd.MultiIndex.from_frame(d[["site_id", "ts"]])
    missing = grid.loc[(elev > 0) & ~pd.MultiIndex.from_frame(grid[["site_id", "ts"]]).isin(existing), ["site_id", "ts"]]
    if len(missing):
        tables = {}
        for provider in ("wa", "wb"):
            table = station_table(provider, sites.reset_index(), weighted_average=stable_average)
            tables[provider] = table.loc[table.ts.isin(missing.ts.unique())].copy()
            del table
            gc.collect()
        filled = build(missing, sites.reset_index(), tables)
        d = pd.concat([d, filled], ignore_index=True)
        del tables, filled
    d = d[["site_id", "ts", *features]]
    gfs = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    if len(gfs.columns) != 60 or not gfs.index.is_unique:
        raise ValueError("공유GFS 조건 위반")
    blocks = []
    for name in HISTORY if build_blocks else []:
        f = fold(name)
        p = pd.read_parquet(PRED_DIR/run_id("gfs60seed", name)/"predictions.parquet",
                            columns=["site_id", "ts", "forecast_kwh"])
        p = p.merge(gen, on=["site_id", "ts"], how="left", validate="one_to_one")
        if p.energy.isna().any() or not p.ts.between(f["val_start"], f["val_end"]).all():
            raise ValueError("OOF 관측/시점 조건 위반")
        weather = sequence_fold_features(d.loc[d.ts.between(f["val_start"], f["val_end"])], sites, f["cutoff"])
        if np.isinf(weather[features].to_numpy()).any():
            raise ValueError("예보 특징 무한값 위반")
        b = Block(p, weather, features, gfs, sites)
        blocks.append(b)
        print(json.dumps(dict(prepared=name, site_days=b.n, full_daylight_rows=len(b.values))), flush=True)
    return blocks, d, features, gfs, sites, gen, len(missing)


def penalty(panel, pred, sites, smp):
    p = panel[["site_id", "ts", "energy"]].copy()
    p["residual"] = pred-p.energy.to_numpy()
    p["group"] = p.site_id.map(sites.current_group)
    if p.group.isna().any():
        raise ValueError("집합 키 누락")
    r = p.groupby(["ts", "group"]).residual.sum()
    w = smp.w.reindex(r.index.get_level_values("ts")).to_numpy()
    if not np.isfinite(w).all():
        raise ValueError("정산 가중치 누락")
    return float(.1*np.dot(np.abs(r.to_numpy()), w))


def main(stage):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    warnings.filterwarnings("error", category=RuntimeWarning)
    Path("/tmp/oibc_sequence_residual.pid").write_text(str(os.getpid()))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(spec=SPEC)
    if result["spec"] != SPEC:
        raise ValueError("사전 고정 설정/체크포인트 불일치")
    blocks, d, features, gfs, sites, gen, restored = prepare()
    result["prepare"] = dict(complete=True, restored_daylight_rows=restored,
        site_days=sum(b.n for b in blocks), channels=SPEC["channels"],
        target_mask_in_input=False, baseline_forecast_in_input=False,
        torch_version=torch.__version__, device=str(device))
    save(PATH, result)
    if stage == "prepare":
        print(json.dumps(result["prepare"]), flush=True)
        return
    if result.get("complete"):
        print("완료된 p57 집계 결과 유지: 재학습하지 않음", flush=True)
        return
    smp = load_smp("train").set_index("ts")
    if "early_stopping" not in result:
        norm = fit_normalizer(blocks[:3])
        records = []
        for seed in SPEC["seeds"]:
            model, epoch, history = train_one(blocks[:3], norm, seed, device,
                validation=blocks[3], cutoff=fold(HISTORY[3])["cutoff"])
            records.append(dict(seed=seed, best_epoch=epoch, history=history))
            del model
        result["early_stopping"] = records
        save(PATH, result)
    norm = fit_normalizer(blocks[:4])
    pred = np.zeros(len(blocks[4].panel))
    for record in result["early_stopping"]:
        model, _, _ = train_one(blocks[:4], norm, record["seed"], device,
            epochs=record["best_epoch"], cutoff=fold(HISTORY[4])["cutoff"])
        pred += predict(model, blocks[4], norm, device)/3
        del model
    b = blocks[4]
    base_penalty = penalty(b.panel, b.panel.forecast_kwh.to_numpy(), sites, smp)
    new_penalty = penalty(b.panel, pred, sites, smp)
    result["oof5"] = dict(base_penalty=round(base_penalty), candidate_penalty=round(new_penalty),
        change_pct=100*(new_penalty/base_penalty-1), passed=new_penalty <= base_penalty)
    save(PATH, result)
    print(json.dumps(dict(oof5=result["oof5"])), flush=True)
    if not result["oof5"]["passed"]:
        result.update(complete=True, selected=False, reason="OOF5 정산 악화: 외부 검증 안함")
        save(PATH, result)
        return
    # 외부 검증은 고정 epoch와 OOF1~5 이력만으로 진행한다.
    norm = fit_normalizer(blocks)
    result["external"] = {}
    grouped = {}
    for name in WINDOWS:
        grouped.setdefault(fold(name)["cutoff"], []).append(name)
    for cutoff, names in grouped.items():
        models = []
        for record in result["early_stopping"]:
            model, _, _ = train_one(blocks, norm, record["seed"], device,
                                    epochs=record["best_epoch"], cutoff=cutoff)
            models.append(model)
        for name in names:
            f = fold(name)
            ctx = evaluation_context(name, gen, sites)
            p = load_reference(name, ctx["actual"], sites)
            weather = sequence_fold_features(d.loc[d.ts.between(f["val_start"], f["val_end"])], sites, cutoff)
            b = Block(p, weather, features, gfs, sites)
            pred = sum(predict(m, b, norm, device)/3 for m in models)
            base_penalty = penalty(p, p.forecast_kwh.to_numpy(), sites, smp)
            new_penalty = penalty(p, pred, sites, smp)
            result["external"][name] = dict(base_penalty=round(base_penalty), candidate_penalty=round(new_penalty),
                change_pct=100*(new_penalty/base_penalty-1))
            save(PATH, result)
            print(json.dumps({name:result["external"][name]}), flush=True)
        del models
    seasons = [result["external"][n] for n in WINDOWS if n != "fold1_static"]
    change = 100*(sum(r["candidate_penalty"] for r in seasons)/sum(r["base_penalty"] for r in seasons)-1)
    result.update(complete=True, seasonal_change_pct=change,
        selected=change <= -1 and sum(r["change_pct"] < 0 for r in seasons) >= 3 and
        max(r["change_pct"] for r in seasons) <= 3 and result["external"]["fold1_static"]["change_pct"] <= 0)
    save(PATH, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "run"], default="prepare")
    args = parser.parse_args()
    try:
        main(args.stage)
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
