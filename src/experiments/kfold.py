"""K-Fold 앙상블 비교(2026-10-17). v4 설정에서 학습 데이터를 날짜 단위 K조각으로 나눠 한 조각씩 뺀 K개 모델의 평균과,
같은 계산량으로 전체 데이터를 쓰고 시드만 바꾼 K개 모델의 평균을 비교한다.

같은 날의 발전소들은 같은 구름 아래에 있으므로 행이 아니라 날짜로 나눈다(행 단위로 섞으면 같은 날이 양쪽에 들어간다).

    python -m src.experiments.kfold [fold ...]
"""
import json
import sys

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR
from ..data import load_sites, load_smp
from ..scoring import composition_penalty, lower_bound, residual_matrix
from ..splits import fold as get_fold

K = 5
CFG = {**T.DEFAULTS, **dict(feats="fullc", target="kt", outage="drop", weight="cap", train_start="2021-01-01")}
OFFICIAL = json.loads((EXP_DIR / "p4_official_baseline" / "metrics.json").read_text())


def penalty(pred: pd.DataFrame, f: dict, smp: pd.Series) -> float:
    if "gen" not in T._cache:
        T._cache["gen"] = T.load_generation()
    gen = T._cache["gen"]
    E = residual_matrix(T.night_filled(pred, gen[gen["ts"].between(f["val_start"], f["val_end"])]))
    w = smp.reindex(E.index).to_numpy()
    sites = load_sites().set_index("site_id").loc[E.columns]
    return composition_penalty(E.to_numpy(), w, (sites["current_group"] == "A").to_numpy())


def main(folds) -> None:
    d = T.load_train_features()
    feats = T.select_features(d, CFG["feats"])
    smp = load_smp("train").set_index("ts")["w"]
    result = {}
    for fold in folds:
        f = get_fold(fold)
        train = d[(d["ts"] >= pd.Timestamp(CFG["train_start"] + " 01:00", tz="Asia/Seoul")) & (d["ts"] <= f["cutoff"])]
        train = T.drop_outage(train, CFG)
        val = d[d["ts"].between(f["val_start"], f["val_end"])]
        cs = T.clear_sky(train["cosz"])
        y = ((train["energy"] / train["cap"]).clip(0, 1.2) / cs).clip(0, 2.0)
        sw = T.sample_weight(train, CFG["weight"], smp) * cs / cs.mean()
        scale = T.clear_sky(val["cosz"]) * val["cap"].to_numpy()
        day = (train["ts"] - pd.Timedelta(hours=1)).dt.normalize()
        days = day.unique()
        part = pd.Series(np.random.default_rng(2026).permutation(len(days)) % K, index=days)
        piece = day.map(part).to_numpy()

        preds = {"kfold": [], "seeds": []}
        for i in range(K):
            m = piece != i                                     # i번째 날짜 조각을 뺀 학습
            model = T.fit_model({**CFG, "seed": 2026 + i}, train.loc[m, feats], y[m], sw[m])
            preds["kfold"].append(model.predict(val[feats]))
            model = T.fit_model({**CFG, "seed": 2026 + i}, train[feats], y, sw)
            preds["seeds"].append(model.predict(val[feats]))
            print(f"  {fold} 모델 {i + 1}/{K} 완료", flush=True)

        off = OFFICIAL[fold]["scores"]["official"]["penalty"]
        res = {"official_penalty": off}
        for name, n in (("seed1", 1), ("seeds3", 3), ("seeds5", K), ("kfold5", K)):
            src = preds["kfold" if name == "kfold5" else "seeds"][:n]
            pred = val[["site_id", "ts"]].copy()
            pred["forecast_kwh"] = np.clip(np.mean(src, axis=0), 0, None) * scale
            p = penalty(pred, f, smp)
            res[name] = dict(penalty=round(p), normalized=round(100 * (1 - p / off), 2))
        # 각 K-Fold 모델 단독(데이터 80%)의 평균 성능: 데이터를 덜 쓰는 손해를 따로 본다
        singles = []
        for pk in preds["kfold"]:
            pred = val[["site_id", "ts"]].copy()
            pred["forecast_kwh"] = np.clip(pk, 0, None) * scale
            singles.append(penalty(pred, f, smp))
        res["kfold_single_mean"] = dict(penalty=round(np.mean(singles)), normalized=round(100 * (1 - np.mean(singles) / off), 2))
        result[fold] = res
        print(fold, {k: v for k, v in res.items() if k != "official_penalty"}, flush=True)
    out = EXP_DIR / "explain"
    (out / "kfold.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"저장: {out / 'kfold.json'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["development", "selection"])
