"""144특징 gfs60에서 사전 고정 설정 두 개만 비교. 집계만 저장."""
import json

import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..gfs_more import load_more
from ..gfs_pca import three_seed_verdict, verdict
from ..gfs_v3 import checked_gfs
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG

VARIANTS = {"all_columns": {"colsample_bytree": 1.0},
            "slow1200": {"learning_rate": 0.03, "n_estimators": 1200}}


def main():
    print("gfs60 설정2개 검증, 2025-05 제외", flush=True)
    gfs, _ = checked_gfs()
    common = pd.concat([gfs, load_more()[0]], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p39_gfs_retune"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    signature = dict(baseline_config=CFG, candidates=VARIANTS, cache_name=CACHE_DIR.name,
                     control="gfs60", rule="14.31", common_columns=list(common))
    result = json.loads(path.read_text()) if path.exists() else dict(signature)
    if any(result.get(k) != v for k, v in signature.items()):
        raise ValueError("체크포인트 설정 불일치")
    if result.get("early_rejected_all"):
        print("두 후보 모두 조기 기각됨, 재학습하지 않음", flush=True)
        return
    rejected = result.get("early_rejected_candidates", {})
    active = [n for n in VARIANTS if n not in rejected]
    if not active:
        print("검증할 후보 없음", flush=True)
        return
    kwargs = dict(variants=VARIANTS, added=common, control_columns=list(common))
    run_stage(d, gen, sites, smp, result, path, 1, active, **kwargs)
    prior = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())
    for name in VARIANTS:
        records = result["seeds1"][name]
        if any(r["vs_matched_v3"]["base_penalty"] !=
               prior["seeds1"]["gfs60"][n]["vs_v3_3seeds"]["direct_penalty"] for n, r in records.items()):
            raise ValueError("동일1시드 대조군 재현 불일치")
    result["verdicts_vs_control"] = {
        name: rejected[name] if name in rejected else verdict(result["seeds1"][name]) for name in VARIANTS}
    save(path, result)
    print("1시드 판정", json.dumps(result["verdicts_vs_control"], ensure_ascii=False), flush=True)
    finalists = [n for n, v in result["verdicts_vs_control"].items() if v["passed"]]
    if finalists:
        run_stage(d, gen, sites, smp, result, path, 3, finalists, **kwargs)
        result["verdicts3_vs_control"] = {
            n: three_seed_verdict(result["seeds3"][n], prior["seeds3"]["gfs60"]) for n in finalists}
        save(path, result)
        print("3시드 판정", json.dumps(result["verdicts3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
