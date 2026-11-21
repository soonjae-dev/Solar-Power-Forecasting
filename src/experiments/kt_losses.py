"""현재 v3 청천 지수 표현에서 L1/Huber 손실 비교. 집계만 저장.

    .venv/bin/python -m src.experiments.kt_losses
"""
import json

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR
from ..data import load_generation,load_sites,load_smp
from ..tune_v3 import END,run_stage,save
from ..weather_teacher import CFG

LOSSES={"kt_l1":{"objective":"l1"},"kt_huber":{"objective":"huber","alpha":.1}}


def main():
    print("현재v3 청천 지수 타깃에서 L1/Huber 비교, holdout 제외",flush=True)
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    gen=load_generation()
    gen=gen[gen.ts<=END]
    d=T.load_train_features()
    cols=["site_id","ts","energy",*T.select_features(d,CFG["feats"])]
    d=d.loc[d.ts<=END,cols].copy()
    T._cache.clear()
    out=EXP_DIR/"p20_kt_losses"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"metrics.json"
    if path.exists():
        result=json.loads(path.read_text())
        if result["baseline_config"]!=CFG or result["loss_candidates"]!=LOSSES or result["cache_name"]!=CACHE_DIR.name:
            raise ValueError("체크포인트 설정이 현재 계획과 다름")
    else:
        result=dict(baseline_config=CFG,loss_candidates=LOSSES,cache_name=CACHE_DIR.name)
    eligible=run_stage(d,gen,sites,smp,result,path,1,list(LOSSES),variants=LOSSES)
    if eligible:
        finalists=run_stage(d,gen,sites,smp,result,path,3,eligible,variants=LOSSES)
        result["selected_candidate"]=min(finalists,key=lambda c:result["seeds3"][c]["fold1_static"]["vs_v3_3seeds"]["direct_penalty"]) if finalists else None
    else:
        result["selected_candidate"]=None
    save(path,result)
    print("최종 후보",result["selected_candidate"],flush=True)


if __name__=="__main__":
    main()
