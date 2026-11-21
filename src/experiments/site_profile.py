"""발전소 고유 편향 및 정상 출력 특성 진단. 집계만 출력·저장.

    .venv/bin/python -m src.experiments.site_profile

과거 OOF 편향의 미래 집합 페널티 이득이 없으면 모델 학습으로 넘어가지 않는다.
"""
import json

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, EXP_DIR, PRED_DIR
from ..correct_total import HISTORY
from ..data import load_generation, load_sites, load_smp
from ..diagnose_season import FOLDS, run_id
from ..direct_groups import metrics
from ..splits import fold
from ..train import clear_sky, outage_mask

END=pd.Timestamp("2025-05-01",tz="Asia/Seoul")
PAIRS=(("janfeb_2023","janfeb_2024"),("janfeb_2024","development"),
       ("spring_2023","spring_2024"),("spring_2024","selection"),
       ("summer_2023","summer_support"),("autumn_2022","autumn_2023"),
       ("autumn_2023","autumn_support"))


def distribution(x):
    x=pd.Series(x).replace([np.inf,-np.inf],np.nan).dropna()
    return dict(count=len(x),quantiles={str(q):round(float(x.quantile(q)),5) for q in (.1,.25,.5,.75,.9)}) if len(x) else dict(count=0)


def isolate(p):
    """실적−예측에서 본인을 제외한 동료들의 청천 정규화 공통 오차를 뺀다."""
    p=p[p.cosz>.3].copy()
    p["scale"]=p.cap*clear_sky(p.cosz)
    p["error"]=p.energy-p.forecast_kwh
    g=p.groupby(["station","ts"])
    n=g.site_id.transform("size")-1
    peer=(g.error.transform("sum")-p.error)/(g.scale.transform("sum")-p.scale)
    allg=p.groupby("ts")
    fleet=(allg.error.transform("sum")-p.error)/(allg.scale.transform("sum")-p.scale)
    p["unique_error"]=p.error-p.scale*peer.where(n>=2,fleet)
    p["day"]=(p.ts-pd.Timedelta(hours=1)).dt.normalize()
    return p


def biases(p):
    days=p.groupby("site_id").day.nunique()
    denom=(p.w*p.forecast_kwh).groupby(p.site_id).sum().replace(0,np.nan)
    raw=(p.w*p.error).groupby(p.site_id).sum()/denom
    unique=(p.w*p.unique_error).groupby(p.site_id).sum()/denom
    return pd.DataFrame(dict(raw=raw,unique=unique,days=days)).replace([np.inf,-np.inf],np.nan).dropna()


def agreement(a,b,column,min_days):
    a=a[a.days>=min_days][column]
    b=b[b.days>=min_days][column]
    j=pd.concat([a.rename("a"),b.rename("b")],axis=1).dropna()
    if len(j)<3:
        return dict(sites=len(j))
    corr=float(j.a.corr(j.b,method="spearman"))
    r=dict(sites=len(j),spearman=round(corr,4) if np.isfinite(corr) else None)
    if column not in ("raw","unique"):
        return r
    material=(j.a.abs()>.005)&(j.b.abs()>.005)
    r.update(
                material_sites=int(material.sum()),
                sign_agreement_pct=round(float(100*(np.sign(j.loc[material,"a"])==np.sign(j.loc[material,"b"])).mean()),2) if material.any() else None)
    return r


def apply_past(p,beta):
    out=p.copy()
    factor=1+out.site_id.map(beta).fillna(0).to_numpy()
    out["forecast_kwh"]=np.maximum(out.forecast_kwh.to_numpy()*factor,0)
    return out


def compare(p,corrected,smp):
    a=p.groupby(["ts","group"]).energy.sum()
    b=p.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
    c=corrected.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
    return metrics(a,b,c,smp)


def profile(d):
    x=d[(d.cosz>.3)&(d.wa_cloud<=30)&(d.wb_cloud<=30)].copy()
    x["kt"]=x.energy.clip(lower=0)/(x.cap*clear_sky(x.cosz))
    x["day"]=(x.ts-pd.Timedelta(hours=1)).dt.normalize()
    g=x.groupby("site_id")
    out=pd.DataFrame(dict(upper=g.kt.quantile(.9),days=g.day.nunique()))
    for label,mask in (("morning",x.azim<150),("midday",x.azim.between(150,210)),("afternoon",x.azim>210)):
        s=x[mask].groupby("site_id")
        q=s.kt.quantile(.9)
        n=s.day.nunique()
        out[label]=q.where(n>=20)/out.upper
    return out


def main():
    print("발전소 고유 편향→미래 집합 손익→정상 출력 특성, holdout 제외",flush=True)
    sites=load_sites().set_index("site_id")
    gen=load_generation()
    gen=gen[gen.ts<=END]
    d=pd.read_parquet(CACHE_DIR/"train_features.parquet",columns=["site_id","ts","cap","cosz","azim","wa_cloud","wb_cloud"])
    d=d[d.ts<=END]
    smp=load_smp("train").set_index("ts")
    panels,isolated,stats={},{},{}
    for name in HISTORY+FOLDS+("fold1_static",):
        f=fold(name)
        assert f["val_end"]<=END
        p=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
        count=len(p)
        p=p.merge(gen,on=["site_id","ts"],validate="one_to_one").merge(d,on=["site_id","ts"],how="left",validate="one_to_one")
        if len(p)!=count:
            raise ValueError("예측과 실적 키 불일치")
        p["cap"]=p.site_id.map(sites.capacity_kw)
        p["station"]=p.site_id.map(sites.wa_station)
        p["group"]=p.site_id.map(sites.current_group)
        p["w"]=p.ts.map(smp.w)
        panels[name]=p
        isolated[name]=isolate(p)
        stats[name]=biases(isolated[name])
        print(f"편향 집계 {name}: {len(stats[name])}개소",flush=True)
    result=dict(bias_pairs={},past_only={},bias_distribution={})
    for a,b in PAIRS:
        result["bias_pairs"][a+"->"+b]={c:agreement(stats[a],stats[b],c,30) for c in ("raw","unique")}
    for n,s in stats.items():
        result["bias_distribution"][n]={c:distribution(s.loc[s.days>=30,c]) for c in ("raw","unique")}
    for name in FOLDS+("fold1_static",):
        eligible=[n for n in HISTORY+FOLDS if fold(n)["val_end"]<=fold(name)["cutoff"]]
        past=biases(pd.concat([isolated[n] for n in eligible],ignore_index=True))
        beta=(past.unique.clip(-.1,.1)*past.days/(past.days+60)).where(past.days>=30,0)
        r=compare(panels[name],apply_past(panels[name],beta),smp)
        r["calibration_folds"]=eligible
        r["factor_distribution"]=distribution(1+beta)
        result["past_only"][name]=r
        print(f"과거 고유 편향 적용 {name}: 페널티 {r['change_pct']:+.3f}%",flush=True)

    # 정상 출력 프로필은 같은 달의 다른 해끼리 비교한다. 예보로만 맑음을 정의한다.
    diagnostic=d.merge(gen,on=["site_id","ts"],validate="one_to_one")
    bad=outage_mask(diagnostic)
    profiles,clean_profiles={},{}
    result["profiles"]={}
    for name in HISTORY+FOLDS:
        f=fold(name)
        m=diagnostic.ts.between(f["val_start"],f["val_end"])
        profiles[name]=profile(diagnostic[m])
        clean_profiles[name]=profile(diagnostic[m&~bad])
        a,b=profiles[name],clean_profiles[name]
        delta=(b.upper/a.upper-1).where((a.days>=20)&(b.days>=20))
        result["profiles"][name]=dict(upper=distribution(a.loc[a.days>=20,"upper"]),
                outage_removed_upper_change=distribution(delta),
                sites_upper_change_over_half_pct=int((delta.abs()>.005).sum()),
                orientation={c:distribution(a.loc[a.days>=20,c]) for c in ("morning","midday","afternoon")})
    result["profile_pairs"]={a+"->"+b:{c:agreement(clean_profiles[a],clean_profiles[b],c,20) for c in ("upper","morning","midday","afternoon")} for a,b in PAIRS}
    pairs=[x["unique"] for x in result["bias_pairs"].values()]
    repeat=sum(x.get("sites",0)>=50 and (x.get("spearman") or -1)>=.3 and (x.get("sign_agreement_pct") or 0)>=60 for x in pairs)
    rows=[result["past_only"][n] for n in FOLDS]
    change=100*(sum(x["direct_penalty"] for x in rows)/sum(x["base_penalty"] for x in rows)-1)
    economic=(change<=-1 and sum(x["change_pct"]<0 for x in rows)>=3 and max(x["change_pct"] for x in rows)<=3 and result["past_only"]["fold1_static"]["change_pct"]<=-.5)
    upper_repeat=sum(x["upper"].get("sites",0)>=50 and (x["upper"].get("spearman") or -1)>=.5 for x in result["profile_pairs"].values())
    result["verdict"]=dict(unique_bias_repeating_pairs=int(repeat),seasonal_change_pct=round(change,3),
                           economic_gate=bool(economic),upper_profile_repeating_pairs=int(upper_repeat),
                           proceed_normalization_model=bool(repeat>=2 and economic and upper_repeat>=2))
    out=EXP_DIR/"p15_site_profile"
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("판정",json.dumps(result["verdict"],ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
