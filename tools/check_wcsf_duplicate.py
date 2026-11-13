"""Local raw-ball screening only; never load competition reference labels."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mtah import atd
from mtah.config import HALVES
from quality_audit import sha


def screen(a,b):
    best=None
    for lag in range(-180,181):
        i=np.arange(1800);j=i+lag;valid=(j>=0)&(j<len(b));x=a[i[valid]];y=b[j[valid]]
        for sx,sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            xx=x*np.array([sx,sy]);ok=np.isfinite(xx).all(1)&np.isfinite(y).all(1)
            if ok.sum()<300:continue
            median=float(np.median(np.linalg.norm(xx[ok]-y[ok],axis=1)))
            if best is None or median<best['median_distance_m']:best={'median_distance_m':median,'lag_seconds':lag,'paired_observed_seconds':int(ok.sum())}
    if best is None:raise ValueError('Not enough comparable observed tracking')
    return dict(best,potential_duplicate=best['median_distance_m']<8)


if __name__=='__main__':
    with np.load(ROOT/'reports/temporal/external/wcsf_geometry.npz') as z:a=(z['ball'][::25,:2]-.5)*z['pitch']
    targets={}
    for name in ['metrica_1','metrica_2']:
        with np.load(ROOT/'external/conv'/(name+'.npz')) as z:targets[name]=(z['ball'][::25,:2]-.5)*z['pitch']
    from fresh_metrica_holdout import parse_tracking
    source=ROOT/'external/holdout_metrica3'
    m3,_,_=parse_tracking(source/'Sample_Game_3_metadata.xml',source/'Sample_Game_3_tracking.txt')
    targets['metrica_3']=(m3['ball'][::25,:2]-.5)*np.asarray(m3['pitch'])
    d=atd.load();start=HALVES[0][1];targets['competition_automatic_tracking_only']=(d['ball'][start::25,:2]-.5)*np.array([105.3,68.])
    result={'purpose':'Heuristic eligibility screening, not an identity proof; no GT or model scores','rule':'Flag median ball-trajectory distance<8m across>=300 observed seconds; first30min, +/-180s, four axis reflections','comparisons':{name:screen(a,b) for name,b in targets.items()}}
    result['eligible_for_anonymous_research_diagnostic']=not any(v['potential_duplicate'] for v in result['comparisons'].values())
    out=ROOT/'docs/experiments/temporal';(out/'wcsf_eligibility.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
