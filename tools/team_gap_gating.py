"""Gap-only team-head blending policies, with frozen dead-ball probabilities."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[key]="1"
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import features2
from mtah.methods import boost,boost_train
from quality_audit import official
from decoder_development import metric
from team_context_trial import FOLDER,OLD,AUDIT


def gap_weight(seen,segments,scale,taper):
    weight=np.zeros(len(seen))
    for a,b in segments:
        if taper:
            since,until,_=features2._gap_info(seen[a:b])
            weight[a:b]=scale*(1-np.exp(-since))*(1-np.exp(-until))
        else:weight[a:b]=scale*(~seen[a:b])
    return weight


def job(item):
    cohort,match=item
    scenarios=["matched_seed0","label_blind_seed1","stress_seed1"] if cohort=="known" else ["natural_detected","label_blind_seed7"]
    candidates=json.loads((FOLDER/"gated_followup_plan.json").read_text())["candidates"]
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/"mtah/methods/v2_params.json").read_text())["decoder"])
    rows=[]
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLD/("known" if cohort=="known" else "skillcorner")/f"{match}__{scenario}__probs.npz") as z:
                pd,ph,gt=z["pd"],z["ph"],z["gt"]
                seen=z["seen"] if "seen" in z else None
            if seen is None:
                with np.load(AUDIT/"features"/f"{match}__{scenario}.npz") as z:seen=z["seen"]
            with np.load(FOLDER/"context_mix/probabilities"/f"{match}__{scenario}.npz") as z:mixed=z["ph"]
            with np.load(OLD/"robust"/f"{match}__{scenario}__probs.npz") as z:robust=z["ph"]
            segments=official.valid_runs(gt)
            for name,(source,scale,kind) in candidates.items():
                w=gap_weight(seen,segments,scale,kind!="hard")
                new=mixed if source=="mixed" else robust
                pred=boost.label(pd,w*new+(1-w)*ph,segments,len(gt),params)
                rows.append({"match":match,"cohort":cohort,"scenario":scenario,"candidate":name,**metric(gt,pred)})
    return match,rows


if __name__=="__main__":
    plan=json.loads((FOLDER/"plan.json").read_text())
    names=json.loads((AUDIT/"protocol.json").read_text())["matches"]
    output=FOLDER/"gated_results.json"
    if output.exists():raise ValueError("Completed policies already exist")
    records=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(job,j) for j in [("known",n) for n in names]+[("development",n) for n in plan["development_skillcorner"]]]):
            name,rows=future.result();records.extend(rows);print(name,"scored",flush=True)
    output.write_text(json.dumps({"records":records},indent=2)+"\n")
    print("PASS",output,flush=True)
