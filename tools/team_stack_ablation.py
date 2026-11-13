"""Control: bypass only the old team stack, keeping dead-ball refinement fixed."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import features2,degrade
from mtah.methods import boost,boost_train
from quality_audit import official,probabilities
from decoder_development import metric
from team_context_trial import FOLDER,OLD,AUDIT,BLIND


def job(item):
    cohort,match=item
    scenarios=["matched_seed0","label_blind_seed1","stress_seed1"] if cohort=="known" else ["natural_detected","label_blind_seed7"]
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/"mtah/methods/v2_params.json").read_text())["decoder"])
    if cohort!="known":
        with (ROOT/"out/training_preflight.pkl").open("rb") as stream:model=cloudpickle.load(stream)["stage1"]
    rows=[]
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLD/("known" if cohort=="known" else "skillcorner")/f"{match}__{scenario}__probs.npz") as z:
                pd,ph,gt=z["pd"],z["ph"],z["gt"]
            if cohort=="known":_,raw=probabilities(AUDIT,(match,),match,scenario)
            else:
                with np.load(OLD/"skillcorner"/(match+".npz")) as z:clean={k:z[k] for k in z.files}
                d=clean if scenario=="natural_detected" else degrade.degrade(clean,seed=7,params=BLIND)
                x,_=features2.build(d,official.valid_runs(gt));xs,_=features2.build(d,official.valid_runs(gt),swap=True)
                _,raw=boost.predict(model,x,xs)
            for name,weight in [("base_team25",.25),("base_team50",.5),("base_team100",1.)]:
                labels=boost.label(pd,weight*raw+(1-weight)*ph,official.valid_runs(gt),len(gt),params)
                rows.append({"match":match,"cohort":cohort,"scenario":scenario,"candidate":name,**metric(gt,labels)})
    return match,rows


if __name__=="__main__":
    plan=json.loads((FOLDER/"plan.json").read_text())
    names=json.loads((AUDIT/"protocol.json").read_text())["matches"]
    jobs=[("known",n) for n in names]+[("development",n) for n in plan["development_skillcorner"]]
    destination=FOLDER/"ablation_results.json"
    if destination.exists():raise ValueError("Ablations already scored")
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(job,x) for x in jobs]):name,records=future.result();rows.extend(records);print(name,"scored",flush=True)
    destination.write_text(json.dumps({"records":rows},indent=2)+"\n")
    print("PASS",destination,flush=True)
