"""Isolate robust team probabilities; retain original stacked dead-ball head."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[k]="1"
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from mtah.methods import boost,boost_train
from quality_audit import official
from decoder_development import metric
from team_context_trial import FOLDER,OLD,AUDIT


def job(item):
    cohort,match=item
    scenarios=["matched_seed0","label_blind_seed1","stress_seed1"] if cohort=="known" else ["natural_detected","label_blind_seed7"]
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/"mtah/methods/v2_params.json").read_text())["decoder"])
    records=[]
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLD/("known" if cohort=="known" else "skillcorner")/f"{match}__{scenario}__probs.npz") as z:pd,ph,gt=z["pd"],z["ph"],z["gt"]
            with np.load(OLD/"robust"/f"{match}__{scenario}__probs.npz") as z:raw=z["ph"]
            for name,weight in [("robust_team25",.25),("robust_team50",.5),("robust_team100",1.)]:
                pred=boost.label(pd,weight*raw+(1-weight)*ph,official.valid_runs(gt),len(gt),params)
                records.append({"match":match,"cohort":cohort,"scenario":scenario,"candidate":name,**metric(gt,pred)})
    return match,records


if __name__=="__main__":
    names=json.loads((AUDIT/"protocol.json").read_text())["matches"]
    dev=json.loads((FOLDER/"plan.json").read_text())["development_skillcorner"]
    output=FOLDER/"robust_team_results.json"
    if output.exists():raise ValueError("Do not overwrite completed comparison")
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(job,j) for j in [("known",n) for n in names]+[("development",n) for n in dev]]):
            name,rows=future.result();records.extend(rows);print(name,"scored",flush=True)
    output.write_text(json.dumps({"records":records},indent=2)+"\n")
    print("PASS",output,flush=True)
