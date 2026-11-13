"""Context-head sensor augmentation, fitted on original external games only."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
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
from mtah import possession_context as context
from quality_audit import official,read_feature,sha
import team_context_trial as trial

FOLDER=ROOT/"reports/v3_search/context_mix"
SCENARIOS=["matched_seed0","label_blind_seed1","clean_source"]


def fit_job(item):
    names,excluded=item
    directory=FOLDER/"models";directory.mkdir(exist_ok=True)
    stem=directory/("exclude_"+excluded if excluded else "all_nine")
    train=[n for n in names if n!=excluded]
    meta={"train":train,"excluded":excluded,"plan_sha256":sha(FOLDER/"plan.json"),"scenarios":SCENARIOS,"stride_per_scenario":12,"context_source_sha256":sha(ROOT/"mtah/possession_context.py")}
    if stem.with_suffix(".json").exists():
        old=json.loads(stem.with_suffix(".json").read_text())
        if any(old[k]!=v for k,v in meta.items()) or sha(stem.with_suffix(".pkl"))!=old["model_sha256"]:raise ValueError("Mixture cache differs")
        return excluded or "all_nine"
    rows=[]
    with threadpool_limits(1):
        for name in train:
            for scenario in SCENARIOS:
                f=read_feature(trial.AUDIT,name,scenario);segments=official.valid_runs(f["gt"])
                x,_=context.extend(f["X"],f["cols"].tolist(),segments)
                xs,_=context.extend(f["Xs"],f["cols"].tolist(),segments)
                rows.append(context.rows(x,xs,f["gt"],stride=12))
                del x,xs,f
        model=context.fit(rows)
    with stem.with_suffix(".pkl").open("wb") as stream:cloudpickle.dump(model,stream)
    meta.update(model_sha256=sha(stem.with_suffix(".pkl")),fitted_utc=datetime.now(timezone.utc).isoformat())
    stem.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    return excluded or "all_nine"


def score_job(item):
    trial.FOLDER=FOLDER
    return trial.score_job(item)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["fit","known","development"]);parser.add_argument("--jobs",type=int,choices=range(1,5),default=4);args=parser.parse_args()
    names=json.loads((trial.AUDIT/"protocol.json").read_text())["matches"]
    plan=json.loads((FOLDER/"plan.json").read_text())
    if args.mode=="fit":function=fit_job;jobs=[(names,n) for n in [None]+names]
    else:function=score_job;jobs=[(args.mode,n) for n in (names if args.mode=="known" else plan["development_skillcorner"])]
    records=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(function,j) for j in jobs]):
            result=future.result()
            if args.mode=="fit":print(result,"trained",flush=True)
            else:name,rows=result;records.extend(rows);print(name,"scored",flush=True)
    if args.mode!="fit":
        output=FOLDER/(args.mode+"_results.json")
        if output.exists():raise ValueError("Do not overwrite completed results")
        output.write_text(json.dumps({"records":records},indent=2)+"\n")
