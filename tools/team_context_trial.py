"""External-only team-context fitting and development comparisons against v2."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import features2,degrade,possession_context as context
from mtah.methods import boost,boost_train
from quality_audit import official,sha,read_feature
from decoder_development import metric

FOLDER=ROOT/"reports/v3_search"
OLD=ROOT/"reports/v2_search"
AUDIT=ROOT/"reports/quality_audit/380987b3029c"
BLIND={"mode":"v1","skip_ball_if_seen_below":0.,"ball_seen_target":.4,"ball_gap_mean_s":6.,
       "frag_mean_s":107.,"frag_gap_mean_s":6.,"pos_noise_m":.3,"speed_noise":.6}


def prepare(name):
    path=FOLDER/"features"/(name+".npz")
    if path.exists():return name
    f=read_feature(AUDIT,name)
    names=f["cols"].tolist();segments=official.valid_runs(f["gt"])
    x,cols=context.extend(f["X"],names,segments);xs,_=context.extend(f["Xs"],names,segments)
    path.parent.mkdir(exist_ok=True)
    np.savez(path,X=x,Xs=xs,gt=f["gt"],cols=np.array(cols))
    return name


def fit_job(job):
    names,excluded=job
    directory=FOLDER/"models";directory.mkdir(exist_ok=True)
    stem=directory/("exclude_"+excluded if excluded else "all_nine")
    train=[n for n in names if n!=excluded]
    provenance={"train":train,"excluded":excluded,"plan_sha256":sha(FOLDER/"plan.json"),
                "context_source_sha256":sha(ROOT/"mtah/possession_context.py")}
    if stem.with_suffix(".json").exists():
        meta=json.loads(stem.with_suffix(".json").read_text())
        if any(meta[k]!=v for k,v in provenance.items()) or sha(stem.with_suffix(".pkl"))!=meta["model_sha256"]:
            raise ValueError("Team-context training cache mismatch")
        return excluded or "all_nine"
    rows=[]
    with threadpool_limits(1):
        for name in train:
            with np.load(FOLDER/"features"/(name+".npz"),allow_pickle=False) as z:
                rows.append(context.rows(z["X"],z["Xs"],z["gt"]))
        model=context.fit(rows)
    with stem.with_suffix(".pkl").open("wb") as stream:cloudpickle.dump(model,stream)
    provenance.update(model_sha256=sha(stem.with_suffix(".pkl")),fitted_utc=datetime.now(timezone.utc).isoformat())
    stem.with_suffix(".json").write_text(json.dumps(provenance,indent=2)+"\n")
    return excluded or "all_nine"


def load_model(match=None):
    stem=FOLDER/"models"/("exclude_"+match if match else "all_nine")
    meta=json.loads(stem.with_suffix(".json").read_text())
    if sha(stem.with_suffix(".pkl"))!=meta["model_sha256"] or (match and match in meta["train"]):
        raise ValueError("Test match leaked or weights changed")
    with stem.with_suffix(".pkl").open("rb") as stream:return cloudpickle.load(stream)


def score_job(job):
    cohort,match=job
    plan=json.loads((FOLDER/"plan.json").read_text())
    model=load_model(match if cohort=="known" else None)
    scenarios=["matched_seed0","label_blind_seed1","stress_seed1"] if cohort=="known" else ["natural_detected","label_blind_seed7"]
    records=[]
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/"mtah/methods/v2_params.json").read_text())["decoder"])
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLD/("known" if cohort=="known" else "skillcorner")/f"{match}__{scenario}__probs.npz") as z:
                pd,ph,gt=z["pd"],z["ph"],z["gt"]
            dest=FOLDER/"probabilities"/f"{match}__{scenario}.npz";dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                with np.load(dest) as z:new_ph=z["ph"]
            else:
                if cohort=="known":
                    f=read_feature(AUDIT,match,scenario)
                    x,names=f["X"],f["cols"].tolist();xs=f["Xs"]
                else:
                    with np.load(OLD/"skillcorner"/(match+".npz")) as z:clean={k:z[k] for k in z.files}
                    d=clean if scenario=="natural_detected" else degrade.degrade(clean,seed=7,params=BLIND)
                    x,names=features2.build(d,official.valid_runs(gt));xs,_=features2.build(d,official.valid_runs(gt),swap=True)
                x,_=context.extend(x,names,official.valid_runs(gt));xs,_=context.extend(xs,names,official.valid_runs(gt))
                new_ph=context.predict(model,x,xs)
                np.savez(dest,ph=new_ph)
            for candidate,weight in {"v2":0.,**plan["candidates"]}.items():
                blended=weight*new_ph+(1-weight)*ph
                labels=boost.label(pd,blended,official.valid_runs(gt),len(gt),params)
                records.append({"match":match,"scenario":scenario,"candidate":candidate,**metric(gt,labels)})
    return match,records


def run_pool(function,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(function,j) for j in jobs]):yield future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["fit","known","development"])
    parser.add_argument("--jobs",type=int,choices=range(1,5),default=4)
    args=parser.parse_args()
    plan=json.loads((FOLDER/"plan.json").read_text())
    if sha(ROOT/"mtah/possession_context.py")!=plan["source_sha256"]:
        raise ValueError("Context features changed after plan registration")
    names=json.loads((AUDIT/"protocol.json").read_text())["matches"]
    if args.mode=="fit":
        for name in run_pool(prepare,names,args.jobs):print(name,"features ready",flush=True)
        for name in run_pool(fit_job,[(names,n) for n in [None]+names],args.jobs):print(name,"trained",flush=True)
        return
    destination=FOLDER/(args.mode+"_results.json")
    if destination.exists():raise ValueError("Preserve completed results; do not silently rescore")
    records=[]
    jobs=[(args.mode,n) for n in (names if args.mode=="known" else plan["development_skillcorner"])]
    for name,rows in run_pool(score_job,jobs,args.jobs):records.extend(rows);print(name,"scored",flush=True)
    destination.write_text(json.dumps({"cohort":args.mode,"plan_sha256":sha(FOLDER/"plan.json"),"records":records},indent=2)+"\n")
    print("PASS",destination,flush=True)


if __name__=="__main__":main()
