"""Fit prespecified sensor-mixture stage one; test only external games."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import degrade, features2
from mtah.methods import boost, boost_train
from quality_audit import SCENARIOS, read_feature, official, sha
from decoder_development import FOLDER, AUDIT, metric

AUGMENTATIONS = ["matched_seed0", "label_blind_seed1", "clean_source"]
BLIND = {"mode":"v1", "skip_ball_if_seen_below":0., "ball_seen_target":.4,
         "ball_gap_mean_s":6., "frag_mean_s":107., "frag_gap_mean_s":6.,
         "pos_noise_m":.3, "speed_noise":.6}


def fit_job(job):
    names, excluded = job
    folder = FOLDER / "robust"
    folder.mkdir(exist_ok=True)
    stem = folder / ("exclude_" + excluded if excluded else "all_nine")
    model_path = stem.with_suffix(".pkl")
    manifest_path = stem.with_suffix(".json")
    train = [n for n in names if n != excluded]
    provenance = {"train":train,"excluded":excluded,"augmentation":AUGMENTATIONS,
                  "stride_per_scenario":12,"training_plan_sha256":sha(FOLDER/"training_plan.json"),
                  "model_parameters":boost.MODEL}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(k) != v for k,v in provenance.items()) or sha(model_path) != existing["model_sha256"]:
            raise ValueError("Robust-model cache provenance mismatch")
        return excluded or "all_nine", existing
    with threadpool_limits(1):
        rows = []
        for name in train:
            for scenario in AUGMENTATIONS:
                f = read_feature(AUDIT, name, scenario)
                rows.append(boost.training_rows(f["X"],f["Xs"],f["gt"],stride=12))
        models = boost.fit(rows)
    with model_path.open("wb") as stream:
        cloudpickle.dump(models,stream)
    provenance.update(model_sha256=sha(model_path),fitted_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(provenance,indent=2)+"\n")
    return excluded or "all_nine",provenance


def robust_score(job):
    cohort, match = job
    stem = FOLDER / "robust" / ("exclude_"+match if cohort=="known" else "all_nine")
    provenance=json.loads(stem.with_suffix(".json").read_text())
    if sha(stem.with_suffix(".pkl"))!=provenance["model_sha256"]:
        raise ValueError("Robust weights failed integrity check")
    if cohort=="known" and (provenance["excluded"]!=match or match in provenance["train"]):
        raise ValueError("Known test game entered robust training")
    with stem.with_suffix(".pkl").open("rb") as stream:
        models = cloudpickle.load(stream)
    specs = [("robust_stage1",1.),("blend50",.5)]
    if cohort == "confirmation":
        selection = json.loads((FOLDER/"selection.json").read_text())
        specs = [s for s in specs if s[0]==selection["candidate"]]
    scenarios = ["matched_seed0","label_blind_seed1","stress_seed1"] if cohort=="known" else ["natural_detected","label_blind_seed7"]
    records=[]
    with threadpool_limits(1):
        for scenario in scenarios:
            destination=FOLDER/"robust"/f"{match}__{scenario}__probs.npz"
            old_folder=FOLDER/("known" if cohort=="known" else "skillcorner")
            with np.load(old_folder/f"{match}__{scenario}__probs.npz",allow_pickle=False) as z:
                qd,qh,gt=z["pd"],z["ph"],z["gt"]
            if destination.exists():
                with np.load(destination,allow_pickle=False) as z: pd,ph=z["pd"],z["ph"]
            else:
                if cohort=="known":
                    f=read_feature(AUDIT,match,scenario)
                    pd,ph=boost.predict(models,f["X"],f["Xs"])
                else:
                    with np.load(FOLDER/"skillcorner"/(match+".npz"),allow_pickle=False) as z:
                        clean={k:z[k] for k in z.files}
                    d=clean if scenario=="natural_detected" else degrade.degrade(clean,seed=7,params=BLIND)
                    segments=official.valid_runs(gt)
                    x,_=features2.build(d,segments);xs,_=features2.build(d,segments,swap=True)
                    pd,ph=boost.predict(models,x,xs)
                np.savez(destination,pd=pd,ph=ph)
            for candidate,weight in specs:
                dead=weight*pd+(1-weight)*qd
                home=weight*ph+(1-weight)*qh
                labels=boost.label(dead,home,official.valid_runs(gt),len(gt),boost_train.decoder_params())
                records.append({"match":match,"scenario":scenario,"candidate":candidate,**metric(gt,labels)})
    return match,records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["fit","known","development","confirmation"])
    parser.add_argument("--jobs",type=int,choices=range(1,5),default=4)
    args=parser.parse_args()
    names=json.loads((AUDIT/"protocol.json").read_text())["matches"]
    plan=json.loads((FOLDER/"plan.json").read_text())
    if args.mode=="fit":
        function=fit_job;jobs=[(names,n) for n in [None]+names]
    else:
        if args.mode=="confirmation":
            selected=json.loads((FOLDER/"selection.json").read_text())
            if selected["plan_sha256"]!=sha(FOLDER/"plan.json") or selected["training_plan_sha256"]!=sha(FOLDER/"training_plan.json"):
                raise ValueError("Frozen selection provenance mismatch")
            if sha(FOLDER/"robust/all_nine.pkl")!=selected["robust_model_sha256"]:
                raise ValueError("Selected robust weights changed after freezing")
        function=robust_score
        jobs=[(args.mode,n) for n in (names if args.mode=="known" else plan[args.mode])]
    destination=FOLDER/(args.mode+"_robust_results.json")
    if destination.exists():
        raise ValueError("Completed experiment already exists; preserve its results")
    records=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=get_context("spawn")) as pool:
        futures=[pool.submit(function,j) for j in jobs]
        for future in as_completed(futures):
            name,rows=future.result()
            if args.mode!="fit":records.extend(rows)
            print(name,"completed",flush=True)
    destination.write_text(json.dumps({"finished_utc":datetime.now(timezone.utc).isoformat(),
                                      "training_plan_sha256":sha(FOLDER/"training_plan.json"),
                                      "cohort":args.mode,"records":records},indent=2)+"\n")
    print("PASS",destination,flush=True)


if __name__=="__main__":
    main()
