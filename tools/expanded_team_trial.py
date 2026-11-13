"""Nineteen-game team-head fitting with complete game exclusion."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import expanded_team as expanded,possession_context as context
from mtah.methods import boost,boost_train
from quality_audit import official,sha,read_feature
from decoder_development import metric

FOLDER=ROOT/"reports/v4_search"
OLD=ROOT/"reports/v2_search"
AUDIT=ROOT/"reports/quality_audit/380987b3029c"


def stem(game):return game.replace("/","__")


def prepare(game):
    output=FOLDER/"rows"/(stem(game)+".npz")
    metadata=output.with_suffix(".json")
    if metadata.exists():
        meta=json.loads(metadata.read_text())
        if meta["plan_sha256"]!=sha(FOLDER/"plan.json") or sha(output)!=meta["rows_sha256"]:raise ValueError("Prepared training rows differ")
        return game
    x,y,cols=expanded.training_rows(game,save_skill_evaluation=FOLDER/"evaluation")
    output.parent.mkdir(exist_ok=True)
    np.savez(output,X=x,y=y,cols=np.array(cols))
    metadata.write_text(json.dumps({"game":game,"plan_sha256":sha(FOLDER/"plan.json"),
         "rows_sha256":sha(output),"rows":len(y),"columns":len(cols),"home_examples":int(y.sum()),
         "away_examples":int((1-y).sum())},indent=2)+"\n")
    return game


def fit_job(excluded):
    names=expanded.games();train=[n for n in names if n!=excluded]
    expanded.assert_training_split(names,train,excluded)
    directory=FOLDER/"models";directory.mkdir(exist_ok=True)
    destination=directory/("exclude__"+stem(excluded) if excluded else "all_nineteen")
    model_path=destination.with_suffix(".pkl");meta_path=destination.with_suffix(".json")
    meta={"train":train,"excluded":excluded,"plan_sha256":sha(FOLDER/"plan.json"),
          "input_rows_sha256":{n:json.loads((FOLDER/"rows"/(stem(n)+".json")).read_text())["rows_sha256"] for n in train}}
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        if any(previous.get(k)!=v for k,v in meta.items()) or sha(model_path)!=previous["model_sha256"]:raise ValueError("Expanded fit cache differs")
        return excluded or "all_nineteen"
    with threadpool_limits(1):
        parts=[]
        for game in train:
            with np.load(FOLDER/"rows"/(stem(game)+".npz"),allow_pickle=False) as z:parts.append((z["X"],z["y"]))
        model=context.fit(parts)
    with model_path.open("wb") as stream:cloudpickle.dump(model,stream)
    meta.update(model_sha256=sha(model_path),fitted_utc=datetime.now(timezone.utc).isoformat())
    meta_path.write_text(json.dumps(meta,indent=2)+"\n")
    return excluded or "all_nineteen"


def load_model(excluded=None):
    name="exclude__"+stem(excluded) if excluded else "all_nineteen"
    model_path=FOLDER/"models"/(name+".pkl")
    meta=json.loads(model_path.with_suffix(".json").read_text())
    expanded.assert_training_split(expanded.games(),meta["train"],excluded)
    if sha(model_path)!=meta["model_sha256"]:raise ValueError("Model integrity check failed")
    with model_path.open("rb") as stream:return cloudpickle.load(stream)


def score_job(game):
    family,name=game.split("/")
    cohort="known" if family=="original" else "development"
    scenarios=["matched_seed0","label_blind_seed1","stress_seed1"] if family=="original" else ["natural_detected","label_blind_seed7"]
    plan=json.loads((FOLDER/"plan.json").read_text());model=load_model(game)
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/"mtah/methods/v2_params.json").read_text())["decoder"])
    records=[]
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLD/("known" if family=="original" else "skillcorner")/f"{name}__{scenario}__probs.npz") as z:pd,ph,gt=z["pd"],z["ph"],z["gt"]
            destination=FOLDER/"probabilities"/f"{name}__{scenario}.npz";destination.parent.mkdir(exist_ok=True)
            if family=="original":
                f=read_feature(AUDIT,name,scenario)
                if not np.array_equal(gt,f["gt"]):raise ValueError("Original reference alignment changed")
                seen=f["seen"]
                if not destination.exists():
                    segments=official.valid_runs(gt)
                    x,_=context.extend(f["X"],f["cols"].tolist(),segments)
                    xs,_=context.extend(f["Xs"],f["cols"].tolist(),segments)
            else:
                with np.load(FOLDER/"evaluation"/f"{name}__{scenario}.npz") as z:
                    if not np.array_equal(gt,z["gt"]):raise ValueError("SkillCorner reference alignment changed")
                    seen=z["seen"]
                    if not destination.exists():x,xs=z["X"],z["Xs"]
            segments=official.valid_runs(gt)
            if destination.exists():
                with np.load(destination) as z:new=z["ph"]
            else:
                new=context.predict(model,x,xs);np.savez(destination,ph=new)
                del x,xs
            specs={"v2":{"weight":0.,"gate":"all"},**plan["candidates"]}
            for candidate,spec in specs.items():
                blended=expanded.blend_team(ph,new,seen,segments,spec)
                pred=boost.label(pd,blended,segments,len(gt),params)
                records.append({"match":name,"game":game,"cohort":cohort,"scenario":scenario,
                                "candidate":candidate,**metric(gt,pred)})
    return game,records


def run_pool(function,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(function,j) for j in jobs]):yield future.result()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["prepare","fit","score"]);parser.add_argument("--jobs",type=int,choices=range(1,5),default=4);args=parser.parse_args()
    plan=json.loads((FOLDER/"plan.json").read_text())
    if expanded.games()!=plan["training_games"]:raise ValueError("Training whitelist changed")
    if sha(ROOT/"mtah/possession_context.py")!=plan["context_source_sha256"]:raise ValueError("Context features changed")
    if args.mode=="prepare":
        for name in run_pool(prepare,expanded.games(),args.jobs):print(name,"rows prepared",flush=True)
    elif args.mode=="fit":
        for name in run_pool(fit_job,[None]+expanded.games(),args.jobs):print(name,"model fitted",flush=True)
    else:
        path=FOLDER/"development_results.json"
        if path.exists():raise ValueError("Do not overwrite finished evaluation")
        rows=[]
        for name,records in run_pool(score_job,expanded.games(),args.jobs):rows.extend(records);print(name,"scored",flush=True)
        path.write_text(json.dumps({"plan_sha256":sha(FOLDER/"plan.json"),"records":rows},indent=2)+"\n")
        print("PASS",path,flush=True)


if __name__=="__main__":main()
