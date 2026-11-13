"""Evaluate one frozen candidate, without selecting alternatives on test games."""
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
from mtah import expanded_team as expanded,features2,stack,degrade,possession_context as context
from mtah.methods import boost,boost_train
from quality_audit import official,sha
from decoder_development import metric
from expanded_team_trial import FOLDER,OLD,load_model
from legacy_skillcorner_reader import period_spans


def team_metrics(gt,pred,seen):
    result={"reference_live_frames":int((gt<2).sum()),"reference_live_missing_ball":int(((gt<2)&~seen).sum())}
    for label,mask in [("all",np.ones(len(gt),bool)),("missing",~seen)]:
        selected=np.where(mask,gt,3)
        cm=official.scored_confusion(official.confusion(selected,pred))
        f1=[official.per_class_prf(cm,k)[2] for k in (0,1)]
        valid=selected<2
        result[label+"_conditional_team_macroF1"]=float(np.mean(f1))
        result[label+"_accuracy"]=float((selected[valid]==pred[valid]).mean()) if valid.any() else None
        result[label+"_false_dead_on_live"]=float((pred[valid]==2).mean()) if valid.any() else None
    return result


def job(item):
    cohort,name=item
    selected=json.loads((FOLDER/"selection.json").read_text())
    if selected["selected_candidate"]=="v2":raise ValueError("No new candidate passed development")
    plan=json.loads((FOLDER/"plan.json").read_text())
    if sha(ROOT/"out/training_preflight.pkl")!=selected["baseline_checkpoint_sha256"] or sha(FOLDER/"models/all_nineteen.pkl")!=selected["model_sha256"]:raise ValueError("Frozen weights changed")
    model=load_model()
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/"mtah/methods/v2_params.json").read_text())["decoder"])
    rows=[]
    with threadpool_limits(1):
        if cohort=="fresh":
            if name not in plan["fresh_games"] or any(name==g.split("/")[1] for g in expanded.games()):raise ValueError("Invalid fresh-game exclusion")
            with np.load(ROOT/"reports/v3_search/legacy"/(name+".npz")) as z:data={k:z[k] for k in z.files}
            scenarios=[("natural_detected",data,period_spans(data["period"]))]
            with (ROOT/"out/training_preflight.pkl").open("rb") as stream:base=cloudpickle.load(stream)
        else:
            config=json.loads((FOLDER/"secondary_protocol.json").read_text())
            if name not in config["cohort"] or "skillcorner/"+name in expanded.games():raise ValueError("Secondary game was trained on")
            with np.load(OLD/"skillcorner"/(name+".npz")) as z:clean={k:z[k] for k in z.files}
            scenarios=[("natural_detected",clean,official.valid_runs(clean["gt"])),
                       ("label_blind_seed7",degrade.degrade(clean,seed=7,params=expanded.BLIND),official.valid_runs(clean["gt"]))]
        for scenario,data,segments in scenarios:
            x,cols=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
            if cohort=="fresh":
                pd,ph=boost.predict(base["stage1"],x,xs);pd,ph=stack.predict(base["stage2"],pd,ph)
            else:
                with np.load(OLD/"skillcorner"/f"{name}__{scenario}__probs.npz") as z:
                    pd,ph=z["pd"],z["ph"]
                    if not np.array_equal(z["gt"],data["gt"]):raise ValueError("Secondary alignment differs")
            x,_=context.extend(x,cols,segments);xs,_=context.extend(xs,cols,segments)
            new=context.predict(model,x,xs);seen=np.isfinite(data["ball"][:,0])
            for method,spec in [("v2",{"weight":0.,"gate":"all"}),(selected["selected_candidate"],selected["candidate_spec"])]:
                blended=expanded.blend_team(ph,new,seen,segments,spec)
                pred=boost.label(pd,blended,segments,len(ph),params)
                scores=team_metrics(data["gt"],pred,seen) if cohort=="fresh" else metric(data["gt"],pred)
                rows.append({"match":name,"scenario":scenario,"candidate":method,**scores})
    return name,rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("cohort",choices=["fresh","secondary"]);parser.add_argument("--jobs",type=int,choices=range(1,5),default=4);args=parser.parse_args()
    selection=json.loads((FOLDER/"selection.json").read_text())
    if selection["plan_sha256"]!=sha(FOLDER/"plan.json") or selection["secondary_protocol_sha256"]!=sha(FOLDER/"secondary_protocol.json"):raise ValueError("Frozen selection no longer matches protocol")
    names=json.loads((FOLDER/"plan.json").read_text())["fresh_games"] if args.cohort=="fresh" else json.loads((FOLDER/"secondary_protocol.json").read_text())["cohort"]
    destination=FOLDER/(args.cohort+"_results.json")
    if destination.exists():raise ValueError("Do not overwrite confirmation")
    rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=get_context("spawn")) as pool:
        for future in as_completed([pool.submit(job,(args.cohort,n)) for n in names]):name,result=future.result();rows.extend(result);print(name,"scored frozen comparison",flush=True)
    destination.write_text(json.dumps({"cohort":args.cohort,"selection_sha256":sha(FOLDER/"selection.json"),"finished_utc":datetime.now(timezone.utc).isoformat(),"records":rows},indent=2)+"\n")
    print("PASS",destination,flush=True)
