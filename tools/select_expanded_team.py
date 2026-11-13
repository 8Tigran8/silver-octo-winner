"""Freeze one nineteen-game candidate using development results only."""
from datetime import datetime,timezone
import json
import statistics
from expanded_team_trial import FOLDER,sha


def main():
    destination=FOLDER/"selection.json"
    if destination.exists():raise ValueError("A candidate is already frozen")
    if (FOLDER/"fresh_results.json").exists() or (FOLDER/"secondary_results.json").exists():raise ValueError("Confirmation already scored")
    data=json.loads((FOLDER/"development_results.json").read_text())
    if data["plan_sha256"]!=sha(FOLDER/"plan.json"):raise ValueError("Protocol identity differs")
    rows=data["records"]
    groups=[("known","matched_seed0"),("known","label_blind_seed1"),("known","stress_seed1"),
            ("development","natural_detected"),("development","label_blind_seed7")]
    scores={}
    for name in sorted({r["candidate"] for r in rows}):
        means=[]
        for group,scenario in groups:
            selected=[r for r in rows if (r["cohort"],r["scenario"],r["candidate"])==(group,scenario,name)]
            if len({r["match"] for r in selected})!=(9 if group=="known" else 10):raise ValueError("Incomplete candidate cells")
            means.append(statistics.mean(r["spell_f1_mean"] for r in selected))
        scores[name]={"means":means,"weighted":.4*means[0]+.2*means[1]+.2*means[3]+.2*means[4]}
    baseline=scores["v2"]
    for name,result in scores.items():
        result["gain"]=result["weighted"]-baseline["weighted"]
        result["nonregression_pass"]=all(a>=b-.005 for a,b in zip(result["means"],baseline["means"]))
        result["pass"]=name!="v2" and result["gain"]>=.005 and result["nonregression_pass"]
    plan=json.loads((FOLDER/"plan.json").read_text())
    eligible=[name for name,result in scores.items() if result["pass"]]
    selected=max(eligible,key=lambda n:(scores[n]["weighted"],-plan["candidates"][n]["weight"])) if eligible else "v2"
    result={"registered_utc":datetime.now(timezone.utc).isoformat(),"selected_candidate":selected,
            "fresh_scores_seen":False,"secondary_scores_seen":False,"plan_sha256":sha(FOLDER/"plan.json"),
            "secondary_protocol_sha256":sha(FOLDER/"secondary_protocol.json"),
            "score_order":[f"{g}/{s}" for g,s in groups],"scores":scores,
            "candidate_spec":plan["candidates"].get(selected),
            "model_sha256":sha(FOLDER/"models/all_nineteen.pkl"),
            "baseline_checkpoint_sha256":plan["baseline_checkpoint_sha256"]}
    destination.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
