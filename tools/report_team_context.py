"""Audit the bounded development search and preserve unused confirmation data."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"reports/v3_search"
OUTPUT=ROOT/"docs/experiments/team-context"


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    files=["known_results.json","development_results.json","ablation_results.json","robust_team_results.json",
           "context_mix/known_results.json","context_mix/development_results.json","gated_results.json"]
    cells={};record_count=0
    for name in files:
        data=json.loads((FOLDER/name).read_text())
        cohort=data.get("cohort",Path(name).name.split("_")[0])
        for r in data["records"]:
            group=r.get("cohort",cohort)
            key=(group,r["scenario"],r["candidate"],r["match"])
            if key in cells and cells[key]["spell_f1_mean"]!=r["spell_f1_mean"]:
                raise ValueError("Repeated baseline does not agree")
            cells[key]=r;record_count+=1
        shutil.copy2(FOLDER/name,OUTPUT/name.replace("/","_"))
    groups=[("known","matched_seed0"),("known","label_blind_seed1"),("known","stress_seed1"),
            ("development","natural_detected"),("development","label_blind_seed7")]
    candidates=sorted({k[2] for k in cells});scores={}
    for candidate in candidates:
        means=[]
        for group,scenario in groups:
            rows=[r for (g,s,c,n),r in cells.items() if (g,s,c)==(group,scenario,candidate)]
            if len(rows)!=(9 if group=="known" else 10):raise ValueError("Incomplete development cells")
            means.append(statistics.mean(r["spell_f1_mean"] for r in rows))
        scores[candidate]={"means":means,"weighted":.4*means[0]+.2*means[1]+.2*means[3]+.2*means[4]}
    baseline=scores["v2"]
    for candidate,result in scores.items():
        result["gain"]=result["weighted"]-baseline["weighted"]
        result["nonregression_pass"]=all(result["means"][i]>=baseline["means"][i]-.005 for i in (0,3,4))
        result["development_pass"]=candidate!="v2" and result["nonregression_pass"] and result["gain"]>=.005
    passing=[c for c in candidates if scores[c]["development_pass"]]
    selected=max(passing,key=lambda c:scores[c]["weighted"]) if passing else "v2"
    best=max(candidates,key=lambda c:scores[c]["weighted"])
    result={"registered_utc":datetime.now(timezone.utc).isoformat(),"selected_candidate":selected,
            "best_development_candidate":best,"fresh_european_model_scores_seen":False,
            "status":"Candidate frozen for conditional European confirmation" if passing else "No candidate clears frozen development gate; retain v2; European games remain unscored",
            "score_order":[f"{a}/{b}" for a,b in groups],"scores":scores,
            "record_count_including_repeated_baselines":record_count,"unique_records":len(cells),
            "selection_threshold_gain":.005,
            "plan_sha256":{name:hashlib.sha256((FOLDER/name).read_bytes()).hexdigest() for name in ["plan.json","ablation_plan.json","followup_plan.json","gated_followup_plan.json","context_mix/plan.json"]}}
    path=FOLDER/"selection.json"
    if path.exists():
        old=json.loads(path.read_text())
        if old["scores"]!=result["scores"] or old["selected_candidate"]!=selected:raise ValueError("Do not reselect after confirmation")
        result=old
    else:path.write_text(json.dumps(result,indent=2)+"\n")
    (OUTPUT/"selection.json").write_text(json.dumps(result,indent=2)+"\n")
    for name in ["diagnostic.json","legacy_eligibility.json","gated_followup_plan.json"]:shutil.copy2(FOLDER/name,OUTPUT/name)
    for subdir,label in [("models","context_training"),("context_mix/models","mixed_context_training")]:
        provenance={p.stem:json.loads(p.read_text()) for p in (FOLDER/subdir).glob("*.json")}
        (OUTPUT/(label+".json")).write_text(json.dumps(provenance,indent=2)+"\n")
    text=["# Team ownership experiments after v2","",
          f"**Decision: {result['status']}.**","","![Development gains](development_gains.png)","",
          "The current submission remains v2 unless a new candidate passes both the predeclared development gate and its later confirmation. This report does not score the competition match, use its public GT for selection, or reuse v2's ten confirmation games as a fresh test.","",
          "## Hypotheses and isolation","",
          "The diagnostic replaces individual probability channels with answers from external reference labels. With the team channel replaced, known matched spellF1 rises from 0.6432 to 0.8934; replacing only dead probabilities gives 0.6695. These are oracle diagnostics, not deployable model scores or strict global upper bounds: the frozen duration decoder still imposes constraints.","",
          "We added 21 observable features: previous/following actually observed team-distance margins, nearest-player distances/counts and ball speeds, decayed endpoint evidence, endpoint agreement/change, and visible-only temporal averages. Carries and windows reset at period boundaries. A team-only boosting head retains v1's model parameters, training on the nine original games; each known test match is excluded. Dead-ball probabilities and cost-9 decoding stay fixed. The original nested stack predictions remain the baseline. No new team head consumes the scored game's labels during fitting.","",
          "Controls separate added features from bypassing the old team stack. Another family retains the old three-sensor robust team head while preserving the original stacked dead head. A mixed-context head trains with matched, label-blind and clean inputs, stride 12 each. Fixed 25/50/100 percent blends test these four families (12 policies). After this bounded set failed the development gate, an explicitly logged follow-up tests five gap-specific policies. Two use hard missing-ball masks; two taper the mixed-context contribution toward zero at each gap edge with one-second exponentials; one uses the original robust head with tapering. These five policies were registered before their scores and keep the original threshold.","",
          "## Development scores, including regressions","",
          "Selection weights known matched / known label-blind / SkillCorner development natural / SkillCorner development label-blind means by 0.4/0.2/0.2/0.2. A candidate needs weighted gain ≥0.005 and may not lose more than 0.005 in known matched, new natural or new label-blind means. Stress is reported separately. These data were already used in earlier development; these are not new independent generalization estimates.","",
          "| Policy | Known matched | Known label-blind | Known stress | Skill natural | Skill label-blind | Weighted gain | Pass |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for candidate,s in sorted(scores.items(),key=lambda pair:-pair[1]["weighted"]):
        values=" | ".join(f"{v:.4f}" for v in s["means"])
        text.append(f"| {candidate} | {values} | {s['gain']:+.4f} | {s['development_pass']} |")
    text += ["","The 17 policies share games and predictions; hundreds of records are not hundreds of independent matches. No interval from this selection search is presented as confirmatory. The known-game live-team frame accuracy check improves during missing-ball frames (about 0.77 to 0.80 for a 50% mixed-context blend), but frame accuracy alone does not establish better possession spells.","",
             "## Nine additional European matches","",
             "We recovered the original 2019/20 SkillCorner open-data release at `1ed50b775f93075c2a6d00d0d795f13b7b89709e`: nine matches in England, Spain, Germany, Italy and France, absent from the implemented training corpus. These are different games from the competition fixture. We verified publisher object hashes and converted observations without using ownership fields as inputs. Anonymous track slots do not interpolate across identity reassignments. Source and licence are pinned in tools/legacy_skillcorner_sources.json and SkillCorner-LICENSE.txt.","",
             "The reference contains automatic ownership only where the provider names H/A. Missing ownership is X, never an invented D label. Prediction periods derive independently from tracking periods. Consequently these games can test conditional H/A performance, including some missing-ball frames, but cannot establish full H/A/D or stoppage quality. Preparing/inspecting schema and coverage is not model scoring. Historical author-wide exposure to this public release cannot be independently audited.","",
             "If no development candidate passes, all nine games remain unscored so they can support a later frozen candidate. If a candidate passes, its conditional H/A macroF1 comparison requires a gain ≥0.005, positive paired 95% lower interval and at least seven wins in nine games. No thresholds may be lowered after results.","",
             "## Reproduce","","Use the pinned original nested-audit cache and v2 development setup described in earlier reports. Run from repository root:","","```bash",
             ".venv/bin/python tools/team_context_diagnostics.py",
             ".venv/bin/python tools/team_context_trial.py fit --jobs 4",
             ".venv/bin/python tools/team_context_trial.py known --jobs 4",
             ".venv/bin/python tools/team_context_trial.py development --jobs 2",
             ".venv/bin/python tools/team_stack_ablation.py",
             ".venv/bin/python tools/team_robust_ablation.py",
             ".venv/bin/python tools/team_context_mixture.py fit --jobs 4",
             ".venv/bin/python tools/team_context_mixture.py known --jobs 4",
             ".venv/bin/python tools/team_context_mixture.py development --jobs 2",
             ".venv/bin/python tools/team_gap_gating.py",
             ".venv/bin/python tools/prepare_legacy_skillcorner.py",
             "reports/plotenv/bin/python tools/report_team_context.py","```","",
             "Copy the published plans to reports/v3_search first (mixed_context_plan.json goes under context_mix/plan.json). Raw arrays, weights and tracking stay local. Training exclusion lists and aggregate per-match metrics accompany this report. Existing v2 code, CSV and reproduction evidence remain available.","",
             "Data attribution: [SkillCorner/Friends of Tracking original release](https://github.com/SkillCorner/opendata/tree/1ed50b775f93075c2a6d00d0d795f13b7b89709e), MIT; original training IDSSE/DFL CC BY 4.0 and Metrica Sports sample-data sources as acknowledged in the technical document.",""]
    (OUTPUT/"REPORT.md").write_text("\n".join(text))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ordered=sorted([(c,s) for c,s in scores.items() if c!="v2"],key=lambda item:item[1]["gain"])
    fig,ax=plt.subplots(figsize=(10,7))
    ax.barh([c for c,s in ordered],[s["gain"] for c,s in ordered],color=["#287a96" if s["nonregression_pass"] else "#b78482" for c,s in ordered])
    ax.axvline(0,color="#888",lw=1);ax.axvline(.005,color="#444",ls="--",label="Frozen selection gate: +0.005")
    ax.set_xlabel("Weighted development spellF1 gain over v2 (reused development games)")
    ax.set_title("Team-head hypotheses: 17 policies, before any European scoring")
    handles,labels=ax.get_legend_handles_labels()
    handles += [Patch(color="#287a96",label="Cohort loss gates pass"),Patch(color="#b78482",label="A cohort loses more than 0.005")]
    ax.legend(handles=handles,frameon=False);fig.tight_layout()
    fig.savefig(OUTPUT/"development_gains.png",dpi=150,bbox_inches="tight");plt.close(fig)
    print(json.dumps({k:result[k] for k in ["status","selected_candidate","best_development_candidate","record_count_including_repeated_baselines"]},indent=2))
    print("Best development gain:",scores[best]["gain"])


if __name__=="__main__":main()
