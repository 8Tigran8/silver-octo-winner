"""Record expanded-training evidence and apply frozen release gates."""
import json
from pathlib import Path
import shutil
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"reports/v4_search"
OUTPUT=ROOT/"docs/experiments/expanded-team"


def paired(records,scenario,metric,candidate,expected):
    base={r["match"]:r for r in records if r["candidate"]=="v2" and r["scenario"]==scenario}
    new={r["match"]:r for r in records if r["candidate"]==candidate and r["scenario"]==scenario}
    if len(base)!=expected or set(base)!=set(new):raise ValueError("Incomplete paired confirmation")
    names=sorted(base)
    delta=np.array([new[n][metric]-base[n][metric] for n in names])
    draws=np.random.default_rng(20261113).integers(0,expected,(10000,expected))
    bounds=np.quantile(delta[draws].mean(axis=1),[.025,.975])
    return {"baseline":float(np.mean([base[n][metric] for n in names])),
            "candidate":float(np.mean([new[n][metric] for n in names])),
            "delta":float(delta.mean()),"descriptive_95ci":bounds.tolist(),
            "wins":int((delta>1e-10).sum()),"losses":int((delta < -1e-10).sum()),"ties":int((abs(delta)<=1e-10).sum())}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    selected=json.loads((FOLDER/"selection.json").read_text())
    name=selected["selected_candidate"]
    result={"selected_candidate":name,"status":"No development candidate passes; retain v2",
            "adopted":False,"fresh_scores_seen":False,"secondary_scores_seen":False}
    fresh_path=FOLDER/"fresh_results.json";secondary_path=FOLDER/"secondary_results.json"
    if name!="v2":
        result["status"]="Confirmation pending; v2 remains release candidate"
        if fresh_path.exists():
            fresh=json.loads(fresh_path.read_text())["records"]
            pair=paired(fresh,"natural_detected","all_conditional_team_macroF1",name,9)
            result.update(fresh_scores_seen=True,fresh_conditional_team_macroF1=pair)
            result["fresh_gate_pass"]=pair["delta"]>=.005 and pair["descriptive_95ci"][0]>0 and pair["wins"]>=7
        if secondary_path.exists():
            secondary=json.loads(secondary_path.read_text())["records"]
            natural=paired(secondary,"natural_detected","spell_f1_mean",name,10)
            blind=paired(secondary,"label_blind_seed7","spell_f1_mean",name,10)
            result.update(secondary_scores_seen=True,secondary_natural=natural,secondary_label_blind=blind)
            result["secondary_gate_pass"]=natural["delta"]>=0 and blind["delta"]>=-.005
        if result["fresh_scores_seen"] and result["secondary_scores_seen"]:
            result["adopted"]=result["fresh_gate_pass"] and result["secondary_gate_pass"]
            result["status"]="All frozen gates pass" if result["adopted"] else "Confirmation gates fail; retain v2"
    (FOLDER/"decision.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUTPUT/"decision.json").write_text(json.dumps(result,indent=2)+"\n")
    files=["plan.json","source_registration.json","secondary_protocol.json","source_eligibility.json","selection.json","development_results.json"]
    files += [p.name for p in [fresh_path,secondary_path] if p.exists()]
    for file in files:shutil.copy2(FOLDER/file,OUTPUT/file)
    provenance={p.stem:json.loads(p.read_text()) for p in (FOLDER/"models").glob("*.json")}
    (OUTPUT/"training_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    rows={p.stem:json.loads(p.read_text()) for p in (FOLDER/"rows").glob("*.json")}
    (OUTPUT/"training_rows.json").write_text(json.dumps(rows,indent=2)+"\n")
    text=["# Nineteen-game team-head training", "",f"**Decision: {result['status']}.**","",
          "![Development comparison](development.png)","",
          "We expand only the team-ownership head from the original nine public matches to nineteen: seven IDSSE/DFL games, two Metrica samples and the ten previously designated SkillCorner development games. The original v2 dead-ball probability channel and cost-9/duration decoder remain fixed. No competition observation, public five-minute GT or private GT enters fitting or selection in this cycle.","",
          "## Fitting and game isolation","",
          "The expanded classifier uses the previous 108 geometry/observed-context features and original 250-iteration histogram gradient boosting settings, random seed 0, no early stopping. Training augments each original game with matched-seed0 / label-blind-seed1 / clean inputs, stride 12 each; each SkillCorner game contributes natural / label-blind-seed7 inputs, stride 8 each. Team-swap augmentation balances home/away examples and inference averages direct and swapped predictions.","",
          f"The full dataset contributes {sum(r['rows'] for r in rows.values()):,} sampled training examples including augmentation and symmetry. Twenty heads are fitted: full nineteen and nineteen leave-one-whole-game-out heads. Every augmentation of a scored game is excluded from its fit; training_provenance.json records these lists and input/model hashes. The original v2 reference probabilities use the earlier nested exclusions on original games and nine-game weights on SkillCorner games. No feature normalization or model statistics are fitted across a held-out game.","",
          "SkillCorner references are automatically generated phase-based H/A proxies, differing from mechanically derived corrected-tracking possession used in the competition. We train the new head only on H/A frames; we do not train a new dead-ball classifier from these proxy gaps. Original IDSSE/DFL labels derive from provider tracking ownership/status; Metrica labels derive from events, as documented in the team method.","",
          "## Development comparison","",
          "We predeclare four policies: 25/50/100 percent expanded-team probabilities at every frame, plus full expanded probabilities tapered within missing-ball gaps. Each keeps the original dead probability channel. The selection weights are 0.4 known matched, 0.2 known label-blind, 0.2 SkillCorner natural and 0.2 SkillCorner label-blind. Required weighted spellF1 gain is ≥0.005 with no mean loss >0.005 in any of the five cohorts below. These games/configuration families were used before; these development results are not fresh independent validation of the entire historical model-selection process.","",
          "| Policy | Known matched | Known label-blind | Known stress | Skill natural | Skill label-blind | Weighted gain | Pass |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for policy,score in sorted(selected["scores"].items(),key=lambda x:-x[1]["weighted"]):
        means=" | ".join(f"{v:.4f}" for v in score["means"])
        text.append(f"| {policy} | {means} | {score['gain']:+.4f} | {score['pass']} |")
    text += ["",f"Frozen candidate: **{name}**. Its selection timestamp, plan hashes and full-nineteen model hash are in selection.json. No runner-up can be selected after confirmation results.","",
             "## Fresh European conditional confirmation","",
             "Nine 2019/20 European SkillCorner matches were held outside all training and model-score inspection until this cycle's selected candidate was frozen. Provider ownership is known only on some H/A frames; unknown ownership remains X and is never invented as D. Prediction periods derive independently from tracking periods, not from reference-label availability. We score the actual decoded H/A/D predictions on known-live reference frames using the mean H/A class F1; predicted D remains a miss. This conditional metric is not full-match spellF1 and cannot validate stoppage quality.","",
             "Source dates exclude overlap with named 2022/23 DFL and 2024/25 SkillCorner games. A raw first-half trajectory screen found no suspicious alignment with the two anonymized Metrica training games; this heuristic is not a proof of anonymized match identities. Historical author-wide exposure to public data cannot be independently audited. We acknowledge these limits rather than calling all provenance independently verified.","",
             "Promotion requires conditional H/A macroF1 gain ≥0.005, positive lower descriptive 95% paired interval and at least seven wins in nine matches. Paired intervals resample whole games 10,000 times, seed 20261113. Different games from one provider remain a small sample; confidence does not transfer automatically to the hidden competition match.",""]
    if result["fresh_scores_seen"]:
        pair=result["fresh_conditional_team_macroF1"]
        text += [f"Conditional H/A macroF1: **{pair['candidate']:.4f} vs v2 {pair['baseline']:.4f}**, difference {pair['delta']:+.4f}, interval [{pair['descriptive_95ci'][0]:+.4f}, {pair['descriptive_95ci'][1]:+.4f}], wins/losses {pair['wins']}/{pair['losses']}. Gate pass: **{result['fresh_gate_pass']}**.",""]
        missing=paired(fresh,"natural_detected","missing_conditional_team_macroF1",name,9)
        text += [f"Additional descriptive breakdown on known-live frames with missing ball tracking: mean conditional H/A macroF1 {missing['baseline']:.4f} → {missing['candidate']:.4f}. This subgroup summary is not an additional selection gate; it uses the already frozen predictions and does not evaluate D reference frames.",""]
    else:text += ["No fresh model scores have been computed in this report state.",""]
    text += ["## Reused full-task secondary check","",
             "The prior ten v2-confirmation games are not training inputs for either method, but their historical use means this is a reused secondary cohort, not a new pristine holdout. Evaluate only the frozen new candidate. Require no natural mean spellF1 regression and at most 0.005 mean loss under added label-blind corruption. This checks complete H/A/D behavior because the European references lack D labels.",""]
    if result["secondary_scores_seen"]:
        text += ["| Scenario | V2 spellF1 | Selected spellF1 | Difference | Descriptive 95% interval | Wins/losses |","|---|---:|---:|---:|---|---:|"]
        for key,label in [("secondary_natural","Natural"),("secondary_label_blind","Label-blind corruption")]:
            pair=result[key]
            text.append(f"| {label} | {pair['baseline']:.4f} | {pair['candidate']:.4f} | {pair['delta']:+.4f} | [{pair['descriptive_95ci'][0]:+.4f}, {pair['descriptive_95ci'][1]:+.4f}] | {pair['wins']}/{pair['losses']} |")
        text += ["",f"Secondary nonregression gate pass: **{result['secondary_gate_pass']}**.",""]
        dead={policy:statistics.mean(r["per_class"]["D"]["f1"] for r in secondary
              if r["candidate"]==policy and r["scenario"]=="natural_detected") for policy in ["v2",name]}
        text += [f"Additional descriptive frame-level D F1 on the same reused natural games changes {dead['v2']:.4f} → {dead[name]:.4f}. Keeping qD fixed does not make decoded D labels identical: changed team probabilities can change the joint temporal path. This small D-class regression is not hidden by the improved overall spellF1.",""]
    else:text += ["Secondary candidate scores have not been computed in this report state.",""]
    text += ["## Reproduce","","Use README setup, pinned source manifests and prior baseline/nested caches. Copy this report's plans into reports/v4_search before running:","","```bash",
             ".venv/bin/python tools/expanded_team_trial.py prepare --jobs 4",
             ".venv/bin/python tools/expanded_team_trial.py fit --jobs 4",
             ".venv/bin/python tools/expanded_team_trial.py score --jobs 4",
             ".venv/bin/python tools/select_expanded_team.py",
             "# Only if development chooses a new candidate:",
             ".venv/bin/python tools/confirm_expanded_team.py fresh --jobs 4",
             ".venv/bin/python tools/confirm_expanded_team.py secondary --jobs 4",
             "reports/plotenv/bin/python tools/report_expanded_team.py","```","",
             "The data whitelist is mtah/expanded_team_training.json. The nine European files stay outside it. Raw tracking, labels, arrays, predictions and fitted weights stay local; only code, immutable source references, hashes and aggregate scores are published. The current submission changes only after the frozen gates and technical reproduction pass.","",
             "Sources: [IDSSE/DFL mirror](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0; [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data), source acknowledged; [SkillCorner open data](https://github.com/SkillCorner/opendata), MIT with SkillCorner attribution; [European release](https://github.com/SkillCorner/opendata/tree/1ed50b775f93075c2a6d00d0d795f13b7b89709e).",""]
    (OUTPUT/"REPORT.md").write_text("\n".join(text))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names=list(selected["scores"]);labels=["Known matched","Known label-blind","Known stress","Skill natural","Skill label-blind"]
    fig,ax=plt.subplots(figsize=(11,6));x=np.arange(5)
    for policy in names:ax.plot(x,selected["scores"][policy]["means"],marker="o",label=policy,linewidth=2 if policy=="v2" else 1)
    ax.set_xticks(x,labels);ax.set_ylim(0,1);ax.set_ylabel("Mean official spellF1 on reused development games")
    ax.set_title("Expanded team training: all game-excluded development cohorts");ax.legend(frameon=False)
    ax.grid(axis="y",alpha=.2);fig.tight_layout();fig.savefig(OUTPUT/"development.png",dpi=150);plt.close(fig)
    if result["fresh_scores_seen"] and result["secondary_scores_seen"]:
        fig,axes=plt.subplots(1,2,figsize=(12,6))
        cohorts=[(fresh_path,"all_conditional_team_macroF1","New European games: conditional H/A macroF1"),
                 (secondary_path,"spell_f1_mean","Reused SkillCorner check: full spellF1")]
        for ax,(path,metric,title) in zip(axes,cohorts):
            records=json.loads(path.read_text())["records"]
            scores={policy:{r["match"]:r[metric] for r in records
                    if r["candidate"]==policy and r["scenario"]=="natural_detected"}
                    for policy in ["v2",name]}
            matches=sorted(scores["v2"]);y=np.arange(len(matches))
            old=np.array([scores["v2"][n] for n in matches])
            new=np.array([scores[name][n] for n in matches])
            ax.hlines(y,old,new,color="#79ad91",linewidth=2)
            ax.scatter(old,y,color="#666666",s=45,label="v2",zorder=3)
            ax.scatter(new,y,color="#13824a",s=45,label="v3 / expanded50",zorder=3)
            ax.set_yticks(y,matches);ax.invert_yaxis();ax.set_xlim(0,1)
            ax.set_xlabel("Score (different metric in each panel)")
            ax.set_ylabel("Public match ID");ax.set_title(title,fontsize=11)
            ax.grid(axis="x",alpha=.2);ax.legend(frameon=False,loc="lower left")
        fig.tight_layout();fig.savefig(OUTPUT/"confirmation.png",dpi=150);plt.close(fig)
        report=OUTPUT/"REPORT.md"
        report.write_text(report.read_text().replace("## Reused full-task secondary check",
            "![Per-game confirmation; metrics differ between panels](confirmation.png)\n\n## Reused full-task secondary check"))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
