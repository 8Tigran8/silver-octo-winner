"""Publish aggregate external results and apply the pre-score adoption rule."""
import csv
import json
from pathlib import Path
import shutil
import statistics
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/"reports/v2_search"
OUTPUT=ROOT/"docs/experiments/skillcorner"


def paired(rows,scenario,candidate):
    a={r["match"]:r for r in rows if r["scenario"]==scenario and r["candidate"]==candidate}
    b={r["match"]:r for r in rows if r["scenario"]==scenario and r["candidate"]=="v1"}
    if len(a)!=10 or set(a)!=set(b):
        raise ValueError("Confirmation needs exactly ten paired games")
    names=sorted(a)
    delta=np.array([a[n]["spell_f1_mean"]-b[n]["spell_f1_mean"] for n in names])
    rng=np.random.default_rng(20261113)
    sample=rng.integers(0,10,size=(10000,10))
    ci=np.quantile(delta[sample].mean(axis=1),[.025,.975])
    return {"v1":float(np.mean([b[n]["spell_f1_mean"] for n in names])),
            "candidate":float(np.mean([a[n]["spell_f1_mean"] for n in names])),
            "delta":float(delta.mean()),"descriptive_95ci":ci.tolist(),
            "wins":int((delta>1e-10).sum()),"losses":int((delta < -1e-10).sum()),
            "v1_macroF1":float(np.mean([b[n]["macro_f1"] for n in names])),
            "candidate_macroF1":float(np.mean([a[n]["macro_f1"] for n in names]))}


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    selection=json.loads((FOLDER/"selection.json").read_text())
    candidate=selection["candidate"]
    dev=[];confirmation=[]
    for cohort,target in [("known",dev),("development",dev),("confirmation",confirmation)]:
        for family in ["decoder","robust"]:
            path=FOLDER/f"{cohort}_{family}_results.json"
            if path.exists():
                data=json.loads(path.read_text());target.extend(data["records"])
                shutil.copy2(path,OUTPUT/path.name)
    natural=paired(confirmation,"natural_detected",candidate)
    stress=paired(confirmation,"label_blind_seed7",candidate)
    adopted=(candidate!="v1" and natural["delta"] >= .005 and natural["descriptive_95ci"][0] > 0
             and natural["wins"] >=7 and stress["delta"] >=-.005)
    summary={"selected_candidate":candidate,"adopted":adopted,
             "submission_candidate":candidate if adopted else "original_v1",
             "confirmation_natural":natural,"confirmation_label_blind":stress,
             "limits":["SkillCorner automatically generated phase labels are proxies, not target corrected GT",
                       "10 Hz coordinates linearly resampled to 25 Hz only across adjacent observations",
                       "Only detected player and ball positions enter the predictor",
                       "Phase intervals intersect metadata periods; gaps inside periods count as dead",
                       "Same provider/league, only ten sealed games; interval is descriptive",
                       "No competition match quality scores or public GT used for this cycle"],
             "development_record_count":len(dev),"confirmation_record_count":len(confirmation)}
    known_risks={}
    for scenario in ["matched_seed0","label_blind_seed1","stress_seed1"]:
        chosen={r["match"]:r["spell_f1_mean"] for r in dev if r["scenario"]==scenario and r["candidate"]==candidate}
        base={r["match"]:r["spell_f1_mean"] for r in dev if r["scenario"]==scenario and r["candidate"]=="v1"}
        delta=np.array([chosen[n]-base[n] for n in sorted(base)])
        picks=np.random.default_rng(20261113).integers(0,len(delta),(10000,len(delta)))
        ci=np.quantile(delta[picks].mean(axis=1),[.025,.975])
        known_risks[scenario]={"delta":float(delta.mean()),"descriptive_95ci":ci.tolist(),
                               "wins":int((delta>0).sum()),"losses":int((delta<0).sum())}
    summary["known_game_tradeoffs"]=known_risks
    (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (FOLDER/"decision.json").write_text(json.dumps(summary,indent=2)+"\n")
    for file in ["plan.json","training_plan.json","adoption_plan.json","selection.json"]:
        shutil.copy2(FOLDER/file,OUTPUT/file)
    provenance={p.stem:json.loads(p.read_text()) for p in sorted((FOLDER/"robust").glob("*.json"))}
    (OUTPUT/"training_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    rows=dev+confirmation
    columns=["cohort","match","scenario","candidate","spell_f1_mean","macro_f1","n_gt_spells","n_pred_spells"]
    with (OUTPUT/"metrics.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,columns,lineterminator="\n");writer.writeheader()
        for cohort,records in [("development_or_known",dev),("confirmation",confirmation)]:
            for row in records:writer.writerow({"cohort":cohort,**{k:row[k] for k in columns[1:]}})
    def mean(records,scenario,method):
        values=[r["spell_f1_mean"] for r in records if r["scenario"]==scenario and r["candidate"]==method]
        return statistics.mean(values)
    text=["# New-provider validation and improvement experiments", "",
          "![Sealed-game confirmation](confirmation.png)","",
          f"Selected before sealed scoring: **{candidate}**. Adoption decision: **{'PROMOTE' if adopted else 'KEEP ORIGINAL V1'}**.","",
          "We fetched 20 SkillCorner A-League matches from pinned source revision `4340d274572876239c154c90bc507a9b3250a656`, under MIT with SkillCorner attribution. None is among the original nine training games. A seeded split fixed ten development and ten confirmation matches before the first new-game score. The original Metrica Sample_Game_3 is not an untouched holdout anymore after its earlier published evaluation; it is not used as the confirmation set here.","",
          "## What changed", "",
          "Nine prespecified decoder configurations include original v1, transition costs 3/6/9, duration 0.4/1 s, and dead-logit offsets ±0.35 in the combinations recorded in plan.json. Two further prespecified learned candidates use the original stage-one boosting settings and original nine training games with matched, label-blind and clean sensor inputs. Stride 12 per input scenario keeps the approximate training-row budget comparable to original stride 4 on one scenario. `robust_stage1` uses these weights directly; `blend50` averages their conditional dead/home probabilities with the original v1 stacked probabilities, with equal fixed weights and the original decoder. No SkillCorner game trains learned weights.","",
          "Known-game scores for robust stage one exclude the scored match from all its training scenarios. Original v1 predictions use the earlier nested stage-two audit. Development selection combines known matched / known label-blind / new natural / new label-blind means with weights 0.4/0.2/0.2/0.2. Gates reject matched or known-stress losses >0.005 and new natural losses >0.01. A candidate needs a weighted gain ≥0.002. We freeze one candidate before any confirmation score; confirmation outcomes cannot select a runner-up.","",
          "## Development results (used for selection)","",
          "| Candidate | Known matched | Known label-blind | Known stress | New natural | New label-blind | Weighted score | Eligible |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for method,data in sorted(selection["development_scores"].items(),key=lambda x:-x[1]["weighted"]):
        vals=" | ".join(f"{x:.4f}" for x in data["means"])
        text.append(f"| {method} | {vals} | {data['weighted']:.4f} | {data['eligible']} |")
    text += ["","Known-game trade-offs are part of the selection evidence, not an independent confirmation. On matched inputs the selected decoder loses six games and wins three: mean change -0.0033, descriptive paired 95% interval [-0.0080, +0.0017]. The interval includes zero and does not establish equivalence. Natural SkillCorner v1 predictions also have the opposite count bias to matched DFL/Metrica: about 1.48 times the reference possession count on the development cohort, versus about 0.76 on the known matched cohort. Increasing possession counts everywhere would therefore be a poor global correction."]
    text += ["","## Confirmation: ten games unseen during selection","",
             "| Scenario | Nearest baseline | Original v1 | Selected candidate | Paired Δ | Descriptive 95% interval | Wins / losses |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for scenario,result in [("natural_detected",natural),("label_blind_seed7",stress)]:
        ci=result["descriptive_95ci"]
        text.append(f"| {scenario} | {mean(confirmation,scenario,'nearest_baseline'):.4f} | {result['v1']:.4f} | {result['candidate']:.4f} | {result['delta']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {result['wins']} / {result['losses']} |")
    text += ["","Paired intervals resample whole games 10,000 times, seed 20261113. They are descriptive: ten matches from one league/provider do not establish performance on the hidden competition match. Promotion requires natural gain ≥0.005, positive lower interval bound, at least seven natural-game wins and no label-blind stress loss >0.005. The rule was registered before confirmation.","",
             "| Confirmation match | Original v1 spellF1 | Selected spellF1 | Difference | Reference / v1 / selected possessions |",
             "|---|---:|---:|---:|---:|"]
    selected={r["match"]:r for r in confirmation if r["scenario"]=="natural_detected" and r["candidate"]==candidate}
    original={r["match"]:r for r in confirmation if r["scenario"]=="natural_detected" and r["candidate"]=="v1"}
    for name in sorted(selected):
        a,b=selected[name],original[name]
        text.append(f"| {name} | {b['spell_f1_mean']:.4f} | {a['spell_f1_mean']:.4f} | {a['spell_f1_mean']-b['spell_f1_mean']:+.4f} | {a['n_gt_spells']} / {b['n_pred_spells']} / {a['n_pred_spells']} |")
    text += ["","## Label and sensor limitations","",
             "SkillCorner phase labels are generated automatically. Phase intervals are treated as H/A; the complement inside independently supplied metadata periods is D, outside is X. This assumes phase coverage is exhaustive and differs from the competition's mechanically derived corrected-tracking ownership. Some provider phase endpoints extend outside metadata halves: we intersect them with their declared period, without shifting or hand-correcting any phase. This is a proxy-label benchmark, not an official competition score.","",
             "The source is 10 Hz. Adjacent finite detected positions are interpolated to 25 Hz; we preserve gaps and derive speeds at the native sampling rate. Extrapolated player/ball positions, tracking possession fields, event-derived coordinates and phase labels never enter model features. Period boundaries supply the eligible spans. Natural detection gaps and camera framing differ from tactical tracking. The label-blind scenario adds geometry-driven ball gaps, player fragmentation and noise without using live/dead classes to choose missing frames.","",
             "We retain original v1 unless the frozen promotion rule passes. We do not use the competition public five-minute GT or hidden full-match GT to tune this cycle. No code or documentation can guarantee qualification; organizers have not published other teams' scores.","",
             "## Reproduce","","Run the original external-data setup from README first, then:","","```bash",
             ".venv/bin/python tools/fetch_skillcorner.py",
             "# Preserve/copy the published plans to reports/v2_search before scoring.",
             ".venv/bin/python tools/decoder_development.py --cohort development --jobs 4",
             ".venv/bin/python tools/decoder_development.py --cohort known --jobs 2",
             ".venv/bin/python tools/robust_training_trial.py fit --jobs 4",
             ".venv/bin/python tools/robust_training_trial.py development --jobs 2",
             ".venv/bin/python tools/robust_training_trial.py known --jobs 4",
             ".venv/bin/python tools/select_external_candidate.py",
             ".venv/bin/python tools/decoder_development.py --cohort confirmation --jobs 4",
             "# If the frozen selection is robust_stage1 or blend50:",
             ".venv/bin/python tools/robust_training_trial.py confirmation --jobs 4",
             "reports/plotenv/bin/python tools/report_skillcorner.py","```","",
             "The known-game stage uses the cached nested audit at reports/quality_audit/380987b3029c; reconstruct it with the original committed audit source and pinned external data before this run. Hashes/plans, per-game aggregate records and training exclusions accompany this report. Raw tracking, frame-level predictions and weights remain local.","",
             "Data: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), [pinned README](https://github.com/SkillCorner/opendata/blob/4340d274572876239c154c90bc507a9b3250a656/README.md), [MIT licence](SkillCorner-LICENSE.txt). We credit SkillCorner and PySport for making these data available.",""]
    (OUTPUT/"REPORT.md").write_text("\n".join(text))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    names=sorted(selected);x=np.arange(10)
    axes[0].bar(x-.18,[original[n]["spell_f1_mean"] for n in names],.36,label="Original v1",color="#8697a0")
    axes[0].bar(x+.18,[selected[n]["spell_f1_mean"] for n in names],.36,label=candidate,color="#247595")
    axes[0].set_xticks(x,names,rotation=45,ha="right");axes[0].set_ylim(0,1)
    axes[0].set_ylabel("Official spellF1 against automatic phase proxies")
    axes[0].set_title("Ten games sealed until candidate selection");axes[0].legend(frameon=False)
    for i,(label,r) in enumerate([("Natural detected",natural),("Extra label-blind corruption",stress)]):
        axes[1].errorbar(r["delta"],i,xerr=[[r["delta"]-r["descriptive_95ci"][0]],[r["descriptive_95ci"][1]-r["delta"]]],fmt="o",capsize=5,color="#247595")
    axes[1].axvline(0,color="#888",lw=1);axes[1].set_yticks([0,1],["Natural detected","Extra label-blind corruption"])
    axes[1].set_xlabel("Selected − original spellF1, descriptive 95% paired interval")
    axes[1].set_ylim(-.5,1.5);axes[1].set_title("Confirmation gain without retuning")
    fig.suptitle(f"SkillCorner new-provider validation: {candidate}")
    fig.tight_layout();fig.savefig(OUTPUT/"confirmation.png",dpi=150,bbox_inches="tight")
    fig.savefig(OUTPUT/"confirmation.pdf",bbox_inches="tight");plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
