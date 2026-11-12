"""Summarize only completed external-audit records; no tracking or submission reads."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MAIN = "nested_stack_viterbi"
BASE = "baseline_smooth50"
SCENARIOS = ["matched_seed0", "matched_seed1", "stress_seed1", "label_blind_seed1", "clean_source"]
METHODS = [BASE, "v0_frozen", "stage1_argmax", "stage1_viterbi", "nested_stack_argmax", MAIN,
           "nested_stack_no_min_duration", "nested_stack_duration_0p4s", "nested_stack_duration_2s"]
LABELS = {BASE: "Nearest + smoothing", "v0_frozen": "v0 heuristic + Viterbi",
          "stage1_argmax": "Boosting only", "stage1_viterbi": "Boosting + Viterbi",
          "nested_stack_argmax": "Nested stack, no decoder", MAIN: "Nested stack + Viterbi (v1)",
          "nested_stack_no_min_duration": "v1, no explicit duration minimum",
          "nested_stack_duration_0p4s": "v1, minimum 0.4 s",
          "nested_stack_duration_2s": "v1, minimum 2 s"}
SEED = 20261112


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Empty match sample")
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[picks].mean(axis=1), [0.025, 0.975])
    return [float(low), float(high)]


def summarize(rows):
    spell = [r["spell_f1_mean"] for r in rows]
    return {"n_matches": len(rows), "spellF1": float(np.mean(spell)), "spellF1_95ci": interval(spell),
            "spellF1_median": float(np.median(spell)), "spellF1_min": float(np.min(spell)),
            "spellF1_max": float(np.max(spell)), "macroF1": float(np.mean([r["macro_f1"] for r in rows])),
            "fragmentation": float(np.mean([r["fragmentation"] for r in rows])),
            "possession_count_ratio": float(np.mean([r["n_pred_spells"] / r["n_gt_spells"] for r in rows])),
            "transitionF1_1s": float(np.mean([r["transition_f1_1s"] for r in rows])),
            "per_class_F1": {k: float(np.mean([r["per_class"][k]["f1"] for r in rows])) for k in ("H", "A", "D")}}


def comparison(a, b):
    delta = np.asarray([x["spell_f1_mean"] - y["spell_f1_mean"] for x, y in zip(a, b)])
    return {"delta_spellF1": float(delta.mean()), "delta_95ci": interval(delta),
            "wins": int((delta > 1e-10).sum()), "losses": int((delta < -1e-10).sum()),
            "ties": int((abs(delta) <= 1e-10).sum())}


def ci_string(bounds):
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--fresh", type=Path, help="additional-game JSON; kept separate from the nine-game mean")
    args = parser.parse_args()
    folder = args.folder.resolve()
    protocol = json.loads((folder / "protocol.json").read_text())
    data = json.loads((folder / "results.json").read_text())
    if data["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("Protocol identity mismatch")
    records, names = data["records"], protocol["matches"]
    lookup = {(r["match"], r["scenario"], r["method"]): r for r in records}
    expected = {(n, s, m) for n in names for s in SCENARIOS for m in METHODS}
    expected |= {(n, "matched_seed0", "leaky_stack_diagnostic") for n in names}
    if set(lookup) != expected or len(lookup) != len(records):
        raise ValueError("Incomplete, duplicate or unexpected experiment cells")
    for r in records:
        if not all(np.isfinite(r[k]) and 0 <= r[k] <= 1 for k in ("spell_f1_mean", "macro_f1")):
            raise ValueError("Invalid score")
    for outer in names:
        fold = json.loads((folder / "scores" / (outer + ".json")).read_text())
        for inner in fold["inner_provenance"]:
            if set(inner["train"]) != set(names) - {outer, inner["inner_test"]}:
                raise ValueError("Nested estimator isolation failed")
    def rows(scenario, method, cohort=None):
        return [lookup[n, scenario, method] for n in (cohort or names)]
    summary = {s: {m: summarize(rows(s, m)) for m in METHODS} for s in SCENARIOS}
    comparisons = {s: comparison(rows(s, MAIN), rows(s, BASE)) for s in SCENARIOS}
    ablations = {m: comparison(rows("matched_seed0", MAIN), rows("matched_seed0", m)) for m in METHODS if m != MAIN}
    leak = comparison(rows("matched_seed0", "leaky_stack_diagnostic"), rows("matched_seed0", MAIN))
    cohorts = {"DFL": [n for n in names if n.startswith("idsse_")],
               "Metrica": [n for n in names if n.startswith("metrica_")],
               "author_reserved": [n for n in names if n in {"idsse_J03WR9", "metrica_1", "metrica_2"}]}
    subgroup = {c: {m: summarize(rows("matched_seed0", m, ns)) for m in (BASE, "stage1_viterbi", MAIN)}
                for c, ns in cohorts.items()}
    result = {"protocol_hash": protocol["protocol_hash"], "source_commit": protocol["source_commit"],
              "primary": {"v1": summary["matched_seed0"][MAIN], "baseline": summary["matched_seed0"][BASE],
                          "paired_comparison": comparisons["matched_seed0"]},
              "scenario_summaries": summary, "comparisons_vs_baseline": comparisons,
              "ablation_comparisons_v1_minus_method": ablations, "subgroups": subgroup,
              "leaky_minus_nested_diagnostic": leak,
              "uncertainty": "95% paired match-bootstrap intervals, 10,000 draws; descriptive, nine reused games; training folds overlap"}
    fresh = json.loads(args.fresh.read_text()) if args.fresh else None
    if fresh:
        if set(fresh["training_games"]) != set(names) or fresh["training_or_retuning_on_holdout"]:
            raise ValueError("Additional-game training provenance mismatch")
        result["additional_game"] = fresh
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ["match", "scenario", "method", "spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells",
               "fragmentation", "alive_dead_f1", "transition_f1_1s", "ball_visible_fraction", "valid_fitted_holdout"]
    with (folder / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in columns})
    text = ["# External quality audit: fixed v1", "",
            f"Protocol `{protocol['protocol_hash']}`; model/evaluation source `{protocol['source_commit']}`.", "",
            "This retrospective audit isolates the held-out match from fitting both learned stages. It does not undo historical hyperparameter/model selection on these known games. It evaluates external games only; competition tracking, the public five-minute GT and the all-data checkpoint are blocked by the audit's I/O guard.", "",
            "For each of nine outer test games, stage one fits the other eight games. Each inner stage-two training game receives first-stage predictions from a fit on seven games excluding both the outer test game and that inner game. Stage two fits the eight inner games, then predicts the outer game. The 36 omitted-pair fits are reused in both directions; nine outer fits bring the total to 45 first-stage fit groups. Fixed hyperparameters and decoder settings are identical across folds.", "",
            "## Primary comparison: nine games, matched sensor simulation", "",
            "Each game has equal weight. Intervals resample whole games in pairs (10,000 draws, seed 20261112, 95% percentiles). With nine reused games and overlapping training folds these are descriptive intervals, not proof of generalization to the hidden competition match.", "",
            "| Method | Mean spellF1 | Descriptive 95% interval | Mean macroF1 | Possession-count ratio |", "|---|---:|---|---:|---:|"]
    for m in METHODS:
        v = summary["matched_seed0"][m]
        text.append(f"| {LABELS[m]} | {v['spellF1']:.4f} | {ci_string(v['spellF1_95ci'])} | {v['macroF1']:.4f} | {v['possession_count_ratio']:.2f} |")
    count_ratio = summary["matched_seed0"][MAIN]["possession_count_ratio"]
    no_stack = ablations["stage1_viterbi"]
    no_decoder = ablations["nested_stack_argmax"]
    duration = ablations["nested_stack_no_min_duration"]
    text += ["", f"The full model predicts {count_ratio:.2f} times as many possession spells as the reference on average. Despite the spellF1 gain, it merges some distinct possessions. Removing the stack costs {no_stack['delta_spellF1']:.4f} spellF1 ({no_stack['wins']} wins, {no_stack['losses']} losses); removing the decoder costs {no_decoder['delta_spellF1']:.4f}. The one-second minimum adds {duration['delta_spellF1']:.4f} over no explicit duration minimum, interval {ci_string(duration['delta_95ci'])}. The duration effect is much smaller than the decoder effect."]
    primary = comparisons["matched_seed0"]
    text += ["", f"V1 minus baseline: **{primary['delta_spellF1']:+.4f} spellF1**, interval {ci_string(primary['delta_95ci'])}; wins/losses/ties: {primary['wins']}/{primary['losses']}/{primary['ties']}.", "",
             "## Per-game results", "", "| Game | Baseline spellF1 | No-stack + Viterbi | Nested v1 spellF1 | V1 macroF1 | True / predicted possessions |",
             "|---|---:|---:|---:|---:|---:|"]
    for n in names:
        b, a, v = [lookup[n, "matched_seed0", m] for m in (BASE, "stage1_viterbi", MAIN)]
        text.append(f"| {n} | {b['spell_f1_mean']:.4f} | {a['spell_f1_mean']:.4f} | {v['spell_f1_mean']:.4f} | {v['macro_f1']:.4f} | {v['n_gt_spells']} / {v['n_pred_spells']} |")
    text += ["", "## Robustness without retraining", "", "Models train on matched seed 0 only. Alternative sensor scenarios change the held-out inputs. Strong corruption halves typical track-fragment lengths, increases gaps/noise and targets 40% ball visibility. Clean source uses unmodified external tracking. The label-blind scenario chooses ball gaps from observable geometry, not live/dead labels.", "",
             "| Scenario | Baseline spellF1 | Nested v1 spellF1 | V1 − baseline | Paired interval | Wins / losses |", "|---|---:|---:|---:|---|---:|"]
    for s in SCENARIOS:
        b, v, c = summary[s][BASE], summary[s][MAIN], comparisons[s]
        text.append(f"| {s} | {b['spellF1']:.4f} | {v['spellF1']:.4f} | {c['delta_spellF1']:+.4f} | {ci_string(c['delta_95ci'])} | {c['wins']} / {c['losses']} |")
    text += ["", "## Source-family differences", "", "Metrica labels are event-derived proxies; DFL labels use provider tracking possession/status fields. Do not treat these as identical annotation processes. The author's reserved subset is historical, not newly untouched data.", "",
             "| Cohort | Games | Baseline spellF1 | Nested v1 spellF1 |", "|---|---:|---:|---:|"]
    for cohort, values in subgroup.items():
        text.append(f"| {cohort} | {values[MAIN]['n_matches']} | {values[BASE]['spellF1']:.4f} | {values[MAIN]['spellF1']:.4f} |")
    text += ["", "## Ablation effects (default v1 minus alternative)", "", "| Alternative | Δ spellF1 | Paired interval |", "|---|---:|---|"]
    for m, c in ablations.items():
        text.append(f"| {LABELS[m]} | {c['delta_spellF1']:+.4f} | {ci_string(c['delta_95ci'])} |")
    text += ["", "No explicit duration minimum still leaves the decoder's one-block (0.2 s) discretization. The 0.4 s / 2 s runs are a prespecified sensitivity analysis, not a new selected submission.", "",
             "## Non-nested stacking diagnostic", "",
             f"Historical leaky-style evaluation minus nested evaluation: {leak['delta_spellF1']:+.4f} spellF1, descriptive paired interval {ci_string(leak['delta_95ci'])}. This diagnostic is excluded from valid quality comparisons; its sign does not change the structural exclusion error. Nested inner models train on seven rather than eight games, so this difference combines training-size effects and contamination; it is not a pure causal estimate of leakage optimism.", "",
             "## Limitations and interpretation", "",
             "- All nine games and configuration choices were already known to the model author. Estimator cross-fitting is isolated, but this is not pristine independent validation of the entire historical selection process.",
             "- Matched v2 corruption uses external live/dead labels to simulate sensor dropouts. These scores are conditional on that assumed sensor model. Label-blind and clean-input results test this dependence; none establishes the actual competition sensor distribution.",
             "- The seven DFL games and two Metrica samples are a small, heterogeneous sample. Thousands of frames are not thousands of independent matches.",
             "- Oracle-equals-GT and constant-dead spellF1 sanity checks passed in each fold. Metrics come from the unmodified pinned organizer evaluator.",
             "- Per-game bootstrap fields in raw records use the organizer's default 5th/95th percentiles (90%). The tables here use a separate descriptive 95% bootstrap over whole games.",
             "- No final CSV or v1 configuration was changed based on this audit. Competitive quality on the hidden full-match GT remains unknown.", "",
             "## Reproduction", "", "```bash", ".venv/bin/python tools/quality_audit.py --jobs 4",
             "# Use the resulting protocol directory:", f"reports/plotenv/bin/python tools/summarize_quality_audit.py {folder.relative_to(folder.parents[2])} --plot", "```", "",
             "Raw inputs are the pinned external sources from tools/external_sources.json. Full records: metrics.csv / results.json. Split provenance and input/source hashes: protocol.json plus stage1/*.json and scores/*.json (local).",
             "[Nested-evaluation background, scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).", ""]
    if fresh:
        extra = ["## Additional game outside the implemented training corpus", "",
                 f"Metrica Sample_Game_3: {fresh['frames']:,} tracking frames, {fresh['graded_frames']:,} graded frames and {fresh['provider_events']:,} provider events. The frozen full-nine-game checkpoint was evaluated without training or retuning on this game. Its hash was registered before scoring. Labels are event-derived proxies; only event-covered period spans are graded. This game is separate from the nine-game mean. Historical author-wide exposure to this public game cannot be independently audited.", "",
                 "| Input scenario | Baseline spellF1 | Frozen v1 spellF1 | V1 macroF1 |", "|---|---:|---:|---:|"]
        for scenario in ["clean_source", "matched_seed0", "label_blind_seed1", "stress_seed1"]:
            b = next(r for r in fresh["records"] if r["scenario"] == scenario and r["method"] == BASE)
            v = next(r for r in fresh["records"] if r["scenario"] == scenario and r["method"] == "frozen_stack_viterbi")
            extra.append(f"| {scenario} | {b['spell_f1_mean']:.4f} | {v['spell_f1_mean']:.4f} | {v['macro_f1']:.4f} |")
        extra.append("")
        anchor = text.index("## Non-nested stacking diagnostic")
        text[anchor:anchor] = extra
    # Keep optional plotting dependencies out of the inference environment.
    anchor = text.index("# Use the resulting protocol directory:")
    text[anchor:anchor] = [".venv/bin/python tools/prepare_fresh_holdout.py --jobs 2",
                          ".venv/bin/python tools/fresh_metrica_holdout.py",
                          "python3.12 -m venv reports/plotenv",
                          "reports/plotenv/bin/python -m pip install -r requirements-quality-plots.txt"]
    for i, line in enumerate(text):
        if line.startswith("reports/plotenv/bin/python tools/summarize_quality_audit.py") and fresh:
            text[i] += " --fresh reports/fresh_holdout/results.json"
    text += ["Data attribution: [IDSSE / DFL / Sportec, PySport mirror](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0, Bassek et al. (2025); [Metrica Sports sample-data](https://github.com/metrica-sports/sample-data), source acknowledged as required by its README.", ""]
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        x = np.arange(len(names)); width = .36
        ax = axes[0, 0]
        ax.bar(x - width / 2, [lookup[n, "matched_seed0", BASE]["spell_f1_mean"] for n in names], width, label="Baseline", color="#df9444")
        ax.bar(x + width / 2, [lookup[n, "matched_seed0", MAIN]["spell_f1_mean"] for n in names], width, label="Nested v1", color="#246d91")
        ax.set_xticks(x, [n.replace("idsse_", "").replace("metrica_", "Metrica ") for n in names], rotation=35, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("Official spellF1"); ax.set_title("Matched simulation: each held-out game"); ax.legend(frameon=False)
        ax = axes[0, 1]
        for method, shift, color in ((BASE, -.16, "#df9444"), (MAIN, .16, "#246d91")):
            v = [summary[s][method] for s in SCENARIOS]
            means = np.array([r["spellF1"] for r in v]); bounds = np.array([r["spellF1_95ci"] for r in v])
            ax.errorbar(means, np.arange(5)+shift, xerr=[np.maximum(0, means-bounds[:,0]), np.maximum(0,bounds[:,1]-means)], fmt="o", capsize=3, color=color, label="Baseline" if method==BASE else "Nested v1")
        ax.set_yticks(np.arange(5), ["Matched, seed 0", "Matched, seed 1", "Stronger corruption", "Label-blind gaps", "Clean input"])
        ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("Mean spellF1 (descriptive 95% match bootstrap)"); ax.set_title("Distribution shift without retraining"); ax.legend(frameon=False)
        ax = axes[1, 0]
        shown = METHODS[:6]
        ax.barh(np.arange(len(shown)), [summary["matched_seed0"][m]["spellF1"] for m in shown], color=["#246d91" if m==MAIN else "#8da3ad" for m in shown])
        ax.set_yticks(np.arange(len(shown)), [LABELS[m] for m in shown]); ax.invert_yaxis(); ax.set_xlim(0,1); ax.set_xlabel("Mean spellF1"); ax.set_title("Component ablations: matched simulation")
        ax = axes[1, 1]
        delta = [lookup[n, "matched_seed0", "leaky_stack_diagnostic"]["spell_f1_mean"]-lookup[n,"matched_seed0",MAIN]["spell_f1_mean"] for n in names]
        ax.axvline(0, color="#999", linewidth=1); ax.scatter(delta, np.arange(9), color="#864a6d")
        ax.set_yticks(np.arange(9), [n.replace("idsse_", "").replace("metrica_", "Metrica ") for n in names]); ax.invert_yaxis(); ax.set_xlabel("Leaky-style minus nested spellF1"); ax.set_title("Validation-protocol diagnostic (not a candidate)")
        for ax in axes.ravel():
            ax.grid(axis="x", alpha=.18); ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
        fig.suptitle("Frozen v1: external-match quality audit", fontsize=18)
        fig.text(.02,.025,"9 reused games; equal game weights. Simulated sensor shifts and proxy labels limit transfer. No competition-match GT used.",fontsize=10)
        fig.tight_layout(rect=(0,.055,1,.95))
        fig.savefig(folder / "quality_audit.png", dpi=160)
        fig.savefig(folder / "quality_audit.pdf")
        plt.close(fig)
        text.insert(4, "![External audit](quality_audit.png)\n")
    (folder / "REPORT.md").write_text("\n".join(text))
    print(json.dumps(result["primary"], indent=2))
    print("Report:", folder / "REPORT.md")


if __name__ == "__main__":
    main()
