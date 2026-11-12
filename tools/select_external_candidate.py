"""Freeze one candidate using development data only, before sealed-game scoring."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from quality_audit import sha
from decoder_development import FOLDER


def main():
    output = FOLDER / "selection.json"
    if output.exists():
        raise ValueError("A candidate is already frozen; do not reselect after confirmation")
    if (FOLDER / "confirmation_decoder_results.json").exists() or (FOLDER / "confirmation_robust_results.json").exists():
        raise ValueError("Confirmation has already been scored")
    cells = {}
    for cohort in ["known", "development"]:
        for family in ["decoder", "robust"]:
            data = json.loads((FOLDER / f"{cohort}_{family}_results.json").read_text())
            for row in data["records"]:
                if row["candidate"] == "nearest_baseline":
                    continue
                key = cohort, row["scenario"], row["candidate"]
                cells.setdefault(key, {})[row["match"]] = row["spell_f1_mean"]
    candidates = sorted({key[2] for key in cells})
    keys = [("known", "matched_seed0"), ("known", "label_blind_seed1"),
            ("known", "stress_seed1"), ("development", "natural_detected"),
            ("development", "label_blind_seed7")]
    scores = {}
    for candidate in candidates:
        means = []
        for cohort, scenario in keys:
            values = cells[cohort, scenario, candidate]
            if len(values) != (9 if cohort == "known" else 10):
                raise ValueError("Incomplete candidate comparison")
            means.append(statistics.mean(values.values()))
        scores[candidate] = {"means": means, "weighted": .4*means[0]+.2*means[1]+.2*means[3]+.2*means[4]}
    base = scores["v1"]["means"]
    for result in scores.values():
        m = result["means"]
        result["eligible"] = m[0] >= base[0]-.005 and m[2] >= base[2]-.005 and m[3] >= base[3]-.01
    eligible = [c for c in candidates if scores[c]["eligible"]]
    best = max(eligible, key=lambda c:scores[c]["weighted"])
    if scores[best]["weighted"] < scores["v1"]["weighted"] + .002:
        best = "v1"
    result = {"registered_utc": datetime.now(timezone.utc).isoformat(), "candidate":best,
              "plan_sha256":sha(FOLDER/"plan.json"),
              "training_plan_sha256":sha(FOLDER/"training_plan.json"),
              "adoption_plan_sha256":sha(FOLDER/"adoption_plan.json"),
              "confirmation_scores_seen":False,
              "score_order":[f"{a}/{b}" for a,b in keys], "development_scores":scores}
    if best in {"robust_stage1","blend50"}:
        result["robust_model_sha256"] = sha(FOLDER/"robust/all_nine.pkl")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
