"""Score VV's registered sensor pilot with whole-fold-out frozen models.

Streams one geometry condition at a time rather than exporting all raw copies.
No training, target-match inputs, final references or candidate selection.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloudpickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import features2, temporal, v4
from mtah.methods import boost
import provider_stress as stress
import pff_mixed_train as train
import pff_mixed_score as score
import strict_fullfold as strict
import temporal_trial as trial
from decoder_development import metric

PLAN = ROOT / "docs/experiments/provider-stress/diagnostic_plan.json"
OUT = ROOT / "reports/provider_stress_diagnostic"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hashes(data):
    result = {}
    for key in stress.KEYS:
        a = np.ascontiguousarray(data[key])
        result[key] = hashlib.sha256(a.tobytes()).hexdigest()
    return result


def frozen_attestation():
    path = ROOT / "docs/experiments/pff-mixed/final_freeze.json"
    frozen = json.loads(path.read_text())
    for name, digest in frozen["files_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen experiment file changed: " + name)
    return {"files": len(frozen["files_sha256"]), "unchanged": True, "freeze_sha256": sha(path)}


def models_for(fold, game):
    folder = ROOT / "reports/strict_fullfold/folds" / f"fold{fold}"
    proof = json.loads((folder / "meta.json").read_text())
    if sha(folder / "models.pkl") != proof["model_sha256"] or proof["plan_sha256"] != sha(strict.PLAN):
        raise ValueError("Inherited model lineage/hash changed")
    _, excluded = trial.game_split(f"fold{fold}")
    if game not in excluded:
        raise ValueError("Wrong exclusion fold")
    strict.assert_component_isolation(proof, excluded)
    with (folder / "models.pkl").open("rb") as stream:
        inherited = cloudpickle.load(stream)
    # Both loaders recompute actual normalization from their training-only rows.
    controls, control_proof = strict.neural_states(fold)
    candidate, candidate_proof = train.load("boundary102", fold)
    return inherited, controls["strict_v4"], candidate, {
        "inherited": proof, "control": control_proof["strict_v4"],
        "candidate": candidate_proof, "complete_transitive_outer_isolation": True,
    }


def predict(data, inherited, control, candidate):
    # Only these geometry/metadata keys reach feature builders. gt is absent.
    observed = {k: data[k] for k in ("players", "ball", "slot_team", "pitch", "period")}
    parts = [(a, b) for a, b, _ in stress.spans(observed["period"])]
    x, cols = features2.build(observed, parts)
    xs, swapped = features2.build(observed, parts, swap=True)
    if cols != swapped:
        raise ValueError("Swapped feature columns differ")
    bd, bh, x, xs = score.inherited_probabilities(inherited, x, xs, cols, parts)
    down = [((a + 4) // 5, (b + 4) // 5) for a, b in parts]
    model, mean, scale = control
    nd, nh = temporal.probabilities(model, x[::5], xs[::5], down, mean, scale)
    nd, nh = v4.neural_upsample(nd, nh, parts, len(x))
    labels = {"strict_v4": boost.label(.5 * bd + .5 * nd, .5 * bh + .5 * nh,
                                      parts, len(x), score.decoder())}
    nd, nh = score.neural(candidate, x, xs, parts)
    labels["boundary102"] = boost.label(.5 * bd + .5 * nd, .5 * bh + .5 * nh,
                                       parts, len(x), score.decoder())
    return labels


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "results.json").exists():
        raise ValueError("Do not overwrite a completed diagnostic")
    plan = json.loads(PLAN.read_text())
    config_path = ROOT / "tools/provider_stress_presets.json"
    config = json.loads(config_path.read_text())
    if sha(config_path) != plan["config_sha256"] or sha(stress.__file__) != plan["converter_sha256"]:
        raise ValueError("Registered converter/config changed")
    if sha(trial.PLAN) != plan["fold_plan_sha256"]:
        raise ValueError("Fold plan changed")
    before = frozen_attestation()
    records, provenance, exports = [], {}, []
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(1):
        for source_name, source in config["sources"].items():
            game, fold = source["fold_key"], source["fold"]
            print("Loading source and excluded-fold models", game, fold, flush=True)
            for item in source["raw_files"].values():
                if sha(ROOT / "external" / item["path"]) != item["sha256"]:
                    raise ValueError("Raw source differs: " + item["path"])
            loader = stress.load_idsse if source["provider"] == "idsse" else stress.load_skillcorner
            original, native = loader(source, ROOT / "external", OUT)
            inherited, control, candidate, proof = models_for(fold, game)
            provenance[game] = proof
            unchanged = array_hashes(original)
            conditions = {"baseline": None, "native_observed_only": dict(
                observation_hz=source["source_hz"], noise_std_m=0,
                ball_bursts_per_minute=0, burst_min_s=2, burst_max_s=6),
                **config["presets"]}
            unique = {}
            for condition, preset in conditions.items():
                started = time.perf_counter()
                if preset is None:
                    data = {k: original[k] for k in stress.KEYS}
                    statistics = {"source_hz": source["source_hz"], "effective_observation_hz": source["source_hz"]}
                    checks = stress.verify(original, data, statistics, native, baseline=True)
                else:
                    data, statistics = stress.make_copy(original, native, source["source_hz"], preset, config["seed"])
                    checks = stress.verify(original, data, statistics, native)
                hashes = array_hashes(data)
                identity = tuple(hashes[k] for k in stress.KEYS)
                alias = unique.get(identity)
                if alias:
                    labels = alias[1]
                else:
                    labels = predict(data, inherited, control, candidate)
                    unique[identity] = (condition, labels)
                # Reference enters the metric only after all predictions exist.
                for name, pred in labels.items():
                    records.append({"game": game, "fold": fold, "scenario": condition,
                                    "candidate": name, **metric(original["gt"], pred),
                                    "prediction_sha256": hashlib.sha256(pred.tobytes()).hexdigest()})
                np.savez_compressed(OUT / f"{source_name}__{condition}_predictions.npz", **labels)
                exports.append({"game": game, "scenario": condition, "array_sha256": hashes,
                                "identical_to": alias[0] if alias else None,
                                "statistics": statistics, "checks": checks})
                print(game, condition, {r["candidate"]: round(r["spell_f1_mean"], 5)
                      for r in records[-2:]}, "seconds", round(time.perf_counter() - started, 2), flush=True)
                if array_hashes(original) != unchanged:
                    raise ValueError("Converter mutated canonical input")
                del data
                gc.collect()
            del original, native, inherited, control, candidate, unique
            gc.collect()
    result = {"plan_sha256": sha(PLAN), "runner_sha256": sha(__file__),
              "scope": plan["scope"], "models_changed": False, "final_reserve_used": False,
              "records": records, "exports": exports, "frozen_before": before,
              "frozen_after": frozen_attestation()}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result["provenance_sha256"] = sha(OUT / "provenance.json")
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Completed all registered stress conditions", flush=True)


if __name__ == "__main__":
    main()
