"""Retrospective external-match audit of a frozen v1 pipeline.

Nested match cross-fitting isolates both learned stages from the outer test match.
This is not fresh-data validation of earlier hyperparameter/model selection.
Competition tracking, its GT and its submission are blocked from this process.
"""
import argparse
import builtins
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mtah import degrade, features, features2, stack
from mtah.methods import baseline, boost, boost_train, viterbi

SCENARIOS = {
    "matched_seed0": {"seed": 0, "params": {}, "label_conditioned_corruption": True},
    "matched_seed1": {"seed": 1, "params": {}, "label_conditioned_corruption": True},
    "stress_seed1": {"seed": 1, "params": {
        "ball_seen_target": 0.40, "skip_ball_if_seen_below": 0.0,
        "dead_jitter_s": 4.0, "p_dead_hide": 0.85, "pos_noise_m": 0.30,
        "speed_noise": 0.60, "frag_median_s": 17.5, "frag_mean_s2": 53.5,
        "frag_gap_mean_s2": 12.0}, "label_conditioned_corruption": True},
    "label_blind_seed1": {"seed": 1, "params": {
        "mode": "v1", "ball_gap_mean_s": 6.0, "frag_mean_s": 107.0,
        "frag_gap_mean_s": 6.0, "skip_ball_if_seen_below": 0.0},
        "label_conditioned_corruption": False},
    "clean_source": {"seed": None, "params": None, "label_conditioned_corruption": False},
}
OFFICIAL_PATH = ROOT / "ref/official/evaluation/possession_metrics.py"
_spec = importlib.util.spec_from_file_location("official_audit_metrics", OFFICIAL_PATH)
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def forbidden(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    p = Path(os.fsdecode(path)).resolve()
    return (p.is_relative_to(ROOT / "data") or p.is_relative_to(ROOT.parent / "data")
            or p.is_relative_to(ROOT / "out/submission")
            or p.name.startswith(("BAR-ATH", "atd_tactical"))
            or p.name in {"training_preflight.pkl", "real_v1_preflight.csv"})


def guard_competition_data():
    for module in (builtins, io):
        original = module.open
        def checked(file, *args, _original=original, **kwargs):
            if forbidden(file):
                raise PermissionError("Competition inputs/GT/checkpoint are blocked in quality audit")
            return _original(file, *args, **kwargs)
        module.open = checked


def split_specs(names):
    names = tuple(sorted(names))
    if len(names) < 3 or len(set(names)) != len(names):
        raise ValueError("Need at least three distinct match groups")
    return [tuple([n]) for n in names] + list(itertools.combinations(names, 2))


def assert_exclusion(all_names, train_names, outer, inner=None):
    omitted = {outer} | ({inner} if inner is not None else set())
    if set(train_names) != set(all_names) - omitted or omitted.intersection(train_names):
        raise ValueError("Held-out match leaked into estimator training")


def tag(excluded):
    return "exclude__" + "__".join(sorted(excluded))


def feature_path(folder, name, scenario):
    return Path(folder) / "features" / f"{name}__{scenario}.npz"


def model_path(folder, excluded):
    return Path(folder) / "stage1" / tag(excluded)


def read_feature(folder, name, scenario="matched_seed0"):
    with np.load(feature_path(folder, name, scenario), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def prepare_match(job):
    guard_competition_data()
    folder, name = job
    started = time.perf_counter()
    source = ROOT / "external/conv" / (name + ".npz")
    with np.load(source, allow_pickle=False) as z:
        clean = {key: z[key] for key in z.files}
    for scenario, cfg in SCENARIOS.items():
        destination = feature_path(folder, name, scenario)
        if destination.exists():
            continue
        if cfg["seed"] is None:
            d = dict(clean)
        else:
            d = degrade.degrade(clean, seed=cfg["seed"], params=cfg["params"])
        segments = official.valid_runs(d["gt"])
        x, cols = features2.build(d, segments)
        xs, cols_s = features2.build(d, segments, swap=True)
        assert cols == cols_s
        base = official.label_codes(baseline.run(d, window=50))
        old = official.label_codes(viterbi.label_segments(
            viterbi.emissions(features.build(d), viterbi.DEFAULTS), viterbi.DEFAULTS,
            segments, len(d["gt"])))
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(destination, X=x, Xs=xs, gt=d["gt"], cols=np.array(cols),
                 seen=np.isfinite(d["ball"][:, 0]), baseline=base, v0=old)
    return name, round(time.perf_counter() - started, 3)


def fit_stage1(job):
    guard_competition_data()
    folder, names, excluded, protocol_hash = job
    destination = model_path(folder, excluded)
    meta_path = destination.with_suffix(".json")
    pred_path = destination.with_suffix(".npz")
    pickle_path = destination.with_suffix(".pkl")
    train = [n for n in names if n not in excluded]
    assert_exclusion(names, train, excluded[0], excluded[1] if len(excluded) == 2 else None)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (meta["protocol_hash"] == protocol_hash and meta["train"] == train
                and sha(pred_path) == meta["predictions_sha256"]
                and sha(pickle_path) == meta["model_sha256"]):
            return tag(excluded), "verified cached"
        raise ValueError("Cached estimator provenance differs")
    started = time.perf_counter()
    with threadpool_limits(1):
        rows = []
        for name in train:
            f = read_feature(folder, name)
            rows.append(boost.training_rows(f["X"], f["Xs"], f["gt"]))
        del f
        models = boost.fit(rows)
        del rows
        output = {}
        # Inner OOF training always uses the fixed base sensor simulation.
        scenarios = list(SCENARIOS) if len(excluded) == 1 else ["matched_seed0"]
        for name in excluded:
            for scenario in scenarios:
                f = read_feature(folder, name, scenario)
                pd, ph = boost.predict(models, f["X"], f["Xs"])
                output[f"{name}__{scenario}__pD"] = pd
                output[f"{name}__{scenario}__pH"] = ph
        destination.parent.mkdir(parents=True, exist_ok=True)
        with pickle_path.open("wb") as stream:
            cloudpickle.dump(models, stream)
        np.savez(pred_path, **output)
    meta = {"protocol_hash": protocol_hash, "train": train, "excluded": list(excluded),
            "model_sha256": sha(pickle_path), "predictions_sha256": sha(pred_path),
            "seconds": round(time.perf_counter() - started, 3)}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return tag(excluded), f"trained in {meta['seconds']:.1f}s"


def probabilities(folder, excluded, name, scenario="matched_seed0"):
    with np.load(model_path(folder, excluded).with_suffix(".npz"), allow_pickle=False) as z:
        return z[f"{name}__{scenario}__pD"], z[f"{name}__{scenario}__pH"]


def argmax_labels(pd, ph, gt):
    out = np.argmax(np.stack([(1 - pd) * ph, (1 - pd) * (1 - ph), pd], axis=1), axis=1).astype(np.int8)
    out[gt == 3] = 3
    return out


def measure(gt, labels, seen, bootstrap=0):
    result = official.evaluate(gt, labels, bootstrap_n=bootstrap)
    wanted = ["spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt_spells", "n_pred_spells",
              "fragmentation", "n_graded", "spell_bootstrap_low", "spell_bootstrap_high"]
    out = {key: result[key] for key in wanted}
    out["per_class"] = result["per_class"]
    out["transition_f1_1s"] = result["transitions"][25]["f1"]
    for name, mask in (("visible", seen), ("missing", ~seen)):
        g = np.where(mask, gt, 3)
        cm = official.scored_confusion(official.confusion(g, labels))
        out[f"macro_f1_ball_{name}"] = official.macro_f1(cm) if (g != 3).any() else None
    return out


def score_outer(job):
    guard_competition_data()
    folder, names, test, protocol_hash = job
    destination = Path(folder) / "scores" / (test + ".json")
    if destination.exists():
        old = json.loads(destination.read_text())
        if old["protocol_hash"] == protocol_hash:
            return test, "verified cached"
        raise ValueError("Score cache identity mismatch")
    started = time.perf_counter()
    nested_rows, leaky_rows = [], []
    provenance = []
    for name in names:
        if name == test:
            continue
        f = read_feature(folder, name)
        omitted = tuple(sorted((test, name)))
        meta = json.loads(model_path(folder, omitted).with_suffix(".json").read_text())
        assert_exclusion(names, meta["train"], test, name)
        pd, ph = probabilities(folder, omitted, name)
        nested_rows.append(stack.rows(pd, ph, f["gt"]))
        provenance.append({"inner_test": name, "train": meta["train"]})
        # Historical non-nested comparison is diagnostic only, not a valid held-out score.
        pd_old, ph_old = probabilities(folder, (name,), name)
        leaky_rows.append(stack.rows(pd_old, ph_old, f["gt"]))
    with threadpool_limits(1):
        stage2 = stack.fit(nested_rows)
        del nested_rows
        leaky_stage2 = stack.fit(leaky_rows)
        del leaky_rows
        records = []
        params = boost_train.decoder_params()
        for scenario in SCENARIOS:
            f = read_feature(folder, test, scenario)
            gt, seen = f["gt"], f["seen"]
            segs = official.valid_runs(gt)
            pd, ph = probabilities(folder, (test,), test, scenario)
            qd, qh = stack.predict(stage2, pd, ph)
            predictions = {
                "baseline_smooth50": f["baseline"], "v0_frozen": f["v0"],
                "stage1_argmax": argmax_labels(pd, ph, gt),
                "stage1_viterbi": boost.label(pd, ph, segs, len(gt), params),
                "nested_stack_argmax": argmax_labels(qd, qh, gt),
                "nested_stack_viterbi": boost.label(qd, qh, segs, len(gt), params),
            }
            for name, seconds in (("no_min_duration", 0.0), ("duration_0p4s", 0.4), ("duration_2s", 2.0)):
                p = dict(params, min_team_s=seconds, min_dead_s=seconds)
                predictions["nested_stack_" + name] = boost.label(qd, qh, segs, len(gt), p)
            if scenario == "matched_seed0":
                ld, lh = stack.predict(leaky_stage2, pd, ph)
                predictions["leaky_stack_diagnostic"] = boost.label(ld, lh, segs, len(gt), params)
                assert measure(gt, gt, seen)["spell_f1_mean"] == 1.0
                dead = np.where(gt == 3, 3, 2).astype(np.int8)
                assert measure(gt, dead, seen)["spell_f1_mean"] == 0.0
            for method, labels in predictions.items():
                draws = 1000 if scenario == "matched_seed0" and method in {"baseline_smooth50", "nested_stack_viterbi"} else 0
                records.append({"match": test, "scenario": scenario, "method": method,
                                "ball_visible_fraction": float(seen[gt != 3].mean()),
                                "valid_fitted_holdout": method != "leaky_stack_diagnostic",
                                **measure(gt, labels, seen, draws)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {"protocol_hash": protocol_hash, "test": test, "inner_provenance": provenance,
              "seconds": round(time.perf_counter() - started, 3), "records": records}
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return test, f"scored in {result['seconds']:.1f}s"


def pool_map(function, jobs, workers):
    from multiprocessing import get_context
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        for future in as_completed(futures):
            name, status = future.result()
            print(f"{name}: {status}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    guard_competition_data()
    names = sorted(p.stem for p in (ROOT / "external/conv").glob("*.npz"))
    if len(names) != 9:
        raise ValueError("Expected the nine frozen external matches")
    sources = ["mtah/features2.py", "mtah/features.py", "mtah/degrade.py", "mtah/stack.py",
               "mtah/methods/boost.py", "mtah/methods/boost_train.py", "mtah/methods/v1_params.json",
               "mtah/methods/viterbi.py", "mtah/methods/baseline.py", "tools/quality_audit.py"]
    protocol = {"design": "Nested match cross-fitting of a frozen pipeline; retrospective",
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "matches": names, "scenarios": SCENARIOS, "model": boost.MODEL,
                "stack_model": stack.MODEL, "decoder": boost_train.decoder_params(),
                "training_scenario": "matched_seed0", "source_sha256": {p: sha(ROOT / p) for p in sources},
                "converted_inputs_sha256": {n: sha(ROOT / "external/conv" / (n + ".npz")) for n in names},
                "official_metric_sha256": sha(OFFICIAL_PATH),
                "limits": ["Previously used matches/hyperparameters; no pristine new-game holdout",
                           "v2 corruption depends on external GT; label-blind and clean tests included",
                           "Metrica event-derived labels differ from DFL tracking possession",
                           "No competition match or public-window quality evaluation"]}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    folder = ROOT / "reports/quality_audit" / protocol_hash[:12]
    folder.mkdir(parents=True, exist_ok=True)
    protocol_path = folder / "protocol.json"
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps({"registered_utc": datetime.now(timezone.utc).isoformat(),
                                             "protocol_hash": protocol_hash, **protocol}, indent=2) + "\n")
    print(f"Protocol fixed before scores: {protocol_path}", flush=True)
    print("Competition data guard enabled. Preparing five scenarios per external match.", flush=True)
    pool_map(prepare_match, [(str(folder), n) for n in names], args.jobs)
    specs = split_specs(names)
    print(f"Training {len(specs)} stage-one fit groups (9 outer + 36 inner pairs).", flush=True)
    pool_map(fit_stage1, [(str(folder), names, s, protocol_hash) for s in specs], args.jobs)
    print("Fitting nested stage two and scoring all ablations with official metrics.", flush=True)
    pool_map(score_outer, [(str(folder), names, n, protocol_hash) for n in names], args.jobs)
    records = []
    for name in names:
        records.extend(json.loads((folder / "scores" / (name + ".json")).read_text())["records"])
    (folder / "results.json").write_text(json.dumps({"protocol_hash": protocol_hash,
                                                     "records": records}, indent=2) + "\n")
    print(f"PASS: {len(records)} per-match experiment records. Results: {folder / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
