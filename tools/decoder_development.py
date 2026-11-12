"""Prespecified decoder development; sealed SkillCorner games require a frozen choice."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import degrade, features2, stack
from mtah.methods import baseline, boost, boost_train
from quality_audit import official, sha, read_feature, probabilities, model_path, assert_exclusion
from skillcorner_reader import read_match

FOLDER = ROOT / "reports/v2_search"
AUDIT = ROOT / "reports/quality_audit/380987b3029c"


def decoder(spec):
    return dict(boost_train.decoder_params(),
                **{k: spec["cost"] for k in ("cost_ha", "cost_in", "cost_same", "cost_other")},
                min_team_s=spec["duration"], min_dead_s=spec["duration"])


def decode(pd, ph, gt, spec):
    if spec["dead_logit_bias"]:
        x = np.clip(pd, 1e-6, 1-1e-6)
        pd = 1 / (1 + np.exp(-(np.log(x) - np.log1p(-x) + spec["dead_logit_bias"])))
    return boost.label(pd, ph, official.valid_runs(gt), len(gt), decoder(spec))


def metric(gt, pred):
    r = official.evaluate(gt, pred, bootstrap_n=0)
    return {k: r[k] for k in ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells", "per_class")}


def score_file(cache, match, scenario, specs):
    with np.load(cache, allow_pickle=False) as z:
        pd, ph, gt = z["pd"], z["ph"], z["gt"]
    return [{"match": match, "scenario": scenario, "candidate": spec["name"],
             **metric(gt, decode(pd, ph, gt, spec))} for spec in specs]


def skill_job(job):
    match, specs = job
    folder = FOLDER / "skillcorner"
    folder.mkdir(exist_ok=True)
    converted = folder / (match + ".npz")
    if not converted.exists():
        clean = read_match(ROOT / "external/skillcorner" / match, match)
        np.savez_compressed(converted, **clean)
    else:
        with np.load(converted, allow_pickle=False) as z:
            clean = {k:z[k] for k in z.files}
    checkpoint = ROOT / "out/training_preflight.pkl"
    plan = json.loads((FOLDER / "plan.json").read_text())
    if sha(checkpoint) != plan["frozen_v1_model_sha256"]:
        raise ValueError("Frozen v1 weights differ")
    with checkpoint.open("rb") as stream:
        models = cloudpickle.load(stream)
    records = []
    with threadpool_limits(1):
        for scenario in plan["scenarios"]:
            cache = folder / f"{match}__{scenario}__probs.npz"
            if not cache.exists():
                d = clean if scenario == "natural_detected" else degrade.degrade(clean, seed=7, params={
                    "mode":"v1", "skip_ball_if_seen_below":0., "ball_seen_target":.4,
                    "ball_gap_mean_s":6., "frag_mean_s":107., "frag_gap_mean_s":6.,
                    "pos_noise_m":.3, "speed_noise":.6})
                gt = d["gt"]
                segs = official.valid_runs(gt)
                x, _ = features2.build(d, segs)
                xs, _ = features2.build(d, segs, swap=True)
                pd, ph = boost.predict(models["stage1"], x, xs)
                pd, ph = stack.predict(models["stage2"], pd, ph)
                np.savez(cache, pd=pd, ph=ph, gt=gt,
                         baseline=official.label_codes(baseline.run(d, window=50)),
                         seen=np.isfinite(d["ball"][:,0]))
                del x, xs
            records.extend(score_file(cache, match, scenario, specs))
            with np.load(cache, allow_pickle=False) as z:
                records.append({"match":match,"scenario":scenario,"candidate":"nearest_baseline",
                                **metric(z["gt"],z["baseline"])})
    return match, records


def known_job(job):
    match, names, specs = job
    folder = FOLDER / "known"
    folder.mkdir(exist_ok=True)
    scenarios = ["matched_seed0", "label_blind_seed1", "stress_seed1"]
    with threadpool_limits(1):
        if not all((folder / f"{match}__{s}__probs.npz").exists() for s in scenarios):
            rows = []
            for inner in names:
                if inner == match:
                    continue
                excluded = tuple(sorted((match, inner)))
                meta = json.loads(model_path(AUDIT, excluded).with_suffix(".json").read_text())
                assert_exclusion(names, meta["train"], match, inner)
                pd, ph = probabilities(AUDIT, excluded, inner)
                f = read_feature(AUDIT, inner)
                rows.append(stack.rows(pd, ph, f["gt"]))
            model = stack.fit(rows)
            del rows
            for scenario in scenarios:
                pd, ph = probabilities(AUDIT, (match,), match, scenario)
                pd, ph = stack.predict(model, pd, ph)
                f = read_feature(AUDIT, match, scenario)
                np.savez(folder / f"{match}__{scenario}__probs.npz", pd=pd, ph=ph, gt=f["gt"])
        records = []
        for scenario in scenarios:
            records.extend(score_file(folder / f"{match}__{scenario}__probs.npz", match, scenario, specs))
    return match, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["development","known","confirmation"], required=True)
    parser.add_argument("--jobs", type=int, choices=range(1,5), default=4)
    args = parser.parse_args()
    plan = json.loads((FOLDER / "plan.json").read_text())
    specs = plan["prespecified_decoder_candidates"]
    if args.cohort == "known":
        names = json.loads((AUDIT / "protocol.json").read_text())["matches"]
        jobs = [(n,names,specs) for n in names]
        function = known_job
    else:
        names = plan[args.cohort]
        if args.cohort == "confirmation":
            frozen = json.loads((FOLDER / "selection.json").read_text())
            if frozen["plan_sha256"] != sha(FOLDER / "plan.json"):
                raise ValueError("Selection/protocol mismatch")
            specs = [s for s in specs if s["name"] in {"v1",frozen["candidate"]}]
        jobs = [(n,specs) for n in names]
        function = skill_job
    destination = FOLDER / (args.cohort + "_decoder_results.json")
    if destination.exists():
        raise ValueError("Preserve completed results before any rerun")
    records = []
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(function,j) for j in jobs]
        for future in as_completed(futures):
            name, rows = future.result()
            records.extend(rows)
            (FOLDER / (args.cohort + "_" + name + ".json")).write_text(json.dumps(rows,indent=2)+"\n")
            print(name, "completed", flush=True)
    destination.write_text(json.dumps({"finished_utc":datetime.now(timezone.utc).isoformat(),
                                      "plan_sha256":sha(FOLDER / "plan.json"),
                                      "cohort":args.cohort,"records":records},indent=2)+"\n")
    print("PASS",destination,flush=True)


if __name__ == "__main__":
    main()
