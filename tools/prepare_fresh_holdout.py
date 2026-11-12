"""Fetch the extra public game and freeze a nine-game checkpoint before evaluating it.

Run external-data fetch/conversion first. The extra game stays outside external/conv.
This command never reads the competition game.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cloudpickle
from fetch_external import fetch
from mtah import external
from mtah.methods import boost_train


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    sources = json.loads((ROOT / "tools/fresh_holdout_sources.json").read_text())
    for item in sources["files"]:
        print(fetch(item, ROOT / "external"), flush=True)
    names = external.names()
    if len(names) != 9 or any("metrica_3" in n for n in names):
        raise ValueError("Expected exactly the nine original external training matches")
    checkpoint = ROOT / "out/training_preflight.pkl"
    metadata = ROOT / "reports/training_preflight.json"
    if checkpoint.exists() and metadata.exists():
        info = json.loads(metadata.read_text())
        if info["matches"] != names or info["model_sha256"] != digest(checkpoint):
            raise ValueError("Existing own checkpoint provenance mismatch")
    else:
        models = boost_train.train(workers=args.jobs)
        checkpoint.parent.mkdir(exist_ok=True)
        metadata.parent.mkdir(exist_ok=True)
        with checkpoint.open("wb") as stream:
            cloudpickle.dump(models, stream)
        info = {"matches": names, "model_sha256": digest(checkpoint),
                "competition_input_used": False, "public_gt_used": False}
        metadata.write_text(json.dumps(info, indent=2)+"\n")
    folder = ROOT / "reports/fresh_holdout"
    folder.mkdir(exist_ok=True)
    protocol_path = folder / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous["model_sha256"] != info["model_sha256"]:
            raise ValueError("Preserve prior holdout protocol before evaluating different weights")
        print("Existing pre-score protocol preserved:", protocol_path)
        return
    protocol = {"registered_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "Metrica Sports Sample_Game_3", "source_revision": sources["source_revision"],
                "frozen_model": str(checkpoint.relative_to(ROOT)), "model_training_games": names,
                "model_sha256": info["model_sha256"], "training_or_retuning_on_holdout": False,
                "scenarios": ["clean_source", "matched_seed0", "label_blind_seed1", "stress_seed1"],
                "files": [{"path": "external/"+f["path"], "size": f["size"], "sha256": f["sha256"]}
                          for f in sources["files"]]}
    protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
    print("Weights and source hashes frozen before holdout evaluation:", protocol_path)


if __name__ == "__main__":
    main()
