"""Regenerate twice from fresh source copies and compare exact CSV bytes.

V1-v3 repeat public-data fitting; v4 verifies frozen external-trained weights
and runs CPU inference. The receipt records this distinction.

Logs, CSVs and the receipt remain local under reports/. Public GT is never scored.
Run after committing the intended source revision and fetching external/reference inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(argv, cwd, env, log):
    with log.open("ab") as stream:
        subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=7200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--method", choices=["v1", "v2", "v3", "v4"], default="v1")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty:
        parser.error("Commit the source changes before checking clean reproduction")
    # Include exactly the files used by the main method, not optional videos.
    inputs = [data / "tactical" / ("BAR-ATH_tactical_atd." + ext) for ext in ("xml", "txt")]
    manifest = json.loads((ROOT / "tools/external_sources.json").read_text())
    if args.method == "v4":
        frozen=json.loads((ROOT/'assets/v4/manifest.json').read_text())
        assets=[ROOT/'assets/v4'/item['path'] for item in frozen['files']]
        if any(not p.is_file() or sha256(p)!=item['sha256'] for p,item in zip(assets,frozen['files'])):
            parser.error('Frozen external-trained model asset verification failed')
        inputs += assets
    else:
        inputs += [ROOT / "external" / item["path"] for item in manifest["files"]]
    extra = []
    if args.method == "v3":
        names = set(json.loads((ROOT / "mtah/expanded_team_training.json").read_text())["skillcorner"])
        extra = [item for item in json.loads((ROOT / "tools/skillcorner_sources.json").read_text())["files"]
                 if item["path"].split("/")[1] in names]
        inputs += [ROOT / "external" / item["path"] for item in extra]
    if any(not p.is_file() for p in inputs):
        parser.error("Required local tracking or pinned external inputs are missing")
    sys.path.insert(0, str(ROOT / "tools"))
    from fetch_external import check
    if args.method != 'v4' and not all(check(ROOT / "external" / item["path"], item) for item in manifest["files"] + extra):
        parser.error("External source verification failed")
    references = json.loads((ROOT / "tools/reference_sources.json").read_text())
    if not all(check(ROOT / item["path"], item) for item in references["files"]):
        parser.error("Run tools/fetch_reference.py to restore the pinned official references")
    before = {str(p): sha256(p) for p in inputs}
    reports = ROOT / "reports" / "reproduction"
    reports.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="check-", dir=reports))
    receipt = {"status": "IN_PROGRESS", "source_commit": commit,
               "official_commit": references["commit"], "jobs": args.jobs, "method": args.method,
               "input_sha256": before, "runs": [], "quality": "Not assessed"}
    receipt['execution_mode']='frozen_external_weights_cpu_inference' if args.method=='v4' else 'raw_input_training_and_inference'
    receipt['training_repeated']=args.method!='v4'
    try:
        archive = folder / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), commit], cwd=ROOT, check=True)
        env = os.environ.copy()
        env.update(MTAH_DATA_DIR=str(data), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
        for number in (1, 2):
            started = time.perf_counter()
            log = folder / f"run-{number}.log"
            output = folder / f"run-{number}.csv"
            with tempfile.TemporaryDirectory(prefix=f"source-{number}-", dir=folder) as temporary:
                clean = Path(temporary)
                with tarfile.open(archive) as source:
                    source.extractall(clean, filter="data")
                if args.method!='v4':
                    (clean / "external").mkdir()
                    for dataset in (("idsse", "metrica", "skillcorner") if args.method == "v3" else ("idsse", "metrica")):
                        (clean / "external" / dataset).symlink_to(ROOT / "external" / dataset,
                                                                  target_is_directory=True)
                    for script in ("tools/ext_idsse.py", "tools/ext_metrica.py"):
                        print(f"Run {number}: converting raw external inputs ({script})", flush=True)
                        run([sys.executable, script], clean, env, log)
                print(f"Run {number}: {'verified frozen-weight CPU inference' if args.method=='v4' else 'training and full-match inference'}", flush=True)
                run([sys.executable, "run.py", args.method, "--jobs", str(args.jobs), "--output", str(output)],
                    clean, env, log)
                validator = ROOT / "ref/official/evaluation/validate_submission.py"
                run([sys.executable, str(validator), str(output)], clean, env, log)
            result = {"run": number, "csv_sha256": sha256(output), "official_format": "PASS",
                      "seconds": round(time.perf_counter() - started, 3)}
            receipt["runs"].append(result)
            print(f"Run {number}: official format PASS, SHA-256 {result['csv_sha256']}", flush=True)
        if before != {str(p): sha256(p) for p in inputs}:
            raise ValueError("Input files changed during reproduction")
        if receipt["runs"][0]["csv_sha256"] != receipt["runs"][1]["csv_sha256"]:
            raise ValueError("Fresh source copies produced different CSV bytes")
        receipt["status"] = "PASS: clean-source reproduction and official CSV format"
        print(receipt["status"], flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["status"] = "FAIL"
        receipt["reason"] = str(error)
        raise
    finally:
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print("Local receipt:", folder / "receipt.json", flush=True)


if __name__ == "__main__":
    main()
