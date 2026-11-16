"""Run committed inference twice in local Linux, with no network or GT mount."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image', required=True, help='Locally built pinned Python/dependency runtime')
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    source.mkdir()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    archive = out / 'source.tar'
    subprocess.run(['git', 'archive', '-o', str(archive), commit], cwd=ROOT, check=True)
    with tarfile.open(archive) as stream:
        stream.extractall(source, filter='data')
    shutil.copytree(ROOT / 'reports/independent_audit/official', source / 'ref/official')
    official_hashes = json.loads((ROOT / 'validation/official_sources.json').read_text())['sha256']
    for name, digest in official_hashes.items():
        if sha(source / 'ref/official' / name) != digest:
            raise ValueError('Unpinned official code: ' + name)
    data = args.data_dir.resolve()
    inputs = {f'tactical/BAR-ATH_tactical_atd.{ext}': sha(data / 'tactical' / f'BAR-ATH_tactical_atd.{ext}') for ext in ['xml', 'txt']}
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image], text=True))[0]
    # Only the two consumed tracking files are mounted, never target truth/video.
    prefix = ['docker', 'run', '--rm', '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,size=256m',
              '--user', f'{os.getuid()}:{os.getgid()}', '--workdir', '/work',
              '-v', f'{source}:/work', '-v', f'{out}:/result']
    for name in inputs:
        prefix += ['-v', f'{data/name}:/data/{name}:ro']
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        prefix += ['-e', name + '=1']
    prefix += ['-e', 'PYTHONHASHSEED=0', image['Id']]
    metadata_code = "import platform,subprocess,json;print(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),dependencies=subprocess.check_output(['python','-m','pip','freeze'],text=True).splitlines())))"
    runtime = json.loads(subprocess.check_output(prefix + ['python', '-c', metadata_code], text=True))
    rows = []
    for i in (1, 2):
        start = time.monotonic()
        target = out / f'linux-run-{i}.csv'
        with (out / f'run-{i}.log').open('w') as log:
            subprocess.run(prefix + ['python', 'tools/generate_mixed_candidate.py', '--bundle', 'assets/pff-mixed-candidate/bundle.json', '--data-dir', '/data', '--output', f'/result/{target.name}'], stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(prefix + ['python', 'ref/official/evaluation/validate_submission.py', f'/result/{target.name}'], stdout=log, stderr=subprocess.STDOUT, check=True)
        rows.append(dict(run=i, seconds=time.monotonic()-start, sha256=sha(target), official_validator='PASS'))
        print(rows[-1], flush=True)
    original = sha(ROOT / 'reports/independent_audit/pre_audit.csv')
    unchanged = inputs == {name: sha(data/name) for name in inputs}
    receipt = dict(source_commit=commit, image_id=image['Id'], image_architecture=image['Architecture'], runtime=runtime,
                   input_sha256=inputs, inputs_unchanged=unchanged, runs=rows, original_csv_sha256=original,
                   all_outputs_identical=len({original, *(r['sha256'] for r in rows)}) == 1,
                   network='disabled during both inference runs', scope='Local Linux container on ARM host, not independent hardware/operator')
    (out / 'reproduction.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not unchanged or not receipt['all_outputs_identical']:
        raise ValueError('Reproduction mismatch; inspect receipt before release')


if __name__ == '__main__':
    main()
