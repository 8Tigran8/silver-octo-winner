"""Download pinned external training matches; verify size and publisher object hashes.

Only public matches from IDSSE and Metrica sample-data are downloaded here.
The confidential competition match must be obtained separately from the organizer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("external_sources.json")


def check(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    digest = hashlib.sha256() if "sha256" in item else hashlib.sha1()
    if "sha256" not in item:
        digest.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item.get("sha256", item.get("git_blob_sha1"))


def fetch(item, destination):
    target = destination / item["path"]
    if check(target, item):
        return item["path"], "verified existing"
    if target.exists():
        raise ValueError(f"Existing file differs from pinned source: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(item["url"], headers={"User-Agent": "Ruses-Cules-reproduction/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if not check(temporary, item):
        raise ValueError(f"Downloaded file failed source hash verification: {target.name}")
    temporary.replace(target)
    return item["path"], "downloaded and verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "external")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    files = manifest["files"]
    print(f"Public external bundle: {len(files)} files, {sum(f['size'] for f in files):,} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for name, status in executor.map(lambda item: fetch(item, args.destination), files):
            print(f"{name}: {status}", flush=True)
    print("PASS: external source hashes verified", flush=True)


if __name__ == "__main__":
    main()
