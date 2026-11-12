"""Fetch verified, pinned public SkillCorner files outside the v1 training directory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from fetch_external import fetch
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1,5), default=4)
    args = parser.parse_args()
    files = json.loads((ROOT / "tools/skillcorner_sources.json").read_text())["files"]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for result in pool.map(lambda item: fetch(item, ROOT / "external"), files):
            print(*result, flush=True)
if __name__ == "__main__":
    main()
