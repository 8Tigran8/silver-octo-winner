"""Fetch pinned public organizer references and the unmodified official validator."""
import json
from pathlib import Path

from fetch_external import fetch

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads(Path(__file__).with_name("reference_sources.json").read_text())
    for item in manifest["files"]:
        name, status = fetch(item, ROOT)
        print(f"{name}: {status}")
    print("Official reference commit:", manifest["commit"])


if __name__ == "__main__":
    main()
