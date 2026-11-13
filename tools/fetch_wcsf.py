"""Fetch pinned public validation-only research source; do not redistribute it."""
import json
from pathlib import Path
from fetch_external import fetch
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 for item in json.loads((ROOT/'tools/wcsf_sources.json').read_text())['files']:print(*fetch(item,ROOT/'external'),flush=True)
