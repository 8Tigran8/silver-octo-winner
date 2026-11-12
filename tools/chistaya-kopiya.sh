#!/usr/bin/env bash
# Compatibility entry point for the clean-copy verifier. No caller directories are deleted.
# Usage: MTAH_DATA_DIR=/absolute/path/to/data bash tools/chistaya-kopiya.sh --jobs 2
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$(pwd)
exec "$SRC/.venv/bin/python" "$SRC/tools/reproduce.py" --data-dir "${MTAH_DATA_DIR:-$SRC/data}" "$@"
