# Clean-room reproduction

PASS for the audited source commit and local macOS ARM platform. `reproduction.json` and root `submission_manifest.json` contain the exact commit, package versions, source/model/input identities and commands.

A new virtual environment was created with Python 3.12.11, then all pinned requirements were installed from package distributions. No packages were copied from the working venv. Cached package wheels are installation artifacts, not model/data caches. A fresh `git archive` was extracted without untracked files, with a minimal execution environment. Only the explicit organizer input directory was provided. No preprocessing cache or external training data was copied.

Run 1: 58.872 s; run 2: 28.555 s. Both use the new environment, run complete preprocessing/inference and pass the unchanged current official validator. The second run exercises the newly content-keyed cache. Both produce SHA-256 `cfbd16ea35075a199d2401c18f162d4c1f953be4049cd776d502ff42ea389c97`, identical to the preserved pre-audit submission. Input hashes remain unchanged.

This proves repeated inference for these source/weight/input bytes on this platform. It does not repeat training, prove all platforms, or establish availability of original training data to every organizer. The earlier VV Linux ARM result concerns the pre-audit source archive; the corrected source has not yet received an independent Linux rerun.

A full-length synthetic XML/TXT fixture also passes the actual archived-source CLI and official CSV validator. This checks parser-to-CSV execution and frame-count handling. The deliberately sparse four-slot stationary-ball fixture produced all D within playing blocks (zero live possessions); it does NOT demonstrate learned recognition of possession from toy geometry. The probability-conditioned semantic tests and full-model execution checks have different scopes.

The bounded all-history scan covers 878 reachable named objects, 525 text blobs, protected asset-name patterns, >15MiB blobs and recognized credential formats. No hits were found. This does not prove absence of disguised data or every secret format. Protected original assets and per-frame outputs remain ignored and local.

Final fail-closed gate: `python tools/final_submission_gate.py --csv "out/audit-ready/Ruses Cules.csv"`. It checks exact CSV bytes, UTF-8/two-column row contract, inference source/model/config hashes and current official validator hashes, then invokes the official validator. Report copies of failure logs normalize trailing whitespace; original logs remain unchanged locally.
