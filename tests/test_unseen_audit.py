"""Pre-opening guards for the prospective six-match audit."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import pff_unseen_audit as audit

class UnseenAuditTests(unittest.TestCase):
    def test_registry_disjoint_and_partial_country_novelty(self):
        protocol,games=audit.registry()
        known={g[side] for split in ['training','development'] for g in protocol[split] for side in ['home','away']}
        self.assertEqual(len(games),6)
        for g in games.values():self.assertEqual(sum(g[s] in known for s in ['home','away']),1)

    def test_existing_content_blocks_unopened_claim(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(audit,'ROOT',Path(folder)):
            audit.ensure_unopened()
            file=Path(folder)/'external/pff/Metadata/10512.json';file.parent.mkdir(parents=True);file.write_text('{}')
            with self.assertRaisesRegex(ValueError,'already present'):audit.ensure_unopened()

    def test_changed_source_invalidates_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);doc=root/'doc';doc.mkdir();file=root/'model';file.write_text('frozen')
            (doc/'freeze.json').write_text(json.dumps({'files_sha256':{'model':audit.sha(file)}}))
            with patch.object(audit,'ROOT',root),patch.object(audit,'DOC',doc):
                audit.check_freeze();file.write_text('changed')
                with self.assertRaisesRegex(ValueError,'changed'):audit.check_freeze()

if __name__=='__main__':unittest.main()
