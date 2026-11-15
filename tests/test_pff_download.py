"""Transport safeguards before consuming the one-shot final reserve."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import fetch_pff_training as fetch


class DownloadTests(unittest.TestCase):
    def session(self,page):
        session=MagicMock();session.__enter__.return_value=session;session.get.return_value=page;return session

    def test_fallback_accepts_normal_public_metadata_response(self):
        page=MagicMock();page.headers={'Content-Type':'application/octet-stream'};page.content=b'[{"id":"demo"}]'
        with tempfile.TemporaryDirectory() as folder,patch.object(fetch.gdown,'download',side_effect=fetch.gdown.exceptions.FileURLRetrievalError('temporary')),patch.object(fetch.requests,'Session',return_value=self.session(page)):
            path=Path(folder)/'metadata.part';fetch.download({'id':'demo','path':'Metadata/demo.json'},path)
            self.assertEqual(json.loads(path.read_bytes()),[{'id':'demo'}])

    def test_unknown_interstitial_and_unexpected_form_are_not_bypassed(self):
        for html in ['Complete CAPTCHA to continue','Google Drive - Virus scan warning <form method="get" action="https://example.org/download"></form>']:
            page=MagicMock();page.headers={'Content-Type':'text/html'};page.text=html;session=self.session(page)
            with self.subTest(html=html),tempfile.TemporaryDirectory() as folder,patch.object(fetch.gdown,'download',side_effect=fetch.gdown.exceptions.FileURLRetrievalError('notice')),patch.object(fetch.requests,'Session',return_value=session):
                path=Path(folder)/'data.part'
                with self.assertRaises(ValueError):fetch.download({'id':'demo','path':'Tracking Data/demo.jsonl.bz2'},path)
                self.assertFalse(path.exists());self.assertEqual(session.get.call_count,1)


if __name__=='__main__':unittest.main()
