import unittest
import numpy as np
from tools.recovery_opened_audit import diagnostic,summary

class OpenedAuditContracts(unittest.TestCase):
    def data(self,n):
        return dict(ball=np.zeros((n,3)),period=np.ones(n,np.int8))
    def test_perfect_and_missing_ball_confusion_are_distinct(self):
        gt=np.repeat([0,2,1],50).astype(np.int8);d=self.data(len(gt));d['ball'][:]=np.nan
        r=diagnostic(gt,gt.copy(),d)
        self.assertEqual(sum(x['matched'] for x in r['short_spells'].values()),2)
        self.assertEqual(sum(r['missed_at_iou05'].values()),0)
        self.assertEqual(r['observation']['missing']['false_dead'],0)
        pred=np.full(len(gt),2,np.int8);r=diagnostic(gt,pred,d)
        self.assertEqual(r['observation']['missing']['false_dead'],100)
        self.assertEqual(r['missed_at_iou05']['mostly_dead'],2)
    def test_merged_possessions_do_not_count_as_perfect(self):
        gt=np.r_[np.zeros(75),np.ones(25),np.zeros(75)].astype(np.int8)
        r=diagnostic(gt,np.zeros(len(gt),np.int8),self.data(len(gt)))
        self.assertEqual(r['predictions_overlapping_multiple_same_team_reference_spells'],1)
        self.assertGreater(sum(r['missed_at_iou05'].values()),0)
        self.assertEqual(r['missed_at_iou05']['mostly_opponent'],1)
    def test_summary_refuses_incomplete_grid(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):summary([])

if __name__=='__main__':unittest.main()

class ReportingRecoveryContracts(unittest.TestCase):
    def test_finish_saved_uses_original_periods_without_touching_predictions(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from unittest.mock import patch
        import tools.recovery_opened_audit as audit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);doc=root/'doc';out=root/'out';old=root/'old'
            for p in [doc,out,old]:p.mkdir()
            folder=out/'fixture';folder.mkdir()
            (doc/'freeze.json').write_text('{}');(doc/'reporting_amendment.json').write_text('{}')
            gt=np.repeat([0,2,1],50).astype(np.int8);preds={s+'__'+m:gt.copy() for s in audit.SCENARIOS for m in ['boundary102','weighted_short_gap']}
            predfile=folder/'predictions.npz';np.savez_compressed(predfile,**preds);before=predfile.read_bytes()
            np.savez_compressed(folder/'diagnostic_inputs.npz',gt=gt,alternate=gt,period=np.ones(len(gt),np.int8),**{s+'__seen':np.ones(len(gt),bool) for s in audit.SCENARIOS})
            np.savez_compressed(old/'fixture_predictions.npz',**preds)
            records=[dict(scenario=s,reference=r,candidate='boundary102',**audit.metric(gt,gt)) for s in audit.SCENARIOS for r in ['canonical','sdk_shadow']]
            (old/'fixture_scores.json').write_text(json.dumps(dict(records=records,source=dict(geometry=dict(frames25=len(gt)),reference=dict(class_frames=dict(H=50,A=50,D=50,X=0))))))
            fixture=dict(game_id='fixture',cohort='bridge6',historical_scores='old/fixture_scores.json')
            with patch.object(audit,'ROOT',root),patch.object(audit,'DOC',doc),patch.object(audit,'OUT',out):result=audit.finish_saved(fixture)
            self.assertEqual(len(result['records']),8);self.assertEqual(len(result['diagnostics']),4)
            self.assertEqual(predfile.read_bytes(),before)
            self.assertTrue(all(r['spell_f1_mean']==1 for r in result['records']))
            self.assertTrue(all(r['per_period'][0]['period']==1 for r in result['diagnostics']))
