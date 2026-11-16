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
