import copy,sys,subprocess,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import final_opened_comparison as audit

class AuditTests(unittest.TestCase):
 def geometry(self):
  n=500;period=np.r_[np.ones(200),np.zeros(100),np.full(200,2)].astype(np.int8)
  players=np.full((n,4,3),.5,np.float32);players[:,:,2]=0;ball=np.full((n,3),.5,np.float32)
  players[period==0]=np.nan;ball[period==0]=np.nan;ball[40:80]=np.nan
  return dict(players=players,ball=ball,period=period,pitch=np.array([105.,68.]),slot_team=np.array([0,0,1,1],np.int8))
 def test_corruption_is_deterministic_preserves_input_gaps_and_clock(self):
  d=self.geometry();before={k:v.copy() for k,v in d.items()}
  a,_=audit.scenario(d,'vv_hz10_noise0p5');b,_=audit.scenario(d,'vv_hz10_noise0p5')
  for k in d:self.assertTrue(np.array_equal(d[k],before[k],equal_nan=True))
  self.assertTrue(np.array_equal(a['players'],b['players'],equal_nan=True));self.assertTrue(np.array_equal(a['ball'],b['ball'],equal_nan=True))
  self.assertTrue(np.isnan(a['ball'][40:80]).all());self.assertTrue(np.isnan(a['players'][d['period']==0]).all());self.assertTrue(np.array_equal(a['period'],d['period']))
 def test_reference_rejected_before_processing(self):
  d=dict(self.geometry(),gt=np.zeros(500))
  with self.assertRaises(ValueError):audit.scenario(d,'natural_detected')
  with self.assertRaises(ValueError):audit.paired(d,[(0,200)],None)
 def test_target_read_guard(self):
  code="import sys;sys.path.insert(0,'tools');import final_opened_comparison as a;a.install_target_guard();open('/tmp/evaluation/gt/forbidden.csv')"
  r=subprocess.run([sys.executable,'-c',code],cwd=ROOT,text=True,capture_output=True)
  self.assertNotEqual(r.returncode,0);self.assertIn('Organizer data/labels blocked',r.stderr)
 def rows(self):
  rows=[];p=audit.plan()
  for g in p['fixtures']:
   for s in p['scenarios']:
    for ref in p['references']:
     for c in [audit.OLD,audit.NEW]:
      rows.append(dict(game=g['game_id'],cohort=g['cohort'],scenario=s,reference=ref,candidate=c,spell_f1_mean=.51 if c==audit.NEW else .5,macro_f1=.8,per_class={k:{'f1':.8} for k in ['H','A','D']},diagnostics=dict(short_reference=10,short_matched=6 if c==audit.NEW else 5,subsecond_reference=1,subsecond_matched=0,missing_live=100,missing_dead=100,missing_false_dead=10,missing_false_live=10,missing_wrong_team=10)))
  return rows
 def test_complete_positive_comparison_retains_new_and_missing_cells_fail(self):
  rows=self.rows();self.assertEqual(audit.summary(rows)['selected'],audit.NEW)
  with self.assertRaises(ValueError):audit.summary(rows[:-1])
  with self.assertRaises(ValueError):audit.summary(rows+[rows[0]])
 def test_single_severe_stress_regression_cannot_hide_in_overall_mean(self):
  rows=self.rows()
  for r in rows:
   if r['candidate']==audit.NEW and r['game']=='10502' and r['scenario']=='vv_hz10_noise2':r['spell_f1_mean']=.4
  result=audit.summary(rows);self.assertEqual(result['selected'],audit.OLD)
  cell=next(x for x in result['comparisons'] if x['cohort']=='all18' and x['scenario']=='vv_hz10_noise2' and x['reference']=='canonical')
  self.assertGreater(cell['gain'],0);self.assertFalse(cell['checks']['worst'])
 def test_false_live_regression_vetoes_mean_spell_gain(self):
  rows=self.rows()
  for r in rows:
   if r['candidate']==audit.NEW and r['scenario']=='natural_detected':r['diagnostics']['missing_false_live']=12
  self.assertEqual(audit.summary(rows)['selected'],audit.OLD)
 def test_short_spell_diagnostic_keeps_dead_as_separator(self):
  gt=np.repeat(np.array([0,2,0,1],np.int8),25);pred=gt.copy();d=audit.diagnostics(gt,pred,np.zeros(len(gt),bool))
  self.assertEqual(d['short_reference'],3);self.assertEqual(d['short_matched'],3);self.assertEqual(d['missing_false_live'],0)
if __name__=='__main__':unittest.main()
