"""Synthetic-only official metric, CSV, decoder and real-predictor contracts."""
import importlib.util,itertools,json,tempfile,unittest,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from mtah import metrics,features2,mixed_candidate,v4,submission
from mtah.methods import boost
from pff_mixed_score import decoder
from tests.test_audit_alignment import fixture
spec=importlib.util.spec_from_file_location('audit_official',ROOT/'ref/official/evaluation/possession_metrics.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)

def seq(s):return np.array(['HADX'.index(x) for x in s],np.int8)
def cases():
 g=seq('D'*30+'H'*50+'A'*40+'D'*25+'H'*20+'X'*10+'A'*50+'D'*20);out={'perfect':(g,g.copy())}
 for shift in [1,3,8,25]:out['shift'+str(shift)]=(g,np.r_[np.full(shift,3),g[:-shift]])
 out.update({
 'split':(seq('HHHHHH'),seq('HHDAHH')),'merged':(seq('HHDAHH'),seq('HHHHHH')),
 'wrong_team':(seq('HHH'),seq('AAA')),'missing_short':(seq('HHHAHHH'),seq('HHHHHHH')),
 'spurious_short':(seq('HHHHHHH'),seq('HHHAHHH')),'wrong_dead':(seq('HHDDDAA'),seq('HHHHHAA')),
 'H_D_H':(seq('HHDDHH'),seq('HHDDHH')),'H_A_H':(seq('HHAAHH'),seq('HHAAHH')),
 'X_inside':(seq('HHHHHH'),seq('HHXXHH')),'X_transition':(seq('HHHAAA'),seq('HHXXAA')),
 'X_start':(seq('HHHAAA'),seq('XXHAAA')),'all_X':(g,np.full(len(g),3)),
 'constant':(g,np.zeros(len(g),np.int8)),'no_possessions':(seq('DDDD'),seq('DDDD')),
 'halftime':(seq('HHHXXXAAA'),seq('HHHXXXAAA'))})
 return out
class MetricParity(unittest.TestCase):
 def test_all_named_cases_and_random_parity(self):
  allcases=cases();rng=np.random.default_rng(178)
  for i in range(100):allcases['random'+str(i)]=(rng.integers(0,4,100,dtype=np.int8),rng.integers(0,4,100,dtype=np.int8))
  records=[]
  for name,(g,p) in allcases.items():
   with self.subTest(name=name):
    a=official.evaluate(g,p,bootstrap_n=0);b=metrics.evaluate(g,p)
    self.assertAlmostEqual(a['spell_f1_mean'],b['spell_f1'],places=14);self.assertAlmostEqual(a['macro_f1'],b['macro_f1'],places=14)
    self.assertEqual(a['n_pred_spells'],b['n_pred_spells']);records.append({'case':name,'spellF1':a['spell_f1_mean'],'macroF1':a['macro_f1']})
  out=ROOT/'reports/independent_audit';out.mkdir(parents=True,exist_ok=True);(out/'metric_cases.json').write_text(json.dumps(records,indent=2)+'\n')
 def test_ungraded_invariance_and_abstention_semantics(self):
  g=seq('HHHXXXAAA');p=g.copy();p[3:6]=0
  self.assertEqual(official.evaluate(g,p)['spell_f1_mean'],1)
  self.assertEqual(official.evaluate(seq('HHHH'),seq('HXXH'))['spell_f1_mean'],1)
  self.assertLess(official.evaluate(seq('HHHH'),seq('HXXH'))['macro_f1'],official.evaluate(seq('HHHH'),seq('HHHH'))['macro_f1'])
  self.assertEqual(official.possession_spells(seq('HHDDHH'),[(0,6)]),[(0,2,0),(4,6,0)])
class SubmissionContract(unittest.TestCase):
 def test_mutations_rejected_and_strict_extra_column(self):
  import pandas as pd
  good=pd.DataFrame({'frame':range(5),'label':list('HADXX')});self.assertIsNone(official.validate_submission(good,5))
  for kind in ['shift','duplicate','blank','bad_label','float','short']:
   d=good.copy()
   if kind=='shift':d['frame']+=1
   elif kind=='duplicate':d.loc[1,'frame']=0
   elif kind=='blank':d.loc[1,'label']=None
   elif kind=='bad_label':d.loc[1,'label']='Q'
   elif kind=='float':d['frame']=d['frame'].astype(float)
   elif kind=='short':d=d.iloc[:-1]
   self.assertIsNotNone(official.validate_submission(d,5),kind)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'x.csv';good.assign(extra=0).to_csv(p,index=False)
   self.assertIsNone(official.validate_submission(official.read_label_csv(p),5))
   self.assertIsNotNone(submission.validate(p,n_expected=5))
class DecoderContracts(unittest.TestCase):
 def test_restart_and_control_scenarios(self):
  params=decoder()
  for name,s in [('pass','H'*100),('interception','H'*50+'A'*50),('deflection','H'*100),('aerial_duel','H'*100),('failed_tackle','H'*100),('recovery','A'*50+'H'*50),('rebound_control','H'*50+'A'*50),('controlled_reception','A'*50+'H'*50)]+[(x,'H'*50+'D'*50+'A'*50) for x in ['throw_in','corner','free_kick','goal_kick','kickoff','goal_restart','referee_stoppage']]:
   gt=seq(s);pd=np.where(gt==2,.9999,.0001);ph=np.where(gt==0,.9999,.0001)
   pred=boost.label(pd,ph,[(0,len(gt))],len(gt),params);np.testing.assert_array_equal(pred,gt,err_msg=name)
 def test_minimum_duration_explicit_short_loss(self):
  gt=seq('H'*100+'A'*5+'H'*100);pd=np.full(len(gt),.01);ph=np.where(gt==0,.99,.01);p=boost.label(pd,ph,[(0,len(gt))],len(gt),decoder());self.assertFalse(np.array_equal(p,gt))
 def test_half_decoder_resets(self):
  pd=np.full(600,.01);ph=np.r_[np.full(300,.99),np.full(300,.01)];a=boost.label(pd,ph,[(0,300),(300,600)],600,decoder());ph[300:]=.99;b=boost.label(pd,ph,[(0,300),(300,600)],600,decoder());np.testing.assert_array_equal(a[:300],b[:300])
class PredictorContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.assets=v4.load();_,cls.state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
 def predict(self,d):return mixed_candidate.probabilities(d,[(0,300),(300,600)],self.assets,self.state)
 def test_predictor_period_isolation(self):
  d=fixture();a=self.predict(d);d['ball'][300:,:2]=.9;d['players'][300:,:,:2]=.8;b=self.predict(d)
  for x,y in zip(a,b):np.testing.assert_array_equal(x[:300],y[:300])
 def test_slot_permutation_and_team_swap(self):
  d=fixture();a=self.predict(d);perm=[3,0,2,1];b=self.predict(dict(d,players=d['players'][:,perm],slot_team=d['slot_team'][perm]))
  for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-7)
  c=self.predict(dict(d,slot_team=1-d['slot_team']));np.testing.assert_allclose(a[0],c[0],atol=1e-7);np.testing.assert_allclose(a[1],1-c[1],atol=1e-7)
 def test_missing_sensor_and_determinism(self):
  for mode in ['ball','team','all']:
   d=fixture()
   if mode in ['ball','all']:d['ball'][100:500]=np.nan
   if mode=='team':d['players'][100:500,:2]=np.nan
   if mode=='all':d['players'][:]=np.nan
   a=self.predict(d);b=self.predict(d)
   for x,y in zip(a,b):self.assertTrue(np.isfinite(x).all());np.testing.assert_array_equal(x,y)
 def test_reference_rejected(self):
  with self.assertRaises(ValueError):self.predict(dict(fixture(),gt=np.zeros(600)))
if __name__=='__main__':unittest.main()
