import unittest
from pathlib import Path
import numpy as np
from mtah import final_candidate
from mtah.recovery_features import activity_features
from mtah.config import PITCH


class FinalInferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle,cls.state=final_candidate.load_bundle(Path(__file__).resolve().parents[1]/'assets/final-candidate/bundle.json')

    def data(self):
        n=500;t=np.arange(n)/25;rng=np.random.default_rng(73)
        players=np.empty((n,22,3),np.float32)
        for j in range(22):
            players[:,j,0]=.1+.8*(j%11)/10+.01*np.sin(t*.3+j)
            players[:,j,1]=.15+.7*(j//11)+.01*np.cos(t*.2+j)
        players[:,:,2]=.7
        ball=players[:,3].copy();ball[:,0]+=.005;ball[80:130,:2]=np.nan
        return dict(ball=ball,players=players,slot_team=np.repeat([0,1],11),pitch=np.asarray(PITCH))

    def test_full_predictor_period_isolation(self):
        d=self.data();parts=[(0,250),(250,500)]
        baseline=final_candidate.probabilities(d,parts,self.state)
        changed=dict(d,ball=d['ball'].copy(),players=d['players'].copy())
        changed['ball'][:250]=np.nan;changed['players'][:250,:,:2]=.8
        actual=final_candidate.probabilities(changed,parts,self.state)
        for before,after in zip(baseline,actual):np.testing.assert_array_equal(before[250:],after[250:])

    def test_team_swap_preserves_dead_probability_and_complements_owner(self):
        d=self.data();parts=[(0,500)]
        pd,ph=final_candidate.probabilities(d,parts,self.state)
        sd,sh=final_candidate.probabilities(dict(d,slot_team=1-d['slot_team']),parts,self.state)
        np.testing.assert_allclose(pd,sd,atol=1e-7,rtol=0)
        np.testing.assert_allclose(ph,1-sh,atol=1e-7,rtol=0)
        self.assertTrue(np.isfinite(pd).all() and ((pd>=0)&(pd<=1)).all())

    def test_rejects_reference_labels(self):
        with self.assertRaises(ValueError):final_candidate.probabilities(dict(self.data(),gt=np.zeros(500)),[(0,500)],self.state)

    def test_observed_probabilities_equal_parent(self):
        from mtah import recovery_candidate
        d=self.data();parts=[(0,500)];p=final_candidate.probabilities(d,parts,self.state);q=recovery_candidate.probabilities(d,parts,self.state[0]);seen=np.isfinite(d['ball'][:,:2]).all(axis=1)
        for a,b in zip(p,q):np.testing.assert_array_equal(a[seen],b[seen])

    def test_atd_default_pitch_and_partial_ball_contracts(self):
        d=self.data();without={k:v for k,v in d.items() if k!='pitch'}
        np.testing.assert_array_equal(activity_features(d,[(0,500)])[0],activity_features(without,[(0,500)])[0])
        one=dict(d,ball=d['ball'].copy());two=dict(d,ball=d['ball'].copy())
        one['ball'][300:330,1]=np.nan;two['ball'][300:330,:2]=np.nan
        for x,y in zip(final_candidate.probabilities(one,[(0,500)],self.state),final_candidate.probabilities(two,[(0,500)],self.state)):
            np.testing.assert_array_equal(x,y)


if __name__=='__main__':unittest.main()
