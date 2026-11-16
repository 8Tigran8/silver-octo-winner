import unittest
import numpy as np
from tools.visible_contact import contact_cue,contact_probabilities


class ContactContracts(unittest.TestCase):
    def data(self):
        n=100;players=np.zeros((n,8,3));players[:,:,:2]=[.8,.8]
        players[:,:4,:2]=[.1,.1];players[:,0,:2]=np.c_[.3+np.arange(n)*.001,np.full(n,.5)]
        ball=players[:,0].copy();ball[:,0]+=.005
        return dict(ball=ball,players=players,slot_team=np.r_[np.zeros(4,int),np.ones(4,int)],pitch=np.array([105.,68.]))

    def test_measured_dribble_cue_and_team_swap(self):
        d=self.data();cue=contact_cue(d,[(0,100)])
        self.assertTrue((cue[10:]==1).all());self.assertTrue((cue[:10]==0).all())
        swapped=dict(d,slot_team=1-d['slot_team'])
        np.testing.assert_array_equal(contact_cue(swapped,[(0,100)]),-cue)
        ph=np.full(100,.6);pd=np.full(100,.1)
        np.testing.assert_allclose(contact_probabilities(1-ph,pd,-cue),1-contact_probabilities(ph,pd,cue))

    def test_missing_ball_and_period_edges_do_not_create_a_contact(self):
        d=self.data();d['ball'][20:25]=np.nan
        cue=contact_cue(d,[(0,50),(50,100)])
        self.assertTrue((cue[20:35]==0).all());self.assertTrue((cue[50:60]==0).all())

    def test_stationary_ball_and_confident_stoppage_are_unchanged(self):
        d=self.data();d['ball'][:]=d['ball'][0]
        self.assertTrue((contact_cue(d,[(0,100)])==0).all())
        ph=np.array([.2,.8]);np.testing.assert_array_equal(contact_probabilities(ph,np.array([.6,.9]),np.array([1,-1])),ph)


if __name__=='__main__':unittest.main()
