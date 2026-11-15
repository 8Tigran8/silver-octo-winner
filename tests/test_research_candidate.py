import unittest
import numpy as np
from mtah import v4,research_candidate


class ResearchAdapterTests(unittest.TestCase):
    def test_adapter_matches_v4_control_and_ignores_reference_labels(self):
        n=900;p=np.zeros((n,22,3),np.float32);p[:,:11,0]=.25;p[:,11:,0]=.75;p[:,:,1]=np.linspace(.1,.9,22)
        data={'players':p,'ball':np.tile([.5,.5,0.],(n,1)),'slot_team':np.repeat([0,1],11),'pitch':(105.,68.),'gt':np.zeros(n,np.int8)}
        spans=[(0,450),(450,n)];assets=v4.load();states=[(assets[2],assets[3],assets[4])];spec={'kind':'fixed','dead':.5,'home':.5}
        bd,bh=v4.probabilities(data,spans,assets);d,h=research_candidate.probabilities(data,spans,assets,states,spec)
        np.testing.assert_array_equal(d,bd);np.testing.assert_array_equal(h,bh)
        data['gt'][:]=3;dd,hh=research_candidate.probabilities(data,spans,assets,states,spec)
        np.testing.assert_array_equal(d,dd);np.testing.assert_array_equal(h,hh)


if __name__=='__main__':unittest.main()
