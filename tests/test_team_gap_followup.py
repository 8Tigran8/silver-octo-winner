import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from team_gap_followup import route

class RoutingContracts(unittest.TestCase):
    def test_missing_only_routing_preserves_seen_probabilities(self):
        c=dict(ph=np.array([.1,.2,.3,.4]),bh=np.array([.5,.6,.7,.8]),seen=np.array([1,0,1,0],bool))
        neural=dict(uniform=np.array([.8,.7,.6,.5]),focus=np.array([.4,.3,.2,.1]))
        for name in ['uniform_missing_only','focus_missing_only']:
            p=route(name,c,neural);assert np.array_equal(p[c['seen']],c['ph'][c['seen']]);assert np.array_equal(p[~c['seen']],neural[name.split('_')[0]][~c['seen']])
        assert np.array_equal(c['ph'],[.1,.2,.3,.4])

    def test_routing_commutes_with_team_swap(self):
        c=dict(ph=np.array([.1,.2,.3,.4]),bh=np.array([.5,.6,.7,.8]),seen=np.array([1,0,1,0],bool));n=dict(uniform=np.array([.8,.7,.6,.5]),focus=np.array([.4,.3,.2,.1]))
        swapped=dict(ph=1-c['ph'],bh=1-c['bh'],seen=c['seen'])
        for name in ['control','uniform_gap100','uniform_missing_only','focus_missing_only']:
            assert np.allclose(route(name,swapped,{k:1-v for k,v in n.items()}),1-route(name,c,n))
