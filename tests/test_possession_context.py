"""Ball-observation context must respect missing evidence and period isolation."""
import unittest
import numpy as np
from mtah import possession_context as context


class ContextTests(unittest.TestCase):
    def fixture(self):
        names=list(dict.fromkeys(["seen"]+list(context.CARRY)))
        x=np.zeros((8,len(names)),np.float32)
        x[:,names.index("seen")]=[1,0,0,1,1,0,0,1]
        x[:,names.index("margin")]=[8,999,999,6,-4,999,999,-2]
        return x,names

    def test_carries_only_actual_observations(self):
        x,names=self.fixture();out,cols=context.extend(x,names,[(0,4),(4,8)])
        np.testing.assert_array_equal(out[:,cols.index("observed_margin_previous")],[8,8,8,6,-4,-4,-4,-2])
        np.testing.assert_array_equal(out[:,cols.index("observed_margin_following")],[8,6,6,6,-4,-2,-2,-2])

    def test_other_half_cannot_change_context(self):
        x,names=self.fixture();a,_=context.extend(x,names,[(0,4),(4,8)])
        x[4:]=10000;b,_=context.extend(x,names,[(0,4),(4,8)])
        np.testing.assert_allclose(a[:4],b[:4],equal_nan=True)

    def test_all_missing_context_stays_unknown(self):
        x,names=self.fixture();x[:,names.index("seen")]=0
        a,_=context.extend(x,names,[(0,8)])
        self.assertTrue(np.isnan(a[:,len(names):]).all())


if __name__=="__main__":unittest.main()
