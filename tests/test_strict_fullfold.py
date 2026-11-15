"""Guard against direct/transitive leakage through inherited model stages."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from strict_fullfold import assert_component_isolation


class StrictLineageTests(unittest.TestCase):
    def setUp(self):
        self.test = ['original/outer','skillcorner/other_outer']
        self.proof = {
            'excluded':self.test.copy(), 'stage1_train':['original/train'],
            'stack_train':['original/train'], 'team_train':['original/train','skillcorner/train'],
            'inner_provenance':[{'inner_test':'original/inner','train':['original/train']}],
        }

    def test_complete_transitive_exclusion_is_accepted(self):
        assert_component_isolation(self.proof,self.test)

    def test_excluding_scored_game_but_training_on_its_fold_peer_is_rejected(self):
        for stage in ['stage1_train','stack_train','team_train']:
            with self.subTest(stage=stage):
                proof = copy.deepcopy(self.proof)
                proof[stage].append('skillcorner/other_outer')
                with self.assertRaises(ValueError):
                    assert_component_isolation(proof,self.test)

    def test_leak_through_inner_first_stage_is_rejected(self):
        for game in self.test+['original/inner']:
            with self.subTest(game=game):
                proof = copy.deepcopy(self.proof)
                proof['inner_provenance'][0]['train'].append(game)
                with self.assertRaises(ValueError):
                    assert_component_isolation(proof,self.test)

    def test_incomplete_exclusion_declaration_is_rejected(self):
        self.proof['excluded'] = self.test[:1]
        with self.assertRaises(ValueError):
            assert_component_isolation(self.proof,self.test)


if __name__=='__main__': unittest.main()
