import unittest
from tools.report_team_gap_specialist import summarize
class OwnershipReportContracts(unittest.TestCase):
    def test_incomplete_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):summarize([],['control'],dict(development_games=['x']))
    def test_gain_does_not_hide_more_wrong_team_errors(self):
        cfg=dict(development_games=['x'],gate_each_scenario=dict(mean_spell_gain_at_least=.02,positive_games_at_least=1,worst_game_spell_gain_at_least=-.02,mean_macro_gain_at_least=-.002));rows=[]
        for s in ['natural_detected','label_blind_seed7']:
            for ref in ['canonical','sdk_shadow']:
                for c in ['control','candidate']:
                    rows.append(dict(game='x',scenario=s,reference=ref,candidate=c,spell_f1_mean=.6 if c=='control' else .65,macro_f1=.8,diagnostics=dict(short_reference=10,short_matched=2,missing_live=100,missing_wrong_team=10 if c=='control' else 11,missing_false_dead=5)))
        result=summarize(rows,['control','candidate'],cfg);self.assertIsNone(result['selected'])
        self.assertTrue(all(r['gates']['mean_gain'] and not r['gates']['missing_wrong_team'] for r in result['summary'] if r['candidate']=='candidate'))
if __name__=='__main__':unittest.main()
