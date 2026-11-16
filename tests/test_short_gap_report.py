import unittest
from tools.report_short_gap import summarize, plan


class RecoveryGateTests(unittest.TestCase):
    def rows(self):
        cfg = plan(); rows = []
        for game in cfg['development_games']:
            for scenario in cfg['scenarios']:
                for reference in ['canonical', 'sdk_shadow']:
                    for candidate in cfg['mechanical_candidates'] + cfg['expert_candidates'][1:]:
                        gain = .03 if candidate == 'short04' else 0.
                        rows.append(dict(game=game, scenario=scenario, reference=reference, candidate=candidate,
                            spell_f1_mean=.6+gain, spell_gain=gain, macro_f1=.8, macro_gain=0.,
                            per_class={k: {'f1': .8} for k in ['H','A','D']},
                            diagnostics=dict(short_reference=20, short_matched=5, missing_live=100,
                                missing_false_dead=20, missing_dead=50, missing_false_live=5)))
        return rows

    def test_average_gain_does_not_hide_a_large_single_game_regression(self):
        rows = self.rows()
        self.assertEqual(summarize(rows)['selected'], 'short04')
        row = next(r for r in rows if r['candidate']=='short04')
        row['spell_gain'] = -.021; row['spell_f1_mean'] = .579
        result = summarize(rows)
        self.assertIsNone(result['selected'])
        affected = next(r for r in result['summary'] if r['candidate']=='short04'
                        and r['scenario']==row['scenario'] and r['reference']==row['reference'])
        self.assertGreater(affected['mean_spell_gain'], .02)
        self.assertFalse(affected['gate_pass'])

    def test_missing_or_duplicate_outcomes_cannot_pass(self):
        rows = self.rows()
        with self.assertRaises(ValueError): summarize(rows[:-1])
        with self.assertRaises(ValueError): summarize(rows + rows[:1])


if __name__ == '__main__': unittest.main()
