import unittest
from tools.report_unseen_audit import components,paired_summary

class UnseenReportTests(unittest.TestCase):
    def test_team_clusters_include_transitive_dependence(self):
        games={'a':dict(home='X',away='Y'),'b':dict(home='Y',away='Z'),'c':dict(home='Z',away='W'),'d':dict(home='K',away='L')}
        self.assertEqual(components(games),[['a','b','c'],['d']])

    def test_constant_paired_gain_bootstrap_is_exact(self):
        rows=[]
        for g in ['a','b','c']:
            for scenario in ['natural_detected','label_blind_seed7']:
                for reference in ['canonical','sdk_shadow']:
                    for candidate,score in [('v4',.5),('boundary102',.625)]:
                        rows.append(dict(game=g,scenario=scenario,reference=reference,candidate=candidate,spell_f1_mean=score,
                                         macro_f1=.8,per_class={k:dict(f1=.8) for k in ['H','A','D']}))
        for result in paired_summary(rows,['a','b','c'],[['a','b'],['c']]).values():
            self.assertEqual(result['spell_gain'],.125);self.assertEqual(result['wins'],3)
            self.assertEqual(result['descriptive_match_bootstrap_gain_95'],[.125,.125])
            self.assertEqual(result['descriptive_shared_team_cluster_bootstrap_gain_95'],[.125,.125])

if __name__=='__main__':unittest.main()
