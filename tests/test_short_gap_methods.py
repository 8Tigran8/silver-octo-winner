import unittest
import numpy as np
from tools.short_gap_methods import activity_features, bridge_dead, policy


class RecoveryContracts(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(42)
        xy = rng.uniform(.1, .9, (120, 6, 3))
        return dict(ball=np.full((120, 3), np.nan), players=xy, pitch=np.array([105., 68.]))

    def test_activity_is_period_local_and_ball_independent(self):
        d = self.data(); parts = [(0, 60), (60, 120)]
        x, names = activity_features(d, parts)
        changed = dict(d, ball=np.ones((120, 3)), players=d['players'].copy())
        changed['players'][:60] = 0
        y, yn = activity_features(changed, parts)
        np.testing.assert_array_equal(x[60:], y[60:]); self.assertEqual(names, yn)
        changed['players'] = d['players']
        np.testing.assert_array_equal(x, activity_features(changed, parts)[0])

    def test_activity_slot_and_pitch_reflection_invariance(self):
        d = self.data(); x, _ = activity_features(d, [(0, 120)])
        other = dict(d, players=d['players'][:, ::-1].copy())
        other['players'][:, :, :2] = 1-other['players'][:, :, :2]
        np.testing.assert_allclose(x, activity_features(other, [(0, 120)])[0], rtol=1e-5, atol=1e-5)

    def test_bridge_does_not_cross_period_or_erase_visible_evidence(self):
        b = np.full((400, 3), .5); pd = np.full(400, .1)
        b[20:50] = np.nan; pd[20:50] = .8
        b[190:210] = np.nan; pd[190:210] = .8
        out = bridge_dead(pd, b, [(0, 200), (200, 400)])
        self.assertTrue((out[20:50] < pd[20:50]).all())
        np.testing.assert_array_equal(out[190:210], pd[190:210])
        np.testing.assert_array_equal(out[np.isfinite(b[:, 0])], pd[np.isfinite(b[:, 0])])

    def test_gap_neural_preserves_seen_frames_and_team_probabilities(self):
        c = dict(pd=np.array([.2, .9]), ph=np.array([.3, .7]), nd=np.array([.4, .6]), seen=np.array([True, False]))
        pd, ph, p = policy('neural', c, {'min_team_s': 1., 'cost_ha': 9.})
        np.testing.assert_array_equal(pd, [.2, .6]); np.testing.assert_array_equal(ph, c['ph'])
        self.assertEqual(p['min_team_s'], 1.)

    def test_short_policy_recovers_clear_brief_control_without_extra_switches(self):
        from mtah.methods import boost
        n = 500; ph = np.full(n, .999); ph[200:210] = .001
        cache = dict(pd=np.full(n, .01), ph=ph)
        base = boost.decode_segment(cache['pd'], ph, boost.DECODER)
        pd, new_ph, params = policy('short04_ha45', cache, boost.DECODER)
        recovered = boost.decode_segment(pd, new_ph, params)
        self.assertEqual(int((base == 1).sum()), 0)
        expected = np.zeros(n, np.int8); expected[200:210] = 1
        np.testing.assert_array_equal(recovered, expected)

    def test_only_registered_train_or_development_games_are_accepted(self):
        from tools.short_gap_study import guard, plan
        for split in ['training_games', 'development_games']:
            guard(plan()[split][0], split)
            for forbidden in ['10512', '10511', '10509', '10507', '10505', '10502', 'target']:
                with self.assertRaises(ValueError): guard(forbidden, split)


if __name__ == '__main__': unittest.main()
