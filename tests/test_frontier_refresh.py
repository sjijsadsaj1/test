import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from avic_bridge.controller import AvicController, AvicControllerConfig
from avic_bridge.frontier_once import refresh_threshold, REFRESH_CONTRACT
from avic_bridge.protocol import validate_request


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, APEXNAV_FRONTIER_REFRESH_DELTA='0.03')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.client = Mock()
        self.requests = []

        def decide(payload):
            validate_request(payload)
            self.requests.append(payload)
            return dict(decision='call_wm', candidate_scores=[dict(
                plan_id=payload['navigation']['candidate_plans'][0]['plan_id'],
                imagined_itm_score=.1 + .01*len(self.requests))])
        self.client.decide.side_effect = decide
        self.c = AvicController(AvicControllerConfig(mode='avic', score_mode='frontier_once'),
                                client=self.client)

    def run_step(self, step, real, query=None, actions=None):
        actions = actions or [1, 1, 1]
        rgb = np.full((24, 32, 3), step % 255, dtype=np.uint8)
        self.c.record_observation(rgb=rgb, robot_x=0, robot_y=0, robot_yaw_rad=0,
            camera_pitch_rad=0, step_index=step, itm_cosine=.2)
        f = dict(rank=0, x=2., y=1., semantic_score=real, first_action=actions[0],
                 path_action_prefix=actions, relative_yaw_rad=0,
                 query_id=query if query is not None else step, query_active=True)
        return self.c.decide_action(rgb=rgb, baseline_action=1, target_category='chair',
            target_room='everywhere', scene_id='scene', episode_id='1', step_index=step,
            camera_pitch_rad=0, itm_cosine=.2, final_state=0, exploration_result=0,
            frontier_candidates=[f])

    def test_accumulation_boundary_both_directions_and_new_reference(self):
        self.run_step(26, .20)
        for step, value in [(27, .21), (28, .22), (29, .229)]:
            self.assertEqual(self.run_step(step, value).cache_hits, 1)
        decision = self.run_step(30, .23)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(decision.reason, REFRESH_CONTRACT)
        self.assertAlmostEqual(decision.frontier_scores[0]['decision_semantic_score'], (.23+.12)/2)
        self.assertEqual(self.run_step(31, .24).cache_hits, 1)
        self.run_step(32, .20)
        self.assertEqual(len(self.requests), 3)
        meta = self.requests[-1]['frontier_refresh']
        self.assertAlmostEqual(meta['previous_real_score'], .23)
        self.assertEqual(meta['prediction_number'], 3)
        self.assertEqual(meta['previous_prediction_step'], 30)

    def test_refresh_uses_current_image_route_and_new_generation_scope(self):
        self.run_step(26, .2)
        self.run_step(27, .25, actions=[2, 1, 3])
        first, later = self.requests
        self.assertNotEqual(first['observation'], later['observation'])
        self.assertEqual([a['id'] for a in later['navigation']['candidate_plans'][0]['actions']], [2, 1, 3])
        self.assertNotEqual(first['episode']['generation_epoch'], later['episode']['generation_epoch'])
        self.assertIn(':refresh:', later['episode']['generation_epoch'])
        self.assertEqual(later['frontier_refresh']['reason'], 'real_score_delta')

    def test_disabled_preserves_once_even_after_large_change(self):
        os.environ['APEXNAV_FRONTIER_REFRESH_DELTA'] = '0'
        self.run_step(26, .1)
        result = self.run_step(1000, .9)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(result.cache_hits, 1)
        self.assertNotIn('frontier_refresh', self.requests[0])

    def test_duplicate_transaction_and_episode_reset(self):
        self.run_step(26, .2, query=1)
        self.run_step(27, .4, query=1)
        self.assertEqual(len(self.requests), 1)
        self.c.reset_episode()
        self.run_step(26, .4, query=1)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[-1]['frontier_refresh']['prediction_number'], 1)
        self.assertEqual(self.requests[-1]['frontier_refresh']['reason'], 'first_observation')

    def test_failed_refresh_does_not_replace_cached_reference(self):
        self.run_step(26, .2)
        before = dict(self.c.once_registry.entries[0])
        self.client.decide.side_effect = RuntimeError('service unavailable')
        with self.assertRaises(RuntimeError):
            self.run_step(27, .25)
        self.assertEqual(self.c.once_registry.entries[0], before)

    def test_invalid_threshold_and_real_score_rejected(self):
        for value in ['nan', 'inf', '-1', '3', 'bad']:
            with patch.dict(os.environ, APEXNAV_FRONTIER_REFRESH_DELTA=value):
                with self.assertRaises(ValueError):
                    refresh_threshold()
        with self.assertRaises(RuntimeError):
            self.run_step(26, float('nan'))


if __name__ == '__main__':
    unittest.main()
