import base64
import copy
import unittest
from unittest.mock import patch

from avic_bridge.service import grounded_view_delta, QwenOpenAIClient, SVCQwenBackend
from avic_bridge.protocol import CURRENT_ROUTE_CONTRACT
from test_avic_frontier_policy import FakeSVC, FakeHorizonClient, payload, frontier


def current_route_payload(frontiers):
    p = payload(frontiers)
    p['score_policy']['mode'] = 'grounded_delta'
    p['episode'] = {'step_index': 26}
    p['navigation']['route_contract'] = CURRENT_ROUTE_CONTRACT
    p['observation']['rgb_jpeg_b64'] = p['navigation']['candidate_plans'][0]['frontier_anchor']['rgb_jpeg_b64']
    for plan in p['navigation']['candidate_plans']:
        plan['actions'] = [{'id': a} for a in plan['source_frontier']['path_action_prefix']]
        plan.update(route_contract=CURRENT_ROUTE_CONTRACT, min_prefix_length=2)
        plan['frontier_anchor'].update(binding_source='current_observation', step_index=26,
                                      rgb_jpeg_b64=p['observation']['rgb_jpeg_b64'])
    return p


class GroundedDeltaTests(unittest.TestCase):
    def setUp(self):
        self.e = dict(reliability=0.8, evidence_gain=0., semantic_relation="real cabinet",
                      reason="No new information; existing cabinet remains visible.", anchor_cue="cabinet",
                      new_cue="", supported_by_real_anchor=True, adds_new_information=False,
                      target_only_in_imagination=False, support_frame_orders=[1,2],
                      consistency_reason='The same room and cabinet geometry is preserved.')
        self.frames = [{"order": 1}, {"order": 2}, {"order": 3}]
        self.policy = dict(real_weight=.25, imagination_weight=.75, max_imagination_bonus=.06)

    def score(self, e=None, values=None, policy=None):
        return grounded_view_delta(e or self.e, self.frames, .20,
                                   values or [.25, .26, .27], policy or self.policy)

    def test_weighted_delta_not_novel_object_and_cap(self):
        b, d = self.score()
        self.assertAlmostEqual(b, .75 * .8 * .055)
        self.assertAlmostEqual(d['future_itm_q25'], .255)
        policy = dict(self.policy, imagination_weight=0., real_weight=1.)
        self.assertEqual(self.score(policy=policy)[0], 0.)
        self.assertAlmostEqual(self.score(policy=dict(self.policy, max_imagination_bonus=.01))[0], .01)

    def test_hallucination_ungrounded_or_low_reliability_stays_zero(self):
        for changes in [dict(target_only_in_imagination=True), dict(supported_by_real_anchor=False),
                        dict(reliability=.2), dict(anchor_cue=''),
                        dict(consistency_reason='a different room'), dict(support_frame_orders=[])]:
            self.assertEqual(self.score(e=dict(self.e, **changes))[0], 0.)

    def test_single_peak_flat_decline_nan_and_invalid_frame_stay_zero(self):
        for values in [[.2,.2,.9], [.20,.20,.20], [.1,.15,.19], [.3,float('nan'),.3]]:
            self.assertEqual(self.score(values=values)[0], 0.)
        self.assertEqual(self.score(e=dict(self.e,support_frame_orders=[9]))[0], 0.)
        self.assertEqual(grounded_view_delta(self.e, [{'order':1}], .2, [.4], self.policy)[0], 0.)

    def test_backend_rejects_nonfinite_blip_without_serializing_nan(self):
        class BadITM:
            def cosine(self, *args):
                return float('nan')
        backend = SVCQwenBackend()
        backend._svc = FakeSVC()
        backend._itm = BadITM()
        backend._qwen = FakeHorizonClient([dict(plan_id='frontier_0',prefix_length=1,reason='test')])
        with self.assertRaisesRegex(RuntimeError, 'BLIP-2 failed'):
            backend.decide(payload([frontier(0,.2,1.,0.,1)]))

    def test_prompt_declares_frame_indices_and_enforces_full_pairing(self):
        q = QwenOpenAIClient.__new__(QwenOpenAIClient)
        q.policy_model = 'fake'
        seen = {}
        def complete(*args, **kwargs):
            seen['content'] = args[1]; seen.update(kwargs); return {}
        q._complete = complete
        c = dict(plan_id='frontier_0', source_frontier={}, frontier_anchor={},
                 future_frames=[dict(order=1,jpeg_b64='a'),dict(order=2,jpeg_b64='b')],frame_itm_scores=[.2,.3])
        q.evaluate_frontier_sequences({'target':{}}, [c])
        self.assertEqual(sum(x['type']=='image_url' for x in seen['content']), 3)
        self.assertIn('allowed support_frame_orders: [1, 2]', str(seen['content']))
        schema = seen['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['evaluations']['items']['properties']['support_frame_orders']['items']['enum'],[1,2])
        with self.assertRaises(ValueError):
            q.evaluate_frontier_sequences({'target':{}}, [dict(c,frame_itm_scores=[.2])])

    def test_backend_anchor_scoring_bonus_transport_and_duplicate_guard(self):
        class TwoFrameSVC(FakeSVC):
            def generate_plan(self, **kwargs):
                d = super().generate_plan(**kwargs)
                d['future_frames'] = [dict(d['future_frames'][0], order=i) for i in (1,2)]
                return d
        class ITM:
            def cosine(self, image, target, room):
                return .2 if base64.b64decode(image).startswith(b'real-anchor') else .3
        p = current_route_payload([frontier(0,.2,1.,0.,1), frontier(1,.21,1.2,.1,2)])
        p['score_policy'].update(self.policy, mode='grounded_delta')
        for duplicate in (False,True):
            backend = SVCQwenBackend()
            backend._svc = TwoFrameSVC(duplicate_frames=duplicate)
            backend._itm = ITM()
            backend._qwen = FakeHorizonClient([
                dict(plan_id='frontier_%d'%i,prefix_length=1,reason='test') for i in (0,1)],
                evaluations=[dict(self.e,plan_id='frontier_%d'%i) for i in (0,1)])
            result = backend.decide(copy.deepcopy(p))
            self.assertEqual(result['workload']['blip_anchor_scores'], 2)
            self.assertEqual(len(result['candidate_scores']),2)
            for s in result['candidate_scores']:
                self.assertAlmostEqual(s['imagination_bonus'], 0. if duplicate else .06)
                self.assertAlmostEqual(s['decision_semantic_score'],s['real_semantic_score']+s['imagination_bonus'])


if __name__ == '__main__':
    unittest.main()
