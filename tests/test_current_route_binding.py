"""Regressions for current-pose route binding and separate view verification."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from avic_bridge.controller import AvicController, AvicControllerConfig
from avic_bridge.protocol import CURRENT_ROUTE_CONTRACT, ProtocolError, validate_request
from avic_bridge.service import SVCQwenBackend, QwenOpenAIClient, validate_view_evidence
from test_avic_bridge import FakeClient
from test_avic_frontier_policy import frontier, FakeHorizonClient
from test_grounded_delta import current_route_payload


class CurrentRouteTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((24, 32, 3), 128, dtype=np.uint8)
        self.client = FakeClient({'decision': 'skip', 'reason': 'test'})
        self.ctrl = AvicController(AvicControllerConfig(
            mode='avic', score_mode='grounded_delta', warmup_steps=0,
            cooldown_steps=0), self.client)
        self.observe()

    def observe(self, step=26, rgb=None, pitch=.2):
        self.ctrl.record_observation(rgb=self.rgb if rgb is None else rgb,
            robot_x=3., robot_y=-2., robot_yaw_rad=.6,
            camera_pitch_rad=pitch, step_index=step, itm_cosine=.2)

    def plans(self, routes):
        fs = [dict(frontier(i,.20+i*.001,3.,0.,r[0],selected=i==0),
                   path_action_prefix=r, query_id=5, query_active=True)
              for i,r in enumerate(routes)]
        return self.ctrl._candidate_plans(1,fs)

    def bind(self, routes):
        plans = self.plans(routes)
        return self.ctrl._bind_current_routes(plans,plans,self.rgb,.2,26)

    def test_real_paths_and_current_pose_survive_binding(self):
        routes = [[1,1,2,1,1,1],[1,1,3,1,1,1]]
        self.ctrl.observation_history.insert(0,dict(self.ctrl.observation_history[0],
            step_index=2, rgb_jpeg_b64='old-image', itm_cosine=.9))
        bound,_ = self.bind(routes)
        self.assertEqual(len(bound),2)
        for p,expected in zip(bound,routes):
            self.assertEqual([a['id'] for a in p['actions']],expected)
            self.assertEqual(p['actions'],p['apexnav_route_actions'])
            self.assertEqual(p['min_prefix_length'],3)
            a=p['frontier_anchor']
            self.assertEqual((a['step_index'],a['robot_x'],a['robot_y'],a['robot_yaw_rad']),
                             (26,3.,-2.,.6))
            self.assertEqual(a['camera_pitch_rad'],.2)
            self.assertEqual(a['rgb_jpeg_b64'],self.ctrl._encode_rgb(self.rgb)[0])

    def test_equal_prefix_and_short_routes_are_deferred(self):
        for routes in ([[1,1,1],[1,1,1]], [[1,1],[1,1,2]], [[1],[2]]):
            self.assertEqual(self.bind(routes)[0],[])

    def test_unique_route_survives_shared_group(self):
        bound,_ = self.bind([[1,1],[1,1],[2,1]])
        self.assertEqual([p['plan_id'] for p in bound],['frontier_2'])

    def test_noneligible_incumbent_still_sets_divergence_bound(self):
        plans=self.plans([[1,1,1,1,1,2],[1,1,1,1,1,3]])
        bound,_=self.ctrl._bind_current_routes(plans[1:],plans,self.rgb,.2,26)
        self.assertEqual(bound[0]['min_prefix_length'],6)

    def test_old_wrong_image_or_wrong_pitch_never_binds(self):
        plans=self.plans([[1,1],[2,1]])
        for rgb,pitch,step in ((self.rgb,.2,27),(self.rgb,.3,26),
                              (np.zeros_like(self.rgb),.2,26)):
            self.assertEqual(self.ctrl._bind_current_routes(plans,plans,rgb,pitch,step)[0],[])

    def decide(self,routes):
        return self.ctrl.decide_action(rgb=self.rgb,baseline_action=1,target_category='chair',
            target_room='everywhere',scene_id='scene',episode_id='1',step_index=26,
            camera_pitch_rad=.2,itm_cosine=.2,final_state=0,exploration_result=0,
            frontier_candidates=[p['source_frontier'] for p in self.plans(routes)])

    def test_controller_payload_is_valid_and_not_a_virtual_probe(self):
        d=self.decide([[1,1,2,1],[1,1,3,1]])
        self.assertTrue(d.used_world_model)
        p=self.client.requests[0]
        validate_request(p)
        self.assertEqual(p['navigation']['route_contract'],CURRENT_ROUTE_CONTRACT)
        self.assertEqual([a['id'] for a in p['navigation']['candidate_plans'][1]['actions']],
                         [1,1,3,1])
        self.assertEqual(d.executed_action,1)

    def test_shared_prefix_skips_before_client_or_budget(self):
        d=self.decide([[1,1,1],[1,1,1]])
        self.assertFalse(d.used_world_model)
        self.assertEqual(self.ctrl.world_model_calls,0)
        self.assertEqual(self.client.requests,[])
        self.assertEqual(d.executed_action,1)

    def test_score_cache_cannot_cross_routes_at_same_anchor(self):
        bound,_=self.bind([[1,1,2],[1,1,3]])
        p=bound[0]
        self.ctrl.current_step=26
        self.ctrl._store_cached_imagination(p,dict(imagined_itm_score=.3,
            imagination_reliability=.8,imagination_bonus=.01))
        self.assertIsNotNone(self.ctrl._find_cached_imagination(p))
        changed=copy.deepcopy(p);changed['actions'][2]['id']=3
        self.assertIsNone(self.ctrl._find_cached_imagination(changed))
        changed=copy.deepcopy(p);changed['min_prefix_length']=2
        self.assertIsNone(self.ctrl._find_cached_imagination(changed))

    def test_protocol_rejects_mixed_epochs_and_rewritten_route(self):
        self.decide([[1,1],[2,1]])
        p=self.client.requests[0]
        for field,value in [('step_index',25),('rgb_jpeg_b64','old'),('camera_pitch_rad',.3)]:
            bad=copy.deepcopy(p)
            bad['navigation']['candidate_plans'][0]['frontier_anchor'][field]=value
            with self.assertRaises(ProtocolError):validate_request(bad)
        bad=copy.deepcopy(p);bad['navigation']['candidate_plans'][0]['actions'][1]['id']=2
        with self.assertRaises(ProtocolError):validate_request(bad)

    def test_horizon_cannot_erase_route_divergence(self):
        plans,_=self.bind([[1,1,1,1,1,2],[1,1,1,1,1,3]])
        b=SVCQwenBackend()
        b._qwen=FakeHorizonClient([dict(plan_id=p['plan_id'],prefix_length=1,reason='short') for p in plans])
        selected,reason,errors=b._select_prefixes({},plans)
        self.assertEqual(selected['frontier_0'],[1,1,1,1,1,2])
        self.assertEqual(selected['frontier_1'],[1,1,1,1,1,3])
        self.assertFalse(errors)
        self.assertIn('preserve route divergence',reason)

    def test_insufficient_frame_budget_rejected_before_model_load(self):
        p=current_route_payload([frontier(0,.2,1.,0.,1)])
        with patch.dict('os.environ',{'AVIC_SVC_MAX_RERANK_FRAMES':'1'}):
            b=SVCQwenBackend()
            with self.assertRaisesRegex(ValueError,'at least two'):
                b.decide(p)
            self.assertIsNone(b._svc)


class ViewPromptTests(unittest.TestCase):
    def capture(self,mode):
        q=QwenOpenAIClient.__new__(QwenOpenAIClient);q.policy_model='fake'
        seen={}
        def complete(*args,**kwargs):seen.update(content=args[1],**kwargs);return {}
        q._complete=complete
        c=dict(plan_id='frontier_0',source_frontier={},frontier_anchor={},
               future_frames=[dict(order=1,jpeg_b64='a'),dict(order=2,jpeg_b64='b')],
               frame_itm_scores=[.2,.3])
        q.evaluate_frontier_sequences({'target':{},'score_policy':{'mode':mode}},[c])
        return seen

    def test_new_prompt_separates_consistency_novelty_and_real_context(self):
        seen=self.capture('grounded_delta');s=str(seen['content'])
        self.assertIn('must NOT lower reliability',s)
        self.assertIn('REAL context/structure',s)
        self.assertNotIn('no frame supports NEW evidence',s)
        self.assertNotIn('anchor_quality=0.000',s)
        schema=seen['response_format']['json_schema']['schema']['properties']['evaluations']['items']
        self.assertIn('consistency_reason',schema['required'])

    def test_strict_prompt_retains_old_evidence_contract(self):
        seen=self.capture('strict_evidence')
        self.assertIn('no frame supports NEW evidence',str(seen['content']))
        schema=seen['response_format']['json_schema']['schema']['properties']['evaluations']['items']
        self.assertNotIn('consistency_reason',schema['properties'])

    def test_missing_consistency_explanation_is_not_valid_negative_evidence(self):
        e=dict(reliability=.8,evidence_gain=0.,semantic_relation='same cue',reason='no new information',
               anchor_cue='chair',new_cue='',supported_by_real_anchor=True,adds_new_information=False,
               target_only_in_imagination=False,support_frame_orders=[1,2])
        frames=[{'order':1},{'order':2}]
        self.assertFalse(validate_view_evidence(e,frames)[0])
        e['consistency_reason']='Room and chair geometry remain consistent.'
        self.assertEqual(validate_view_evidence(e,frames)[:2],(True,True))


if __name__=='__main__':unittest.main()
