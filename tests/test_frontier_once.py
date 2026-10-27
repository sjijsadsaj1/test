import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np

from avic_bridge.frontier_once import FrontierRegistry
from avic_bridge.controller import AvicController, AvicControllerConfig, filter_applicable_frontier_scores
from avic_bridge.protocol import validate_request


class OnceTests(unittest.TestCase):
    def test_identity_rank_split_reappear_and_no_chain_drift(self):
        r=FrontierRegistry()
        a,b=r.associate([dict(x=0,y=0),dict(x=2,y=0)])
        self.assertEqual([e['id'] for e in r.associate([dict(x=2.1,y=0),dict(x=.1,y=0)])],[1,0])
        self.assertEqual(len(set(e['id'] for e in r.associate([dict(x=.1,y=0),dict(x=.2,y=0)]))),2)
        r.associate([])
        self.assertIs(r.associate([dict(x=0,y=0)])[0],a)
        r=FrontierRegistry();r.associate([dict(x=0,y=0)])
        r.associate([dict(x=.7,y=0)])
        self.assertEqual(r.associate([dict(x=1.4,y=0)])[0]['id'],1)

    def test_controller_once_all_frontiers_reuse_reset_and_negative(self):
        client=Mock()
        def decide(payload):
            validate_request(payload)
            return dict(decision='call_wm', candidate_scores=[dict(
                plan_id=payload['navigation']['candidate_plans'][0]['plan_id'], imagined_itm_score=.1)])
        client.decide.side_effect=decide
        c=AvicController(AvicControllerConfig(mode='avic',score_mode='frontier_once'),client=client)
        def run(step,query):
            rgb=np.zeros((240,320,3),dtype=np.uint8)
            c.record_observation(rgb=rgb,robot_x=0,robot_y=0,robot_yaw_rad=0,
                camera_pitch_rad=0,step_index=step,itm_cosine=.2)
            f=[dict(rank=i,x=i*2.,y=1.,semantic_score=.3,first_action=1,
                path_action_prefix=[1,1],relative_yaw_rad=0,query_id=query,query_active=True) for i in range(5)]
            return c.decide_action(rgb=rgb,baseline_action=1,target_category='chair',target_room='everywhere',
                scene_id='scene',episode_id='1',step_index=step,camera_pitch_rad=0,itm_cosine=.2,
                final_state=0,exploration_result=0,frontier_candidates=f)
        first=run(26,1)
        self.assertEqual(client.decide.call_count,5)
        self.assertEqual(len(filter_applicable_frontier_scores(first.frontier_scores)),5)
        self.assertAlmostEqual(first.frontier_scores[0]['decision_semantic_score'],.2)
        later=run(1000,2)
        self.assertEqual(client.decide.call_count,5)
        self.assertEqual(later.cache_hits,5)
        c.reset_episode();run(26,1)
        self.assertEqual(client.decide.call_count,10)

    def test_old_filter_still_positive_only(self):
        self.assertEqual(filter_applicable_frontier_scores([dict(real_semantic_score=.3,
            decision_semantic_score=.2,imagination_bonus=-.1)]),[])

    def test_backend_q25_no_verifier(self):
        from avic_bridge.frontier_once_backend import evaluate_once
        b=Mock()
        b._select_prefixes.return_value=({'track_0':[1,1]},'two steps',[])
        b._get_itm.return_value.cosine.side_effect=[.1,.3]
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'request'/'track_0';folder.mkdir(parents=True)
            b._get_svc.return_value.generate_plan.return_value=dict(artifact=str(folder/'svc_final.jpg'),
                source_video=str(folder/'pred.mp4'),future_frames=[dict(jpeg_b64='a',order=1),dict(jpeg_b64='b',order=2)])
            p=dict(request_id='test',target=dict(category='chair',related_room='everywhere'),
                camera={},episode=dict(generation_epoch='test'),navigation=dict(baseline_action=dict(id=1),
                candidate_plans=[dict(plan_id='track_0',actions=[dict(id=1),dict(id=1)],
                    source_frontier=dict(semantic_score=.3),frontier_anchor=dict(rgb_jpeg_b64='test'))]))
            p['frontier_refresh'] = dict(reason='real_score_delta', threshold=.03,
                                         previous_real_score=.25, current_real_score=.3)
            p['episode']['generation_epoch'] = 'test:refresh:test'
            result=evaluate_once(b,p)
            self.assertAlmostEqual(result['candidate_scores'][0]['imagined_itm_score'],.15)
            self.assertAlmostEqual(result['candidate_scores'][0]['decision_semantic_score'],.225)
            b._get_qwen.assert_not_called()
            self.assertTrue((folder.parent/'verification.json').exists())
            audit = json.loads((folder.parent/'verification.json').read_text())
            self.assertEqual(audit['frontier_refresh'], p['frontier_refresh'])
            self.assertEqual(b._get_svc.return_value.generate_plan.call_args.kwargs['generation_scope'],
                             'test:refresh:test')


if __name__=='__main__': unittest.main()
