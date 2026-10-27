#!/usr/bin/env python3
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from avic_bridge.frontier_dictionary import CONTRACT,DictionaryFrontierModel,dictionary,validate,real_inputs
from avic_bridge.frontier_crop import project


class Tests(unittest.TestCase):
    def scene(self):
        image=Image.new('RGB',(640,480),(80,90,100));buf=io.BytesIO();image.save(buf,format='JPEG')
        f=dict(x=3.,y=0.,rank=0,semantic_score=.1,query_id=4,query_active=True,
            cluster_contract='original_cluster_projection_v1',cluster_cells_xy=[[3.,float(y)] for y in np.arange(-.5,.51,.05)])
        v=dict(rgb_jpeg_b64=base64.b64encode(buf.getvalue()).decode(),width=640,height=480,
            projection_depth=np.full((480,640),4.),robot_x=0.,robot_y=0.,robot_yaw_rad=0.,camera_pitch_rad=0.,
            step_index=26,age_steps=0,bearing_rad=0.)
        cfg=dict(width=640,height=480,hfov=79.,camera_height=.88,min_depth=0.,max_depth=10.,normalize_depth=False)
        return f,v,cfg

    def test_independent_marginals_not_normalized(self):
        r=dict(frontier_id='f0',visible_evidence='table',predicted_space='dining room',probabilities=dict(chair='0.8',table='0.7'))
        out=validate(r,'f0',['chair','table']);self.assertEqual(out['probabilities'],dict(chair=.8,table=.7))
        self.assertAlmostEqual(sum(out['probabilities'].values()),1.5)
        for p in ('nan','1.5',True,-.2):
            r['probabilities']['chair']=p
            with self.assertRaises(ValueError):validate(r,'f0',['chair','table'])

    def test_single_candidate_identity_owned_by_program(self):
        r=dict(frontier_id='invented_by_model',visible_evidence='wall',predicted_space='unknown',probabilities=dict(sink='0.3'))
        self.assertEqual(validate(r,'real_f2',['sink'])['frontier_id'],'real_f2')

    def test_crop_identity_occlusion_and_no_generation_size_gate(self):
        f,v,cfg=self.scene();r=project(f,v,cfg);self.assertIsNotNone(r['box']);v['roi']=r
        image,context,crop,_=real_inputs(dict(frontier=f,view=v))
        self.assertEqual(crop.tobytes(),image.crop(r['box']).tobytes())
        f['x']=4.
        with self.assertRaises(ValueError):real_inputs(dict(frontier=f,view=v))
        f,v,cfg=self.scene();v['projection_depth'][:]=1.;self.assertIsNone(project(f,v,cfg)['box'])
        f,v,cfg=self.scene();f['cluster_cells_xy']=[[3.,0.]]
        r=project(f,v,cfg);self.assertIsNotNone(r['box']);self.assertLess(r['box'][2]-r['box'][0],48)

    def test_model_request_has_crop_dictionary_and_no_goal_or_generation(self):
        f,v,cfg=self.scene();v['roi']=project(f,v,cfg);v.pop('projection_depth')
        class Client:
            model='fixture';calls=[]
            def request(self,content,max_tokens):
                self.calls.append(content)
                return dict(frontier_id='f0',visible_evidence='fixture',predicted_space='fixture',
                    probabilities={o:'0.8' for o in dictionary()['objects']})
        client=Client();payload=dict(contract=CONTRACT,request_id='test',episode=dict(step_index=26),target='TARGET_ONLY_SECRET',
            dictionary=dictionary(),candidates=[dict(id='f0',frontier=f,view=v)])
        with tempfile.TemporaryDirectory() as temp:
            result=DictionaryFrontierModel(client=client).predict(payload,Path(temp)/'output')
            self.assertNotIn('TARGET_ONLY_SECRET',json.dumps(client.calls))
            self.assertNotIn('semantic_score',json.dumps(client.calls))
            self.assertEqual(sum(x['type']=='image_url' for x in client.calls[0]),2)
            self.assertAlmostEqual(sum(result['candidates'][0]['probabilities'].values()),21*.8)
            self.assertEqual(result['generated_images'],0)
            self.assertTrue((Path(temp)/'output/candidate_0_crop.png').is_file())

    def test_no_generator_imports(self):
        import ast
        root=Path(__file__).resolve().parents[1]/'avic_bridge'
        for name in ('frontier_dictionary.py','frontier_crop.py','vlm_transport.py'):
            tree=ast.parse((root/name).read_text())
            modules=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            self.assertFalse(any(x and any(word in x for word in ('image_glm_backend','local_semantic_client','diffusers','interior_reference')) for x in modules))

    def test_partial_retry_preserves_values_and_asks_only_missing(self):
        class Client:
            model='fixture';calls=[]
            def request(self,content,max_tokens):
                self.calls.append(content)
                if len(self.calls)==1:
                    return dict(visible_evidence='wall',predicted_space='room',probabilities=dict(chair='0.8',table='nan'))
                return dict(probabilities=dict(chair='0.01',table='0.7',invented='0.9'))
        client=Client()
        with tempfile.TemporaryDirectory() as t:
            row,calls=DictionaryFrontierModel(client=client).complete_candidate(
                [dict(type='text',text='real input')],'f0',['chair','table'],Path(t),'candidate_0')
            self.assertEqual(row['probabilities'],dict(chair=.8,table=.7))
            self.assertEqual(calls,2)
            req=json.loads((Path(t)/'candidate_0_attempt_1_request.json').read_text())
            self.assertEqual(req['requested_objects'],['table'])

    def test_repeated_omissions_use_single_object_recovery_without_defaults(self):
        class Client:
            model='fixture';n=0
            def request(self,content,max_tokens):
                self.n+=1
                probs={} if self.n<=3 else ({'chair':'0.8'} if self.n==4 else {'table':'0.7'})
                return dict(visible_evidence='wall',predicted_space='room',probabilities=probs)
        with tempfile.TemporaryDirectory() as t:
            row,calls=DictionaryFrontierModel(client=Client()).complete_candidate(
                [dict(type='text',text='real input')],'f0',['chair','table'],Path(t),'candidate_0')
            self.assertEqual(calls,5)
            self.assertEqual(row['probabilities'],dict(chair=.8,table=.7))
        class BadClient:
            model='fixture'
            def request(self,content,max_tokens):
                return dict(visible_evidence='wall',predicted_space='room',probabilities={})
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                DictionaryFrontierModel(client=BadClient()).complete_candidate(
                    [],'f0',['chair'],Path(t),'candidate_0')

    def test_archived_failure_keeps_first_valid_values(self):
        root=Path(__file__).resolve().parents[1]
        archived=root/'experiments/runs/frontier_dictionary30/20261026_123344_323566/predictions/98bd8c52dd0e4a53a1564fda6e88e713'
        if not archived.exists():self.skipTest('archived failed run unavailable')
        first=json.loads((archived/'candidate_2_attempt_0.json').read_text())
        second=json.loads((archived/'candidate_2_attempt_1.json').read_text())
        self.assertNotIn('picture',first['probabilities'])
        class Client:
            model='archived-fixture';n=0
            def request(self,content,max_tokens):
                self.n+=1
                if self.n==1:return first
                if self.n==2:return second
                return dict(probabilities=dict(picture='0.17'))
        with tempfile.TemporaryDirectory() as t:
            row,calls=DictionaryFrontierModel(client=Client()).complete_candidate(
                [],'f2',dictionary()['objects'],Path(t),'candidate_2')
            self.assertEqual(calls,3)
            self.assertEqual(row['probabilities']['picture'],.17)
            for key,val in first['probabilities'].items():self.assertEqual(row['probabilities'][key],float(val))

    def test_input_padding_preserves_pixels_and_handles_extreme_aspect_ratio(self):
        from avic_bridge.vlm_transport import model_image
        for size in [(2,430),(1,480),(640,1),(3,349),(640,480)]:
            original=Image.new('RGB',size,(15,40,80))
            padded,layout=model_image(original)
            self.assertGreaterEqual(min(padded.size),32)
            self.assertLess(max(padded.size)/min(padded.size),200)
            self.assertEqual(padded.crop(layout['content_box']).tobytes(),original.tobytes())
        self.assertFalse(model_image(Image.new('RGB',(640,480)))[1]['padding'])

    def test_preprocessor_failure_not_retried_as_missing_dictionary(self):
        class Client:
            model='fixture';n=0;last_answer='stale previous response'
            def request(self,content,max_tokens):
                self.n+=1
                raise ValueError('absolute aspect ratio must be smaller than 200')
        c=Client()
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError,'aspect ratio'):
                DictionaryFrontierModel(client=c).complete_candidate([], 'f0', ['chair'], Path(t),'candidate_0')
            self.assertEqual(c.n,1)
            self.assertEqual((Path(t)/'candidate_0_attempt_0_raw.txt').read_text(),'')

    def test_malformed_model_response_is_repairable(self):
        from avic_bridge.vlm_transport import VLMResponseError
        class Client:
            model='fixture';n=0
            def request(self,content,max_tokens):
                self.n+=1
                if self.n==1:raise VLMResponseError('invalid JSON response')
                return dict(visible_evidence='wall',predicted_space='room',probabilities=dict(chair='0.4'))
        with tempfile.TemporaryDirectory() as t:
            row,n=DictionaryFrontierModel(client=Client()).complete_candidate([], 'f0',['chair'],Path(t),'candidate_0')
            self.assertEqual(n,2)
            self.assertEqual(row['probabilities']['chair'],.4)

    def test_controller_publishes_exact_target_probability_without_reward(self):
        from avic_bridge.controller import AvicController,AvicControllerConfig,filter_applicable_frontier_scores
        from avic_bridge.text_frontier import binding
        class Client:
            def decide(self,payload):
                rs=[]
                for i,c in enumerate(payload['candidates']):
                    probs={x:.7 for x in payload['dictionary']['objects']};probs['sink']=.02+i*.03
                    rs.append(dict(id=c['id'],probabilities=probs,visible_evidence='fixture',predicted_space='fixture'))
                return dict(contract=CONTRACT,binding_sha256=binding(payload),dictionary=payload['dictionary'],
                            candidates=rs,inference_seconds=.1,vlm_calls=len(rs))
        flags=dict(APEXNAV_TEXT_FRONTIER='1',APEXNAV_TEXT_METHOD='dictionary',APEXNAV_IMAGE_GLM='0',
                   APEXNAV_RELAXED_IMAGINATION='0',APEXNAV_ALIGNED_BASE_EXPERIMENT='0',APEXNAV_FRONTIER_ONCE='0')
        with patch.dict(os.environ,flags):
            c=AvicController(AvicControllerConfig(mode='avic',score_mode='text_frontier'),client=Client())
            f,v,cfg=self.scene();c.frontier_camera_config=cfg;c.rgb_hfov_deg=79
            c.record_observation(rgb=np.zeros((480,640,3),dtype=np.uint8),robot_x=0.,robot_y=0.,robot_yaw_rad=0.,
                camera_pitch_rad=0.,step_index=26,projection_depth=v['projection_depth'])
            fs=[f,dict(f,rank=1,x=3.1)]
            result=c.decide_action(rgb=np.zeros((480,640,3),dtype=np.uint8),baseline_action=1,
                target_category='sink',target_room='',target_related_objects=[],scene_id='fixture',episode_id='0',
                step_index=26,camera_pitch_rad=0.,itm_cosine=.1,final_state=0,exploration_result=0,frontier_candidates=fs)
            rows=filter_applicable_frontier_scores(result.frontier_scores)
            self.assertEqual(len(rows),2)
            self.assertEqual([r['decision_semantic_score'] for r in rows],[.02,.05])
            self.assertTrue(all('imagination_bonus' not in r for r in rows))
            self.assertEqual(result.selected_plan_id,'f1')

if __name__=='__main__':unittest.main()
