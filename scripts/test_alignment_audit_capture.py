"""Pure diagnostic contract tests: no Habitat/model/service loading."""
import base64
import ast
import __future__
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('alignment',Path(__file__).resolve().parents[1]/'avic_bridge/alignment_audit.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class CaptureTests(unittest.TestCase):
    def controller_functions(self):
        source=Path(__file__).resolve().parents[1]/'avic_bridge/controller.py'
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AvicController')
        funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
               ('record_observation','_temporary_score_record','_store_cached_imagination')]
        from avic_bridge import controller
        namespace=dict(vars(controller))
        exec(compile(ast.Module(body=funcs,type_ignores=[]),str(source),'exec',flags=__future__.annotations.compiler_flag),namespace)
        return namespace

    def test_disabled_actual_controller_record_score_cache_have_no_diagnostic_fields(self):
        from avic_bridge.controller import AvicControllerConfig
        funcs=self.controller_functions()
        obj=NS(enabled=True,current_step=0,observation_history=[],frontier_imagination_cache=[],
            config=AvicControllerConfig(),_find_cached_imagination=lambda plan:None,
            _encode_rgb=lambda rgb:(base64.b64encode(b'jpeg').decode(),2,2))
        plan={'plan_id':'p','source_frontier':{'x':0.,'y':1.,'semantic_score':.2,'rank':0,'query_id':7},'actions':[{'id':1}]}
        score={'imagined_itm_score':.3,'imagination_reliability':.6,'imagination_bonus':.01,'evidence_gain':.2}
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'0'}):
            funcs['record_observation'](obj,rgb=None,robot_x=0,robot_y=1,robot_yaw_rad=0,camera_pitch_rad=0,step_index=0,
                                       alignment_pose={'test':'must not persist'})
            result=funcs['_temporary_score_record'](obj,plan,score,cache_hit=False)
            funcs['_store_cached_imagination'](obj,plan,score)
        self.assertNotIn('alignment_pose',obj.observation_history[0])
        self.assertNotIn('alignment_source_request',result)
        self.assertNotIn('alignment_source_request',obj.frontier_imagination_cache[0])
        self.assertAlmostEqual(result['decision_semantic_score'],.21)

    def test_disabled_no_payload_change_or_io(self):
        payload={'episode':{'generation_epoch':'e'}}
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'0'}), patch.object(a.Path,'write_text',side_effect=AssertionError('IO')):
            a.bind_request(payload,[{'query_id':4}]);self.assertEqual(payload,{'episode':{'generation_epoch':'e'}})
            self.assertIsNone(a.save_json('/must/not/write',{}))
            self.assertIsNone(a.capture_pose(None,None,None))

    def fake_pose(self):
        q=NS(x=0.,y=0.,z=0.,w=1.)
        state=NS(position=[1.,2.,3.],rotation=q,sensor_states={'rgb':NS(position=[1.,2.88,3.],rotation=q)})
        rgb=type('HabitatSimRGBSensor',(),{})();rgb.uuid='rgb'
        env=NS(sim=NS(get_agent_state=lambda:state,sensor_suite=NS(sensors={'rgb':rgb})))
        cfg=NS(habitat=NS(simulator=NS(forward_step_size=.25,turn_angle=30,
            agents={'main':NS(sim_sensors={'rgb_sensor':NS(type='HabitatSimRGBSensor')})})))
        return env,cfg

    def test_actual_sensor_not_body_and_ros_coordinates(self):
        env,cfg=self.fake_pose()
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'1'}):
            p=a.capture_pose(env,cfg,{'gps':[2.,0.,4.],'compass':[.3]})
        self.assertEqual(p['rgb_sensor']['position'][1],2.88)
        self.assertEqual(p['body']['position'][1],2.)
        self.assertEqual(p['ros_planar'],dict(x=-4.,y=-2.,yaw_rad=.3))

    def test_quaternion_yaw_and_pitch_matrix(self):
        q=NS(x=0.,y=math.sin(math.pi/4),z=0.,w=math.cos(math.pi/4))
        p=a.pose([0,0,0],q);self.assertAlmostEqual(p['c2w'][0][2],1)
        q=NS(x=math.sin(math.pi/4),y=0.,z=0.,w=math.cos(math.pi/4))
        p=a.pose([0,0,0],q);self.assertAlmostEqual(p['c2w'][1][2],-1)

    def test_missing_sensor_is_explicit_not_body_fallback(self):
        env,cfg=self.fake_pose();env.sim.get_agent_state().sensor_states.clear()
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'1'}),redirect_stdout(io.StringIO()):
            p=a.capture_pose(env,cfg,{'gps':[0,0,0],'compass':[0]})
        self.assertIn('RGB sensor state missing',p['missing_reason'])
        self.assertNotIn('rgb_sensor',p)

    def test_multiple_rgb_sensors_are_explicitly_ambiguous(self):
        env,cfg=self.fake_pose();sensor=type('HabitatSimRGBSensor',(),{})();sensor.uuid='second'
        env.sim.sensor_suite.sensors['second']=sensor
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'1'}),redirect_stdout(io.StringIO()):
            p=a.capture_pose(env,cfg,{'gps':[0,0,0],'compass':[0]})
        self.assertIn('ambiguous',p['missing_reason']);self.assertNotIn('rgb_sensor',p)

    def test_depth_sensor_does_not_confuse_rgb_identity(self):
        env,cfg=self.fake_pose();sensor=type('HabitatSimDepthSensor',(),{})();sensor.uuid='depth'
        env.sim.sensor_suite.sensors['depth']=sensor
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'1'}):
            p=a.capture_pose(env,cfg,{'gps':[0,0,0],'compass':[0]})
        self.assertEqual(p['sensor_uuid'],'rgb');self.assertEqual(p['rgb_sensor']['position'][1],2.88)

    def test_original_vs_reencoded_hash_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'1'}):
            root=Path(tmp);input_path=root/'input.jpg';input_path.write_bytes(b'reencoded')
            original=b'original jpeg bytes';b64=base64.b64encode(original).decode()
            meta=a.capture_generation(root,b64,input_path,request_id='r',plan_id='p',scope='epoch',
                generation_key='f',cache_hit=True,cache_origin={'request_id':'old','plan_id':'p','contract_path':'old/path'},
                samples=[{'video_frame_index':15,'target_pose_index':None}])
            data=json.loads(Path(meta['camera_contract_path']).read_text())
            self.assertEqual(Path(data['original_anchor']['path']).read_bytes(),original)
            self.assertEqual(data['original_anchor']['sha256'],hashlib.sha256(original).hexdigest())
            self.assertNotEqual(data['original_anchor']['sha256'],data['recompressed_input']['sha256'])
            self.assertEqual(data['cache_origin']['request_id'],'old')
            self.assertIsNone(data['samples'][0]['target_pose_index'])

    def test_logging_failure_does_not_raise(self):
        with patch.dict(os.environ,{'APEXNAV_ALIGNMENT_AUDIT':'1'}),redirect_stdout(io.StringIO()):
            result=a.capture_generation('/missing/path','invalid','/missing/input')
        self.assertIn('missing_reason',result)


if __name__=='__main__':unittest.main()
