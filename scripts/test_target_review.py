#!/usr/bin/env python3
import json
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from avic_bridge.target_review import TargetReviewer


class Tests(unittest.TestCase):
    def reviewer(self):
        r=TargetReviewer.__new__(TargetReviewer);r.url='http://fixture/v1/review';r.reset();r.commands=[]
        def proxy(episode,step,op,ident):
            if op:r.commands.append((step,op,ident))
            return SimpleNamespace(accepted=True,instance_id=7,detection_instances=[7],action=3 if op==3 else 2,
                final_state=0 if op==1 else 1,exploration_result=1,reason='fixture')
        r.proxy=proxy
        return r

    def step(self,r,step,action=0,state=4):
        mask=np.zeros((480,640),np.uint8);mask[100:200,200:300]=1
        return r.process(episode='test',step=step,action=action,final_state=state,target='chair',alias='chair',
            rgb=np.full((480,640,3),step%255,np.uint8),masks=[mask],labels=[0],pose=(0.,0.,step*.1))

    def response(self,verdict):
        def request(req,timeout):
            payload=json.loads(req.data)
            return io.BytesIO(json.dumps(dict(request_id=payload['request_id'],instance_id=payload['instance_id'],
                verdict=verdict,visible_evidence='legs and seat',identified_category='chair',reason='fixture')).encode())
        return request

    def test_confirm_keeps_original_stop(self):
        r=self.reviewer()
        with patch('urllib.request.urlopen',side_effect=self.response('target')):
            self.assertEqual(self.step(r,50),(0,4))
            self.assertEqual(self.step(r,51),(0,4))
        self.assertEqual(r.calls,1);self.assertEqual(r.commands,[])

    def test_negative_rejects_instance_not_category(self):
        r=self.reviewer()
        with patch('urllib.request.urlopen',side_effect=self.response('confusable')):
            self.assertEqual(self.step(r,50),(2,0))
        self.assertEqual(r.commands,[(50,1,7)])

    def test_uncertain_scan_once_four_counted_turns_then_fallback(self):
        r=self.reviewer()
        with patch('urllib.request.urlopen',side_effect=self.response('uncertain')):
            outputs=[self.step(r,s) for s in range(50,56)]
        self.assertEqual([a for a,_ in outputs],[2,3,3,2,0,0])
        self.assertEqual(r.calls,2);self.assertEqual(r.extra_actions,4)

    def test_no_call_during_exploration_or_budget_stop(self):
        r=self.reviewer()
        with patch('urllib.request.urlopen') as call:
            self.step(r,40,action=1,state=0);self.step(r,499)
            call.assert_not_called()

    def test_vlm_failure_falls_back_without_extra_action(self):
        r=self.reviewer()
        with patch('urllib.request.urlopen',side_effect=TimeoutError):
            self.assertEqual(self.step(r,50),(0,4))
        self.assertEqual(r.commands,[])

    def test_different_instance_history_is_not_borrowed(self):
        r=self.reviewer()
        r.proxy=lambda *args:SimpleNamespace(accepted=True,instance_id=8,detection_instances=[7],exploration_result=1)
        with patch('urllib.request.urlopen') as call:
            self.assertEqual(self.step(r,50),(0,4))
            call.assert_not_called()

if __name__=='__main__':unittest.main()
