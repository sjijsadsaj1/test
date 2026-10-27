"""A complete simulator frame and its decision cross ROS as one transaction."""

import hashlib
import json
import io
import queue
import threading

import rospy
import numpy as np
from plan_env.srv import PlanObservation, PlanObservationRequest


class SynchronousPlanner:
    def __init__(self, publisher, timeout_s=120.0):
        self.publisher = publisher
        self.timeout_s = timeout_s
        rospy.wait_for_service("/apexnav/plan_observation", timeout=timeout_s)
        self.proxy = rospy.ServiceProxy("/apexnav/plan_observation", PlanObservation)

    def plan(self, *, observations, episode, step, executed_action, itm_score,
             confidence_threshold, detections):
        req = PlanObservationRequest()
        req.episode, req.step = episode, step
        req.executed_action = executed_action
        req.itm_score = -1.0 if itm_score is None else itm_score
        req.confidence_threshold = confidence_threshold
        # Sensor timestamps identify simulation steps, independent of inference time.
        stamp = rospy.Time(step + 1)
        req.depth = self.publisher.publish_depth(stamp, observations["depth"], publish=False)
        req.camera_pose = self.publisher.publish_camera_odom(
            stamp, observations["gps"], observations["compass"],
            observations["camera_pitch"], publish=False)
        req.robot_pose = self.publisher.publish_robot_odom(
            stamp, observations["gps"], observations["compass"], publish=False)
        req.detections = detections
        result_queue = queue.Queue(maxsize=1)

        def call():
            try:
                result_queue.put(self.proxy(req))
            except Exception as exc:
                result_queue.put(exc)

        threading.Thread(target=call, daemon=True).start()
        try:
            result = result_queue.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            raise RuntimeError("observation commit timeout; abort evaluation (state unknown)") from exc
        if isinstance(result, Exception):
            raise RuntimeError("observation commit transport failure") from result
        if not result.accepted or result.episode != episode or result.step != step:
            raise RuntimeError("observation commit rejected/mismatched: " + result.reason)
        def message_hash(message):
            buffer = io.BytesIO()
            message.serialize(buffer)
            return hashlib.sha256(buffer.getvalue()).hexdigest()
        # Hash only the detector fields consumed by C++, excluding wall-time
        # headers and the unused/random fourth PointCloud2 padding field.
        detection_digest = hashlib.sha256(json.dumps([
            list(req.detections.confidence_scores), list(req.detections.label_indices)
        ]).encode())
        for cloud in req.detections.point_clouds:
            detection_digest.update(str((cloud.height, cloud.width)).encode())
            if cloud.data:
                for field in cloud.fields:
                    if field.name in ("x", "y", "z"):
                        values = np.ndarray((cloud.height, cloud.width),
                            dtype=">f4" if cloud.is_bigendian else "<f4", buffer=cloud.data,
                            offset=field.offset, strides=(cloud.row_step, cloud.point_step))
                        detection_digest.update(field.name.encode() + values.astype("<f4").tobytes())
        print("[SYNC_PLAN] " + json.dumps({
            "episode": episode, "step": step, "action": result.action,
            "final_state": result.final_state, "exploration_result": result.exploration_result,
            "depth_sha256": hashlib.sha256(req.depth.data).hexdigest(),
            "camera_pose_sha256": message_hash(req.camera_pose),
            "robot_pose_sha256": message_hash(req.robot_pose),
            "detections_sha256": detection_digest.hexdigest(),
            "confidence_threshold": req.confidence_threshold,
            "itm_score": req.itm_score,
            "frontiers": list(result.frontiers.data),
        }), flush=True)
        return result
