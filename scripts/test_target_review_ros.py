#!/usr/bin/env python3
"""Synthetic instance review command integration test on an isolated ROS master.

No Habitat, GPU or learned models. The out-of-depth cloud deliberately has a
nearby XY projection to exercise arrival without simulating a full trajectory.
It tests control flow, not perception accuracy or navigation success rate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import time

from test_sync_planner import quaternion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11439)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--case", choices=("unconfirmed", "confirmed", "confirm_during_scan", "soft_arrival"),
                        default="confirmed")
    args = parser.parse_args()
    if args.port == 11311:
        raise ValueError("use an isolated master port")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    root = Path(tempfile.mkdtemp(prefix="target-confirm-ros-"))
    (root / "tsp").mkdir()
    os.environ.update(ROS_MASTER_URI=f"http://127.0.0.1:{args.port}",
                      ROS_IP="127.0.0.1", ROS_HOME=str(root / "ros"),
                      ROS_LOG_DIR=str(root / "ros_logs"),
                      APEXNAV_TARGET_CONFIRMATION="false" if args.disabled else "true",
                      APEXNAV_STAGNATION_RECOVERY="false", APEXNAV_TARGET_REVIEW="1", APEXNAV_TEXT_FRONTIER="0")
    os.environ.pop("ROS_HOSTNAME", None)
    print("Test artifacts: " + str(root), flush=True)
    import rospy
    from geometry_msgs.msg import Quaternion
    from sensor_msgs.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header
    from plan_env.srv import PlanObservation, PlanObservationRequest, ReviewTarget
    traces = []; review_checks=[]
    with (root / "launch.log").open("w") as log:
        process = subprocess.Popen([
            "roslaunch", "-p", str(args.port), "exploration_manager", "exploration.launch",
            "tsp_dir:=" + str(root / "tsp")], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            rospy.init_node("target_confirmation_regression", anonymous=True, disable_signals=True)
            rospy.wait_for_service("/apexnav/plan_observation", timeout=45)
            assert rospy.get_param("/exploration_node/exploration/target_confirmation") == (not args.disabled)
            proxy = rospy.ServiceProxy("/apexnav/plan_observation", PlanObservation)
            review = rospy.ServiceProxy("/apexnav/review_target", ReviewTarget)
            # Two fresh episodes exercise actual FSM/map reset, not just helper reset.
            for repeat in range(2):
                trace = []
                for step in range(33):
                    req = PlanObservationRequest()
                    req.episode, req.step = f"unconfirmed-{repeat}", step
                    req.executed_action = trace[-1][0] if trace else 0
                    if args.case == "soft_arrival" and step == 27:
                        req.executed_action = 1  # blocked forward within 0.45m
                    req.itm_score = -1 if step == 0 else 0.20
                    req.confidence_threshold = 0.5
                    req.depth.height, req.depth.width = 480, 640
                    req.depth.encoding, req.depth.step = "32FC1", 640 * 4
                    req.depth.data = struct.pack("<f", 1.0) * (640 * 480)
                    req.depth.header.stamp = rospy.Time(step + 1)
                    yaw = (step % 12) * math.pi / 6
                    req.robot_pose.pose.pose.orientation = Quaternion(*quaternion(0, 0, yaw))
                    req.camera_pose.pose.pose.position.z = 0.88
                    req.camera_pose.pose.pose.orientation = Quaternion(*quaternion(math.pi / 3, math.pi, yaw + math.pi / 2))
                    if step >= (24 if args.case == "confirmed" else 26):
                        # A valid camera quaternion passing the original detection gate.
                        req.camera_pose.pose.pose.orientation = Quaternion(*quaternion(2 * math.pi / 3, 0, 0))
                        points = [(0.35 if args.case == "soft_arrival" else 0.10, 0.0, 6.0)]
                        if args.case == "confirmed" or (args.case == "confirm_during_scan" and step >= 27):
                            # Synthetic RGB detection must coincide with occupied
                            # REAL-depth cells: the original mapper rejects free-space objects.
                            req.depth.data = struct.pack("<f", 0.10) * (640 * 480)
                            points = []
                            for u in range(200, 441, 10):
                                for v in range(120, 361, 10):
                                    x, y, z = (u - 320) * 0.5 / 388.191, (v - 240) * 0.5 / 422.048, 0.5
                                    points.append((x, -0.5*y - math.sqrt(3)/2*z,
                                                   0.88 + math.sqrt(3)/2*y - 0.5*z))
                        req.detections.point_clouds = [create_cloud_xyz32(
                            Header(stamp=req.depth.header.stamp, frame_id="world"), points)]
                        req.detections.confidence_scores = [0.95]
                        req.detections.label_indices = [0]
                    res = proxy(req)
                    assert res.accepted, res.reason
                    duplicate = proxy(req)
                    assert duplicate.accepted and duplicate.action == res.action
                    if step in (27,28):
                        snap=review(req.episode,step,0,-1)
                        assert snap.accepted and snap.instance_id>=0, str(snap)
                        bad=review(req.episode,step,1,snap.instance_id+1000)
                        assert not bad.accepted, 'wrong instance accepted'
                        stale=review(req.episode,step-1,1,snap.instance_id)
                        assert not stale.accepted, 'stale observation accepted'
                        operation=2 if step==27 else 1
                        decision=review(req.episode,step,operation,snap.instance_id)
                        assert decision.accepted, str(decision)
                        duplicate_review=review(req.episode,step,operation,snap.instance_id)
                        assert duplicate_review.accepted and duplicate_review.action==decision.action
                        res=proxy(req)
                        assert res.action==decision.action and res.final_state==decision.final_state
                        review_checks.append(dict(repeat=repeat,step=step,instance=snap.instance_id,operation=operation,action=res.action))
                    trace.append([res.action, res.final_state, res.exploration_result])
                    if res.action == 0:
                        break  # Habitat ends an episode on STOP too.
                traces.append(trace)
                assert trace == traces[0], "confirmation state leaked across episodes"
            time.sleep(0.3)
            launch_text = (root / "launch.log").read_text()
            assert len(review_checks)==4
            assert all(row[1]!=4 for row in traces[0][28:]), traces[0][26:]
            result=dict(status='passed',checks=review_checks,traces=traces,
                        meaning='Synthetic control-flow test, not navigation SR or visual accuracy')
            (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        finally:
            rospy.signal_shutdown("regression complete")
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
