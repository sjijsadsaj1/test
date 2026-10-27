#!/usr/bin/env python3
"""Synthetic STOP/confirmation integration test on an isolated ROS master.

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
    parser.add_argument("--port", type=int, default=11432)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--case", choices=("unconfirmed", "confirmed", "confirm_during_scan", "soft_arrival"),
                        default="unconfirmed")
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
                      APEXNAV_STAGNATION_RECOVERY="false")
    os.environ.pop("ROS_HOSTNAME", None)
    print("Test artifacts: " + str(root), flush=True)
    import rospy
    from geometry_msgs.msg import Quaternion
    from sensor_msgs.point_cloud2 import create_cloud_xyz32
    from std_msgs.msg import Header
    from plan_env.srv import PlanObservation, PlanObservationRequest
    traces = []
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
                    trace.append([res.action, res.final_state, res.exploration_result])
                    if res.action == 0:
                        break  # Habitat ends an episode on STOP too.
                traces.append(trace)
                assert trace == traces[0], "confirmation state leaked across episodes"
            time.sleep(0.3)
            launch_text = (root / "launch.log").read_text()
            if args.disabled:
                assert "stage=start episode=unconfirmed-" not in launch_text
                assert traces[0][27][:2] == [0, 4], traces[0][26:]
            elif args.case == "confirmed":
                assert traces[0][26][2] == 1, traces[0][26:]
                assert traces[0][27][:2] == [0, 4], traces[0][26:]
                assert "stage=start episode=unconfirmed-" not in launch_text
            elif args.case == "confirm_during_scan":
                assert traces[0][27][0] == 2, traces[0][26:]
                assert traces[0][28][:2] == [0, 4], traces[0][26:]
                assert "stage=confirmed episode=unconfirmed-0 step=28" in launch_text
            else:
                assert [row[0] for row in traces[0][27:31]] == [2, 3, 3, 2], traces[0][26:]
                assert "stage=reject episode=unconfirmed-0 step=31 reason=no_confirmed_target" in launch_text
                assert "stage=excluded" in launch_text
                assert all(row[1] != 4 for row in traces[0][27:]), traces[0][26:]
                assert "stage=start episode=unconfirmed-1 step=27" in launch_text
            result = dict(status="passed", enabled=not args.disabled, case=args.case, traces=traces,
                          checks=["duplicate idempotency", "episode reset", args.case,
                                  "rollback STOP" if args.disabled else "confirmation control flow"])
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
