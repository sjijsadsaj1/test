#!/usr/bin/env python3
"""Isolated real-ROS transaction regression; no Habitat/GPU/models required.

Run after source devel/setup.bash with system python3. Own ROS master, nodes,
TSP files and logs; never contacts or stops the running experiment on port 11311.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import socket
import struct
import subprocess
import tempfile
import time


def quaternion(roll, pitch, yaw):
    sr, cr = math.sin(roll / 2), math.cos(roll / 2)
    sp, cp = math.sin(pitch / 2), math.cos(pitch / 2)
    sy, cy = math.sin(yaw / 2), math.cos(yaw / 2)
    return sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11431)
    parser.add_argument("--exercise-recovery", action="store_true",
                        help="synthetic repeated-position fixture, 80 frames; requires recovery=true")
    parser.add_argument("--exercise-route", action="store_true",
                        help="inject synthetic positive scores to test bounded goal holding (no models)")
    parser.add_argument("--exercise-tsp-failure", action="store_true",
                        help="stop only this isolated TSP node; verify transaction rejects instead of navigating")
    parser.add_argument("--exercise-once", action="store_true",
                        help="isolated all-frontier/negative-fusion diagnostic regression")
    parser.add_argument("--exercise-semantic", action="store_true",
                        help="1.0 pure semantic policy; kill isolated TSP and verify score-driven routing")
    parser.add_argument('--exercise-aligned-base',choices=['off','on'],
                        help='synthetic real-ROS downgrade filter regression, isolated owned nodes only')
    parser.add_argument('--exercise-relaxed', choices=['off','on'],
                        help='real-ROS signed-score filter regression; no model or SR claim')
    parser.add_argument('--exercise-image',action='store_true',
                        help='image/GLM isolated signed-score acceptance, no GPU/API required')
    parser.add_argument('--exercise-image-strong',choices=['off','on'],
                        help='near-equal synthetic image scores: original Closest vs direct semantic winner')
    parser.add_argument('--exercise-cycle-memory',action='store_true',
                        help='isolated synthetic move-away/return; requires genuine suppress/fallback evidence')
    args = parser.parse_args()
    if args.exercise_image_strong:
        assert not any((args.exercise_image,args.exercise_relaxed,args.exercise_aligned_base,
                        args.exercise_once,args.exercise_semantic,args.exercise_route,
                        args.exercise_recovery,args.exercise_tsp_failure,args.exercise_cycle_memory))
        os.environ.update(APEXNAV_IMAGE_GLM='1',APEXNAV_RELAXED_IMAGINATION='0',
                          APEXNAV_IMAGE_GLM_STRONG='1' if args.exercise_image_strong=='on' else '0',
                          APEXNAV_ALIGNED_BASE_EXPERIMENT='0',APEXNAV_FRONTIER_CYCLE_MEMORY='0')
    if args.exercise_image:
        assert not any((args.exercise_relaxed,args.exercise_aligned_base,args.exercise_once,
                        args.exercise_semantic,args.exercise_route,args.exercise_recovery,
                        args.exercise_tsp_failure,args.exercise_cycle_memory))
        os.environ.update(APEXNAV_IMAGE_GLM='1',APEXNAV_RELAXED_IMAGINATION='0',
                          APEXNAV_ALIGNED_BASE_EXPERIMENT='0',APEXNAV_FRONTIER_CYCLE_MEMORY='0')
    if args.exercise_relaxed:
        assert not any((args.exercise_once,args.exercise_semantic,args.exercise_recovery,
                        args.exercise_route,args.exercise_tsp_failure,args.exercise_aligned_base,
                        args.exercise_cycle_memory))
        os.environ['APEXNAV_RELAXED_IMAGINATION'] = '1' if args.exercise_relaxed == 'on' else '0'
        os.environ['APEXNAV_ALIGNED_BASE_EXPERIMENT'] = '0'
        os.environ['APEXNAV_FRONTIER_CYCLE_MEMORY'] = '0'
        os.environ.pop('APEXNAV_FRONTIER_ONCE', None)
    if args.exercise_cycle_memory:
        assert not any((args.exercise_once,args.exercise_semantic,args.exercise_recovery,args.exercise_route,args.exercise_tsp_failure,args.exercise_aligned_base))
        os.environ['APEXNAV_FRONTIER_CYCLE_MEMORY']='1'
    if args.exercise_aligned_base:
        assert not any((args.exercise_once,args.exercise_semantic,args.exercise_recovery,args.exercise_route,args.exercise_tsp_failure))
        os.environ['APEXNAV_ALIGNED_BASE_EXPERIMENT']='1' if args.exercise_aligned_base=='on' else '0'
        os.environ.pop('APEXNAV_FRONTIER_ONCE',None)
    if args.exercise_semantic:
        assert args.exercise_route and not args.exercise_once and not args.exercise_tsp_failure
        os.environ['APEXNAV_V1_FRONTIER_POLICY'] = 'semantic_only'
    if args.exercise_once:
        os.environ['APEXNAV_FRONTIER_ONCE'] = '1'
    if args.port == 11311:
        raise ValueError("use an isolated master port, not 11311")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))  # fail before starting if occupied
    root = Path(tempfile.mkdtemp(prefix="stable-sr-ros-"))
    (root / "tsp").mkdir()
    os.environ.update(ROS_MASTER_URI="http://127.0.0.1:%d" % args.port,
                      ROS_IP="127.0.0.1", ROS_HOME=str(root / "ros"),
                      ROS_LOG_DIR=str(root / "ros_logs"))
    os.environ.pop("ROS_HOSTNAME", None)
    print("Test artifacts: " + str(root), flush=True)
    import rospy
    from geometry_msgs.msg import Quaternion
    from plan_env.srv import PlanObservation, PlanObservationRequest
    from std_msgs.msg import Float32MultiArray
    with (root / "launch.log").open("w") as log:
        process = subprocess.Popen([
            "roslaunch", "-p", str(args.port), "exploration_manager", "exploration.launch",
            "tsp_dir:=" + str(root / "tsp")], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            rospy.init_node("sync_regression", anonymous=True, disable_signals=True)
            rospy.wait_for_service("/apexnav/plan_observation", timeout=45)
            if args.exercise_semantic:
                import rosnode
                killed, failed = rosnode.kill_nodes(['/tsp_solver'])
                assert '/tsp_solver' in killed and not failed
            expected_recovery = os.environ.get("APEXNAV_STAGNATION_RECOVERY", "false") == "true"
            assert rospy.get_param("/exploration_node/exploration/stagnation_recovery") == expected_recovery
            if args.exercise_recovery:
                assert expected_recovery, "exercise requires APEXNAV_STAGNATION_RECOVERY=true"
            if args.exercise_route:
                assert not args.exercise_recovery
                assert rospy.get_param("/exploration_node/exploration/avic_route_commit_steps") == 6
            proxy = rospy.ServiceProxy("/apexnav/plan_observation", PlanObservation)
            ack_queue = queue.Queue()
            subscriber = rospy.Subscriber("/apexnav/avic_frontier_ack", Float32MultiArray,
                                           lambda message: ack_queue.put(message.data))
            publisher = rospy.Publisher("/apexnav/avic_frontier_scores", Float32MultiArray, queue_size=10)

            def request(episode, step):
                req = PlanObservationRequest()
                req.episode, req.step = episode, step
                req.executed_action = 0 if step == 0 else 2
                req.itm_score = -1 if step == 0 else 0.20
                req.confidence_threshold = 0.5
                req.depth.height, req.depth.width = 480, 640
                req.depth.encoding, req.depth.step = "32FC1", 640 * 4
                req.depth.data = struct.pack("<f", 1.0) * (640 * 480)
                req.depth.header.stamp = rospy.Time(step + 1)
                yaw = (step % 12) * math.pi / 6
                if args.exercise_recovery and step >= 26:
                    yaw = math.pi / 3
                    # Small nonzero oscillation avoids the legacy collision and
                    # stationary-goal counters, while never making net progress.
                    req.robot_pose.pose.pose.position.x = 0.04 if step % 2 else -0.04
                    req.camera_pose.pose.pose.position.x = req.robot_pose.pose.pose.position.x
                req.robot_pose.pose.pose.orientation = Quaternion(*quaternion(0, 0, yaw))
                req.camera_pose.pose.pose.position.z = 0.88
                req.camera_pose.pose.pose.orientation = Quaternion(*quaternion(math.pi / 3, math.pi, yaw + math.pi / 2))
                return req

            traces = []
            no_op_count = 0
            for repeat in range(3):
                trace = []
                episode = "sync-regression-%d" % repeat
                for step in range(80 if args.exercise_recovery else 32):
                    req = request(episode, step)
                    res = proxy(req)
                    assert res.accepted, res.reason
                    duplicate = proxy(req)
                    assert duplicate.accepted and duplicate.action == res.action
                    assert list(duplicate.frontiers.data) == list(res.frontiers.data)
                    changed = copy.deepcopy(req)
                    changed.itm_score = 0.21
                    assert not proxy(changed).accepted, "different-payload duplicate accepted"
                    if step == 0:
                        assert not proxy(request(episode, 2)).accepted, "out-of-order accepted"
                        bad = request(episode, 1)
                        bad.robot_pose.pose.pose.orientation.w = float("nan")
                        assert not proxy(bad).accepted, "invalid pose accepted"
                        if repeat:
                            assert not proxy(request("sync-regression-0", 0)).accepted, "retired episode accepted"
                    trace.append([res.action, res.final_state, res.exploration_result, list(res.frontiers.data)])
                    # Third repeat tries a bonus for ApexNav's existing target.
                    # It must ACK false and leave future decisions identical.
                    if repeat == 2 and res.frontiers.data and not args.exercise_recovery:
                        width = res.frontiers.layout.dim[1].size
                        rows = [res.frontiers.data[i:i+width] for i in range(0, len(res.frontiers.data), width)]
                        for rank, row in enumerate(rows):
                            if round(row[6]) and round(row[8]):
                                deadline = time.monotonic() + 5
                                while not publisher.get_num_connections() and time.monotonic() < deadline:
                                    time.sleep(0.02)
                                publisher.publish(Float32MultiArray(data=[row[7], 1, rank, row[0], row[1], row[2], 0.5, row[2] + 0.004]))
                                ack = ack_queue.get(timeout=30)
                                assert round(ack[0]) == round(row[7]) and not round(ack[2]), ack
                                assert round(ack[1]) == res.action, ack
                                no_op_count += 1
                    time.sleep(0.005 if repeat == 0 else 0.04)
                traces.append(trace)
                assert trace == traces[0], "decision/frontier divergence in repeat %d" % repeat
            assert any(row[3] for row in traces[0][26:]), "test did not reach frontier planning"
            if not args.exercise_recovery:
                assert no_op_count > 0, "test did not exercise same-target rollback"
            if args.exercise_recovery:
                launch_text = (root / "launch.log").read_text()
                assert "stage=replan" in launch_text, "fixture did not trigger recovery"
                assert "stage=switch" in launch_text, "fixture did not reach stage 2"
            if args.exercise_once:
                rospy.set_param('/exploration_node/exploration/avic_top_k', 1)
                exercised = False
                for step in range(32):
                    res = proxy(request('frontier-once-negative-regression', step))
                    assert res.accepted, res.reason
                    width = res.frontiers.layout.dim[1].size if res.frontiers.layout.dim else 18
                    values = list(res.frontiers.data)
                    rows = [values[i:i+width] for i in range(0, len(values), width)]
                    if not exercised and len(rows) > 1 and rows[0][8]:
                        # all frontiers must be exported even with top_k=1.
                        packet = [rows[0][7], len(rows)]
                        for rank, row in enumerate(rows):
                            packet.extend([rank, row[0], row[1], row[2], row[2]-.1, row[2]-.05])
                        publisher.publish(Float32MultiArray(data=packet))
                        ack = ack_queue.get(timeout=30)
                        assert round(ack[0]) == round(rows[0][7]), ack
                        exercised = True
                assert exercised, 'fixture did not expose multiple frontiers'
                time.sleep(.2)
                text = (root/'launch.log').read_text()
                assert '[FRONTIER_ONCE_CONFIG] enabled=1' in text
                assert 'ignoring non-positive gain' not in text
                assert '[AVIC_SCORE] frontier=' in text
            route_cases = []
            if args.exercise_cycle_memory:
                offset=(root/'launch.log').stat().st_size
                avic_cycle_rejection_exercised=False
                selected_history=[];cycle_trials=[]
                for episode in ['cycle-memory-return','cycle-memory-new-episode']:
                    previous_action=0
                    for step in range(80 if episode=='cycle-memory-return' else 27):
                        req=request(episode,step)
                        req.executed_action=previous_action if step else 0
                        if step>=26:
                            loop=[(0,0),(.25,0),(.5,0),(.75,0),(.75,.25),(.75,.5),(.75,.75),
                                  (.5,.75),(.25,.75),(0,.75),(0,.5),(0,.25)]
                            x,y=loop[(step-26)%len(loop)]
                            req.robot_pose.pose.pose.position.x=x;req.robot_pose.pose.pose.position.y=y
                            req.camera_pose.pose.pose.position.x=x;req.camera_pose.pose.pose.position.y=y
                        res=proxy(req);assert res.accepted,res.reason
                        duplicate=proxy(req);assert duplicate.accepted and duplicate.action==res.action
                        safe_action=res.action
                        width=res.frontiers.layout.dim[1].size if res.frontiers.layout.dim else 18
                        values=list(res.frontiers.data)
                        rows=[values[i:i+width] for i in range(0,len(values),width)]
                        if episode=='cycle-memory-return' and not avic_cycle_rejection_exercised and res.frontiers.data:
                            robot=(req.robot_pose.pose.pose.position.x,req.robot_pose.pose.pose.position.y)
                            old_selected=[old for old in reversed(selected_history) if step-old['step']>=12 and
                                math.hypot(robot[0]-old['robot'][0],robot[1]-old['robot'][1])<=.5]
                            matches=[(rank,row) for rank,row in enumerate(rows) if step>=47 and row[8] and not round(row[6]) and
                                any(math.hypot(row[0]-old['candidate'][0],row[1]-old['candidate'][1])<=.5 for old in old_selected)]
                            if matches:
                                rank,row=matches[0]
                                deadline=time.monotonic()+5
                                while not publisher.get_num_connections() and time.monotonic()<deadline:time.sleep(.02)
                                assert publisher.get_num_connections(),'score publisher not connected'
                                assert ack_queue.empty(),'unexpected pending ACK before cycle trial'
                                # Deliberately dominate the original Hybrid semantic ranking.
                                # This is an interface fixture, not a model or score-policy change.
                                publisher.publish(Float32MultiArray(data=[row[7],1,rank,row[0],row[1],row[2],1.,row[2]+10.]))
                                ack=ack_queue.get(timeout=30)
                                assert round(ack[0])==round(row[7]) and not round(ack[2]),ack
                                safe_action=round(ack[1])
                                cycle_trials.append(dict(episode=episode,step=step,query=round(row[7])))
                                avic_cycle_rejection_exercised=True
                        if episode=='cycle-memory-return':
                            for row in rows:
                                if round(row[6]):selected_history.append(dict(step=step,
                                    robot=(req.robot_pose.pose.pose.position.x,req.robot_pose.pose.pose.position.y),candidate=(row[0],row[1])))
                        previous_action=safe_action
                        time.sleep(.005)
                # All observations are complete. Shut down ONLY this fixture's
                # owned launch group before asserting its buffered output.
                rospy.signal_shutdown('cycle fixture observations complete')
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGINT)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
                text=(root/'launch.log').read_text()[offset:]
                assert '[FRONTIER_CYCLE] stage=suppress' in text or '[FRONTIER_CYCLE] stage=fallback' in text, \
                    'fixture did not exercise a genuine loop; do not count this as cycle-memory validation'
                assert avic_cycle_rejection_exercised,'fixture never injected an active-query score into a genuinely suppressed candidate'
                for trial in cycle_trials:
                    assert re.search(r'\[FRONTIER_CYCLE\] stage=avic_reject episode='+re.escape(trial['episode'])+r' step='+str(trial['step'])+r'\b',text), \
                        'trial was not rejected by cycle memory; same-target rollback is insufficient evidence: '+str(trial)
                new_lines=[line for line in text.splitlines() if 'episode=cycle-memory-new-episode' in line]
                assert not any('stage=suppress' in line or 'stage=fallback' in line for line in new_lines), 'cross-episode cycle history contamination'
                print('Cycle-memory genuine loop and episode reset regression passed',flush=True)
            if args.exercise_aligned_base or args.exercise_relaxed or args.exercise_image:
                downgrade_mode = 'on' if args.exercise_image else args.exercise_relaxed or args.exercise_aligned_base
                offset=(root/'launch.log').stat().st_size
                exercised=False
                for step in range(40):
                    res=proxy(request('aligned-base-downgrade-regression',step))
                    assert res.accepted,res.reason
                    values=list(res.frontiers.data)
                    width=res.frontiers.layout.dim[1].size if res.frontiers.layout.dim else 18
                    rows=[values[i:i+width] for i in range(0,len(values),width)]
                    if not rows or not rows[0][8]:continue
                    deadline=time.monotonic()+5
                    while not publisher.get_num_connections() and time.monotonic()<deadline:time.sleep(.02)
                    query=round(rows[0][7]);row=rows[0]
                    publisher.publish(Float32MultiArray(data=[query-1,1,0,row[0],row[1],row[2],.2,-.05]))
                    time.sleep(.15);assert ack_queue.empty(),'stale packet unexpectedly ACKed'
                    publisher.publish(Float32MultiArray(data=[query,1,0,row[0],row[1],row[2],.2,float('nan')]))
                    time.sleep(.15);assert ack_queue.empty(),'nonfinite packet unexpectedly ACKed'
                    packet=[query,len(rows)]
                    for rank,row in enumerate(rows):
                        lowered=-.05 if downgrade_mode=='on' else max(0.,row[2]-.05)
                        packet.extend([rank,row[0],row[1],row[2],.2,lowered])
                    publisher.publish(Float32MultiArray(data=packet));ack=ack_queue.get(timeout=30)
                    assert round(ack[0])==query,ack
                    if downgrade_mode=='off':assert not round(ack[2]),ack
                    exercised=True;break
                assert exercised,'synthetic depth did not expose an active frontier query'
                # Flush owned ROS output before checking actual C++ score matching.
                if args.exercise_relaxed or args.exercise_image:
                    rospy.signal_shutdown('relaxed downgrade observations complete')
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGINT)
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM); process.wait(timeout=10)
                time.sleep(.2)
                text=(root/'launch.log').read_text()[offset:]
                assert 'Ignoring stale AVIC frontier scores' in text
                assert 'Ignoring non-finite AVIC frontier score' in text
                if downgrade_mode=='on':
                    assert '[AVIC_SCORE] frontier=' in text,'finite lowering was not matched to a reachable planning frontier'
                else:
                    assert '[AVIC_SCORE] frontier=' not in text,'default accepted lowering'
                print('ROS downgrade/stale/nonfinite regression passed: '+
                      ('image_glm=on' if args.exercise_image else 'relaxed=' + args.exercise_relaxed if args.exercise_relaxed else
                       'aligned_base=' + args.exercise_aligned_base),flush=True)
            if args.exercise_route:
                # Synthetic map transactions, not Habitat trajectories or an SR test.
                # Mild positional jitter keeps legacy stationary counters from
                # terminating the lifetime fixture; the blocked fixture instead
                # reports a forward action with exactly zero displacement.
                for case in ("lifetime", "blocked", "reset"):
                    episode = "route-regression-" + case
                    applied_step = None
                    previous_request = None
                    selected_goal = None
                    for step in range(45):
                        req = request(episode, step)
                        if step >= 26:
                            for pose in (req.robot_pose, req.camera_pose):
                                pose.pose.pose.position.x = 0.04 if step % 2 else -0.04
                        if case == "blocked" and applied_step is not None and step == applied_step + 1:
                            req.robot_pose = copy.deepcopy(previous_request.robot_pose)
                            req.camera_pose = copy.deepcopy(previous_request.camera_pose)
                            req.executed_action = 1
                        res = proxy(req)
                        assert res.accepted, res.reason
                        previous_request = req
                        duplicate = proxy(req)
                        assert duplicate.accepted and duplicate.action == res.action
                        if applied_step is not None:
                            if case == "lifetime" and step < applied_step + 6:
                                width = res.frontiers.layout.dim[1].size
                                rows = [res.frontiers.data[i:i+width]
                                        for i in range(0, len(res.frontiers.data), width)]
                                assert rows and all(not round(row[8]) for row in rows), rows
                            if step >= applied_step + 7:
                                break
                            continue
                        if not res.frontiers.data:
                            continue
                        width = res.frontiers.layout.dim[1].size
                        rows = [res.frontiers.data[i:i+width]
                                for i in range(0, len(res.frontiers.data), width)]
                        alternates = [(rank, row) for rank, row in enumerate(rows)
                                      if not round(row[6]) and round(row[8])]
                        if not alternates:
                            continue
                        rank, row = alternates[-1]
                        test_decision = (0.25 * row[2] + 0.75 * 0.95
                            if os.environ.get('APEXNAV_RELAXED_IMAGINATION') == '1' or os.environ.get('APEXNAV_IMAGE_GLM') == '1'
                            else row[2] + 10.0)
                        publisher.publish(Float32MultiArray(data=[row[7], 1, rank,
                            row[0], row[1], row[2], 0.95, test_decision]))
                        ack = ack_queue.get(timeout=30)
                        assert round(ack[0]) == round(row[7])
                        if round(ack[2]):
                            applied_step = step
                            selected_goal = list(row[:2])
                            if case == "reset":
                                # Switch episodes with an active lease. No goal or
                                # action changes may leak into the fresh episode.
                                fresh_trace = []
                                for fresh_step in range(32):
                                    fresh = proxy(request("route-regression-after-reset", fresh_step))
                                    assert fresh.accepted, fresh.reason
                                    fresh_trace.append([fresh.action, fresh.final_state,
                                        fresh.exploration_result, list(fresh.frontiers.data)])
                                assert fresh_trace == traces[0], "route leaked across episodes"
                                break
                    assert applied_step is not None, "fixture never changed frontier: " + case
                    time.sleep(0.2)
                    lines = [line for line in (root / "launch.log").read_text().splitlines()
                             if "[AVIC_ROUTE]" in line and "episode=" + episode + " " in line]
                    assert any("stage=start" in line for line in lines), lines
                    if case == "lifetime":
                        assert any("stage=hold" in line for line in lines), lines
                        assert any("reason=budget_complete" in line for line in lines), lines
                    elif case == "blocked":
                        assert any("reason=forward_blocked" in line for line in lines), lines
                        assert not any("stage=hold" in line for line in lines), lines
                    route_cases.append(dict(case=case, applied_step=applied_step,
                                            selected_goal=selected_goal, events=lines))
            image_strong_case = None
            if args.exercise_image_strong:
                # Inject close, legal 25/75 scores into the actual C++ callback.
                # No generated image, API, Habitat success or SR is claimed here.
                for step in range(40):
                    res=proxy(request('image-strong-near-score',step));assert res.accepted,res.reason
                    values=list(res.frontiers.data)
                    width=res.frontiers.layout.dim[1].size if res.frontiers.layout.dim else 18
                    rows=[values[i:i+width] for i in range(0,len(values),width)]
                    alternates=[(rank,row) for rank,row in enumerate(rows) if row[8] and not round(row[6])]
                    if not alternates:continue
                    rank,winner=alternates[-1]
                    base=max(row[2] for row in rows)
                    decisions=[base+(0.001 if i==rank else 0.) for i in range(len(rows))]
                    packet=[rows[0][7],len(rows)]
                    for i,row in enumerate(rows):
                        image_score=(decisions[i]-.25*row[2])/.75
                        assert -1<=image_score<=1
                        packet.extend([i,row[0],row[1],row[2],image_score,decisions[i]])
                    deadline=time.monotonic()+5
                    while not publisher.get_num_connections() and time.monotonic()<deadline:time.sleep(.02)
                    offset=(root/'launch.log').stat().st_size
                    publisher.publish(Float32MultiArray(data=packet));ack=ack_queue.get(timeout=30)
                    assert round(ack[0])==round(rows[0][7]),ack
                    expected=args.exercise_image_strong=='on'
                    assert bool(round(ack[2]))==expected,(args.exercise_image_strong,ack)
                    image_strong_case=dict(mode=args.exercise_image_strong,step=step,
                        query=round(rows[0][7]),rows=rows,decisions=decisions,winner_rank=rank,
                        target_changed=bool(round(ack[2])),actual_action=round(ack[1]))
                    break
                assert image_strong_case,'fixture did not expose an active alternate frontier'
                rospy.signal_shutdown('near-score observations complete')
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGINT)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=10)
                text=(root/'launch.log').read_text()[offset:]
                if args.exercise_image_strong=='on':
                    expected_xy='selected_candidate=(%.3f,%.3f)'%(winner[0],winner[1])
                    assert '[IMAGE_GLM_STRONG_SELECT]' in text and expected_xy in text,text
                    assert 'bypass=Hybrid_TSP_Closest' in text
                else:
                    assert '[IMAGE_GLM_STRONG_SELECT]' not in text
                    assert 'Explore the environment (Closest)' in text,text
                print('Near-score policy regression passed: '+json.dumps(image_strong_case),flush=True)
            tsp_failure = None
            if args.exercise_tsp_failure:
                import rosnode
                rospy.set_param('/exploration_node/exploration/policy', 3)  # force actual LKH
                for step in range(32):
                    res = proxy(request('tsp-real-solver-regression', step))
                    assert res.accepted, res.reason
                time.sleep(0.2)
                assert '[TSP_RPC] stage=success' in (root / 'launch.log').read_text(), 'real LKH was not exercised'
                # Only this test's ROS_MASTER_URI is contacted, never port 11311.
                killed, failed = rosnode.kill_nodes(['/tsp_solver'])
                assert '/tsp_solver' in killed and not failed
                time.sleep(0.3)
                for step in range(27):
                    res = proxy(request('tsp-outage-regression', step))
                    if step < 26:
                        assert res.accepted, res.reason
                    else:
                        assert not res.accepted, 'TSP outage silently became a navigation action'
                        assert 'TSP infrastructure failure' in res.reason, res.reason
                        again = proxy(request('tsp-outage-regression', step))
                        assert not again.accepted and 'previous transaction failed' in again.reason
                        tsp_failure = res.reason
            result = dict(status="passed", repeats=3, frames_per_repeat=32,
                          no_op_rollbacks=no_op_count, identical_decisions_and_frontiers=True,
                          checks=["duplicate", "payload mismatch", "out of order", "invalid pose",
                                  "retired episode", "cross-episode reset", "wall-time delay", "same-target rollback"])
            result["frames_per_repeat"] = 80 if args.exercise_recovery else 32
            result["recovery_exercise"] = args.exercise_recovery
            result["frontier_once_exercise"] = args.exercise_once
            if args.exercise_once:
                result["checks"].extend(["all frontiers despite top_k=1", "negative fusion accepted by C++"])
            result["route_cases"] = route_cases
            if args.exercise_semantic:
                text = (root / 'launch.log').read_text()
                assert '[V1_POLICY_CONFIG] version=1.0 policy=semantic_only' in text
                assert '[V1_SEMANTIC_SELECT]' in text
                assert 'augmented=1' in text
                assert '[TSP_RPC]' not in text, 'semantic experiment invoked TSP'
                assert '[Exploration Mode] Working on Hybrid Mode' not in text
                result['semantic_policy'] = 'passed with isolated TSP stopped and AVIC route changes'
            result["tsp_failure"] = tsp_failure
            result['image_strong_case'] = image_strong_case
            if args.exercise_recovery:
                result["checks"].remove("same-target rollback")
                result["checks"].extend(["recovery replan", "recovery switch", "SVC off"])
            (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (root / "traces.json").write_text(json.dumps(traces) + "\n")
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
