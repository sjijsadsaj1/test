"""
Habitat ObjectNav Evaluation Script for HM3D/MP3D Datasets

This script evaluates object navigation performance using the Habitat simulator
with support for HM3D-v1, HM3D-v2, and MP3D datasets. It communicates with ROS for
real-time planning and decision making, incorporates vision-language models
for object detection and image-text matching, and generates comprehensive
evaluation metrics.

Usage:
    # Run with HM3D-v1 dataset
    python habitat_evaluation.py --dataset hm3dv1

    # Run with HM3D-v2 dataset (default)
    python habitat_evaluation.py --dataset hm3dv2

    # Run with MP3D dataset
    python habitat_evaluation.py --dataset mp3d

    # Test specific episode
    python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10

Author: Zager-Zhang
"""

# Standard library imports
import argparse
import gzip
import json
import os
import signal
import time
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from omegaconf import DictConfig
from prettytable import PrettyTable
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Int32, Int32MultiArray, Float32MultiArray, Float64
import tqdm

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    images_to_video,
    observations_to_image,
    overlay_frame,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from basic_utils.failure_check.count_files import count_files_in_directory
from basic_utils.failure_check.failure_check import check_failure, is_on_same_floor
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from basic_utils.record_episode.read_record import read_record
from basic_utils.record_episode.write_record import write_record
from basic_utils.episode_selection import select_episodes_from_manifest
from avic_bridge import AvicController, AvicControllerConfig
from habitat2ros import habitat_publisher
from llm.answer_reader.answer_reader import read_answer
from params import HABITAT_STATE, ROS_STATE, ACTION, RESULT_TYPES
from vlm.Labels import MP3D_ID_TO_NAME
from vlm.utils.get_itm_message import get_itm_message_cosine
from vlm.utils.get_object_utils import get_object


def publish_int32(publisher, data):
    msg = Int32()
    msg.data = data
    publisher.publish(msg)


def publish_float64(publisher, data):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)


def publish_int32_array(publisher, data_list):
    msg = Int32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def publish_float32_array(publisher, data_list):
    msg = Float32MultiArray()
    msg.data = data_list
    publisher.publish(msg)


def signal_handler(sig, frame):
    """Handle Ctrl+C signal for graceful shutdown"""
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):
    """Convert RGB image to BGR format"""
    return image[:, :, [2, 1, 0]]


def observation_planar_pose(observations):
    """Return the same planar world pose published to ApexNav over ROS."""
    gps = np.asarray(observations["gps"], dtype=float).reshape(-1)
    compass = np.asarray(observations["compass"], dtype=float).reshape(-1)
    if gps.size < 3 or compass.size < 1:
        raise ValueError("Habitat GPS/compass observation has an unexpected shape")
    return float(-gps[2]), float(-gps[0]), float(compass[0])


def detected_object_records(target_label, related_objects, scores, label_indices):
    """Map ApexNav detector indices back to names for the AVIC evidence prompt."""
    records = []
    related = [str(item) for item in (related_objects or [])]
    for score, label_index in zip(scores, label_indices):
        index = int(label_index)
        if index == 0:
            name = str(target_label)
        elif 1 <= index <= len(related):
            name = related[index - 1]
        else:
            continue
        records.append({"label": name, "score": float(score)})
    return records


def publish_observations(event):
    """Timer callback to publish habitat observations and trigger messages"""
    global msg_observations, fusion_threshold
    global ros_pub, trigger_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)
    trigger = PoseStamped()
    trigger_pub.publish(trigger)


def ros_action_callback(msg):
    global global_action
    global_action = msg.data


def ros_state_callback(msg):
    global ros_state
    ros_state = msg.data


def ros_final_state_callback(msg):
    global final_state
    final_state = msg.data


def ros_expl_result_callback(msg):
    global expl_result
    expl_result = msg.data


def frontier_candidates_callback(msg):
    """Decode a read-only ApexNav frontier decision snapshot."""
    global latest_topk_frontiers
    fields = 11
    if len(msg.layout.dim) >= 2:
        declared_fields = int(msg.layout.dim[1].size)
        if declared_fields in (7, 9, 11) or declared_fields >= 12:
            fields = declared_fields
    values = list(msg.data)
    candidates = []
    for offset in range(0, len(values) - fields + 1, fields):
        row = values[offset : offset + fields]
        candidate = {
            "rank": len(candidates),
            "x": float(row[0]),
            "y": float(row[1]),
            "semantic_score": float(row[2]),
            "path_length": float(row[3]),
            "relative_yaw_rad": float(row[4]),
            "first_action": int(round(row[5])),
            "selected_by_apexnav": bool(round(row[6])),
            "query_id": -1,
            "query_active": False,
        }
        if fields >= 9:
            candidate["query_id"] = int(round(row[7]))
            candidate["query_active"] = bool(round(row[8]))
        candidate["observation_evidence"] = float(row[9]) if fields >= 11 else 0.0
        candidate["evidence_observed"] = bool(round(row[10])) if fields >= 11 else False
        if fields >= 12:
            prefix_count = max(0, min(int(round(row[11])), fields - 12))
            candidate["path_action_prefix"] = [
                int(round(value)) for value in row[12 : 12 + prefix_count]
            ]
        else:
            candidate["path_action_prefix"] = []
        candidates.append(candidate)
    latest_topk_frontiers = candidates
    if candidates:
        print("[AVIC_FRONTIER] " + json.dumps(candidates, ensure_ascii=False), flush=True)


def avic_frontier_ack_callback(msg):
    """Receive the action recomputed by ApexNav after frontier validation."""
    global latest_avic_frontier_ack
    if len(msg.data) < 3:
        return
    latest_avic_frontier_ack = {
        "query_id": int(round(msg.data[0])),
        "safe_action": int(round(msg.data[1])),
        "applied": bool(round(msg.data[2])),
    }


def _parse_dataset_arg():
    """Parse CLI to choose dataset and capture remaining Hydra overrides."""
    parser = argparse.ArgumentParser(
        description="Habitat ObjectNav Evaluation", add_help=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hm3dv1", "hm3dv2", "mp3d"],
        default="hm3dv2",
        help="Choose dataset: hm3dv1, hm3dv2 or mp3d (default: hm3dv2)",
    )
    parser.add_argument(
        "--avic-mode",
        choices=["apexnav", "avic"],
        default="apexnav",
        help="Navigation mode: original ApexNav or AVIC-augmented frontier scoring",
    )
    parser.add_argument(
        "--avic-service-url",
        default="http://127.0.0.1:8091",
        help="Base URL of the independent AVIC service",
    )
    parser.add_argument("--avic-timeout", type=float, default=180.0)
    parser.add_argument("--avic-warmup-steps", type=int, default=26)
    parser.add_argument("--avic-max-calls", type=int, default=20)
    parser.add_argument("--avic-cooldown-steps", type=int, default=5)
    parser.add_argument("--avic-score-low", type=float, default=0.191149)
    parser.add_argument("--avic-score-high", type=float, default=0.227590)
    parser.add_argument("--avic-max-score-gap", type=float, default=0.01)
    parser.add_argument("--avic-real-weight", type=float, default=0.5)
    parser.add_argument("--avic-imagination-weight", type=float, default=0.5)
    parser.add_argument("--avic-min-score-spread", type=float, default=0.0001)
    parser.add_argument("--avic-cache-radius", type=float, default=0.5)
    parser.add_argument("--avic-cache-yaw-deg", type=float, default=15.0)
    parser.add_argument("--avic-anchor-fov-deg", type=float, default=90.0)
    parser.add_argument("--avic-anchor-max-distance", type=float, default=8.0)
    parser.add_argument("--avic-anchor-min-quality", type=float, default=0.35)
    parser.add_argument("--avic-max-bonus", type=float, default=0.02)
    parser.add_argument("--avic-min-reliability", type=float, default=0.55)
    parser.add_argument("--avic-min-evidence-gain", type=float, default=0.20)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Evaluate only the first N episodes; 0 means the complete split",
    )
    parser.add_argument(
        "--episode-manifest",
        default="",
        help="JSON manifest selecting an explicit reproducible episode subset",
    )
    parser.add_argument(
        "--result-dir",
        default="",
        help="Independent record/video directory for this evaluation condition",
    )
    # Keep unknown so users can still pass Hydra-style overrides (e.g., key=value)
    args, unknown = parser.parse_known_args()
    return args, unknown


def main(cfg: DictConfig, cli_args: argparse.Namespace) -> None:
    global msg_observations, global_action, ros_state, fusion_threshold
    global ros_pub, trigger_pub, obj_point_cloud_pub, confidence_threshold_pub
    global final_state, expl_result, latest_topk_frontiers, latest_avic_frontier_ack

    # Load MP3D validation data for object category mapping
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }

    start_time = time.time()

    final_state = 0
    expl_result = 0
    latest_topk_frontiers = []
    latest_avic_frontier_ack = None
    result_list = [0] * len(RESULT_TYPES)

    cfg = patch_config(cfg)

    # Extract configuration parameters
    video_output_path = cfg.video_output_path.format(split=cfg.habitat.dataset.split)
    if cli_args.result_dir:
        video_output_path = os.path.abspath(os.path.expanduser(cli_args.result_dir))
    need_video = cfg.need_video
    record_file_path = os.path.join(video_output_path, cfg.record_file_name)
    continue_path = os.path.join(video_output_path, cfg.continue_file_name)
    max_episode_steps = cfg.habitat.environment.max_episode_steps
    success_distance = cfg.habitat.task.measurements.success.success_distance

    detector_cfg = cfg.detector

    avic_controller = AvicController(
        AvicControllerConfig(
            mode=cli_args.avic_mode,
            service_url=cli_args.avic_service_url,
            timeout_s=cli_args.avic_timeout,
            warmup_steps=cli_args.avic_warmup_steps,
            max_calls_per_episode=cli_args.avic_max_calls,
            cooldown_steps=cli_args.avic_cooldown_steps,
            score_band_low=cli_args.avic_score_low,
            score_band_high=cli_args.avic_score_high,
            max_score_gap=cli_args.avic_max_score_gap,
            real_score_weight=cli_args.avic_real_weight,
            imagination_score_weight=cli_args.avic_imagination_weight,
            min_imagination_score_spread=cli_args.avic_min_score_spread,
            cache_match_radius_m=cli_args.avic_cache_radius,
            cache_yaw_tolerance_rad=np.deg2rad(cli_args.avic_cache_yaw_deg),
            anchor_horizontal_fov_rad=np.deg2rad(cli_args.avic_anchor_fov_deg),
            anchor_max_distance_m=cli_args.avic_anchor_max_distance,
            anchor_min_quality=cli_args.avic_anchor_min_quality,
            max_imagination_bonus=cli_args.avic_max_bonus,
            min_imagination_reliability=cli_args.avic_min_reliability,
            min_imagination_evidence_gain=cli_args.avic_min_evidence_gain,
        )
    )
    if avic_controller.enabled:
        try:
            health = avic_controller.check_health()
            if not health.get("ok", False):
                raise RuntimeError("service reports not ready: %s" % health)
            print("[AVIC] service ready: " + json.dumps(health, ensure_ascii=False))
        except Exception as exc:
            print("[AVIC] service unavailable; disabling AVIC for this run: %s" % exc)
            avic_controller.config.mode = "apexnav"

    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = llm_cfg.llm_response_path

    # Single test parameters
    env_num_once = cfg.test_epi_num  # Which episode to test for single run
    flag_once = env_num_once != -1  # Whether to run single test
    if cli_args.episode_manifest and flag_once:
        raise ValueError(
            "--episode-manifest cannot be combined with the test_epi_num override"
        )

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)
    os.makedirs(video_output_path, exist_ok=True)

    # Add top_down_map and collisions visualization
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    env = habitat.Env(cfg)
    print("Environment creation successful")
    if cli_args.episode_manifest:
        selected_episodes = select_episodes_from_manifest(
            env.episodes, cli_args.episode_manifest
        )
        env.episodes = selected_episodes
        print(
            f"Loaded {len(selected_episodes)} selected episodes from "
            f"{os.path.abspath(os.path.expanduser(cli_args.episode_manifest))}"
        )
    number_of_episodes = len(env.episodes)
    if cli_args.max_episodes < 0:
        raise ValueError("--max-episodes must be non-negative")
    target_episode_count = number_of_episodes
    if cli_args.max_episodes > 0:
        target_episode_count = min(cli_args.max_episodes, number_of_episodes)

    # Read previous records and set initial values
    (
        num_total,
        num_success,
        spl_all,
        soft_spl_all,
        distance_to_goal_all,
        distance_to_goal_reward_all,
        last_time,
    ) = read_record(continue_path, flag_once)

    if not flag_once and num_total >= target_episode_count:
        raise ValueError("Already finished the requested episodes.")

    pbar = tqdm.tqdm(total=target_episode_count)

    env_count = num_total if not flag_once else env_num_once
    while env_count:
        pbar.update()
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    # Initialize ROS publishers, subscribers, and timers
    obj_point_cloud_pub = rospy.Publisher(
        "habitat/object_point_cloud", PointCloud2, queue_size=10
    )
    ros_pub = habitat_publisher.ROSPublisher()
    rospy.Subscriber("/habitat/plan_action", Int32, ros_action_callback, queue_size=10)
    rospy.Subscriber("/ros/state", Int32, ros_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_state", Int32, ros_final_state_callback, queue_size=10)
    rospy.Subscriber("/ros/expl_result", Int32, ros_expl_result_callback, queue_size=10)
    rospy.Subscriber(
        "/apexnav/topk_frontiers",
        Float32MultiArray,
        frontier_candidates_callback,
        queue_size=10,
    )
    rospy.Subscriber(
        "/apexnav/avic_frontier_ack",
        Float32MultiArray,
        avic_frontier_ack_callback,
        queue_size=10,
    )
    state_pub = rospy.Publisher("/habitat/state", Int32, queue_size=10)
    trigger_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    cld_with_score_pub = rospy.Publisher(
        "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
    )
    progress_pub = rospy.Publisher("/habitat/progress", Int32MultiArray, queue_size=10)
    record_pub = rospy.Publisher("/habitat/record", Float32MultiArray, queue_size=10)
    executed_action_pub = rospy.Publisher(
        "/habitat/executed_action", Int32, queue_size=10
    )
    avic_frontier_scores_pub = rospy.Publisher(
        "/apexnav/avic_frontier_scores", Float32MultiArray, queue_size=10
    )

    episodes_this_run = 1 if flag_once else target_episode_count - num_total
    for epi in range(episodes_this_run):
        # Publish progress information
        publish_int32_array(progress_pub, [num_total, target_episode_count])

        if flag_once:
            while env_count:
                env.current_episode = next(env.episode_iterator)
                env_count -= 1

        # Initialize episode variables
        pass_object = 0.0
        near_object = 0.0
        global_action = None
        cld_with_score_msg = MultipleMasksWithConfidence()
        count_steps = 0

        camera_pitch = 0.0
        observations = env.reset()
        current_rgb_for_avic = np.array(observations["rgb"], copy=True)
        last_itm_cosine = None
        latest_topk_frontiers = []
        latest_avic_frontier_ack = None
        avic_controller.reset_episode()
        observations["camera_pitch"] = camera_pitch
        msg_observations = deepcopy(observations)
        del observations["camera_pitch"]
        label = env.current_episode.object_category

        # Convert object category to coco name format
        if label in category_to_coco:
            coco_id = category_to_coco[label]
            label = id_to_name.get(coco_id, label)

        # Get LLM answer and fusion threshold for the target object
        llm_answer, room, fusion_threshold = read_answer(
            llm_answer_path, llm_response_path, label, llm_client
        )
        initial_x, initial_y, initial_yaw = observation_planar_pose(observations)
        avic_controller.record_observation(
            rgb=current_rgb_for_avic,
            robot_x=initial_x,
            robot_y=initial_y,
            robot_yaw_rad=initial_yaw,
            camera_pitch_rad=camera_pitch,
            step_index=0,
        )

        # Initialize video frame collection
        vis_frames = []
        info = env.get_metrics()
        if need_video:
            frame = observations_to_image(observations, info)
            info.pop("top_down_map")
            frame = overlay_frame(frame, info)
            vis_frames = [frame]

        # Start publishing basic information and trigger messages
        pub_timer = rospy.Timer(rospy.Duration(0.25), publish_observations)

        print("Agent is waiting in the environment!!!")

        # Wait for ROS system to be ready
        rate = rospy.Rate(10)
        ros_state = ROS_STATE.INIT
        while ros_state == ROS_STATE.INIT or ros_state == ROS_STATE.WAIT_TRIGGER:
            if ros_state == ROS_STATE.INIT:
                print("Waiting for ROS to get odometry...")
            elif ros_state == ROS_STATE.WAIT_TRIGGER:
                print("Waiting for ROS trigger...")
            rate.sleep()

        # Stop timer publishing when starting action execution
        pub_timer.shutdown()

        print("Agent is ready to go!!!!")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not env.episode_over:
            # Skip episode if target is not on the same floor
            is_feasible = 0
            for goal in env.current_episode.goals:
                height = goal.position[1]
                is_feasible += is_on_same_floor(
                    height=height, episode=env.current_episode
                )
            if not is_feasible:
                break

            # Parse action from decision system
            action = None
            if global_action is not None:
                if count_steps == max_episode_steps - 1:
                    global_action = ACTION.STOP

                baseline_action = int(global_action)
                avic_decision = avic_controller.decide_action(
                    rgb=current_rgb_for_avic,
                    baseline_action=baseline_action,
                    target_category=label,
                    target_room=room,
                    target_related_objects=llm_answer,
                    scene_id=env.current_episode.scene_id,
                    episode_id=env.current_episode.episode_id,
                    step_index=count_steps,
                    camera_pitch_rad=camera_pitch,
                    itm_cosine=last_itm_cosine,
                    final_state=final_state,
                    exploration_result=expl_result,
                    frontier_candidates=deepcopy(latest_topk_frontiers),
                )
                # AVIC never directly controls Habitat or chooses a frontier.
                # It sends temporary semantic scores back to C++; ApexNav then
                # reruns its unchanged Hybrid semantic/geometric policy.
                global_action = baseline_action
                # Cached imagination scores must be reapplied when the same
                # physical frontier re-enters the middle score band, without
                # issuing another SVC request.
                if avic_decision.frontier_scores:
                    score_rows = avic_decision.frontier_scores
                    query_id = int(score_rows[0]["query_id"])
                    score_message = [float(query_id), float(len(score_rows))]
                    for score in score_rows:
                        score_message.extend(
                            [
                                float(score["rank"]),
                                float(score["x"]),
                                float(score["y"]),
                                float(score["real_semantic_score"]),
                                float(score["imagined_itm_score"]),
                                float(score["decision_semantic_score"]),
                            ]
                        )
                    latest_avic_frontier_ack = None
                    publish_float32_array(avic_frontier_scores_pub, score_message)
                    print(
                        "[AVIC_FRONTIER_SCORES] "
                        + json.dumps(
                            {
                                "query_id": query_id,
                                "scores": score_rows,
                                "confidence": avic_decision.confidence,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    # A timeout fails open to the original ApexNav action.
                    ack_deadline = time.monotonic() + 2.0
                    while time.monotonic() < ack_deadline and not rospy.is_shutdown():
                        ack = latest_avic_frontier_ack
                        if ack is not None and ack["query_id"] == query_id:
                            if ack["applied"]:
                                global_action = ack["safe_action"]
                            print(
                                "[AVIC_FRONTIER_ACK] "
                                + json.dumps(ack, ensure_ascii=False),
                                flush=True,
                            )
                            break
                        time.sleep(0.01)
                publish_int32(executed_action_pub, global_action)
                avic_controller.record_executed_action(global_action)

                if global_action == ACTION.MOVE_FORWARD:
                    action = HabitatSimActions.move_forward
                elif global_action == ACTION.TURN_LEFT:
                    action = HabitatSimActions.turn_left
                elif global_action == ACTION.TURN_RIGHT:
                    action = HabitatSimActions.turn_right
                elif global_action == ACTION.TURN_DOWN:
                    action = HabitatSimActions.look_down
                    camera_pitch = camera_pitch - np.pi / 6.0
                elif global_action == ACTION.TURN_UP:
                    action = HabitatSimActions.look_up
                    camera_pitch = camera_pitch + np.pi / 6.0
                elif global_action == ACTION.STOP:
                    action = HabitatSimActions.stop

                global_action = None

            if action is None:
                continue

            count_steps += 1
            print(f"\n--------------Step: {count_steps}--------------")
            print(f"Finding [{label}]; Action: {action};")

            # Notify ROS system that action execution is starting
            publish_int32(state_pub, HABITAT_STATE.ACTION_EXEC)

            observations = env.step(action)
            current_rgb_for_avic = np.array(observations["rgb"], copy=True)

            # Calculate ITM cosine similarity score
            cosine = get_itm_message_cosine(observations["rgb"], label, room)
            last_itm_cosine = cosine
            print(f"Target related room: {room}")
            print(f"ITM cosine similarity: {cosine:.3f}")

            publish_float64(itm_score_pub, cosine)

            # Detect objects in the current observation
            observations["rgb"], score_list, object_masks_list, label_list = get_object(
                label, observations["rgb"], detector_cfg, llm_answer
            )
            robot_x, robot_y, robot_yaw = observation_planar_pose(observations)
            avic_controller.record_observation(
                rgb=current_rgb_for_avic,
                robot_x=robot_x,
                robot_y=robot_y,
                robot_yaw_rad=robot_yaw,
                camera_pitch_rad=camera_pitch,
                step_index=count_steps,
                itm_cosine=cosine,
                detected_objects=detected_object_records(
                    label, llm_answer, score_list, label_list
                ),
            )

            # Publish habitat observations to ROS
            observations["camera_pitch"] = camera_pitch
            msg_observations = deepcopy(observations)
            del observations["camera_pitch"]
            ros_pub.habitat_publish_ros_topic(msg_observations)

            # Generate and publish object point clouds
            obj_point_cloud_list = get_object_point_cloud(
                cfg, observations, object_masks_list
            )

            # Publish detection-related information
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_pub.publish(cld_with_score_msg)

            # Generate video frame
            info = env.get_metrics()
            if need_video:
                frame = observations_to_image(observations, info)
                info.pop("top_down_map")
                frame = overlay_frame(frame, info)
                vis_frames.append(frame)

            # Track if agent has passed close to the target
            distance_to_goal = info["distance_to_goal"]
            if distance_to_goal <= success_distance and pass_object == 0:
                pass_object = 1

            # Notify ROS system that action execution is complete
            publish_int32(state_pub, HABITAT_STATE.ACTION_FINISH)
            rate.sleep()

        # Notify ROS system that current episode evaluation is complete
        publish_int32(state_pub, HABITAT_STATE.EPISODE_FINISH)

        # Collect evaluation metrics
        info = env.get_metrics()
        spl = info["spl"]
        soft_spl = info["soft_spl"]
        distance_to_goal = info["distance_to_goal"]
        distance_to_goal_reward = info["distance_to_goal_reward"]
        success = info["success"]

        # Check if agent got close to the target object
        if distance_to_goal <= success_distance:
            near_object = 1

        # Determine episode result
        if success == 1:
            num_success += 1
            result_text = "success"
        else:
            result_text = check_failure(
                env.current_episode,
                final_state,
                expl_result,
                count_steps,
                max_episode_steps,
                pass_object,
                near_object,
            )

        # Update cumulative statistics
        num_total += 1
        spl_all += spl
        soft_spl_all += soft_spl
        distance_to_goal_all += distance_to_goal
        distance_to_goal_reward_all += distance_to_goal_reward

        # Generate video file
        scene_id = env.current_episode.scene_id
        episode_id = env.current_episode.episode_id
        video_name = f"{os.path.basename(scene_id)}_{episode_id}"
        time_spend = time.time() - start_time + last_time

        img2video_output_path = os.path.join(video_output_path, result_text)

        if flag_once:
            img2video_output_path = "videos"
            video_name = "video_once"

        if need_video:
            images_to_video(
                vis_frames, img2video_output_path, video_name, fps=6, quality=9
            )
        vis_frames.clear()

        # Display average performance metrics
        table1 = PrettyTable(["Metric", "Average"])
        table1.add_row(["Average Success", f"{num_success/num_total * 100:.2f}%"])
        table1.add_row(["Average SPL", f"{spl_all/num_total * 100:.2f}%"])
        table1.add_row(["Average Soft SPL", f"{soft_spl_all/num_total * 100:.2f}%"])
        table1.add_row(
            ["Average Distance to Goal", f"{distance_to_goal_all/num_total:.4f}"]
        )
        print(table1)
        print(f"Episode {num_total} data written to {record_file_path}")
        print(f"Result: {result_text}")

        # Display total performance metrics
        table2 = PrettyTable(["Metric", "Total"])
        table2.add_row(["Total Success", f"{num_success}"])
        table2.add_row(["Total SPL", f"{spl_all:.2f}"])
        table2.add_row(["Total Soft SPL", f"{soft_spl_all:.2f}"])
        table2.add_row(["Total Distance to Goal", f"{distance_to_goal_all:.4f}"])

        if flag_once:
            break

        # Write results to record file
        write_record(
            scene_id,
            episode_id,
            table1,
            result_text,
            label,
            num_total,
            time_spend,
            record_file_path,
        )

        # Write results to continue file
        write_record(
            scene_id,
            episode_id,
            table2,
            result_text,
            label,
            num_total,
            time_spend,
            continue_path,
        )

        # Count files in each result category folder
        for i in range(len(RESULT_TYPES)):
            folder = RESULT_TYPES[i]  # Get current category (folder name)
            folder_path = os.path.join(video_output_path, folder)  # Build folder path
            file_count = count_files_in_directory(folder_path)  # Count files in folder
            result_list[i] = file_count

        # Publish comprehensive record data
        record_data = [
            num_success / num_total * 100,
            spl_all / num_total * 100,
            soft_spl_all / num_total * 100,
            distance_to_goal_all / num_total,
        ]
        record_data.extend(result_list)
        publish_float32_array(record_pub, record_data)

        pbar.update()
        if num_total >= target_episode_count:
            break
        try:
            env.current_episode = next(env.episode_iterator)
        except StopIteration:
            break
        rospy.sleep(0.1)  # wait a moment

    env.close()
    pbar.close()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_eval_node", anonymous=True)

    try:
        cli_args, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{cli_args.dataset}"
        # Compose the chosen config and pass through extra Hydra overrides
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg, cli_args)
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
