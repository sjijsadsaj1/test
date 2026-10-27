
#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_fsm.h>
#include <exploration_manager/exploration_data.h>
#include <vis_utils/planning_visualization.h>
#include <plan_env/map_ros.h>

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <sensor_msgs/point_cloud2_iterator.h>

namespace apexnav_planner {
namespace {
bool literatureMethod(const char* name) {
  const char* value=std::getenv("APEXNAV_LITERATURE_METHOD");
  return value && std::string(value)==name;
}
bool terminalRecoveryOnly() {
  const char* value=std::getenv("APEXNAV_OPEN_ESCAPE_TRIGGER");
  return value && std::string(value)=="terminal";
}
bool frontierOnceExperiment() {
  const char* mode = std::getenv("APEXNAV_FRONTIER_ONCE");
  return mode && std::string(mode) == "1";
}
bool alignedBaseExperiment() {
  const char* mode = std::getenv("APEXNAV_ALIGNED_BASE_EXPERIMENT");
  return mode && std::string(mode) == "1";
}
bool relaxedImagination() {
  const char* mode=std::getenv("APEXNAV_RELAXED_IMAGINATION");
  return mode && std::string(mode)=="1";
}
bool imageGLMExperiment() {
  const char* mode=std::getenv("APEXNAV_IMAGE_GLM");
  return mode && std::string(mode)=="1";
}
bool textFrontierExperiment() {
  const char* mode=std::getenv("APEXNAV_TEXT_FRONTIER");
  return mode && std::string(mode)=="1";
}
bool extremeSafetyExperiment() {
  const char* mode = std::getenv("APEXNAV_EXTREME_SAFETY");
  return mode && std::string(mode) == "1";
}
bool hypothesisProgressExperiment() {
  const char* mode = std::getenv("APEXNAV_HYPOTHESIS_PROGRESS");
  return mode && std::string(mode) == "1";
}
}
void ExplorationFSM::init(ros::NodeHandle& nh)
{
  nh_ = nh;
  fresh_confirmation_.reset();fresh_target_frames_.clear();
  open_escape_deadline_=-1;open_escape_cooldown_=-1;open_escape_attempts_=0;
  open_escape_handoff_=RecoveryHandoff();
  ROS_INFO("[LITERATURE_CONFIG] reperception=%d open_escape=%d",
      literatureMethod("reperception"),literatureMethod("open_escape"));
  if(literatureMethod("open_escape"))
    ROS_INFO("[OPEN_ESCAPE_CONFIG] version=4 abort_blocked=1 preserve_legacy_prefix=1 terminal_only=%d min_forward_displacement=0.05",terminalRecoveryOnly());
  hypothesis_progress_.reset();
  ROS_INFO("[HYPOTHESIS_PROGRESS_CONFIG] enabled=%d window=16 progress_m=0.25 cooldown=80",
      hypothesisProgressExperiment());
  ROS_INFO("[EXTREME_SAFETY_CONFIG] enabled=%d version=local_cost_v1", extremeSafetyExperiment());
  ROS_INFO("[ALIGNED_BASE_CONFIG] enabled=%d", alignedBaseExperiment());
  ROS_INFO("[RELAXED_IMAGINATION_CONFIG] enabled=%d contract=relaxed_blend_v1",relaxedImagination());
  ROS_INFO("[IMAGE_GLM_CONFIG] enabled=%d contract=image_glm_single_view_v1",imageGLMExperiment());
  ROS_INFO("[FRONTIER_ONCE_CONFIG] enabled=%d contract=frontier_once_spatial_q25_half_v1_20260923",
      frontierOnceExperiment());
  nh.param("exploration/stagnation_recovery", stagnation_recovery_enabled_, false);
  stagnation_recovery_ = StagnationRecovery();
  recovery_exclusion_until_ = -1;
  ROS_INFO("[RECOVERY_CONFIG] enabled=%d window=20 progress_m=0.5 repeat_ratio=0.7 "
           "observe=10 cooldown=30 max_attempts=2 exclusion_steps=30",
      stagnation_recovery_enabled_);
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new ExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));
  fp_->vis_scale_ = expl_manager_->sdf_map_->getResolution() * FSMConstants::VIS_SCALE_FACTOR;
  nh.param("exploration/avic_top_k", fp_->avic_top_k_, 3);
  nh.param("exploration/avic_route_commit_steps", avic_route_commit_steps_, 6);
  avic_route_commit_steps_ = std::max(0, std::min(6, avic_route_commit_steps_));
  avic_route_commitment_.reset();
  nh.param("exploration/target_confirmation", target_confirmation_enabled_, true);
  target_confirmation_.reset();
  ROS_INFO("[TARGET_CONFIRM_CONFIG] enabled=%d version=search_point_v1_20260919 "
           "scan_steps=4 max_attempts=3 exclusion_steps=30 exclusion_radius_m=0.75 synchronous_only=1",
      target_confirmation_enabled_);
  ROS_INFO("[AVIC_ROUTE_CONFIG] commit_steps=%d match_tolerance_m=0.5 synchronous_only=1",
      avic_route_commit_steps_);
  nh.param("exploration/avic_score_match_tolerance", fp_->avic_score_match_tolerance_, 0.10);
  nh.param("exploration/avic_observation_evidence_threshold",
      fp_->avic_observation_evidence_threshold_, 0.55);

  frontier_query_id_ = 0;
  frontier_query_active_ = false;
  frontier_query_eligible_this_cycle_ = false;
  have_last_frontier_target_ = false;
  last_exploration_result_ = 0;
  frontier_snapshot_ = std_msgs::Float32MultiArray();

  state_ = ROS_STATE::INIT;

  /* ROS Timer */
  exec_timer_ = nh.createTimer(
      ros::Duration(FSMConstants::EXEC_TIMER_DURATION), &ExplorationFSM::FSMCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(FSMConstants::FRONTIER_TIMER_DURATION),
      &ExplorationFSM::frontierCallback, this);

  /* ROS Subscriber */
  trigger_sub_ = nh.subscribe("/move_base_simple/goal", 10, &ExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 10, &ExplorationFSM::odometryCallback, this);
  habitat_state_sub_ =
      nh.subscribe("/habitat/state", 10, &ExplorationFSM::habitatStateCallback, this);
  executed_action_sub_ =
      nh.subscribe("/habitat/executed_action", 10, &ExplorationFSM::executedActionCallback, this);
  avic_frontier_scores_sub_ = nh.subscribe("/apexnav/avic_frontier_scores", 10,
      &ExplorationFSM::avicFrontierScoresCallback, this);
  confidence_threshold_sub_ = node_.subscribe(
      "/detector/confidence_threshold", 10, &ExplorationFSM::confidenceThresholdCallback, this);

  /* ROS Publisher */
  ros_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/state", 10);
  expl_state_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_state", 10);
  action_pub_ = nh.advertise<std_msgs::Int32>("/habitat/plan_action", 10);
  expl_result_pub_ = nh.advertise<std_msgs::Int32>("/ros/expl_result", 10);
  frontier_candidates_pub_ =
      nh.advertise<std_msgs::Float32MultiArray>("/apexnav/topk_frontiers", 10);
  avic_frontier_ack_pub_ =
      nh.advertise<std_msgs::Float32MultiArray>("/apexnav/avic_frontier_ack", 10);
  robot_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/robot", 10);
  if (!observation_service_)
    observation_service_ = nh.advertiseService(
        "/apexnav/plan_observation", &ExplorationFSM::planObservation, this);
  if(!review_service_)
    review_service_=nh.advertiseService("/apexnav/review_target",&ExplorationFSM::reviewTarget,this);
  review_command_step_=-1;review_detection_instances_.clear();
}

bool ExplorationFSM::reviewTarget(plan_env::ReviewTarget::Request& req,plan_env::ReviewTarget::Response& res) {
  const char* flag=std::getenv("APEXNAV_TARGET_REVIEW");
  if(!flag || std::string(flag)!="1" || !synchronous_observations_ || observation_failed_ ||
     req.episode!=observation_episode_ || int(req.step)!=observation_step_) {
    res.reason="disabled or stale review transaction";return true;
  }
  res.instance_id=expl_manager_->review_object_id_;
  res.action=fd_->newest_action_;res.final_state=fd_->final_result_;
  res.exploration_result=last_exploration_result_;
  res.detection_instances=review_detection_instances_;
  Vector3d lower,upper;
  if(expl_manager_->object_map2d_->reviewBounds(res.instance_id,lower,upper))
    res.bounds={lower.x(),lower.y(),lower.z(),upper.x(),upper.y(),upper.z()};
  if(req.operation==0){res.accepted=true;res.reason="read-only real instance snapshot";return true;}
  if(review_command_step_==observation_step_) {
    if(review_command_op_==req.operation && review_command_id_==req.instance_id)res=review_command_response_;
    else res.reason="different command in same observation";
    return true;
  }
  if(req.instance_id<0 || req.instance_id!=res.instance_id || last_exploration_result_!=EXPL_RESULT::SEARCH_BEST_OBJECT ||
     (res.final_state!=FINAL_RESULT::SEARCH_OBJECT && res.final_state!=FINAL_RESULT::REACH_OBJECT) ||
     req.operation<1 || req.operation>3) {res.reason="review instance/state mismatch";return true;}
  const Vector2d pos(fd_->odom_pos_.x(),fd_->odom_pos_.y());
  if(expl_manager_->sdf_map_->getOccupancy(pos)!=SDFMap2D::FREE) {
    res.reason="in-place observation requires known free robot position";return true;
  }
  releaseAvicRoute("target_review");
  if(req.operation==1) {
    expl_manager_->object_map2d_->reviewReject(req.instance_id);
    fd_->replan_flag_=true;fd_->final_result_=FINAL_RESULT::EXPLORE;
    expl_manager_->ed_->next_best_path_.clear();
  } else fd_->final_result_=FINAL_RESULT::SEARCH_OBJECT;
  fd_->newest_action_=req.operation==3?ACTION::TURN_RIGHT:ACTION::TURN_LEFT;
  res.action=fd_->newest_action_;res.final_state=fd_->final_result_;res.accepted=true;
  res.reason=req.operation==1?"instance excluded for 60 steps; resume exploration":"in-place observation";
  observation_response_.action=res.action;observation_response_.final_state=res.final_state;
  review_command_step_=observation_step_;review_command_op_=req.operation;review_command_id_=req.instance_id;
  review_command_response_=res;
  ROS_INFO("[TARGET_REVIEW_COMMAND] step=%d instance=%d operation=%d action=%d",observation_step_,req.instance_id,req.operation,res.action);
  return true;
}

bool ExplorationFSM::planObservation(plan_env::PlanObservation::Request& req,
    plan_env::PlanObservation::Response& res)
{
  res.episode = req.episode;
  res.step = req.step;
  const auto valid_pose = [](const nav_msgs::Odometry& odom) {
    const auto& p = odom.pose.pose.position;
    const auto& q = odom.pose.pose.orientation;
    const double norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) &&
        std::isfinite(norm) && std::fabs(norm - 1.0) < 1e-3;
  };
  std::vector<uint8_t> request_bytes(ros::serialization::serializationLength(req));
  ros::serialization::OStream request_stream(request_bytes.data(), request_bytes.size());
  ros::serialization::serialize(request_stream, req);
  if (req.episode.empty() || retired_observation_episodes_.count(req.episode) ||
      (req.episode != observation_episode_ && req.step != 0) ||
      req.executed_action < ACTION::STOP || req.executed_action > ACTION::TURN_UP ||
      req.step > 1000000 || !valid_pose(req.camera_pose) || !valid_pose(req.robot_pose) ||
      !std::isfinite(req.itm_score) || !std::isfinite(req.confidence_threshold) ||
      req.depth.encoding != "32FC1" || req.depth.width != 640 || req.depth.height != 480 ||
      req.depth.step != req.depth.width * sizeof(float) ||
      req.depth.data.size() != static_cast<size_t>(req.depth.step) * req.depth.height ||
      req.detections.point_clouds.size() != req.detections.confidence_scores.size() ||
      req.detections.point_clouds.size() != req.detections.label_indices.size()) {
    res.reason = "invalid or retired observation";
    return true;
  }
  if (req.episode == observation_episode_) {
    if (observation_failed_) {
      res.reason = "previous transaction failed; restart episode";
      return true;
    }
    if (static_cast<int>(req.step) == observation_step_) {
      if (request_bytes != observation_request_bytes_) {
        res.reason = "same episode/step with different payload";
        return true;
      }
      res = observation_response_;  // no map fusion and no replanning on retry
      res.reason = "duplicate observation; cached decision";
      return true;
    }
    if (static_cast<int>(req.step) != observation_step_ + 1) {
      res.reason = "out-of-order observation";
      return true;
    }
  }
  else {
    if (!observation_episode_.empty()) {
      retired_observation_episodes_.insert(observation_episode_);
      ++benchmark_episode_index_;
      benchmark_decision_index_ = 0;
      init(nh_);
    }
    observation_episode_ = req.episode;
    observation_step_ = -1;
    observation_failed_ = false;
  }
  synchronous_observations_ = true;
  exec_timer_.stop();
  frontier_timer_.stop();
  odom_sub_.shutdown();
  confidence_threshold_sub_.shutdown();
  executed_action_sub_.shutdown();
  habitat_state_sub_.shutdown();
  trigger_sub_.shutdown();
  try {
    // Keep failures terminal: a partly committed observation cannot be retried safely.
    observation_failed_ = true;
    expl_manager_->confirmCycleDecision(observation_step_,req.executed_action,observation_episode_);
    expl_manager_->path_finder_->useDeterministicBudget();
    odometryCallback(boost::make_shared<nav_msgs::Odometry>(req.robot_pose));
    expl_manager_->object_map2d_->setConfidenceThreshold(req.confidence_threshold);
    fd_->have_confidence_ = true;
    fd_->newest_action_ = req.executed_action;
    if(literatureMethod("reperception")) {
      std::vector<Vector2d> points;
      for(size_t i=0;i<req.detections.point_clouds.size();++i) {
        if(i>=req.detections.label_indices.size() || i>=req.detections.confidence_scores.size() ||
            req.detections.label_indices[i]!=0 || req.detections.confidence_scores[i]<req.confidence_threshold)continue;
        const auto& cloud=req.detections.point_clouds[i];
        if(!cloud.width || !cloud.height)continue;
        sensor_msgs::PointCloud2ConstIterator<float> x(cloud,"x"),y(cloud,"y");
        size_t index=0;
        for(;x!=x.end();++x,++y,++index)
          if(index%16==0 && std::isfinite(*x) && std::isfinite(*y))points.emplace_back(*x,*y);
      }
      fresh_target_frames_.push_back({int(req.step),std::move(points)});
      if(fresh_target_frames_.size()>6)fresh_target_frames_.erase(fresh_target_frames_.begin());
    }
    expl_manager_->sdf_map_->observationInterface()->commitObservation(
        req.depth, req.camera_pose, req.detections, req.itm_score, req.step == 0);
    expl_manager_->object_map2d_->reviewStep(req.step);
    const char* review_flag=std::getenv("APEXNAV_TARGET_REVIEW");
    if(review_flag && std::string(review_flag)=="1") {
      review_detection_instances_.clear();
      for(const auto& cloud:req.detections.point_clouds) {
        pcl::PointCloud<pcl::PointXYZ> points;
        if(cloud.width && cloud.height) {
          sensor_msgs::PointCloud2ConstIterator<float> x(cloud,"x"),y(cloud,"y");
          int n=0;for(;x!=x.end();++x,++y,++n)if(n%16==0)points.emplace_back(*x,*y,0.0f);
        }
        review_detection_instances_.push_back(expl_manager_->object_map2d_->reviewInstance(points));
      }
    }
    frontier_snapshot_ = std_msgs::Float32MultiArray();
    frontier_query_active_ = false;
    if (fd_->init_action_count_ < 26) {
      const int index = fd_->init_action_count_++;
      fd_->newest_action_ = index == 0 ? ACTION::TURN_DOWN :
          (index == 13 ? ACTION::TURN_UP : ACTION::TURN_LEFT);
      updateFrontierAndObject();
    }
    else {
      fd_->start_pt_ = fd_->odom_pos_;
      fd_->start_yaw_(0) = fd_->odom_yaw_;
      const auto planning_start = std::chrono::steady_clock::now();
      fd_->final_result_ = callActionPlanner();
      updateFrontierQueryState();
      publishTopKFrontiers(Vector2d(fd_->start_pt_(0), fd_->start_pt_(1)), fd_->start_yaw_(0));
      ROS_INFO("[BENCH] component=apexnav_plan episode_seq=%llu decision_seq=%llu "
               "latency_ms=%.3f final_result=%d frontier_count=%zu",
          static_cast<unsigned long long>(benchmark_episode_index_),
          static_cast<unsigned long long>(benchmark_decision_index_++),
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - planning_start).count(),
          fd_->final_result_, expl_manager_->ed_->frontiers_.size());
      if (fd_->final_result_ != FINAL_RESULT::EXPLORE &&
          fd_->final_result_ != FINAL_RESULT::SEARCH_OBJECT)
        fd_->newest_action_ = ACTION::STOP;
    }
    state_ = ROS_STATE::WAIT_ACTION_FINISH;
    res.accepted = true;
    res.reason = "observation committed and planned";
    res.action = fd_->newest_action_;
    res.final_state = fd_->final_result_;
    res.exploration_result = last_exploration_result_;
    res.frontiers = frontier_snapshot_;
    observation_step_ = static_cast<int>(req.step);
    observation_response_ = res;
    observation_request_bytes_ = std::move(request_bytes);
    observation_failed_ = false;
    ROS_INFO("[OBSERVATION_COMMIT] episode=%s step=%u action=%d query=%u",
        req.episode.c_str(), req.step, res.action, frontier_query_id_);
  }
  catch (const std::exception& exc) {
    res.accepted = false;
    res.reason = exc.what();
  }
  return true;
}

// FSM between ROS and Habitat for action planning and execution
void ExplorationFSM::FSMCallback(const ros::TimerEvent& e)
{
  if (synchronous_observations_) return;
  exec_timer_.stop();
  std_msgs::Int32 ros_state_msg;
  ros_state_msg.data = state_;
  ros_state_pub_.publish(ros_state_msg);
  switch (state_) {
    case ROS_STATE::INIT: {
      // Wait for odometry and target confidence threshold
      if (!fd_->have_odom_ || !fd_->have_confidence_) {
        ROS_WARN_THROTTLE(1.0, "No odom || No target confidence threshold.");
        exec_timer_.start();
        return;
      }
      // Go to WAIT_TRIGGER when prerequisites are ready
      clearVisMarker();
      transitState(ROS_STATE::WAIT_TRIGGER, "FSM");
      break;
    }

    case ROS_STATE::WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "Wait for trigger.");
      break;
    }

    case ROS_STATE::FINISH: {
      if (!fd_->have_finished_) {
        fd_->have_finished_ = true;
        clearVisMarker();
        std_msgs::Int32 action_msg;
        action_msg.data = ACTION::STOP;
        action_pub_.publish(action_msg);
      }
      ROS_WARN_THROTTLE(1.0, "Finish One Episode!!!");
      break;
    }

    case ROS_STATE::PLAN_ACTION: {
      // Initial action sequence: perform orientation calibration turns
      if (fd_->init_action_count_ < 1 + 12 + 1 + 12) {
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACTION::TURN_DOWN;
        else if (fd_->init_action_count_ < 1 + 12)
          fd_->newest_action_ = ACTION::TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + 12 + 1)
          fd_->newest_action_ = ACTION::TURN_UP;
        else
          fd_->newest_action_ = ACTION::TURN_LEFT;
        ROS_WARN("Init Mode Process -----> (%d/26)", fd_->init_action_count_);
        fd_->init_action_count_++;
        transitState(ROS_STATE::PUB_ACTION, "FSM");
        updateFrontierAndObject();
      }
      else {
        // Main planning phase: determine robot pose and call action planner
        fd_->start_pt_ = fd_->odom_pos_;
        fd_->start_yaw_(0) = fd_->odom_yaw_;

        // Use a monotonic wall clock. ROS time can be simulated and therefore
        // is not suitable for comparing planner latency across methods.
        const auto t1 = std::chrono::steady_clock::now();
        fd_->final_result_ = callActionPlanner();
        updateFrontierQueryState();
        publishTopKFrontiers(
            Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1)), fd_->start_yaw_(0));
        const double call_action_planner_time_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t1)
                .count();
        ROS_INFO_THROTTLE(
            10.0, "[Calculating Time] Planning process time = %.3f s",
            call_action_planner_time_ms / 1000.0);
        ROS_INFO("[BENCH] component=apexnav_plan episode_seq=%llu decision_seq=%llu "
                 "latency_ms=%.3f final_result=%d frontier_count=%zu",
            static_cast<unsigned long long>(benchmark_episode_index_),
            static_cast<unsigned long long>(benchmark_decision_index_),
            call_action_planner_time_ms, fd_->final_result_,
            expl_manager_->ed_->frontiers_.size());
        ++benchmark_decision_index_;

        std_msgs::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        expl_state_pub_.publish(expl_state_msg);
        if (fd_->final_result_ == FINAL_RESULT::EXPLORE ||
            fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT)
          transitState(ROS_STATE::PUB_ACTION, "FSM");
        else
          transitState(ROS_STATE::FINISH, "FSM");
      }
      visualize();
      break;
    }

    case ROS_STATE::PUB_ACTION: {
      std_msgs::Int32 action_msg;
      action_msg.data = fd_->newest_action_;
      action_pub_.publish(action_msg);
      transitState(ROS_STATE::WAIT_ACTION_FINISH, "FSM");
      break;
    }

    case ROS_STATE::WAIT_ACTION_FINISH: {
      exec_timer_.start();
      break;
    }
  }
  exec_timer_.start();
}

/**
 * @brief Plan the next action based on current state and environment
 * @return Final result indicating the planned action type and exploration state
 *
 * This is the core planning function that decides what action the robot should take next.
 * It handles obstacle avoidance, frontier exploration, object search, and stuck recovery.
 */
int ExplorationFSM::callActionPlanner()
{
  const double stucking_distance = FSMConstants::STUCKING_DISTANCE;
  const double reach_distance = FSMConstants::REACH_DISTANCE;
  const double soft_reach_distance = FSMConstants::SOFT_REACH_DISTANCE;
  frontier_query_eligible_this_cycle_ = false;

  // Record the action Habitat actually executed, not the previous proposed action.
  const bool had_avic_route = avic_route_commitment_.active;
  if (had_avic_route) {
    avic_route_commitment_.observe(observation_step_ + 1, fd_->newest_action_);
    if (avic_route_commitment_.expired(observation_step_ + 1))
      releaseAvicRoute("budget_complete");
    else if (fd_->dormant_frontier_flag_)
      releaseAvicRoute("frontier_dormant");
  }

  bool frontier_change_flag = updateFrontierAndObject();

  int expl_res, final_res;
  Eigen::Vector2d current_pos = Eigen::Vector2d(fd_->start_pt_(0), fd_->start_pt_(1));
  Eigen::Vector2d last_pos = Eigen::Vector2d(fd_->last_start_pos_(0), fd_->last_start_pos_(1));
  double current_yaw = fd_->start_yaw_(0);
  fd_->last_start_pos_ = fd_->start_pt_;

  if (hypothesisProgressExperiment() && synchronous_observations_) {
    expl_manager_->setConfirmationStep(observation_step_ + 1);
    const bool uncertain = fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (last_exploration_result_ == EXPL_RESULT::SEARCH_EXTREME ||
         last_exploration_result_ == EXPL_RESULT::SEARCH_SUSPICIOUS_OBJECT ||
         last_exploration_result_ == EXPL_RESULT::SEARCH_OVER_DEPTH_OBJECT);
    if (!uncertain || target_confirmation_.active) hypothesis_progress_.reset();
    else if (hypothesis_progress_.observe(observation_step_ + 1, current_pos,
                                         expl_manager_->ed_->next_pos_)) {
      const Vector2d rejected = expl_manager_->ed_->next_pos_;
      expl_manager_->rejectStalledHypothesis(rejected);
      releaseAvicRoute("stalled_hypothesis");
      fd_->replan_flag_ = true;
      ROS_INFO("[HYPOTHESIS_PROGRESS] stage=exclude episode=%s step=%d goal=(%.3f,%.3f) distance=%.3f",
          observation_episode_.c_str(), observation_step_ + 1, rejected.x(), rejected.y(),
          (current_pos-rejected).norm());
      // Existing collision recovery retains ownership; do not blindly move or reset STOP counters.
    }
  }

  const bool confirmation_guard = target_confirmation_enabled_ && synchronous_observations_;
  if(literatureMethod("reperception") && synchronous_observations_) {
    expl_manager_->setConfirmationStep(observation_step_+1);
    const bool arrived=fd_->final_result_==FINAL_RESULT::SEARCH_OBJECT &&
      ((current_pos-expl_manager_->ed_->next_pos_).norm()<reach_distance ||
       (fd_->newest_action_==ACTION::MOVE_FORWARD && (current_pos-last_pos).norm()<stucking_distance &&
        (current_pos-expl_manager_->ed_->next_pos_).norm()<soft_reach_distance));
    if(fresh_confirmation_.active || (arrived && last_exploration_result_==EXPL_RESULT::SEARCH_BEST_OBJECT)) {
      const int step=observation_step_+1;
      const Vector2d goal=fresh_confirmation_.active?fresh_confirmation_.goal:expl_manager_->ed_->next_pos_;
      int frames=0;
      for(const auto& frame:fresh_target_frames_) {
        bool nearby=false;
        for(const auto& point:frame.second)if((point-goal).norm()<0.85){nearby=true;break;}
        if(nearby)++frames;
      }
      ROS_INFO("[REPERCEPTION] step=%d matched_frames=%d goal=(%.3f,%.3f) active=%d",
          step,frames,goal.x(),goal.y(),fresh_confirmation_.active);
      if(frames>=2)fresh_confirmation_.cancel();
      else {
        if(!fresh_confirmation_.active)fresh_confirmation_.begin(step,goal);
        const int action=fresh_confirmation_.active?fresh_confirmation_.action(step):-1;
        if(action>=0) {
          releaseAvicRoute("reperception");fd_->newest_action_=action;
          return FINAL_RESULT::SEARCH_OBJECT;
        }
        expl_manager_->rejectStalledHypothesis(goal);
        fresh_confirmation_.cancel();fd_->replan_flag_=true;
        fd_->final_result_=FINAL_RESULT::EXPLORE;
        ROS_INFO("[REPERCEPTION] step=%d stage=reject cooldown=80",step);
      }
    }
  }
  if (confirmation_guard) {
    expl_manager_->setConfirmationStep(observation_step_ + 1);
    const bool searching = fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT;
    const bool arrived = searching &&
        ((current_pos - expl_manager_->ed_->next_pos_).norm() < reach_distance ||
         (fd_->newest_action_ == ACTION::MOVE_FORWARD &&
          (current_pos - last_pos).norm() < stucking_distance &&
          (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance));
    // Over-depth/suspicious/extreme search points are hypotheses, not confirmed
    // targets. The legacy collision escape still owns an already-active escape.
    if (!fd_->escape_stucking_flag_ && (target_confirmation_.active ||
        (arrived && last_exploration_result_ != EXPL_RESULT::SEARCH_BEST_OBJECT))) {
      if (confirmSearchPoint(current_pos)) return FINAL_RESULT::SEARCH_OBJECT;
    }
    else if (fd_->escape_stucking_flag_) target_confirmation_.cancel();
  }

  const bool may_report_reached = !confirmation_guard ||
      last_exploration_result_ == EXPL_RESULT::SEARCH_BEST_OBJECT;

  // Reach the object - check if close enough to target object
  if (may_report_reached && fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
      (current_pos - expl_manager_->ed_->next_pos_).norm() < reach_distance) {
    ROS_ERROR("Reach the object successfully!!!");
    releaseAvicRoute("object_reached");
    final_res = FINAL_RESULT::REACH_OBJECT;
    return final_res;
  }

  // A bounded, map-planned retreat before the legacy fixed turn/forward sequence.
  if(literatureMethod("open_escape") && synchronous_observations_) {
    const int step=observation_step_+1;
    const bool blocked=fd_->newest_action_==ACTION::MOVE_FORWARD && (current_pos-last_pos).norm()<stucking_distance;
    if(open_escape_deadline_>=0)
      open_escape_handoff_.observe(fd_->newest_action_,(current_pos-last_pos).norm());
    // Observe the action actually executed by Habitat. A geometrically valid
    // A* route is not evidence that its next forward action succeeded.
    // Leave newest_action untouched: legacy collision handling below must see
    // that failed MOVE_FORWARD, rather than a newly proposed turn/forward.
    const bool abort_blocked=open_escape_deadline_>=0 && blocked;
    if(abort_blocked) {
      ROS_INFO("[OPEN_ESCAPE] stage=abort_blocked step=%d displacement=%.6f distance=%.3f handoff=legacy_collision",
          step,(current_pos-last_pos).norm(),(current_pos-open_escape_goal_).norm());
      open_escape_deadline_=-1;
      fd_->replan_flag_=true;
      const int resume=open_escape_handoff_.resumeIndex();
      if(resume>=0) {
        fd_->escape_stucking_flag_=true;
        fd_->escape_stucking_count_=resume;
        fd_->escape_stucking_pos_=open_escape_legacy_pos_;
        fd_->escape_stucking_yaw_=open_escape_legacy_yaw_;
      }
      ROS_INFO("[OPEN_ESCAPE] stage=handoff step=%d resume_legacy_index=%d",step,resume);
    }
    if(!terminalRecoveryOnly() && !abort_blocked && open_escape_deadline_<0 && blocked && step>=open_escape_cooldown_ && open_escape_attempts_<2 &&
        last_exploration_result_!=EXPL_RESULT::SEARCH_BEST_OBJECT) {
      vector<Vector2d> path;
      ++open_escape_attempts_;open_escape_cooldown_=step+40;
      if(expl_manager_->openSpaceRecovery(current_pos,open_escape_goal_,path)) {
        open_escape_handoff_.start(fd_->escape_stucking_flag_?fd_->escape_stucking_count_:0);
        open_escape_legacy_pos_=fd_->escape_stucking_flag_?fd_->escape_stucking_pos_:current_pos;
        open_escape_legacy_yaw_=fd_->escape_stucking_flag_?fd_->escape_stucking_yaw_:current_yaw;
        open_escape_deadline_=step+20;fd_->escape_stucking_flag_=false;
        releaseAvicRoute("open_space_retreat");
        ROS_INFO("[OPEN_ESCAPE] stage=begin step=%d goal=(%.3f,%.3f)",step,open_escape_goal_.x(),open_escape_goal_.y());
      } else ROS_INFO("[OPEN_ESCAPE] stage=no_path step=%d",step);
    }
    if(open_escape_deadline_>=0) {
      vector<Vector2d> path;
      if(step<open_escape_deadline_ && (current_pos-open_escape_goal_).norm()>0.25 &&
          expl_manager_->recoveryToPoint(current_pos,open_escape_goal_,path)) {
        fd_->newest_action_=planNextBestAction(current_pos,current_yaw,path);
        ROS_INFO("[OPEN_ESCAPE] stage=execute step=%d action=%d distance=%.3f",step,fd_->newest_action_,(current_pos-open_escape_goal_).norm());
        return FINAL_RESULT::EXPLORE;
      }
      ROS_INFO("[OPEN_ESCAPE] stage=end step=%d distance=%.3f",step,(current_pos-open_escape_goal_).norm());
      open_escape_deadline_=-1;fd_->replan_flag_=true;
    }
  }

  /*******  Escape-from-stuck logic START *******/
  // Detect if robot is stuck and initiate escape sequence
  int last_action = fd_->newest_action_;
  if (!fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() < stucking_distance &&
      last_action == ACTION::MOVE_FORWARD) {
    releaseAvicRoute("forward_blocked");
    if (may_report_reached && fd_->final_result_ == FINAL_RESULT::SEARCH_OBJECT &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() < soft_reach_distance) {
      ROS_ERROR("Reach the object successfully!!!");
      final_res = FINAL_RESULT::REACH_OBJECT;
      return final_res;
    }

    bool past_stucking_flag = false;
    for (auto stucking_point : fd_->stucking_points_) {
      Vector2d stucking_pos = Vector2d(stucking_point(0), stucking_point(1));
      double stucking_yaw = stucking_point(2);
      if ((stucking_pos - current_pos).norm() < stucking_distance &&
          fabs(stucking_yaw - current_yaw) < FSMConstants::ACTION_ANGLE) {
        past_stucking_flag = true;
        ROS_ERROR("Still stuck at the same place");
        break;
      }
    }
    if (!past_stucking_flag) {
      fd_->escape_stucking_flag_ = true;
      fd_->escape_stucking_count_ = 0;
      fd_->escape_stucking_pos_ = current_pos;
      fd_->escape_stucking_yaw_ = current_yaw;
    }
  }

  if (fd_->escape_stucking_flag_ && (current_pos - last_pos).norm() >= stucking_distance) {
    ROS_ERROR("Escaped from stuck state.");
    fd_->escape_stucking_flag_ = false;
  }

  if (fd_->escape_stucking_flag_) {
    releaseAvicRoute("stuck_recovery");
    if (stagnation_recovery_enabled_) stagnation_recovery_.interrupt();
    ROS_ERROR("Escaping stuck...");
    if (fd_->escape_stucking_count_ == 0)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 1)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 2)
      fd_->newest_action_ = ACTION::TURN_RIGHT;
    else if (fd_->escape_stucking_count_ == 3)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 4)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 5)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 6)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 7)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else if (fd_->escape_stucking_count_ == 8)
      fd_->newest_action_ = ACTION::TURN_LEFT;
    else if (fd_->escape_stucking_count_ == 9)
      fd_->newest_action_ = ACTION::MOVE_FORWARD;
    else {
      // Failed to escape - mark area as occupied and add to stuck points
      ROS_ERROR("Cannot escape stuck state.");
      fd_->escape_stucking_flag_ = false;
      expl_manager_->sdf_map_->setForceOccGrid(current_pos);
      double forward_distance = FSMConstants::FORWARD_DISTANCE;
      Eigen::Vector2d forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      forward_distance = FSMConstants::FORWARD_DISTANCE * 2.0;
      forward_pos = fd_->escape_stucking_pos_;
      forward_pos(0) += forward_distance * cos(fd_->escape_stucking_yaw_);
      forward_pos(1) += forward_distance * sin(fd_->escape_stucking_yaw_);
      expl_manager_->sdf_map_->setForceOccGrid(forward_pos);
      fd_->dormant_frontier_flag_ = true;
      Vector3d stucking_point(
          fd_->escape_stucking_pos_(0), fd_->escape_stucking_pos_(1), fd_->escape_stucking_yaw_);
      fd_->stucking_points_.push_back(stucking_point);
    }

    if (fd_->escape_stucking_flag_) {
      fd_->escape_stucking_count_++;
      return fd_->final_result_;
    }
  }

  /*******  Decide whether to replan path (stability heuristic) START *******/
  // Use path stability to reduce oscillation between different frontier targets
  vector<Vector2d> last_next_best_path = expl_manager_->ed_->next_best_path_;
  Vector2d last_next_pos = expl_manager_->ed_->next_pos_;
  if (fd_->dormant_frontier_flag_) {
    fd_->replan_flag_ = true;
    fd_->dormant_frontier_flag_ = false;
  }
  else if (fd_->final_result_ == FINAL_RESULT::EXPLORE && !frontier_change_flag)
    fd_->replan_flag_ = false;

  expl_res = expl_manager_->planNextBestPoint(fd_->start_pt_, fd_->start_yaw_(0));
  last_exploration_result_ = expl_res;

  if (expl_res != EXPL_RESULT::EXPLORATION) {
    fd_->replan_flag_ = true;
  }
  if (expl_res == EXPL_RESULT::EXPLORATION && !fd_->replan_flag_ && !had_avic_route) {
    expl_manager_->ed_->next_best_path_ = last_next_best_path;
    expl_manager_->ed_->next_pos_ = last_next_pos;
    fd_->replan_flag_ = true;
  }
  /*******  Decide whether to replan path (stability heuristic) END *******/

  // Object pursuit wins. Otherwise recheck the leased frontier in the current
  // map; never restore yesterday's path or carry its imagination score forward.
  applyAvicRoute(current_pos, expl_res);
  if(synchronous_observations_ && expl_res==EXPL_RESULT::EXPLORATION &&
      !avic_route_commitment_.active && !fd_->escape_stucking_flag_)
    expl_manager_->suppressCycleGoal(current_pos,observation_step_+1,observation_episode_);

  // Publish exploration result to monitor
  std_msgs::Int32 expl_result_msg;
  expl_result_msg.data = expl_res;
  expl_result_pub_.publish(expl_result_msg);

  // Determine current high-level state based on exploration results
  if (expl_res == EXPL_RESULT::EXPLORATION)
    final_res = FINAL_RESULT::EXPLORE;
  else if (expl_res == EXPL_RESULT::NO_COVERABLE_FRONTIER ||
           expl_res == EXPL_RESULT::NO_PASSABLE_FRONTIER)
    final_res = FINAL_RESULT::NO_FRONTIER;
  else
    final_res = FINAL_RESULT::SEARCH_OBJECT;

  if (final_res == FINAL_RESULT::NO_FRONTIER || expl_manager_->ed_->next_best_path_.empty()) {
    if (stagnation_recovery_enabled_) stagnation_recovery_.interrupt();
    ROS_WARN("No (passable) frontier");
    return final_res;
  }

  if (stagnation_recovery_enabled_ && synchronous_observations_ &&
      !avic_route_commitment_.active)
    applyStagnationRecovery(current_pos, final_res == FINAL_RESULT::EXPLORE &&
        (current_pos - expl_manager_->ed_->next_pos_).norm() >= 0.5);

  Eigen::Vector2d end_pos = expl_manager_->ed_->next_pos_;
  Eigen::Vector2d last_end_pos = fd_->last_next_pos_;
  fd_->last_next_pos_ = end_pos;
  double min_dist = (current_pos - end_pos).norm();
  ROS_WARN("To the next point (%.2fm %.2fm), distance = %.2f m", end_pos(0), end_pos(1), min_dist);

  // Handling being stuck while exploring toward a specific frontier
  if (final_res == FINAL_RESULT::EXPLORE) {
    // Force dormant if very close to target but still exploring
    if (min_dist < FSMConstants::FORCE_DORMANT_DISTANCE) {
      releaseAvicRoute("frontier_reached");
      ROS_ERROR("Force set dormant frontier.");
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }

    // Count consecutive times with same target position while stuck
    if ((end_pos - last_end_pos).norm() < 1e-3 &&
        (current_pos - last_pos).norm() < stucking_distance) {
      fd_->stucking_next_pos_count_++;
      ROS_ERROR_COND(fd_->stucking_next_pos_count_ > 8, "stucking_next_pos_count_ = %d",
          fd_->stucking_next_pos_count_);
    }
    else
      fd_->stucking_next_pos_count_ = 0;

    // Mark frontier as dormant if stuck too long with same target
    if (fd_->stucking_next_pos_count_ >= FSMConstants::MAX_STUCKING_NEXT_POS_COUNT) {
      releaseAvicRoute("frontier_stagnation");
      ROS_ERROR("Set dormant frontier.");
      fd_->stucking_action_count_ = 0;
      fd_->stucking_next_pos_count_ = 0;
      expl_manager_->frontier_map2d_->setForceDormantFrontier(end_pos);
      fd_->dormant_frontier_flag_ = true;
    }
  }

  // Track consecutive stuck actions globally
  if ((current_pos - last_pos).norm() < stucking_distance) {
    fd_->stucking_action_count_++;
    ROS_ERROR_COND(fd_->stucking_action_count_ > 15, "Stucking action count = %d",
        fd_->stucking_action_count_);
  }
  else
    fd_->stucking_action_count_ = 0;

  // If stuck for too long globally, terminate episode
  if (fd_->stucking_action_count_ >= FSMConstants::MAX_STUCKING_COUNT) {
    // Isolated v4 trial: leave ordinary collision recovery in control. Only
    // replace the planner's STUCKING termination, never REACH_OBJECT/STOP.
    // No simulator goal distance or success flag is consulted here.
    if(literatureMethod("open_escape") && terminalRecoveryOnly() && synchronous_observations_ &&
        open_escape_deadline_<0 && open_escape_attempts_<2) {
      vector<Vector2d> path;
      const int step=observation_step_+1;
      ++open_escape_attempts_;
      if(expl_manager_->openSpaceRecovery(current_pos,open_escape_goal_,path)) {
        open_escape_handoff_.start(-1);
        open_escape_deadline_=step+20;
        open_escape_cooldown_=step+40;
        fd_->escape_stucking_flag_=false;
        fd_->stucking_action_count_=0;
        fd_->stucking_next_pos_count_=0;
        fd_->replan_flag_=true;
        releaseAvicRoute("terminal_open_space_retreat");
        fd_->newest_action_=planNextBestAction(current_pos,current_yaw,path);
        ROS_INFO("[OPEN_ESCAPE] stage=begin step=%d trigger=planner_stucking goal=(%.3f,%.3f)",
            step,open_escape_goal_.x(),open_escape_goal_.y());
        return FINAL_RESULT::EXPLORE;
      }
      ROS_INFO("[OPEN_ESCAPE] stage=no_path step=%d trigger=planner_stucking",step);
    }
    releaseAvicRoute("episode_stuck");
    ROS_ERROR("Stuck for too long, stopping episode.");
    final_res = FINAL_RESULT::STUCKING;
    return final_res;
  }

  // Plan specific action based on exploration result
  if (expl_res == EXPL_RESULT::SEARCH_EXTREME) {
    const int legacy_action = planNextBestAction(
        current_pos, current_yaw, expl_manager_->ed_->next_best_path_, false);
    fd_->newest_action_ = extremeSafetyExperiment() ? planNextBestAction(
        current_pos, current_yaw, expl_manager_->ed_->next_best_path_, true) : legacy_action;
    ROS_INFO("[EXTREME_SAFETY] enabled=%d step=%d legacy_action=%d action=%d changed=%d distance=%.3f",
        extremeSafetyExperiment(), observation_step_ + 1, legacy_action, fd_->newest_action_,
        legacy_action != fd_->newest_action_, min_dist);
  }
  else
    fd_->newest_action_ =
        planNextBestAction(current_pos, current_yaw, expl_manager_->ed_->next_best_path_);

  // Only a normal exploration-policy decision may open an AVIC query.  Early
  // returns for object pursuit, collision/stuck recovery or termination retain
  // exclusive ApexNav control.
  frontier_query_eligible_this_cycle_ = final_res == FINAL_RESULT::EXPLORE &&
      !avic_route_commitment_.active;
  if(synchronous_observations_ && final_res==FINAL_RESULT::EXPLORE && !fd_->escape_stucking_flag_)
    expl_manager_->stageCycleDecision(current_pos,observation_step_+1,fd_->newest_action_,observation_episode_);
  return final_res;
}

bool ExplorationFSM::confirmSearchPoint(const Vector2d& current_pos)
{
  const int step = observation_step_ + 1;
  // Check the newly fused REAL object map every time, including after the last
  // scan action. A confirmed reachable object may preempt an uncertain point.
  if (expl_manager_->planConfirmedObject(fd_->start_pt_)) {
    ROS_INFO("[TARGET_CONFIRM] stage=confirmed episode=%s step=%d attempts=%d "
             "goal=(%.3f,%.3f)", observation_episode_.c_str(), step,
        target_confirmation_.attempts, expl_manager_->ed_->next_pos_.x(),
        expl_manager_->ed_->next_pos_.y());
    target_confirmation_.cancel();
    last_exploration_result_ = EXPL_RESULT::SEARCH_BEST_OBJECT;
    fd_->replan_flag_ = true;
    return false;  // use the REAL confirmed approach point for normal arrival/planning
  }

  if (!target_confirmation_.active) {
    const Vector2d point = expl_manager_->ed_->next_pos_;
    if (!target_confirmation_.begin(step, point)) {
      expl_manager_->rejectUnconfirmedGoal(point);
      ROS_INFO("[TARGET_CONFIRM] stage=reject episode=%s step=%d reason=attempt_budget "
               "goal=(%.3f,%.3f)", observation_episode_.c_str(), step, point.x(), point.y());
      fd_->replan_flag_ = true;
      return false;
    }
    releaseAvicRoute("target_confirmation");
    if (stagnation_recovery_enabled_) stagnation_recovery_.interrupt();
    ROS_INFO("[TARGET_CONFIRM] stage=start episode=%s step=%d search_kind=%d "
             "attempt=%d goal=(%.3f,%.3f)", observation_episode_.c_str(), step,
        last_exploration_result_, target_confirmation_.attempts, point.x(), point.y());
  }
  const int action = target_confirmation_.action(step);
  if (action >= 0 && (current_pos - target_confirmation_.goal).norm() <= 0.75) {
    fd_->newest_action_ = action;
    ROS_INFO("[TARGET_CONFIRM] stage=observe episode=%s step=%d action=%d",
        observation_episode_.c_str(), step, action);
    return true;
  }

  expl_manager_->rejectUnconfirmedGoal(target_confirmation_.goal);
  ROS_INFO("[TARGET_CONFIRM] stage=reject episode=%s step=%d reason=%s goal=(%.3f,%.3f)",
      observation_episode_.c_str(), step, action < 0 ? "no_confirmed_target" : "moved_away",
      target_confirmation_.goal.x(), target_confirmation_.goal.y());
  target_confirmation_.cancel();
  fd_->replan_flag_ = true;
  return false;  // replan; never claim REACH_OBJECT merely for reaching a search point
}

void ExplorationFSM::releaseAvicRoute(const char* reason)
{
  if (!avic_route_commitment_.active) return;
  const auto& route = avic_route_commitment_;
  ROS_INFO("[AVIC_ROUTE] stage=release episode=%s step=%d reason=%s "
           "start=%d deadline=%d executed=%d matching_prefix=%d expected=%zu goal=(%.3f,%.3f)",
      observation_episode_.c_str(), observation_step_ + 1, reason,
      route.start_step, route.deadline, route.executed, route.matching_prefix,
      route.expected_actions.size(), route.goal.x(), route.goal.y());
  avic_route_commitment_.reset();
  // Also stop the original stability heuristic retaining an expired/invalid lease.
  fd_->replan_flag_ = true;
}

void ExplorationFSM::applyAvicRoute(const Vector2d& current_pos, int exploration_result)
{
  if (!avic_route_commitment_.active) return;
  if (exploration_result != EXPL_RESULT::EXPLORATION) {
    releaseAvicRoute("object_or_nonexploration");
    return;
  }
  if ((current_pos - avic_route_commitment_.goal).norm() < FSMConstants::FORCE_DORMANT_DISTANCE) {
    releaseAvicRoute("frontier_reached");
    return;
  }
  Vector2d selected;
  vector<Vector2d> path;
  // Modest frontier-centroid drift is allowed, relative to the original goal.
  if (!expl_manager_->selectReachableFrontier(
          current_pos, avic_route_commitment_.goal, 0.5, selected, path)) {
    releaseAvicRoute("frontier_missing_or_unreachable");
    return;
  }
  auto& data = *expl_manager_->ed_;
  const Vector2d baseline_goal = data.next_pos_;
  data.next_pos_ = selected;
  data.next_best_path_ = path;
  data.tsp_tour_.clear();  // the ordinary-policy tour no longer describes this decision
  ROS_INFO("[AVIC_ROUTE] stage=hold episode=%s step=%d deadline=%d "
           "baseline=(%.3f,%.3f) selected=(%.3f,%.3f) executed=%d matching_prefix=%d",
      observation_episode_.c_str(), observation_step_ + 1, avic_route_commitment_.deadline,
      baseline_goal.x(), baseline_goal.y(), selected.x(), selected.y(),
      avic_route_commitment_.executed, avic_route_commitment_.matching_prefix);
}

void ExplorationFSM::applyStagnationRecovery(const Vector2d& pos, bool eligible)
{
  const int step = observation_step_ + 1;  // current transaction; retries never enter here
  auto& data = *expl_manager_->ed_;
  if (!eligible) {
    stagnation_recovery_.observe(step, false, pos, data.next_pos_, data.next_best_path_);
    recovery_exclusion_until_ = -1;  // object pursuit always has priority
    return;
  }
  Vector2d selected;
  vector<Vector2d> path;
  if (step < recovery_exclusion_until_ &&
      (data.next_pos_ - recovery_excluded_goal_).norm() <= 0.5) {
    const bool found = expl_manager_->recoveryPath(pos, recovery_excluded_goal_, true, selected, path);
    if (found) { data.next_pos_ = selected; data.next_best_path_ = path; }
    ROS_INFO("[RECOVERY] episode=%s step=%d stage=exclusion applied=%d",
        observation_episode_.c_str(), step, found);
  }
  const auto decision = stagnation_recovery_.observe(step, true, pos, data.next_pos_, data.next_best_path_);
  if (decision == StagnationRecovery::NONE) return;
  const Vector2d old_goal = data.next_pos_;
  const bool alternate = decision == StagnationRecovery::SWITCH;
  const bool found = expl_manager_->recoveryPath(pos, old_goal, alternate, selected, path);
  // Compare the remaining route, ignoring the already travelled prefix.
  auto suffix = [&pos](const vector<Vector2d>& route) {
    size_t nearest = 0;
    for (size_t i = 1; i < route.size(); ++i)
      if ((route[i] - pos).squaredNorm() < (route[nearest] - pos).squaredNorm()) nearest = i;
    return vector<Vector2d>(route.begin() + nearest, route.end());
  };
  bool changed = false;
  if (found) {
    const auto old_suffix = suffix(data.next_best_path_);
    const auto new_suffix = suffix(path);
    changed = old_suffix.size() != new_suffix.size();
    for (size_t i = 0; !changed && i < old_suffix.size(); ++i)
      changed = (old_suffix[i] - new_suffix[i]).norm() > 1e-3;
    if (changed) { data.next_pos_ = selected; data.next_best_path_ = path; }
    if (alternate && changed) {
      recovery_excluded_goal_ = old_goal;
      recovery_exclusion_until_ = step + 30;
    }
  }
  ROS_INFO("[RECOVERY] episode=%s step=%d attempt=%d stage=%s reachable=%d "
           "applied=%d old_goal=(%.3f,%.3f) new_goal=(%.3f,%.3f)",
      observation_episode_.c_str(), step, stagnation_recovery_.attempts(),
      alternate ? "switch" : "replan", found, changed, old_goal.x(), old_goal.y(),
      data.next_pos_.x(), data.next_pos_.y());
}

void ExplorationFSM::updateFrontierQueryState()
{
  frontier_query_active_ = false;
  if (!frontier_query_eligible_this_cycle_ || fd_->final_result_ != FINAL_RESULT::EXPLORE ||
      expl_manager_->ed_->next_best_path_.empty())
    return;

  const Vector2d selected = expl_manager_->ed_->next_pos_;
  const bool changed = !have_last_frontier_target_ ||
      (selected - last_frontier_target_).norm() > 0.10;
  if (!changed && !frontierOnceExperiment())
    return;

  ++frontier_query_id_;
  frontier_query_active_ = true;
  last_frontier_target_ = selected;
  have_last_frontier_target_ = true;
  ROS_INFO("[AVIC_FRONTIER] new adaptive decision query=%u apexnav_target=(%.3f, %.3f)",
      frontier_query_id_, selected(0), selected(1));
}

void ExplorationFSM::publishTopKFrontiers(const Vector2d& current_pos, double current_yaw)
{
  std_msgs::Float32MultiArray msg;
  vector<SemanticFrontier> semantic_frontiers;
  expl_manager_->getSortedSemanticFrontiers(
      current_pos, expl_manager_->ed_->frontier_averages_, semantic_frontiers);

  const int count = frontierOnceExperiment() ? static_cast<int>(semantic_frontiers.size()) :
      std::min(fp_->avic_top_k_, static_cast<int>(semantic_frontiers.size()));
  vector<int> candidate_indices;
  candidate_indices.reserve(count);
  for (int index = 0; index < count; ++index)
    candidate_indices.push_back(index);

  // The geometric/Closest winner can sit outside the semantic top-k.  Always
  // include ApexNav's actually selected frontier so G1 can compare the two
  // policies instead of reasoning over an incomplete candidate set.
  int selected_index = -1;
  for (int index = 0; index < static_cast<int>(semantic_frontiers.size()); ++index) {
    if ((semantic_frontiers[index].position - expl_manager_->ed_->next_pos_).norm() < 0.10) {
      selected_index = index;
      break;
    }
  }
  if (selected_index >= count && !candidate_indices.empty())
    candidate_indices.back() = selected_index;

  constexpr int route_action_count = FSMConstants::AVIC_ROUTE_PREFIX_ACTIONS;
  // Read-only export of ORIGINAL cluster cells, bound to the same sorted
  // candidate snapshot. Do not re-cluster, resample or change selection.
  const bool export_cells = (std::getenv("AVIC_IMAGE_FRONTIER_GEOMETRY") &&
      std::string(std::getenv("AVIC_IMAGE_FRONTIER_GEOMETRY")) == "1") ||
      (textFrontierExperiment() && std::getenv("APEXNAV_TEXT_METHOD") &&
       std::string(std::getenv("APEXNAV_TEXT_METHOD")) == "dictionary");
  vector<vector<Vector2d>> exported_cells(count);
  int max_cells = 0;
  if (export_cells) {
    const auto& data = *expl_manager_->ed_;
    for (int rank = 0; rank < count; ++rank) {
      const auto& position = semantic_frontiers[candidate_indices[rank]].position;
      for (size_t j = 0; j < data.frontier_averages_.size() && j < data.frontiers_.size(); ++j) {
        if ((position - data.frontier_averages_[j]).norm() < 1e-6) {
          exported_cells[rank] = data.frontiers_[j];
          break;
        }
      }
      max_cells = std::max(max_cells, static_cast<int>(exported_cells[rank].size()));
    }
  }
  const int field_count = 12 + route_action_count + (export_cells ? 2 + 2 * max_cells : 0);
  msg.layout.dim.resize(2);
  msg.layout.dim[0].label = "frontier";
  msg.layout.dim[0].size = count;
  msg.layout.dim[0].stride = count * field_count;
  msg.layout.dim[1].label =
      "x,y,semantic,path_length,relative_yaw,first_action,selected,query_id,query_active,"
      "observation_evidence,evidence_observed,route_action_count,route_action_0..5";
  if (export_cells) msg.layout.dim[1].label += ",cluster_cells_v1,count,xy_pairs_padded";
  msg.layout.dim[1].size = field_count;
  msg.layout.dim[1].stride = field_count;
  msg.data.reserve(count * field_count);

  const Vector2d saved_local_pos = fd_->local_pos_;
  for (int rank = 0; rank < count; ++rank) {
    const auto& frontier = semantic_frontiers[candidate_indices[rank]];
    const int first_action =
        planNextBestAction(current_pos, current_yaw, frontier.path, true);
    vector<int> route_actions = buildFrontierActionPrefix(
        current_pos, current_yaw, frontier.path, route_action_count);
    // The first route action must be exactly the action produced by ApexNav's
    // current A* path controller.  A mismatch means the future simulation
    // drifted before its first step, so keep the authoritative first action.
    if (route_actions.empty())
      route_actions.push_back(first_action);
    else
      route_actions.front() = first_action;
    const Vector2d local_target = fd_->local_pos_;
    double relative_yaw =
        std::atan2(local_target(1) - current_pos(1), local_target(0) - current_pos(0)) -
        current_yaw;
    wrapAngle(relative_yaw);
    const bool selected =
        (frontier.position - expl_manager_->ed_->next_pos_).norm() < 0.10;

    msg.data.push_back(static_cast<float>(frontier.position(0)));
    msg.data.push_back(static_cast<float>(frontier.position(1)));
    msg.data.push_back(static_cast<float>(frontier.semantic_value));
    msg.data.push_back(static_cast<float>(frontier.path_length));
    msg.data.push_back(static_cast<float>(relative_yaw));
    msg.data.push_back(static_cast<float>(first_action));
    msg.data.push_back(selected ? 1.0F : 0.0F);
    msg.data.push_back(static_cast<float>(frontier_query_id_));
    msg.data.push_back(frontier_query_active_ ? 1.0F : 0.0F);
    msg.data.push_back(static_cast<float>(frontier.observation_evidence));
    const bool evidence_observed =
        frontier.observation_evidence >= fp_->avic_observation_evidence_threshold_;
    msg.data.push_back(evidence_observed ? 1.0F : 0.0F);
    msg.data.push_back(static_cast<float>(route_actions.size()));
    for (int action_index = 0; action_index < route_action_count; ++action_index) {
      const int action = action_index < static_cast<int>(route_actions.size())
          ? route_actions[action_index]
          : -1;
      msg.data.push_back(static_cast<float>(action));
    }
    if (export_cells) {
      msg.data.push_back(1.0F);
      msg.data.push_back(static_cast<float>(exported_cells[rank].size()));
      for (int j = 0; j < max_cells; ++j) {
        const Vector2d cell = j < static_cast<int>(exported_cells[rank].size())
            ? exported_cells[rank][j] : Vector2d::Zero();
        msg.data.push_back(static_cast<float>(cell.x()));
        msg.data.push_back(static_cast<float>(cell.y()));
      }
    }

    std::ostringstream route_stream;
    for (size_t action_index = 0; action_index < route_actions.size(); ++action_index) {
      if (action_index > 0) route_stream << ',';
      route_stream << route_actions[action_index];
    }
    const std::string route_string = route_stream.str();

    ROS_INFO("[AVIC_FRONTIER] rank=%d x=%.3f y=%.3f semantic=%.6f path_length=%.3f "
             "relative_yaw=%.3f first_action=%d selected=%d query=%u active=%d "
             "evidence=%.3f observed=%d route_actions=%s",
        rank, frontier.position(0), frontier.position(1), frontier.semantic_value,
        frontier.path_length, relative_yaw, first_action, selected ? 1 : 0,
        frontier_query_id_, frontier_query_active_ ? 1 : 0, frontier.observation_evidence,
        evidence_observed ? 1 : 0, route_string.c_str());
  }
  fd_->local_pos_ = saved_local_pos;
  frontier_snapshot_ = msg;
  frontier_candidates_pub_.publish(msg);
}

int ExplorationFSM::planNextBestAction(
    Vector2d current_pos, double current_yaw, const vector<Vector2d>& path, bool need_safety)
{
  const double local_distance = FSMConstants::LOCAL_DISTANCE;

  // Update target position based on path and local distance
  Vector2d local_pos = selectLocalTarget(current_pos, path, local_distance);
  fd_->local_pos_ = local_pos;

  // Compute the best step considering obstacles and safety
  Vector2d best_step;
  if ((current_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE && need_safety)
    best_step = computeBestStep(current_pos, current_yaw, local_pos);
  else
    best_step = local_pos;

  // Calculate target orientation from best step direction
  double target_yaw = std::atan2(best_step(1) - current_pos(1), best_step(0) - current_pos(0));
  return decideNextAction(current_yaw, target_yaw);
}

vector<int> ExplorationFSM::buildFrontierActionPrefix(const Vector2d& current_pos,
    double current_yaw, const vector<Vector2d>& path, int max_actions)
{
  vector<int> actions;
  if (path.empty() || max_actions <= 0)
    return actions;

  Vector2d simulated_pos = current_pos;
  double simulated_yaw = current_yaw;
  actions.reserve(max_actions);

  // Roll ApexNav's own local A* path follower forward in Habitat's discrete
  // action units.  This is only a route description for SVC; it does not alter
  // odometry, map state, the selected frontier, or the executed action.
  for (int index = 0; index < max_actions; ++index) {
    const Vector2d local_target =
        selectLocalTarget(simulated_pos, path, FSMConstants::LOCAL_DISTANCE);
    Vector2d best_step = local_target;
    if ((simulated_pos - path.back()).norm() > FSMConstants::ACTION_DISTANCE)
      best_step = computeBestStep(simulated_pos, simulated_yaw, local_target);

    const double target_yaw =
        std::atan2(best_step(1) - simulated_pos(1), best_step(0) - simulated_pos(0));
    const int action = decideNextAction(simulated_yaw, target_yaw);
    if (action != ACTION::MOVE_FORWARD && action != ACTION::TURN_LEFT &&
        action != ACTION::TURN_RIGHT)
      break;
    actions.push_back(action);

    if (action == ACTION::TURN_LEFT) {
      simulated_yaw += FSMConstants::ACTION_ANGLE;
      wrapAngle(simulated_yaw);
    }
    else if (action == ACTION::TURN_RIGHT) {
      simulated_yaw -= FSMConstants::ACTION_ANGLE;
      wrapAngle(simulated_yaw);
    }
    else {
      simulated_pos(0) += FSMConstants::ACTION_DISTANCE * std::cos(simulated_yaw);
      simulated_pos(1) += FSMConstants::ACTION_DISTANCE * std::sin(simulated_yaw);
      if ((simulated_pos - path.back()).norm() <= FSMConstants::REACH_DISTANCE)
        break;
    }
  }
  return actions;
}

Vector2d ExplorationFSM::selectLocalTarget(
    const Vector2d& current_pos, const vector<Vector2d>& path, const double& local_distance)
{
  Vector2d target_pos = path.back();

  // Find the closest path point to current position as starting search index
  int start_path_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos = path[i];
    if ((pos - current_pos).norm() < min_dist) {
      min_dist = (pos - current_pos).norm();
      start_path_id = i + 1;
    }
  }

  // Select a local target position within the specified distance
  double len = (path[start_path_id] - current_pos).norm();
  for (int i = start_path_id + 1; i < (int)path.size(); i++) {
    len += (path[i] - path[i - 1]).norm();
    if (len > local_distance && (current_pos - path[i - 1]).norm() > 0.30) {
      target_pos = path[i - 1];
      break;
    }
  }

  return target_pos;
}

Vector2d ExplorationFSM::computeBestStep(
    const Vector2d& current_pos, double current_yaw, const Vector2d& target_pos)
{
  Vector2d best_step = target_pos;

  double min_cost = std::numeric_limits<double>::max();
  for (auto step : fp_->action_steps_) {
    double cost = computeActionTotalCost(current_pos, current_yaw, target_pos, step);
    if (cost < min_cost) {
      best_step = current_pos + step;
      min_cost = cost;
    }
  }

  return best_step;
}

// Compute total cost of taking a step towards target
// Considers distance-to-target, movement efficiency, and collision safety
double ExplorationFSM::computeActionTotalCost(const Vector2d& current_pos, double current_yaw,
    const Vector2d& target_pos, const Vector2d& step)
{
  const double traget_weight = FSMConstants::TARGET_WEIGHT;
  const double traget_close_weight1 = FSMConstants::TARGET_CLOSE_WEIGHT_1;
  const double traget_close_weight2 = FSMConstants::TARGET_CLOSE_WEIGHT_2;
  const double safety_weight = FSMConstants::SAFETY_WEIGHT;
  double cost = 0.0;

  // Distance-to-target cost
  Vector2d step_pos = current_pos + step;
  double target_cost = traget_weight * (step_pos - target_pos).norm();

  // Change-in-distance cost (negative if moving closer)
  double target_close_cost = (step_pos - target_pos).norm() - (current_pos - target_pos).norm();
  if (target_close_cost > 0)
    target_close_cost *= traget_close_weight1;
  else
    target_close_cost *= traget_close_weight2;

  // Safety distance cost
  double safety_cost = safety_weight * computeActionSafetyCost(current_pos, step);

  cost += target_cost + target_close_cost + safety_cost;
  return cost;
}

// Compute safety cost along the step using SDF distance to obstacles
// Returns higher cost for paths that go too close to obstacles
double ExplorationFSM::computeActionSafetyCost(const Vector2d& current_pos, const Vector2d& step)
{
  const double min_safe_distance = FSMConstants::MIN_SAFE_DISTANCE;
  const double sample_num = FSMConstants::SAMPLE_NUM;

  Vector2d dir = step;
  double len = dir.norm();
  dir.normalize();

  double safety_cost = 0.0;
  for (double l = len / sample_num; l < len; l += len / sample_num) {
    Vector2d ckpt = current_pos + l * dir;
    Vector2d grad;
    double dist_to_occ = expl_manager_->sdf_map_->getDistWithGrad(ckpt, grad);
    if (dist_to_occ < min_safe_distance)
      safety_cost += 1 / (dist_to_occ + 1e-2);
  }

  return safety_cost;
}

// Decide whether to turn or move forward based on yaw difference
// Uses action angle threshold to determine if orientation adjustment is needed
int ExplorationFSM::decideNextAction(double current_yaw, double target_yaw)
{
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  int next_action;
  if (std::fabs(yaw_diff) > FSMConstants::ACTION_ANGLE / 1.9) {
    if (yaw_diff > 0)
      next_action = ACTION::TURN_LEFT;
    else
      next_action = ACTION::TURN_RIGHT;
  }
  else
    next_action = ACTION::MOVE_FORWARD;

  return next_action;
}

void ExplorationFSM::visualize()
{
  auto ed_ptr = expl_manager_->ed_;

  // Lambda function to convert 2D vectors to 3D for visualization
  auto vec2dTo3d = [](const vector<Eigen::Vector2d>& vec2d, double z = 0.15) {
    vector<Eigen::Vector3d> vec3d;
    for (auto v : vec2d) vec3d.push_back(Vector3d(v(0), v(1), z));
    return vec3d;
  };

  // Draw frontier
  static int last_ftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->frontiers_[i]), fp_->vis_scale_,
        visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 1.0), "frontier", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
  }
  last_ftr2d_num = ed_ptr->frontiers_.size();

  // Draw dormant frontier
  static int last_dftr2d_num = 0;
  for (int i = 0; i < (int)ed_ptr->dormant_frontiers_.size(); ++i) {
    visualization_->drawCubes(vec2dTo3d(ed_ptr->dormant_frontiers_[i]), fp_->vis_scale_,
        Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  for (int i = ed_ptr->dormant_frontiers_.size(); i < last_dftr2d_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
  }
  last_dftr2d_num = ed_ptr->dormant_frontiers_.size();

  // Draw object
  // static int last_obj_num = 0;
  // for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
  //   visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
  //       visualization_->getColor(double(i) / ed_ptr->objects_.size(), 1.0), "object", i, 4);
  // }
  // for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
  //   visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  // }
  // last_obj_num = ed_ptr->objects_.size();

  static int last_obj_num = 0;
  for (int i = 0; i < (int)ed_ptr->objects_.size(); ++i) {
    int label = ed_ptr->object_labels_[i];
    visualization_->drawCubes(vec2dTo3d(ed_ptr->objects_[i]), fp_->vis_scale_,
        visualization_->getColor(double(label) / 5.0, 1.0), "object", i, 4);
  }
  for (int i = ed_ptr->objects_.size(); i < last_obj_num; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }
  last_obj_num = ed_ptr->objects_.size();

  // Draw next best path
  visualization_->drawLines(vec2dTo3d(ed_ptr->next_best_path_), fp_->vis_scale_,
      Vector4d(1, 0.2, 0.2, 1), "next_path", 1, 6);

  // Draw next local point
  vector<Vector2d> local_points;
  local_points.push_back(fd_->local_pos_);
  visualization_->drawSpheres(vec2dTo3d(local_points), fp_->vis_scale_ * 3,
      Vector4d(0.2, 0.2, 1.0, 1), "local_point", 1, 6);

  visualization_->drawLines(vec2dTo3d(ed_ptr->tsp_tour_), fp_->vis_scale_ / 1.25,
      Vector4d(0.2, 1, 0.2, 1), "tsp_tour", 0, 6);

  visualization_->drawSpheres(vec2dTo3d(fd_->traveled_path_), fp_->vis_scale_ * 1.5,
      Vector4d(2.0 / 255.0, 111.0 / 255.0, 197.0 / 255.0, 1), "traveled_path", 1, 6);
}

void ExplorationFSM::clearVisMarker()
{
  auto ed_ptr = expl_manager_->ed_;
  for (int i = 0; i < 500; ++i) {
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "dormant_frontier", i, 4);
    visualization_->drawCubes({}, fp_->vis_scale_, Vector4d(0, 0, 0, 1), "object", i, 4);
  }

  visualization_->drawLines({}, fp_->vis_scale_, Vector4d(0, 0, 1, 1), "next_path", 1, 6);
}

bool ExplorationFSM::updateFrontierAndObject()
{
  bool change_flag = false;
  auto frt_map = expl_manager_->frontier_map2d_;
  auto obj_map = expl_manager_->object_map2d_;
  auto ed = expl_manager_->ed_;
  const Eigen::Vector2d sensor_pos(fd_->odom_pos_(0), fd_->odom_pos_(1));

  change_flag = frt_map->isAnyFrontierChanged();
  frt_map->searchFrontiers();
  change_flag |= frt_map->dormantSeenFrontiers(sensor_pos, fd_->odom_yaw_);
  frt_map->getFrontiers(ed->frontiers_, ed->frontier_averages_);
  frt_map->getDormantFrontiers(ed->dormant_frontiers_, ed->dormant_frontier_averages_);
  obj_map->getObjects(ed->objects_, ed->object_averages_, ed->object_labels_);

  return change_flag;
}

// Receive Habitat state messages
void ExplorationFSM::habitatStateCallback(const std_msgs::Int32ConstPtr& msg)
{
  if (msg->data == HABITAT_STATE::ACTION_FINISH && state_ == ROS_STATE::WAIT_ACTION_FINISH)
    transitState(PLAN_ACTION, "Habitat Finish Action");
  if (msg->data == HABITAT_STATE::EPISODE_FINISH) {
    ++benchmark_episode_index_;
    benchmark_decision_index_ = 0;
    init(nh_);
  }
  return;
}

// Habitat remains the authority on what was actually executed.  AVIC no longer
// publishes primitive actions, but this check also protects future controllers.
void ExplorationFSM::executedActionCallback(const std_msgs::Int32ConstPtr& msg)
{
  if (msg->data < ACTION::STOP || msg->data > ACTION::TURN_UP) {
    ROS_WARN("Ignoring invalid executed action: %d", msg->data);
    return;
  }
  if (fd_->newest_action_ != msg->data) {
    ROS_INFO("Executed action differs from last published action: planned=%d executed=%d",
        fd_->newest_action_,
        msg->data);
  }
  fd_->newest_action_ = msg->data;
}

void ExplorationFSM::avicFrontierScoresCallback(
    const std_msgs::Float32MultiArrayConstPtr& msg)
{
  constexpr size_t header_count = 2;
  constexpr size_t score_field_count = 6;
  if (msg->data.size() < header_count) {
    ROS_WARN("Ignoring malformed AVIC frontier scores with %zu fields", msg->data.size());
    return;
  }

  const int query_value = static_cast<int>(std::lround(msg->data[0]));
  if (query_value < 0 || static_cast<std::uint32_t>(query_value) != frontier_query_id_ ||
      !frontier_query_active_) {
    ROS_WARN("Ignoring stale AVIC frontier scores query=%d current=%u active=%d", query_value,
        frontier_query_id_, frontier_query_active_ ? 1 : 0);
    return;
  }

  const int score_count = static_cast<int>(std::lround(msg->data[1]));
  if (score_count < 1 || msg->data.size() !=
          header_count + static_cast<size_t>(score_count) * score_field_count) {
    ROS_WARN("Ignoring malformed AVIC frontier score count=%d fields=%zu", score_count,
        msg->data.size());
    return;
  }

  vector<TemporaryFrontierScore> scores;
  scores.reserve(score_count);
  for (int index = 0; index < score_count; ++index) {
    const size_t offset = header_count + static_cast<size_t>(index) * score_field_count;
    const double x = static_cast<double>(msg->data[offset + 1]);
    const double y = static_cast<double>(msg->data[offset + 2]);
    const double real_score = static_cast<double>(msg->data[offset + 3]);
    const double imagined_itm_score = static_cast<double>(msg->data[offset + 4]);
    const double decision_score = static_cast<double>(msg->data[offset + 5]);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(real_score) ||
        !std::isfinite(imagined_itm_score) || !std::isfinite(decision_score) ||
        (!frontierOnceExperiment() && !alignedBaseExperiment() && !relaxedImagination() && !imageGLMExperiment() && !textFrontierExperiment() && decision_score < 0.0) ||
        (textFrontierExperiment() && (decision_score<0.0 || decision_score>1.0))) {
      ROS_WARN("Ignoring non-finite AVIC frontier score at index=%d", index);
      return;
    }
    TemporaryFrontierScore score;
    score.position = Vector2d(x, y);
    score.real_semantic_value = real_score;
    score.imagined_semantic_value = imagined_itm_score;
    score.decision_semantic_value = decision_score;
    constexpr double kMinImaginationGain = 1e-6;
    if (!frontierOnceExperiment() && !alignedBaseExperiment() && !relaxedImagination() && !imageGLMExperiment() && !textFrontierExperiment() && score.decision_semantic_value <=
        score.real_semantic_value + kMinImaginationGain) {
      ROS_INFO("[AVIC_SCORE] ignoring non-positive gain query=%d index=%d "
               "real=%.6f decision=%.6f",
          query_value, index, score.real_semantic_value,
          score.decision_semantic_value);
      continue;
    }
    scores.push_back(score);
  }

  frontier_query_active_ = false;
  if (scores.empty()) {
    const int safe_action = fd_->newest_action_;
    std_msgs::Float32MultiArray ack;
    ack.data = { static_cast<float>(query_value), static_cast<float>(safe_action), 0.0F };
    avic_frontier_ack_pub_.publish(ack);
    ROS_INFO("[AVIC_SCORE] no positive-gain scores query=%d; keeping action=%d "
             "without replanning",
        query_value, safe_action);
    return;
  }
  ROS_INFO("[AVIC_SCORE] received query=%d score_count=%d applicable_count=%zu",
      query_value, score_count, scores.size());

  const Vector2d original_next_pos = expl_manager_->ed_->next_pos_;
  const vector<Vector2d> original_next_path = expl_manager_->ed_->next_best_path_;
  const vector<Vector2d> original_tsp_tour = expl_manager_->ed_->tsp_tour_;
  const Vector2d original_local_pos = fd_->local_pos_;
  const int original_action = fd_->newest_action_;
  int matched_score_count = 0;
  const int exploration_result = expl_manager_->planNextBestPointWithTemporaryScores(
      fd_->start_pt_, fd_->start_yaw_(0), scores, fp_->avic_score_match_tolerance_,
      matched_score_count);
  const bool target_changed =
      (expl_manager_->ed_->next_pos_ - original_next_pos).norm() > 0.10;
  bool path_changed = original_next_path.size() != expl_manager_->ed_->next_best_path_.size();
  if (!path_changed)
    for (size_t i = 0; i < original_next_path.size(); ++i)
      path_changed |= (original_next_path[i] - expl_manager_->ed_->next_best_path_[i]).norm() > 1e-6;
  const Vector2d cycle_robot(fd_->odom_pos_(0),fd_->odom_pos_(1));
  const bool cycle_rejected=synchronous_observations_ && exploration_result==EXPL_RESULT::EXPLORATION &&
      expl_manager_->rejectCycleGoal(cycle_robot,observation_step_,observation_episode_);
  if(cycle_rejected)
    ROS_INFO("[FRONTIER_CYCLE] stage=avic_reject episode=%s step=%d robot=(%.3f,%.3f) original=(%.3f,%.3f) candidate=(%.3f,%.3f)",
        observation_episode_.c_str(),observation_step_,cycle_robot.x(),cycle_robot.y(),original_next_pos.x(),original_next_pos.y(),
        expl_manager_->ed_->next_pos_.x(),expl_manager_->ed_->next_pos_.y());
  const bool applied = !cycle_rejected && target_changed && exploration_result == EXPL_RESULT::EXPLORATION &&
      matched_score_count > 0 && !expl_manager_->ed_->next_best_path_.empty();

  const Vector2d current_pos(fd_->odom_pos_(0), fd_->odom_pos_(1));
  int safe_action = fd_->newest_action_;
  if (applied) {
    safe_action = planNextBestAction(
        current_pos, fd_->odom_yaw_, expl_manager_->ed_->next_best_path_, true);
    fd_->newest_action_ = safe_action;
    if(synchronous_observations_)
      expl_manager_->stageCycleDecision(current_pos,observation_step_,safe_action,observation_episode_);
    fd_->replan_flag_ = true;
    last_frontier_target_ = expl_manager_->ed_->next_pos_;
    have_last_frontier_target_ = true;
    if (synchronous_observations_) {
      const auto prefix = buildFrontierActionPrefix(current_pos, fd_->odom_yaw_,
          expl_manager_->ed_->next_best_path_, FSMConstants::AVIC_ROUTE_PREFIX_ACTIONS);
      avic_route_commitment_.begin(observation_step_, expl_manager_->ed_->next_pos_,
          avic_route_commit_steps_, prefix);
      if (avic_route_commitment_.active) {
        if (stagnation_recovery_enabled_) stagnation_recovery_.interrupt();
        ROS_INFO("[AVIC_ROUTE] stage=start episode=%s step=%d query=%d deadline=%d "
                 "goal=(%.3f,%.3f) expected=%zu",
            observation_episode_.c_str(), observation_step_, query_value,
            avic_route_commitment_.deadline, avic_route_commitment_.goal.x(),
            avic_route_commitment_.goal.y(), prefix.size());
      }
    }
    std_msgs::Int32 action_msg;
    action_msg.data = safe_action;
    if (!synchronous_observations_) action_pub_.publish(action_msg);
  }
  else {
    expl_manager_->ed_->next_pos_ = original_next_pos;
    expl_manager_->ed_->next_best_path_ = original_next_path;
    expl_manager_->ed_->tsp_tour_ = original_tsp_tour;
    fd_->local_pos_ = original_local_pos;
  }

  std_msgs::Float32MultiArray ack;
  ack.data = { static_cast<float>(query_value), static_cast<float>(safe_action),
    applied ? 1.0F : 0.0F };
  avic_frontier_ack_pub_.publish(ack);
  ROS_INFO("[AVIC_EFFECT] query=%d target_changed=%d trial_path_changed=%d action_changed=%d applied=%d",
      query_value, target_changed, path_changed, safe_action != original_action, applied);
  ROS_INFO("[AVIC_SCORE] ack query=%d applied=%d matched=%d apexnav_safe_action=%d "
           "selected=(%.3f, %.3f)",
      query_value, applied ? 1 : 0, matched_score_count, safe_action,
      expl_manager_->ed_->next_pos_(0), expl_manager_->ed_->next_pos_(1));
}

// Periodically update frontiers and visualize in idle states
void ExplorationFSM::frontierCallback(const ros::TimerEvent& e)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER && state_ != ROS_STATE::FINISH)
    return;

  updateFrontierAndObject();
  visualize();
}

// Receive user trigger to start exploration
void ExplorationFSM::triggerCallback(const geometry_msgs::PoseStampedConstPtr& msg)
{
  if (state_ != ROS_STATE::WAIT_TRIGGER)
    return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN_ACTION, "triggerCallback");
}

// Receive robot odometry and update traveled path + marker
void ExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg)
{
  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;

  Vector2d odom_pos2d = Vector2d(fd_->odom_pos_(0), fd_->odom_pos_(1));
  if (fd_->traveled_path_.empty())
    fd_->traveled_path_.push_back(odom_pos2d);
  else if ((fd_->traveled_path_.back() - odom_pos2d).norm() > 1e-2)
    fd_->traveled_path_.push_back(odom_pos2d);

  publishRobotMarker();
}

void ExplorationFSM::publishRobotMarker()
{
  const double robot_height = FSMConstants::ROBOT_HEIGHT;
  const double robot_radius = FSMConstants::ROBOT_RADIUS;

  // Create robot body cylinder marker
  visualization_msgs::Marker robot_marker;
  robot_marker.header.frame_id = "world";
  robot_marker.header.stamp = ros::Time::now();
  robot_marker.ns = "robot_position";
  robot_marker.id = 0;
  robot_marker.type = visualization_msgs::Marker::CYLINDER;
  robot_marker.action = visualization_msgs::Marker::ADD;

  // Set cylinder position
  robot_marker.pose.position.x = fd_->odom_pos_(0);
  robot_marker.pose.position.y = fd_->odom_pos_(1);
  robot_marker.pose.position.z = fd_->odom_pos_(2) + robot_height / 2.0;

  // Set cylinder orientation
  robot_marker.pose.orientation.x = fd_->odom_orient_.x();
  robot_marker.pose.orientation.y = fd_->odom_orient_.y();
  robot_marker.pose.orientation.z = fd_->odom_orient_.z();
  robot_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set cylinder dimensions
  robot_marker.scale.x = robot_radius * 2;  // Diameter
  robot_marker.scale.y = robot_radius * 2;  // Diameter
  robot_marker.scale.z = robot_height;      // Height

  // Set cylinder color (blue)
  robot_marker.color.r = 50.0 / 255.0;
  robot_marker.color.g = 50.0 / 255.0;
  robot_marker.color.b = 255.0 / 255.0;
  robot_marker.color.a = 1.0;

  // Create direction arrow marker
  visualization_msgs::Marker arrow_marker;
  arrow_marker.header.frame_id = "world";
  arrow_marker.header.stamp = ros::Time::now();
  arrow_marker.ns = "robot_direction";
  arrow_marker.id = 1;
  arrow_marker.type = visualization_msgs::Marker::ARROW;
  arrow_marker.action = visualization_msgs::Marker::ADD;

  // Set arrow position
  arrow_marker.pose.position.x = fd_->odom_pos_(0);
  arrow_marker.pose.position.y = fd_->odom_pos_(1);
  arrow_marker.pose.position.z = fd_->odom_pos_(2) + robot_height;

  // Set arrow orientation
  arrow_marker.pose.orientation.x = fd_->odom_orient_.x();
  arrow_marker.pose.orientation.y = fd_->odom_orient_.y();
  arrow_marker.pose.orientation.z = fd_->odom_orient_.z();
  arrow_marker.pose.orientation.w = fd_->odom_orient_.w();

  // Set arrow dimensions
  arrow_marker.scale.x = robot_radius + 0.13;  // Arrow length
  arrow_marker.scale.y = 0.08;                 // Arrow width
  arrow_marker.scale.z = 0.08;                 // Arrow thickness

  // Set arrow color (green)
  arrow_marker.color.r = 10.0 / 255.0;
  arrow_marker.color.g = 255.0 / 255.0;
  arrow_marker.color.b = 10.0 / 255.0;
  arrow_marker.color.a = 1.0;

  // Publish both markers
  robot_marker_pub_.publish(robot_marker);
  robot_marker_pub_.publish(arrow_marker);
}

void ExplorationFSM::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  if (fd_->have_confidence_)
    return;
  fd_->have_confidence_ = true;
  expl_manager_->sdf_map_->object_map2d_->setConfidenceThreshold(msg->data);
}

// Transition FSM state and log the change
void ExplorationFSM::transitState(ROS_STATE new_state, string pos_call)
{
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[ " + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " +
              fd_->state_str_[int(new_state)]
       << endl;
}
}  // namespace apexnav_planner
