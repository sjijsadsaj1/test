/**
 * @file exploration_manager.cpp
 * @brief Implementation of exploration manager for autonomous semantic navigation
 * @author Zager-Zhang
 *
 * This file implements the ExplorationManager class that handles various
 * exploration strategies including distance-based, semantic-based, hybrid,
 * and TSP-optimized frontier selection for autonomous robot exploration.
 */

#include <exploration_manager/exploration_manager.h>
#include <exploration_manager/exploration_data.h>
#include <exploration_manager/tsp_service.h>
#include <lkh_mtsp_solver/SolveMTSP.h>
#include <plan_env/map_ros.h>
#include <path_searching/kino_astar.h>
#include <trajectory_manager/optimizer.h>

#include <algorithm>
#include <cstdlib>
#include <queue>
#include <set>

using namespace Eigen;

namespace apexnav_planner {
namespace {
bool textFrontierEnabled() {
  const char* flag=std::getenv("APEXNAV_TEXT_FRONTIER");
  return flag && std::string(flag)=="1";
}
bool imageGLMStrongRequested() {
  const char* flag=std::getenv("APEXNAV_IMAGE_GLM_STRONG");
  return flag && std::string(flag)=="1";
}
bool imageGLMStrongEnabled() {
  const char* flag=std::getenv("APEXNAV_IMAGE_GLM");
  return imageGLMStrongRequested() && flag && std::string(flag)=="1";
}
}

ExplorationManager::~ExplorationManager() = default;

void ExplorationManager::initialize(ros::NodeHandle& nh)
{
  const bool dictionary_mode=std::getenv("APEXNAV_TEXT_METHOD") && std::string(std::getenv("APEXNAV_TEXT_METHOD"))=="dictionary";
  ROS_INFO("[TEXT_FRONTIER_CONFIG] enabled=%d contract=%s",textFrontierEnabled(),
      dictionary_mode ? "frontier_object_marginals_v1" : "text_frontier_belief_v1");
  ROS_INFO("[IMAGE_GLM_STRONG_CONFIG] enabled=%d requested=%d contract=image_glm_strong_semantic_v1 scope=current_temporary_score_request",
      imageGLMStrongEnabled(),imageGLMStrongRequested());
  cycle_memory_.reset();
  ROS_INFO("[FRONTIER_CYCLE_CONFIG] enabled=%d robot_m=0.5 candidate_m=0.5 score_delta=0.01 min_steps=12 max_history=512",
      cycleMemoryEnabled());
  // Initialize SDF map and get object map reference
  sdf_map_.reset(new SDFMap2D);
  sdf_map_->initMap(nh);
  object_map2d_ = sdf_map_->object_map2d_;

  // Initialize frontier map and path finder
  frontier_map2d_.reset(new FrontierMap2D(sdf_map_, nh));
  path_finder_.reset(new Astar2D);
  path_finder_->init(nh, sdf_map_);

  // Initialize exploration data and parameter containers
  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  // Load exploration parameters from ROS parameter server
  nh.param("exploration/policy", ep_->policy_mode_, 0);
  nh.param("exploration/sigma_threshold", ep_->sigma_threshold_, 0.030);
  nh.param("exploration/max_to_mean_threshold", ep_->max_to_mean_threshold_, 1.2);
  nh.param("exploration/max_to_mean_percentage", ep_->max_to_mean_percentage_, 0.95);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));

  // Get map parameters for ray casting initialization
  double resolution = sdf_map_->getResolution();
  Eigen::Vector2d origin, size;
  sdf_map_->getRegion(origin, size);

  // Initialize ray caster for collision checking and TSP service client
  ray_caster2d_.reset(new RayCaster2D);
  ray_caster2d_->setParams(resolution, origin);

  // Initialize KinoAstar and GCopter for real-world trajectory planning
  kinoastar_.reset(new KinoAstar(nh, sdf_map_));
  kinoastar_->init();
  
  Config gcopter_config(nh);
  gcopter_.reset(new Gcopter(gcopter_config, nh, sdf_map_, kinoastar_));
  
  ROS_INFO("[ExplorationManager] KinoAstar and GCopter initialized for real-world mode");
}

bool ExplorationManager::cycleMemoryEnabled() const {
  const char* flag=std::getenv("APEXNAV_FRONTIER_CYCLE_MEMORY");
  return flag && std::string(flag)=="1";
}

bool ExplorationManager::realCycleContext(FrontierCycleMemory::Point& context) {
  vector<double> values;
  for(const auto& frontier:ed_->frontier_averages_) {
    Vector2i idx;sdf_map_->posToIndex(frontier,idx);
    double value=sdf_map_->value_map_->getValue(idx);
    for(const auto& nbr:allNeighbors(idx,2)) {
      if(sdf_map_->getInflateOccupancy(nbr)==1 || sdf_map_->getOccupancy(nbr)==SDFMap2D::OCCUPIED)continue;
      value=std::max(value,sdf_map_->value_map_->getValue(nbr));
    }
    if(!std::isfinite(value))return false;
    values.push_back(value);
  }
  if(values.empty())return false;
  std::sort(values.begin(),values.end(),std::greater<double>());
  context={{values[0],values.size()>1?values[1]:values[0]}};return true;
}

bool ExplorationManager::cycleDecision(const Vector2d& robot,const Vector2d& goal,int step,int action,
    FrontierCycleMemory::Decision& decision) {
  if(!cycleMemoryEnabled())return false;
  const Vector2d* nearest=nullptr;double distance=.5+1e-12;
  for(const auto& candidate:ed_->frontier_averages_)
    if((candidate-goal).norm()<=distance) {distance=(candidate-goal).norm();nearest=&candidate;}
  FrontierCycleMemory::Point context;
  if(!nearest || !realCycleContext(context))return false;
  decision={{{robot.x(),robot.y()}},{{nearest->x(),nearest->y()}},context,step,action};return true;
}

void ExplorationManager::confirmCycleDecision(int previous_step,int executed_action,const std::string& episode) {
  if(!cycleMemoryEnabled())return;
  const auto d=cycle_memory_.pending();
  if(cycle_memory_.confirm(previous_step,executed_action))
    ROS_INFO("[FRONTIER_CYCLE] stage=commit episode=%s step=%d robot=(%.3f,%.3f) candidate=(%.3f,%.3f) top2=(%.6f,%.6f) history=%zu",episode.c_str(),previous_step,
        d.robot[0],d.robot[1],d.candidate[0],d.candidate[1],d.context[0],d.context[1],cycle_memory_.size());
}

void ExplorationManager::stageCycleDecision(const Vector2d& robot,int step,int action,const std::string& episode) {
  if(!cycleMemoryEnabled())return;
  FrontierCycleMemory::Decision d;
  if(action<=0 || !cycleDecision(robot,ed_->next_pos_,step,action,d)) {
    cycle_memory_.clearPending();
    ROS_INFO("[FRONTIER_CYCLE] stage=skip_association episode=%s step=%d goal=(%.3f,%.3f)",episode.c_str(),step,ed_->next_pos_.x(),ed_->next_pos_.y());return;
  }
  if(cycle_memory_.continuing(d))
    ROS_INFO("[FRONTIER_CYCLE] stage=continue episode=%s step=%d robot=(%.3f,%.3f) candidate=(%.3f,%.3f) top2=(%.6f,%.6f)",
        episode.c_str(),step,robot.x(),robot.y(),d.candidate[0],d.candidate[1],d.context[0],d.context[1]);
  cycle_memory_.stage(d);
}

bool ExplorationManager::rejectCycleGoal(const Vector2d& robot,int step,const std::string& episode) {
  FrontierCycleMemory::Decision d;
  if(!cycleDecision(robot,ed_->next_pos_,step,1,d))return false;
  const bool cyclic=cycle_memory_.cyclic(d);
  if(cyclic)ROS_INFO("[FRONTIER_CYCLE] stage=avic_reject_context episode=%s step=%d robot=(%.3f,%.3f) candidate=(%.3f,%.3f) top2=(%.6f,%.6f)",
      episode.c_str(),step,robot.x(),robot.y(),d.candidate[0],d.candidate[1],d.context[0],d.context[1]);
  return cyclic;
}

void ExplorationManager::suppressCycleGoal(const Vector2d& robot,int step,const std::string& episode) {
  if(!cycleMemoryEnabled())return;
  FrontierCycleMemory::Decision original;
  if(!cycleDecision(robot,ed_->next_pos_,step,1,original)) {
    ROS_INFO("[FRONTIER_CYCLE] stage=skip_association episode=%s step=%d",episode.c_str(),step);return;
  }
  if(!cycle_memory_.cyclic(original))return;
  const Vector2d old_goal=ed_->next_pos_;const auto old_path=ed_->next_best_path_;const auto old_tour=ed_->tsp_tour_;
  vector<Vector2d> candidates;
  for(const auto& candidate:ed_->frontier_averages_) {
    FrontierCycleMemory::Decision d={original.robot,{{candidate.x(),candidate.y()}},original.context,step,1};
    if(!cycle_memory_.cyclic(d))candidates.push_back(candidate);
  }
  Vector2d goal=old_goal;vector<Vector2d> path;bool replaced=false;
  if(!candidates.empty()) {
    try {
      ed_->tsp_tour_.clear();
      chooseExplorationPolicy(robot,candidates,goal,path);
      FrontierCycleMemory::Decision d;
      replaced=!path.empty() && cycleDecision(robot,goal,step,1,d) && !cycle_memory_.cyclic(d);
    } catch(const std::exception& exc) {
      ed_->next_pos_=old_goal;ed_->next_best_path_=old_path;ed_->tsp_tour_=old_tour;
      ROS_ERROR("[FRONTIER_CYCLE] stage=policy_error episode=%s step=%d restored=1 error=%s",episode.c_str(),step,exc.what());
      throw;
    }
  }
  if(replaced) {ed_->next_pos_=goal;ed_->next_best_path_=path;}
  else {ed_->next_pos_=old_goal;ed_->next_best_path_=old_path;ed_->tsp_tour_=old_tour;}
  ROS_INFO("[FRONTIER_CYCLE] stage=%s episode=%s step=%d robot=(%.3f,%.3f) original=(%.3f,%.3f) candidate=(%.3f,%.3f) final=(%.3f,%.3f) top2=(%.6f,%.6f) filtered=%zu",
      replaced?"suppress":"fallback",episode.c_str(),step,robot.x(),robot.y(),old_goal.x(),old_goal.y(),original.candidate[0],original.candidate[1],
      ed_->next_pos_.x(),ed_->next_pos_.y(),original.context[0],original.context[1],ed_->frontier_averages_.size()-candidates.size());
}

bool ExplorationManager::recoveryPath(const Vector2d& pos, const Vector2d& goal,
    bool alternate, Vector2d& selected, vector<Vector2d>& path)
{
  path.clear();
  if (!alternate) return searchFrontierPath(pos, goal, selected, path) && !path.empty();
  vector<Vector2d> candidates;
  for (const auto& frontier : ed_->frontier_averages_)
    if ((frontier - goal).norm() > 0.5) candidates.push_back(frontier);
  if (candidates.empty()) return false;
  // Same configured policy and real map; no score/map writes, no object fallback.
  const auto saved_tour = ed_->tsp_tour_;
  chooseExplorationPolicy(pos, candidates, selected, path);
  ed_->tsp_tour_ = saved_tour;
  return !path.empty() && (selected - goal).norm() > 0.5;
}

bool ExplorationManager::recoveryToPoint(const Vector2d& pos, const Vector2d& goal,
    vector<Vector2d>& path)
{
  path.clear();
  if (sdf_map_->getOccupancy(goal)!=SDFMap2D::FREE || sdf_map_->getInflateOccupancy(goal)==1)
    return false;
  path_finder_->reset();
  if(path_finder_->astarSearch(pos,goal,0.15,0.05)!=Astar2D::REACH_END) return false;
  path=path_finder_->getPath();
  return !path.empty();
}

// TriHelper-inspired retreat. Only the observed, robot-connected free component;
// never use a privileged simulator map or a centroid inside an obstacle.
bool ExplorationManager::openSpaceRecovery(const Vector2d& pos, Vector2d& goal,
    vector<Vector2d>& path)
{
  Vector2i start; sdf_map_->posToIndex(pos,start);
  std::queue<Vector2i> queue;
  std::set<int> visited;
  queue.push(start); visited.insert(sdf_map_->toAddress(start));
  vector<Vector2d> cells;
  const Vector2i delta[4]={Vector2i(1,0),Vector2i(-1,0),Vector2i(0,1),Vector2i(0,-1)};
  while(!queue.empty() && visited.size()<40000) {
    const Vector2i id=queue.front();queue.pop();
    Vector2d point;sdf_map_->indexToPos(id,point);
    if(sdf_map_->getOccupancy(id)==SDFMap2D::FREE && sdf_map_->getInflateOccupancy(id)!=1)
      cells.push_back(point);
    for(const auto& offset:delta) {
      Vector2i next=id+offset;
      if(!sdf_map_->isInMap(next))continue;
      Vector2d p;sdf_map_->indexToPos(next,p);
      if((p-pos).norm()>6. || sdf_map_->getOccupancy(next)!=SDFMap2D::FREE ||
          sdf_map_->getInflateOccupancy(next)==1)continue;
      if(visited.insert(sdf_map_->toAddress(next)).second)queue.push(next);
    }
  }
  if(cells.empty())return false;
  Vector2d centroid=Vector2d::Zero();for(const auto& p:cells)centroid+=p;
  centroid/=double(cells.size());
  vector<Vector2d> candidates;
  for(const auto& p:cells)
    if((p-pos).norm()>=0.75 && sdf_map_->getDistance(p)>=0.3)candidates.push_back(p);
  std::sort(candidates.begin(),candidates.end(),[&](const Vector2d& a,const Vector2d& b){
    const double x=(a-centroid).squaredNorm(),y=(b-centroid).squaredNorm();
    if(x!=y)return x<y;
    return a.x()!=b.x()?a.x()<b.x():a.y()<b.y();
  });
  for(size_t i=0;i<std::min<size_t>(20,candidates.size());++i) {
    if(recoveryToPoint(pos,candidates[i],path)) {
      goal=candidates[i];
      ROS_INFO("[OPEN_ESCAPE_MAP] component=%zu centroid=(%.3f,%.3f) selected=(%.3f,%.3f) clearance=%.3f",
          cells.size(),centroid.x(),centroid.y(),goal.x(),goal.y(),sdf_map_->getDistance(goal));
      return true;
    }
  }
  return false;
}

bool ExplorationManager::planConfirmedObject(const Vector3d& pos)
{
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> clouds;
  // Existing target-label, confidence and observation-count gates; no lowering.
  object_map2d_->getTopConfidenceObjectCloud(clouds);
  for (const auto& cloud : clouds) {
    Vector2d goal;
    vector<Vector2d> path;
    if (searchObjectPath(pos, cloud, goal, path) && !path.empty()) {
      const char* trial = std::getenv("APEXNAV_LITERATURE_METHOD");
      if (trial && std::string(trial)=="reperception" &&
          unconfirmed_exclusions_.contains(goal, confirmation_step_)) continue;
      ed_->next_pos_ = goal;
      ed_->next_best_path_ = path;
      review_object_id_=object_map2d_->reviewInstance(*cloud);
      ed_->tsp_tour_.clear();
      return true;
    }
  }
  return false;  // retain the previous search point while taking real views
}

bool ExplorationManager::searchUnconfirmedObjectPath(const Vector3d& pos,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& cloud, bool extreme)
{
  const bool found = extreme ? searchObjectPathExtreme(pos, cloud, ed_->next_pos_, ed_->next_best_path_) :
      searchObjectPath(pos, cloud, ed_->next_pos_, ed_->next_best_path_);
  if (found && unconfirmed_exclusions_.contains(ed_->next_pos_, confirmation_step_)) {
    ROS_INFO("[TARGET_CONFIRM] stage=excluded step=%d goal=(%.3f,%.3f)",
        confirmation_step_, ed_->next_pos_.x(), ed_->next_pos_.y());
    ed_->next_best_path_.clear();
    return false;
  }
  return found;
}

int ExplorationManager::planNextBestPoint(const Vector3d& pos, const double& yaw)
{
  review_object_id_=-1;
  Vector2d pos2d = Vector2d(pos(0), pos(1));
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;

  // Clear previous planning results
  ed_->tsp_tour_.clear();
  ed_->next_best_path_.clear();
  vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> object_clouds;
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds);

  // ==================== Navigation Mode: High-Confidence Objects ====================
  if (!object_clouds.empty()) {
    ROS_WARN("[Navigation Mode] Get object_cloud num = %ld", object_clouds.size());

    // Try to find path to each detected object in order of confidence
    for (auto object_cloud : object_clouds) {
      if (searchObjectPath(pos, object_cloud, ed_->next_pos_, ed_->next_best_path_)) {
        const char* trial = std::getenv("APEXNAV_LITERATURE_METHOD");
        if (trial && std::string(trial)=="reperception" &&
            unconfirmed_exclusions_.contains(ed_->next_pos_, confirmation_step_)) {
          ed_->next_best_path_.clear();
          continue;
        }
        review_object_id_=object_map2d_->reviewInstance(*object_cloud);
        return SEARCH_BEST_OBJECT;
      }
    }
  }

  // ==================== Navigation Mode: Over-Depth Objects ====================
  if (!object_map2d_->over_depth_object_cloud_->points.empty()) {
    ROS_WARN("[Navigation Mode (Over Depth)] Get over depth object cloud");
    if (searchUnconfirmedObjectPath(pos, object_map2d_->over_depth_object_cloud_))
      return SEARCH_OVER_DEPTH_OBJECT;
  }

  // ==================== Exploration Mode: Frontier-Based Planning ====================
  sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false);
  pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (object_clouds.size() >= 1)
    top_object_cloud = object_clouds[0];

  // Apply selected exploration policy to choose next frontier
  Eigen::Vector2d next_best_pos;
  std::vector<Eigen::Vector2d> next_best_path;
  chooseExplorationPolicy(pos2d, ed_->frontier_averages_, next_best_pos, next_best_path);

  // Handle case when no passable frontiers are found
  if (next_best_path.empty()) {
    ROS_WARN("Maybe no passable frontier.");

    // Try suspicious objects as backup
    if (!top_object_cloud->points.empty() &&
        searchUnconfirmedObjectPath(pos, top_object_cloud))
      return SEARCH_SUSPICIOUS_OBJECT;
    else
      // Try dormant frontiers as last resort
      chooseExplorationPolicy(
          pos2d, ed_->dormant_frontier_averages_, next_best_pos, next_best_path);

    // Extreme search mode when all normal options fail
    if (next_best_path.empty()) {
      ROS_ERROR("search exterme case!!!");

      // Try extreme object search with relaxed constraints
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchUnconfirmedObjectPath(pos, object_cloud, true))
          return SEARCH_EXTREME;
      }

      // Include lower confidence objects in extreme search
      sdf_map_->object_map2d_->getTopConfidenceObjectCloud(object_clouds, false, true);
      for (auto object_cloud : object_clouds) {
        if (!object_cloud->points.empty() &&
            searchUnconfirmedObjectPath(pos, object_cloud, true))
          return SEARCH_EXTREME;
      }

      // Try cached over-depth objects as final option
      if (!last_over_depth_object_cloud_ || !object_map2d_->over_depth_object_cloud_->points.empty())
        last_over_depth_object_cloud_ = object_map2d_->over_depth_object_cloud_;

      if (!last_over_depth_object_cloud_->points.empty() &&
          searchUnconfirmedObjectPath(pos, last_over_depth_object_cloud_, true)) {
        return SEARCH_EXTREME;
      }
    }

    // Final error handling when no valid targets exist
    if (next_best_path.empty()) {
      if (ed_->frontiers_.empty()) {
        ROS_ERROR("No coverable frontier!!");
        return NO_COVERABLE_FRONTIER;
      }
      else {
        ROS_ERROR("No passable frontier!!");
        return NO_PASSABLE_FRONTIER;
      }
    }
  }

  // Store successful planning results
  ed_->next_pos_ = next_best_pos;
  ed_->next_best_path_ = next_best_path;

  // Performance monitoring
  double total_time = (ros::Time::now() - t2).toSec();
  ROS_ERROR_COND(total_time > 0.25, "[Plan NBV] Total time %.2lf s too long!!!", total_time);

  return EXPLORATION;
}

int ExplorationManager::planNextBestPointWithTemporaryScores(const Vector3d& pos,
    const double& yaw, const vector<TemporaryFrontierScore>& scores, double match_tolerance,
    int& matched_score_count)
{
  temporary_frontier_scores_ = scores;
  temporary_score_match_tolerance_ = match_tolerance;
  temporary_score_match_count_ = 0;
  temporary_scores_active_ = !temporary_frontier_scores_.empty();

  int result;
  try { result = planNextBestPoint(pos, yaw); }
  catch (...) {
    temporary_scores_active_=false;temporary_frontier_scores_.clear();temporary_score_match_count_=0;
    throw;
  }
  matched_score_count = temporary_score_match_count_;

  // Imagination is scoped to exactly one planner decision.  It is never
  // retained in the Semantic Score Map or reused after the candidate snapshot.
  temporary_scores_active_ = false;
  temporary_frontier_scores_.clear();
  temporary_score_match_count_ = 0;
  return result;
}

void ExplorationManager::chooseExplorationPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  if (temporary_scores_active_ && textFrontierEnabled()) {
    vector<SemanticFrontier> reachable;
    getSortedSemanticFrontiers(cur_pos,frontiers,reachable);
    // Probabilities and unscored BLIP similarities must NEVER compete as if
    // they had the same scale. This query only ranks scored reachable branches.
    reachable.erase(std::remove_if(reachable.begin(),reachable.end(),
        [](const SemanticFrontier& f){return !f.score_augmented;}),reachable.end());
    if (!reachable.empty()) {
      const auto best=std::max_element(reachable.begin(),reachable.end(),
          [](const SemanticFrontier& a,const SemanticFrontier& b){
            if(a.semantic_value!=b.semantic_value)return a.semantic_value<b.semantic_value;
            if(a.real_semantic_value!=b.real_semantic_value)return a.real_semantic_value<b.real_semantic_value;
            return a.path_length>b.path_length;
          });
      next_best_path=best->path;next_best_pos=best->path.back();ed_->tsp_tour_.clear();
      ROS_INFO("[TEXT_FRONTIER_SELECT] candidate=(%.3f,%.3f) goal=(%.3f,%.3f) belief=%.6f real=%.6f scored_reachable=%zu",
          best->position.x(),best->position.y(),next_best_pos.x(),next_best_pos.y(),
          best->semantic_value,best->real_semantic_value,reachable.size());
      return;
    }
    ROS_WARN("[TEXT_FRONTIER_SELECT] fallback=original_policy reason=no_reachable_scored_frontier");
  }
  if (temporary_scores_active_ && imageGLMStrongEnabled()) {
    vector<SemanticFrontier> reachable;
    getSortedSemanticFrontiers(cur_pos,frontiers,reachable);
    const bool matched=std::any_of(reachable.begin(),reachable.end(),
        [](const SemanticFrontier& frontier){return frontier.score_augmented;});
    if (matched) {
      const auto best=std::max_element(reachable.begin(),reachable.end(),
          [](const SemanticFrontier& a,const SemanticFrontier& b){
            if(a.semantic_value!=b.semantic_value)return a.semantic_value<b.semantic_value;
            return a.path_length>b.path_length;
          });
      next_best_path=best->path;
      next_best_pos=best->path.back(); // A* reachable endpoint, not an invented goal.
      ed_->tsp_tour_.clear();
      ROS_INFO("[IMAGE_GLM_STRONG_SELECT] selected_candidate=(%.3f,%.3f) reachable_goal=(%.3f,%.3f) fused_score=%.6f real=%.6f matched=%d reachable_count=%zu reason=highest_current_fused_semantic bypass=Hybrid_TSP_Closest",
          best->position.x(),best->position.y(),next_best_pos.x(),next_best_pos.y(),best->semantic_value,
          best->real_semantic_value,best->score_augmented,reachable.size());
      return;
    }
    ROS_INFO("[IMAGE_GLM_STRONG] fallback=original_policy reason=no_reachable_matched_temporary_score");
  }
  switch (ep_->policy_mode_) {
    case ExplorationParam::DISTANCE:
      ROS_WARN("[Exploration Mode] Find Closest Frontier");
      findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::SEMANTIC:
      ROS_WARN("[Exploration Mode] Find Highest Semantic Value Frontier");
      findHighestSemanticsFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::HYBRID:
      ROS_WARN("[Exploration Mode] Working on Hybrid Mode");
      hybridExplorePolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    case ExplorationParam::TSP_DIST:
      ROS_WARN("[Exploration Mode] Working on TSP Distance Mode");
      findTSPTourPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
      break;

    default:
      ROS_WARN("[Exploration Mode] Unknown Mode");
      break;
  }
}

bool ExplorationManager::selectReachableFrontier(const Vector2d& cur_pos,
    const Vector2d& requested_pos, double match_tolerance, Vector2d& selected_pos,
    vector<Vector2d>& selected_path)
{
  vector<SemanticFrontier> semantic_frontiers;
  getSortedSemanticFrontiers(cur_pos, ed_->frontier_averages_, semantic_frontiers);
  if (semantic_frontiers.empty())
    return false;

  const SemanticFrontier* nearest = nullptr;
  double nearest_distance = std::numeric_limits<double>::max();
  for (const auto& frontier : semantic_frontiers) {
    const double distance = (frontier.position - requested_pos).norm();
    if (distance < nearest_distance) {
      nearest = &frontier;
      nearest_distance = distance;
    }
  }
  if (nearest == nullptr || nearest_distance > match_tolerance || nearest->path.empty())
    return false;

  // The path originates from ApexNav's real occupancy map and A* search.  The
  // temporary imagination score is intentionally absent from this method and
  // can never modify the Semantic Score Map or bypass reachability checks.
  selected_pos = nearest->position;
  selected_path = nearest->path;
  return true;
}

void ExplorationManager::hybridExplorePolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  double std_dev_threshold = ep_->sigma_threshold_;
  double max_to_mean_threshold = ep_->max_to_mean_threshold_;
  vector<SemanticFrontier> sem_frontiers;
  getSortedSemanticFrontiers(cur_pos, frontiers, sem_frontiers);
  if (sem_frontiers.empty())
    return;

  double std_dev, max_to_mean, mean;
  calcSemanticFrontierInfo(sem_frontiers, std_dev, max_to_mean, mean);

  const int augmented_count = static_cast<int>(std::count_if(sem_frontiers.begin(),
      sem_frontiers.end(), [](const SemanticFrontier& frontier) {
        return frontier.score_augmented;
      }));
  if (augmented_count > 0) {
    ROS_INFO("[AVIC_SCORE] Hybrid input augmented=%d sigma=%.6f max_to_mean=%.6f mean=%.6f",
        augmented_count, std_dev, max_to_mean, mean);
  }

  // Decide between exploitation and exploration based on semantic statistics
  if (std_dev > std_dev_threshold && max_to_mean > max_to_mean_threshold) {
    ROS_WARN("Exploit the semantic value (TSP)!!");
    vector<Vector2d> high_sem_frontiers;

    // Select high-value frontiers for TSP optimization
    for (auto sem_frontier : sem_frontiers) {
      double auto_max_to_mean_threshold =
          max(max_to_mean_threshold, ep_->max_to_mean_percentage_ * max_to_mean);
      if (sem_frontier.semantic_value / mean < auto_max_to_mean_threshold)
        break;
      high_sem_frontiers.push_back(sem_frontier.position);
    }
    findTSPTourPolicy(cur_pos, high_sem_frontiers, next_best_pos, next_best_path);
  }
  else {
    ROS_WARN("Explore the environment (Closest)!!");
    findClosestFrontierPolicy(cur_pos, frontiers, next_best_pos, next_best_path);
  }
}

void ExplorationManager::findHighestSemanticsFrontierPolicy(Vector2d cur_pos,
    vector<Vector2d> frontiers, Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Container for frontier-value pairs for sorting
  vector<pair<Vector2d, double>> frontier_values;

  // Compute semantic value for each frontier
  for (auto frontier : frontiers) {
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 neighborhood

    // Find maximum semantic value in local neighborhood
    double value = sdf_map_->value_map_->getValue(idx);
    for (auto nbr : nbrs) value = max(value, sdf_map_->value_map_->getValue(nbr));

    frontier_values.emplace_back(frontier, value);
  }

  // Sort by semantic value (descending), then by distance (ascending)
  auto compareFrontiers = [&cur_pos](
                              const pair<Vector2d, double>& a, const pair<Vector2d, double>& b) {
    if (fabs(a.second - b.second) > 1e-5) {
      return a.second > b.second;  // Higher semantic value first
    }
    else {
      double dist_a = (a.first - cur_pos).norm();
      double dist_b = (b.first - cur_pos).norm();
      return dist_a < dist_b;  // Closer distance first for tie-breaking
    }
  };

  std::sort(frontier_values.begin(), frontier_values.end(), compareFrontiers);

  // Update frontier list with sorted order
  frontiers.clear();
  for (const auto& fv : frontier_values) {
    frontiers.push_back(fv.first);
  }

  // Select first reachable frontier from sorted list
  for (int i = 0; i < (int)frontiers.size(); i++) {
    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;
    next_best_pos = tmp_pos;
    next_best_path = tmp_path;
    break;
  }
}

void ExplorationManager::findClosestFrontierPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();

  // Sort frontiers by Euclidean distance for efficient processing
  std::sort(frontiers.begin(), frontiers.end(), [&cur_pos](const Vector2d& a, const Vector2d& b) {
    return (a - cur_pos).norm() < (b - cur_pos).norm();
  });

  double min_len = std::numeric_limits<double>::max();

  // Find the frontier with shortest actual path length
  for (int i = 0; i < (int)frontiers.size(); i++) {
    // Skip if Euclidean distance already exceeds best path length
    if ((frontiers[i] - cur_pos).norm() >= min_len)
      continue;

    std::vector<Eigen::Vector2d> tmp_path;
    Eigen::Vector2d tmp_pos;

    // Attempt path planning to this frontier
    if (!searchFrontierPath(cur_pos, frontiers[i], tmp_pos, tmp_path))
      continue;

    // Update best solution if this path is shorter
    double len = Astar2D::pathLength(tmp_path);
    if (len < min_len) {
      min_len = len;
      next_best_pos = tmp_pos;
      next_best_path = tmp_path;
    }
  }
}

void ExplorationManager::findTSPTourPolicy(Vector2d cur_pos, vector<Vector2d> frontiers,
    Vector2d& next_best_pos, vector<Vector2d>& next_best_path)
{
  next_best_path.clear();
  vector<Vector2d> filter_frontiers;
  for (auto frontier : frontiers) {
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path))
      filter_frontiers.push_back(frontier);
  }

  vector<int> indices;
  computeATSPTour(cur_pos, filter_frontiers, indices);
  ed_->tsp_tour_.push_back(cur_pos);
  for (auto idx : indices) ed_->tsp_tour_.push_back(filter_frontiers[idx]);

  if (!indices.empty()) {
    for (auto idx : indices) {
      Vector2d next_bext_frontier = filter_frontiers[idx];
      if (searchFrontierPath(cur_pos, next_bext_frontier, next_best_pos, next_best_path))
        break;
    }
  }
}

double ExplorationManager::computePathCost(const Vector2d& pos1, const Vector2d& pos2)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(pos1, pos2, 0.25, 0.002) == Astar2D::REACH_END)
    return Astar2D::pathLength(path_finder_->getPath());
  return 10000.0;
}

void ExplorationManager::computeATSPCostMatrix(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, Eigen::MatrixXd& mat)
{
  int dimen = frontiers.size() + 1;
  mat.resize(dimen, dimen);

  // Agent to frontiers
  for (int i = 1; i < dimen; i++) {
    mat(0, i) = computePathCost(cur_pos, frontiers[i - 1]);
    mat(i, 0) = 0;
  }

  // Costs between frontiers
  for (int i = 1; i < dimen; ++i) {
    for (int j = i + 1; j < dimen; ++j) {
      double cost = computePathCost(frontiers[i - 1], frontiers[j - 1]);
      mat(i, j) = cost;
      mat(j, i) = cost;
    }
  }

  // Diag
  for (int i = 0; i < dimen; ++i) {
    mat(i, i) = 100000.0;
  }
}

void ExplorationManager::computeATSPTour(
    const Vector2d& cur_pos, const vector<Vector2d>& frontiers, vector<int>& indices)
{
  indices.clear();
  if (frontiers.empty()) {
    ROS_ERROR("No frontier to compute tsp!");
    return;
  }
  else if (frontiers.size() == 1) {
    indices.push_back(0);
    return;
  }
  /* change ATSP to lhk3 */
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  computeATSPCostMatrix(cur_pos, frontiers, cost_mat);
  const int dimension = cost_mat.rows();

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Initialize ATSP par file
  // Create problem file
  ofstream file(ep_->tsp_dir_ + "/atsp_tour.atsp");
  if (!file) throw std::runtime_error("TSP problem file is not writable: " + ep_->tsp_dir_);
  file << "NAME : amtsp\n";
  file << "TYPE : ATSP\n";
  file << "DIMENSION : " + to_string(dimension) + "\n";
  file << "EDGE_WEIGHT_TYPE : EXPLICIT\n";
  file << "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n";
  file << "EDGE_WEIGHT_SECTION\n";
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = 100 * cost_mat(i, j);
      file << int_cost << " ";
    }
    file << "\n";
  }
  file.close();

  // Create par file
  if (!file) throw std::runtime_error("Failed to write TSP problem file: " + ep_->tsp_dir_);
  const int drone_num = 1;
  file.open(ep_->tsp_dir_ + "/atsp_tour.par");
  if (!file) throw std::runtime_error("TSP parameter file is not writable: " + ep_->tsp_dir_);
  file << "SPECIAL\n";
  file << "PROBLEM_FILE = " + ep_->tsp_dir_ + "/atsp_tour.atsp\n";
  file << "SALESMEN = " << to_string(drone_num) << "\n";
  file << "MTSP_OBJECTIVE = MINSUM\n";
  file << "RUNS = 1\n";
  file << "TRACE_LEVEL = 0\n";
  file << "TOUR_FILE = " + ep_->tsp_dir_ + "/atsp_tour.tour\n";
  file.close();

  if (!file) throw std::runtime_error("Failed to write TSP parameter file: " + ep_->tsp_dir_);
  ros::NodeHandle nh;
  indices = solveTspWithRecovery(nh, ep_->tsp_dir_ + "/atsp_tour.tour", frontiers.size());

  // for (auto idx : indices) ROS_WARN("ATSP idx = %d", idx);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("[ATSP Tour] Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}

Vector2d ExplorationManager::findNearestObjectPoint(
    const Vector3d& start, const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud)
{
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(object_cloud);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);

  pcl::PointXYZ cur_pt;
  cur_pt.x = start(0);
  cur_pt.y = start(1);
  cur_pt.z = start(2);

  if (kdtree.nearestKSearch(cur_pt, 1, pointIdxNKNSearch, pointNKNSquaredDistance) <= 0) {
    ROS_ERROR("[Bug] No nearest object point found.");
    return Vector2d(-1000.0, -1000.0);  // Error indicator
  }

  int nearest_idx = pointIdxNKNSearch[0];
  auto nearest_point = object_cloud->points[nearest_idx];
  return Vector2d(nearest_point.x, nearest_point.y);
}

bool ExplorationManager::trySearchObjectPathWithDistance(const Vector2d& start2d,
    const Vector2d& object_pose, double distance, double max_search_time,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path,
    const std::string& debug_msg)
{
  path_finder_->reset();
  if (path_finder_->astarSearch(start2d, object_pose, distance, max_search_time) ==
      Astar2D::REACH_END) {
    std::vector<Eigen::Vector2d> path = path_finder_->getPath();
    Vector2d tmp_pos(-1000.0, -1000.0);

    // Find valid position along the path (from end to start)
    for (int i = path.size() - 1; i >= 0; i--) {
      if (sdf_map_->getOccupancy(path[i]) != SDFMap2D::OCCUPIED &&
          sdf_map_->getOccupancy(path[i]) != SDFMap2D::UNKNOWN &&
          sdf_map_->getInflateOccupancy(path[i]) != 1) {
        tmp_pos = path[i];
        break;
      }
    }

    // Search path to the valid position
    path_finder_->reset();
    if (path_finder_->astarSearch(start2d, tmp_pos, 0.2, max_search_time) == Astar2D::REACH_END) {
      refined_path = path_finder_->getPath();
      refined_pos = tmp_pos;
      if (!debug_msg.empty()) {
        ROS_WARN("%s", debug_msg.c_str());
      }
      return true;
    }
  }
  return false;
}

bool ExplorationManager::searchObjectPath(const Vector3d& start,
    const pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_cloud,
    Eigen::Vector2d& refined_pos, std::vector<Eigen::Vector2d>& refined_path)
{
  const double max_search_time = 0.2;  // Maximum planning time per attempt
  Vector2d start2d = Vector2d(start(0), start(1));

  // Find nearest accessible point in object cloud
  Vector2d object_pose = findNearestObjectPoint(start, object_cloud);
  if (object_pose.x() < -999.0)
    return false;  // Error indicator from findNearestObjectPoint

  // Try different safety distances in order of preference
  const std::vector<double> distances = { 0.5, 0.70, 0.85 };
  const std::vector<std::string> debug_messages = { "I'm going to the object! dist = 0.5m!",
    "I'm going to the object! dist = 0.70m!", "I'm going to the object! dist = 0.85m!" };

  // Attempt path planning with each safety distance
  for (size_t i = 0; i < distances.size(); ++i) {
    if (trySearchObjectPathWithDistance(start2d, object_pose, distances[i], max_search_time,
            refined_pos, refined_path, debug_messages[i])) {
      return true;
    }
  }

  ROS_ERROR("Failed to find object path.");
  return false;
}

void ExplorationManager::getSortedSemanticFrontiers(const Vector2d& cur_pos,
    const vector<Vector2d>& frontiers, vector<SemanticFrontier>& sem_frontiers)
{
  // Filter and sort frontiers based on semantic values and reachability
  sem_frontiers.clear();

  for (auto& frontier : frontiers) {
    SemanticFrontier sem_frontier;
    sem_frontier.position = frontier;

    // Compute semantic value from local neighborhood
    Vector2i idx;
    sdf_map_->posToIndex(frontier, idx);
    auto nbrs = allNeighbors(idx, 2);  // 5x5 grid neighborhood
    double value = sdf_map_->value_map_->getValue(idx);
    double evidence_sum = sdf_map_->value_map_->getObservationEvidence(idx);
    int evidence_count = 1;

    // Find maximum semantic value in neighborhood (ignoring occupied cells)
    for (auto& nbr : nbrs) {
      if (sdf_map_->getInflateOccupancy(nbr) == 1 ||
          sdf_map_->getOccupancy(nbr) == SDFMap2D::OCCUPIED)
        continue;
      value = std::max(value, sdf_map_->value_map_->getValue(nbr));
      evidence_sum += sdf_map_->value_map_->getObservationEvidence(nbr);
      ++evidence_count;
    }
    sem_frontier.semantic_value = value;
    sem_frontier.real_semantic_value = value;
    sem_frontier.observation_evidence = evidence_sum / static_cast<double>(evidence_count);

    if (temporary_scores_active_) {
      const TemporaryFrontierScore* matched = nullptr;
      double nearest_distance = std::numeric_limits<double>::max();
      for (const auto& candidate_score : temporary_frontier_scores_) {
        const double distance = (candidate_score.position - frontier).norm();
        if (distance < nearest_distance) {
          nearest_distance = distance;
          matched = &candidate_score;
        }
      }
      if (matched != nullptr && nearest_distance <= temporary_score_match_tolerance_) {
        sem_frontier.semantic_value = matched->decision_semantic_value;
        sem_frontier.imagined_semantic_value = matched->imagined_semantic_value;
        sem_frontier.score_augmented = true;
        ++temporary_score_match_count_;
        ROS_INFO("[AVIC_SCORE] frontier=(%.3f, %.3f) real_map=%.6f real_request=%.6f "
                 "imagined_itm=%.6f decision=%.6f match_distance=%.3f",
            frontier(0), frontier(1), value, matched->real_semantic_value,
            matched->imagined_semantic_value, matched->decision_semantic_value, nearest_distance);
      }
    }

    // Validate reachability and compute path cost
    Vector2d tmp_pos;
    vector<Vector2d> tmp_path;
    if (!searchFrontierPath(cur_pos, frontier, tmp_pos, tmp_path)) {
      // Assign high cost penalty for unreachable frontiers
      sem_frontier.path_length = 1000000;
      sem_frontier.path.clear();
    }
    else {
      sem_frontier.path_length = Astar2D::pathLength(tmp_path);
      sem_frontier.path = tmp_path;
    }

    // Only include frontiers with valid paths
    if (!sem_frontier.path.empty())
      sem_frontiers.push_back(sem_frontier);
  }

  // Sort by semantic value (desc) then by path length (asc)
  std::sort(sem_frontiers.begin(), sem_frontiers.end());
}

void ExplorationManager::calcSemanticFrontierInfo(const vector<SemanticFrontier>& sem_frontiers,
    double& std_dev, double& max_to_mean, double& mean, bool if_print)
{
  // Handle empty frontier list
  if (sem_frontiers.empty()) {
    std::cout << "No semantic frontiers available." << std::endl;
    max_to_mean = 1.0;  // Neutral ratio
    std_dev = 0.0;      // No variation
    return;
  }

  // Compute mean and maximum semantic values
  double sum = 0.0;
  double max_value = 0.0;
  for (const auto& frontier : sem_frontiers) {
    sum += frontier.semantic_value;
    max_value = max(max_value, frontier.semantic_value);
  }
  mean = sum / sem_frontiers.size();

  // Compute standard deviation
  double variance_sum = 0.0;
  for (const auto& frontier : sem_frontiers)
    variance_sum += (frontier.semantic_value - mean) * (frontier.semantic_value - mean);

  max_to_mean = max_value / mean;
  std_dev = std::sqrt(variance_sum / sem_frontiers.size());

  // Print summary statistics
  std::cout << "Mean Value: " << std::fixed << std::setprecision(3) << mean;
  std::cout << " , Standard Deviation: " << std::fixed << std::setprecision(3) << std_dev;
  std::cout << " , Max-to-Mean: " << std::fixed << std::setprecision(3) << max_to_mean << std::endl;

  // Print detailed frontier values if requested
  if (if_print) {
    for (const auto& sem_frontier : sem_frontiers)
      std::cout << "Value: " << std::fixed << std::setprecision(3) << sem_frontier.semantic_value
                << std::endl;
  }
}

bool ExplorationManager::planTrajectory(
    const Eigen::VectorXd& start, const Eigen::VectorXd& end, const Vector3d& ctrl)
{
  if (!gcopter_ || !kinoastar_) {
    ROS_WARN_THROTTLE(1.0, "[ExplorationManager] GCopter or KinoAstar not initialized for real-world mode");
    return false;
  }
  
  Eigen::VectorXd goal_state, current_state;
  Vector3d control = ctrl;
  goal_state = end;
  current_state = start;

  // Kinodynamic A* search
  kinoastar_->reset();
  kinoastar_->search(goal_state, current_state, control);
  kinoastar_->getKinoNode();
  
  if (kinoastar_->has_path_) {
    kinoastar_->kinoastarFlatPathPub(kinoastar_->flat_trajs_);
    gcopter_->minco_plan();
    std::vector<Trajectory<7, 3>> final_trajes = gcopter_->final_trajes;
    gcopter_->mincoPathPub(gcopter_->final_trajes, gcopter_->final_singuls);
    return true;
  }
  
  return false;
}

}  // namespace apexnav_planner
