#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

namespace apexnav_planner {
// Observation-counted, deterministic and independent of ROS wall time.
// Stage 2 belongs to the same attempt as stage 1 (not a third recovery).
class StagnationRecovery {
public:
  enum Decision { NONE, REPLAN, SWITCH };
  static constexpr int WINDOW = 20, OBSERVE = 10, COOLDOWN = 30, MAX_ATTEMPTS = 2;
  static constexpr double PROGRESS = 0.5, RADIUS = 0.5;

  void interrupt() { active_ = false; pending_ = false; }
  int attempts() const { return attempts_; }

  Decision observe(int step, bool eligible, const Eigen::Vector2d& pos,
      const Eigen::Vector2d& goal, const std::vector<Eigen::Vector2d>& path) {
    if (step <= last_step_) return NONE;
    const bool consecutive = last_step_ < 0 || step == last_step_ + 1;
    last_step_ = step;
    if (!eligible || path.empty()) { interrupt(); return NONE; }
    if (!consecutive || !active_ || (goal - goal_).norm() > RADIUS) {
      begin(step, pos, goal, path);
      return NONE;
    }
    // Measure progress against a FROZEN route, not changing A* lengths.
    // Nearest-segment remaining arc length also works on detours around walls.
    const double distance = remaining(pos, route_);
    if (start_distance_ - distance >= PROGRESS) {
      begin(step, pos, goal, path);
      return NONE;
    }
    positions_.push_back(pos);
    const int elapsed = step - start_step_;
    if (elapsed < (pending_ ? OBSERVE : WINDOW)) return NONE;
    int repeated = 0;
    for (const auto& p : positions_)
      if ((p - pos).norm() <= RADIUS) ++repeated;
    const bool revisiting = repeated * 10 >= static_cast<int>(positions_.size()) * 7;
    if (pending_) {
      pending_ = false;
      active_ = false;
      return revisiting ? SWITCH : NONE;
    }
    if (!revisiting || attempts_ >= MAX_ATTEMPTS || step - last_attempt_ < COOLDOWN)
      return NONE;
    ++attempts_;
    last_attempt_ = step;
    begin(step, pos, goal, path);
    pending_ = true;
    return REPLAN;
  }

  static double remaining(const Eigen::Vector2d& pos,
      const std::vector<Eigen::Vector2d>& path) {
    if (path.empty()) return 0.;
    double tail = 0., nearest = (pos - path.back()).squaredNorm();
    double result = std::sqrt(nearest);
    for (int i = static_cast<int>(path.size()) - 2; i >= 0; --i) {
      const Eigen::Vector2d segment = path[i + 1] - path[i];
      const double length = segment.norm();
      if (length < 1e-9) continue;
      const double t = std::max(0., std::min(1., (pos - path[i]).dot(segment) / (length * length)));
      const double d = (pos - (path[i] + t * segment)).squaredNorm();
      if (d < nearest) { nearest = d; result = std::sqrt(d) + (1. - t) * length + tail; }
      tail += length;
    }
    return result;
  }

private:
  void begin(int step, const Eigen::Vector2d& pos, const Eigen::Vector2d& goal,
      const std::vector<Eigen::Vector2d>& path) {
    active_ = true; pending_ = false; start_step_ = step; goal_ = goal;
    route_ = path; start_distance_ = remaining(pos, route_); positions_ = {pos};
  }
  bool active_{false}, pending_{false};
  int last_step_{-1}, start_step_{0}, last_attempt_{-COOLDOWN}, attempts_{0};
  double start_distance_{0.};
  Eigen::Vector2d goal_{Eigen::Vector2d::Zero()};
  std::vector<Eigen::Vector2d> route_, positions_;
};
}  // namespace apexnav_planner
