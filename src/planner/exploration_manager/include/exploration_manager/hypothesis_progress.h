#pragma once
#include <Eigen/Core>
#include <cmath>

namespace apexnav_planner {
// Fixed reference window: do not reset the goal anchor for tiny map jitter.
// Only uncertain-object pursuit may call observe; confirmed targets are exempt.
class HypothesisProgress {
  bool active_{false};
  Eigen::Vector2d anchor_{Eigen::Vector2d::Zero()};
  double best_{0.0};
  int since_{0}, last_{-1};
public:
  void reset() { *this = HypothesisProgress(); }
  bool observe(int step, const Eigen::Vector2d& robot, const Eigen::Vector2d& goal) {
    if (step < 0 || !robot.allFinite() || !goal.allFinite()) { reset(); return false; }
    if (step <= last_) return false;
    const double distance = (robot-goal).norm();
    if (!active_ || (goal-anchor_).norm() > 0.75) {
      active_=true; anchor_=goal; best_=(robot-anchor_).norm(); since_=step; last_=step;
      return false;
    }
    last_=step;
    const double anchored_distance=(robot-anchor_).norm();
    if (anchored_distance <= best_-0.25) { best_=anchored_distance; since_=step; }
    // Near arrivals remain owned by existing confirmation logic.
    if (distance > 0.75 && step-since_ >= 16) { reset(); return true; }
    return false;
  }
};
}
