#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <vector>

namespace apexnav_planner {

// One observation transaction per step. Never emits STOP or moves blindly.
class TargetConfirmation {
public:
  static constexpr int kScanSteps = 4;
  static constexpr int kMaxAttempts = 3;
  bool active{false};
  int start_step{-1}, attempts{0};
  Eigen::Vector2d goal{Eigen::Vector2d::Zero()};

  void reset() { *this = TargetConfirmation(); }
  void cancel() { active = false; }
  bool begin(int step, const Eigen::Vector2d& point) {
    if (active || step < 0 || !point.allFinite() || attempts >= kMaxAttempts) return false;
    active = true;
    start_step = step;
    goal = point;
    ++attempts;
    return true;
  }
  int action(int step) const {
    // +30, back to 0, -30, back to 0. Four fresh real observations.
    static const int turns[kScanSteps] = {2, 3, 3, 2};
    const int index = step - start_step;
    return active && index >= 0 && index < kScanSteps ? turns[index] : -1;
  }
};

class UnconfirmedGoalExclusions {
  struct Entry { Eigen::Vector2d point; int until; };
  std::vector<Entry> entries_;
public:
  static constexpr int kCooldown = 30;
  void reject(const Eigen::Vector2d& point, int step, int cooldown = kCooldown) {
    prune(step);
    entries_.push_back({point, step + cooldown});
  }
  void prune(int step) {
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
        [step](const Entry& e) { return step >= e.until; }), entries_.end());
  }
  bool contains(const Eigen::Vector2d& point, int step) const {
    for (const auto& e : entries_)
      if (step < e.until && (point - e.point).norm() <= 0.75) return true;
    return false;
  }
};

}  // namespace apexnav_planner
