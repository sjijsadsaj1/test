#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <vector>

namespace apexnav_planner {

// A bounded goal lease, NOT a persisted imagination score or open-loop action plan.
// Observation s selects the action executed before observation s+1 arrives.
class AvicRouteCommitment {
public:
  bool active{false};
  Eigen::Vector2d goal{Eigen::Vector2d::Zero()};
  int start_step{-1}, deadline{-1}, executed{0}, matching_prefix{0};
  std::vector<int> expected_actions;

  void reset() { *this = AvicRouteCommitment(); }

  void begin(int step, const Eigen::Vector2d& target, int max_actions,
      const std::vector<int>& route)
  {
    reset();
    const int budget = std::min({max_actions, 6, static_cast<int>(route.size())});
    if (step < 0 || budget <= 0 || !target.allFinite()) return;
    active = true;
    goal = target;  // fixed origin: matching drift cannot accumulate across steps
    start_step = step;
    deadline = step + budget;
    expected_actions.assign(route.begin(), route.begin() + budget);
  }

  void observe(int step, int action)
  {
    if (!active || step <= start_step || step > deadline) return;
    const int index = step - start_step - 1;
    if (index < executed) return;  // retries do not spend additional budget
    if (index == executed && matching_prefix == executed &&
        expected_actions[index] == action) ++matching_prefix;
    executed = index + 1;
  }

  bool expired(int step) const { return active && step >= deadline; }
};

}  // namespace apexnav_planner
