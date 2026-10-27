#include <exploration_manager/avic_route_commitment.h>
#include <cassert>
#include <limits>

using apexnav_planner::AvicRouteCommitment;

int main()
{
  AvicRouteCommitment route;
  assert(!route.active);
  // Audited Tb162-style shared first action: the goal must survive long enough
  // for the next (turning) action, with the initial action included in the budget.
  route.begin(112, Eigen::Vector2d(-6.091, -1.950), 6, {1, 3, 1, 1, 1, 1});
  assert(route.active && route.deadline == 118);
  for (int step = 113; step <= 118; ++step) {
    route.observe(step, step == 114 ? 3 : 1);
    assert(route.executed == step - 112);
    assert(route.matching_prefix == route.executed);
    route.observe(step, 2);  // duplicate must not consume the lease twice
    assert(route.matching_prefix == route.executed);
    assert(route.expired(step) == (step == 118));
  }
  route.reset();
  assert(!route.active && route.expected_actions.empty());
  route.begin(260, Eigen::Vector2d(-1.45, -4.35), 6, {1, 1, 3});
  assert(route.deadline == 263);  // shorter real prefix bounds the lease
  route.observe(261, 1);
  route.observe(262, 3);  // closed-loop safety may disagree; audit, do not force it
  route.observe(263, 3);
  assert(route.executed == 3 && route.matching_prefix == 1);
  route.begin(0, Eigen::Vector2d::Zero(), 0, {1, 2});
  assert(!route.active);  // diagnostic off switch
  route.begin(0, Eigen::Vector2d::Zero(), 99, {1,1,1,1,1,1,1,1});
  assert(route.deadline == 6);
  route.begin(-1, Eigen::Vector2d::Zero(), 6, {1});
  assert(!route.active);
  route.begin(2, Eigen::Vector2d::Zero(), 6, {});
  assert(!route.active);
  route.begin(2, Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0), 6, {1});
  assert(!route.active);
  route.begin(2, Eigen::Vector2d(1,2), 6, {1,1});
  route.reset();  // episode reset / safety cancellation
  route.observe(3, 1);
  assert(!route.active && route.executed == 0);
}
