#include <exploration_manager/target_confirmation.h>
#include <cassert>
#include <limits>

using namespace apexnav_planner;

int main()
{
  TargetConfirmation confirm;
  const Eigen::Vector2d goal(1.0, 2.0);
  assert(confirm.action(0) == -1);
  assert(!confirm.begin(-1, goal));
  assert(!confirm.begin(0, Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(), 0)));
  assert(confirm.begin(347, goal));  // historical premature-stop stage
  const int expected[] = {2, 3, 3, 2};
  for (int i = 0; i < 4; ++i) {
    assert(confirm.action(347 + i) == expected[i]);
    assert(confirm.action(347 + i) != 0 && confirm.action(347 + i) != 1);
    assert(confirm.action(347 + i) == expected[i]);  // duplicate: no budget change
    assert(!confirm.begin(347 + i, Eigen::Vector2d(5, 5)));
  }
  assert(confirm.attempts == 1 && confirm.action(351) == -1);
  confirm.cancel();
  assert(confirm.action(351) == -1 && !confirm.active);
  assert(confirm.begin(355, goal));
  confirm.cancel();  // a newly confirmed real object preempts the scan
  assert(confirm.begin(360, goal));
  confirm.cancel();
  assert(!confirm.begin(400, goal));  // global finite scan budget
  confirm.reset();
  assert(confirm.attempts == 0 && confirm.begin(0, goal));

  UnconfirmedGoalExclusions exclusions;
  exclusions.reject(goal, 351);
  assert(exclusions.contains(goal, 351));
  assert(exclusions.contains(goal + Eigen::Vector2d(0.5, 0), 380));
  assert(!exclusions.contains(goal + Eigen::Vector2d(1, 0), 360));
  assert(!exclusions.contains(goal, 381));
  exclusions.prune(381);
  assert(!exclusions.contains(goal, 382));
  exclusions.reject(goal, 400);
  exclusions = UnconfirmedGoalExclusions();  // new episode must not inherit rejects
  assert(!exclusions.contains(goal, 0));
}
