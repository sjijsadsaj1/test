#include <exploration_manager/stagnation_recovery.h>
#include <cassert>
#include <iostream>
using apexnav_planner::StagnationRecovery;
using Eigen::Vector2d;
using R = StagnationRecovery;
int main() {
  const Vector2d pos(0, 0), goal(5, 0);
  const std::vector<Vector2d> path{pos, goal};
  R r;
  for (int s = 0; s < 20; ++s) assert(r.observe(s, true, pos, goal, path) == R::NONE);
  assert(r.observe(20, true, pos, goal, path) == R::REPLAN);
  assert(r.observe(20, true, pos, goal, path) == R::NONE); // retry
  for (int s = 21; s < 30; ++s) assert(r.observe(s, true, pos, goal, path) == R::NONE);
  assert(r.observe(30, true, pos, goal, path) == R::SWITCH);
  for (int s = 31; s < 51; ++s) assert(r.observe(s, true, pos, goal, path) == R::NONE);
  assert(r.observe(51, true, pos, goal, path) == R::REPLAN);
  for (int s = 52; s < 61; ++s) assert(r.observe(s, true, pos, goal, path) == R::NONE);
  assert(r.observe(61, true, pos, goal, path) == R::SWITCH);
  for (int s = 62; s < 500; ++s) assert(r.observe(s, true, pos, goal, path) == R::NONE);
  assert(r.attempts() == 2);
  r = R();
  for (int s = 0; s < 100; ++s)
    assert(r.observe(s, true, Vector2d(s * 0.05, 0), goal, path) == R::NONE);
  r = R();
  for (int s = 0; s < 20; ++s) r.observe(s, true, pos, goal, path);
  assert(r.observe(20, false, pos, goal, path) == R::NONE); // pursuit/escape clears window
  for (int s = 21; s < 41; ++s) assert(r.observe(s, true, pos, goal, path) == R::NONE);
  assert(r.observe(41, true, pos, goal, path) == R::REPLAN);
  assert(r.observe(42, true, Vector2d(0.6, 0), goal, path) == R::NONE); // progress cancels switch
  for (int s = 43; s <= 52; ++s)
    assert(r.observe(s, true, Vector2d(0.6, 0), goal, path) == R::NONE);
  r = R();
  for (int s = 0; s < 100; ++s)
    assert(r.observe(s, true, pos, Vector2d(5 + s, 0), path) == R::NONE); // changing targets
  r = R();
  for (int s = 0; s < 20; ++s) r.observe(s, true, pos, goal, path);
  assert(r.observe(22, true, pos, goal, path) == R::NONE); // observation gap
  assert(r.attempts() == 0);
  r = R();
  for (int s = 0; s <= 20; ++s) r.observe(s, true, pos, goal, path);
  r.observe(21, true, Vector2d(0.6, 0), goal, path); // cancel first pending escalation
  for (int s = 22; s < 50; ++s)
    assert(r.observe(s, true, Vector2d(0.6, 0), goal, path) == R::NONE);
  assert(r.observe(50, true, Vector2d(0.6, 0), goal, path) == R::REPLAN); // exact cooldown
  r = R();
  for (int s = 0; s < 100; ++s)
    assert(r.observe(s, true, Vector2d(0, s % 2 ? -1 : 1), goal, path) == R::NONE);
  assert(r.attempts() == 0); // no progress, but no 70% local repetition
  r = R();
  for (int s = 0; s < 20; ++s) r.observe(s, true, pos, goal, path);
  assert(r.observe(20, true, pos, goal, {}) == R::NONE); // empty path clears window
  assert(r.attempts() == 0);
  assert(std::abs(R::remaining(Vector2d(0, 1), {pos, Vector2d(0, 2), Vector2d(2, 2)}) - 3.) < 1e-9);
  assert(R::remaining(pos, {}) == 0.);
  std::cout << "Recovery detector regression checks passed\n";
}
