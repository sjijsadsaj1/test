#pragma once
#include <ros/ros.h>
#include <lkh_mtsp_solver/SolveMTSP.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace apexnav_planner {
// Preserve LKH order; reject partial, stale or malformed output.
inline std::vector<int> readCheckedTspTour(std::istream& input, int frontier_count)
{
  std::string line;
  bool section = false, ended = false;
  int dimension = -1;
  std::vector<bool> seen(frontier_count + 2, false);
  std::vector<int> indices;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!section) {
      if (line.find("DIMENSION") == 0) {
        auto colon = line.find(':');
        if (colon != std::string::npos) {
          std::istringstream value(line.substr(colon + 1));
          value >> dimension;
        }
      }
      if (line == "TOUR_SECTION") section = true;
      continue;
    }
    std::istringstream value(line);
    int id;
    std::string extra;
    if (!(value >> id) || (value >> extra)) throw std::runtime_error("malformed TSP tour entry");
    if (id == -1) { ended = true; break; }
    if (id < 1 || id > frontier_count + 1 || seen[id])
      throw std::runtime_error("duplicate/out-of-range TSP tour entry");
    seen[id] = true;
    if (id != 1) indices.push_back(id - 2);
  }
  if (!section || !ended || dimension != frontier_count + 1 || !seen[1] ||
      indices.size() != static_cast<size_t>(frontier_count))
    throw std::runtime_error("missing/incomplete TSP tour");
  return indices;
}

inline std::vector<int> solveTspWithRecovery(ros::NodeHandle& nh,
    const std::string& tour_file, int frontier_count)
{
  std::string reason = "service_unavailable";
  for (int attempt = 1; attempt <= 3 && ros::ok(); ++attempt) {
    // Connect at call time, never reuse a stale persistent startup connection.
    auto client = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp", false);
    if (!client.waitForExistence(ros::Duration(1.0))) {
      reason = "service_unavailable";
    } else {
      // A successful RPC without a newly written tour cannot reuse the old file.
      std::ofstream cleared(tour_file, std::ios::trunc);
      cleared.close();
      if (!cleared) throw std::runtime_error("TSP output is not writable: " + tour_file);
      lkh_mtsp_solver::SolveMTSP service;
      service.request.prob = 1;
      if (!client.call(service)) reason = "service_call_failed";
      else {
        try {
          std::ifstream input(tour_file);
          auto tour = readCheckedTspTour(input, frontier_count);
          ROS_INFO("[TSP_RPC] stage=success attempt=%d frontiers=%d", attempt, frontier_count);
          return tour;
        } catch (const std::exception& exc) {
          reason = std::string("invalid_output: ") + exc.what();
        }
      }
    }
    client.shutdown();
    ROS_WARN("[TSP_RPC] stage=retry attempt=%d max_attempts=3 reason=%s", attempt, reason.c_str());
  }
  ROS_ERROR("[TSP_RPC] stage=failed reason=%s; refusing navigation fallback", reason.c_str());
  // Synchronous handler rejects the transaction instead of counting a false
  // navigation failure. This bounds retries/discovery, not ROS1 RPC duration.
  throw std::runtime_error("TSP infrastructure failure after recovery: " + reason);
}
}  // namespace apexnav_planner
