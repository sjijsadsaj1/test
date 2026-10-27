#include <exploration_manager/tsp_service.h>
#include <cassert>
#include <atomic>
#include <chrono>
#include <thread>

using namespace apexnav_planner;
const std::string valid = "DIMENSION : 3\nTOUR_SECTION\n1\n3\n2\n-1\nEOF\n";

void parserTests()
{
  std::istringstream good(valid);
  assert((readCheckedTspTour(good, 2) == std::vector<int>{1, 0}));
  for (const auto& text : {std::string(), std::string("TOUR_SECTION\n1\n2\n3\n-1\n"),
      std::string("DIMENSION : 3\nTOUR_SECTION\n1\n2\n2\n-1\n"),
      std::string("DIMENSION : 3\nTOUR_SECTION\n1\n4\n2\n-1\n"),
      std::string("DIMENSION : 3\nTOUR_SECTION\n1\n2\n-1\n"),
      std::string("DIMENSION : 3\nTOUR_SECTION\n1\n2\n3\n"),
      std::string("DIMENSION : 3\nTOUR_SECTION\n1\n2garbage\n3\n-1\n")}) {
    bool rejected = false;
    try { std::istringstream input(text); readCheckedTspTour(input, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
}

int main(int argc, char** argv)
{
  parserTests();
  if (argc == 2 && std::string(argv[1]) == "--parse-only") return 0;
  assert(argc == 2);
  const std::string output = argv[1];
  ros::init(argc, argv, "tsp_recovery_regression");
  ros::NodeHandle nh;
  ros::AsyncSpinner spinner(1);
  spinner.start();
  std::atomic<int> mode{0}, calls{0};
  boost::function<bool(lkh_mtsp_solver::SolveMTSP::Request&, lkh_mtsp_solver::SolveMTSP::Response&)> callback =
      [&](lkh_mtsp_solver::SolveMTSP::Request& req, lkh_mtsp_solver::SolveMTSP::Response&) {
    assert(req.prob == 1);
    const int call = ++calls;
    if (mode == 1 && call == 1) return false;
    if (mode == 2) return true;  // stale file must NOT be accepted
    std::ofstream file(output);
    file << (mode == 3 ? "DIMENSION : 3\nTOUR_SECTION\n1\n2\n2\n-1\n" : valid);
    file.close();
    return true;
  };
  ros::ServiceServer server;
  auto delayedStart = [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    server = nh.advertiseService("/solve_tsp", callback);
  };
  auto expectSuccess = [&]() {
    assert((solveTspWithRecovery(nh, output, 2) == std::vector<int>{1, 0}));
  };
  auto expectFailure = [&]() {
    bool rejected = false;
    try { solveTspWithRecovery(nh, output, 2); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  };
  auto legacy = nh.serviceClient<lkh_mtsp_solver::SolveMTSP>("/solve_tsp", true);
  std::thread startup(delayedStart);
  expectSuccess();
  startup.join();
  assert(calls == 1);
  lkh_mtsp_solver::SolveMTSP legacy_request;
  legacy_request.request.prob = 1;
  ROS_INFO("LEGACY_EARLY_PERSISTENT_CALL success=%d", legacy.call(legacy_request));
  mode = 1; calls = 0;
  expectSuccess();
  assert(calls == 2);
  server.shutdown();
  mode = 0; calls = 0;
  std::thread restart(delayedStart);
  expectSuccess();
  restart.join();
  assert(calls == 1);
  mode = 2; calls = 0;
  expectFailure();  // previous valid tour exists before this call
  assert(calls == 3);
  std::ifstream empty(output);
  assert(empty.peek() == std::ifstream::traits_type::eof());
  mode = 3; calls = 0;
  expectFailure();
  assert(calls == 3);
  server.shutdown();
  auto start = std::chrono::steady_clock::now();
  expectFailure();
  assert(std::chrono::steady_clock::now() - start < std::chrono::seconds(8));
  ROS_INFO("TSP_RECOVERY_TEST_PASSED delayed_start failed_call restart stale_output malformed_output unavailable");
}
