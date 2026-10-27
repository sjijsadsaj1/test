#pragma once

namespace apexnav_planner {
// Resume legacy recovery only if ACTUAL actions match its exact prefix and
// translation has not already ended that legacy escape. Never infer from plans.
class RecoveryHandoff {
public:
  void start(int index) { index_=index;valid_=index>=0 && index<=10; }
  void observe(int executed_action,double displacement) {
    if(!valid_)return;
    if(displacement>=0.05 || index_>=10 || executed_action!=action(index_)) {
      valid_=false;return;
    }
    ++index_;
  }
  int resumeIndex() const {return valid_?index_:-1;}
  static int action(int index) {
    // Same ROS action sequence as ExplorationFSM's legacy stuck branch.
    const int actions[10]={3,1,3,1,2,2,2,1,2,1};
    return index>=0 && index<10?actions[index]:-1;
  }
private:
  int index_{0};
  bool valid_{false};
};
}
