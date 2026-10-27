#pragma once
#include <array>
#include <cmath>
#include <deque>
#include <cstddef>

namespace apexnav_planner {
// VLFM-inspired state/action context, with explicit engineering tolerances.
// No scores, maps, or permanent exclusions are modified.
class FrontierCycleMemory {
public:
  using Point = std::array<double,2>;
  struct Decision { Point robot, candidate, context; int step, action; };
  void reset() { history_.clear(); pending_valid_=false;tracking_valid_=false;segment_=0; }
  static bool near(const Point& a,const Point& b) {
    return std::hypot(a[0]-b[0],a[1]-b[1])<=.5+1e-12;
  }
  static bool sameContext(const Point& a,const Point& b) {
    return std::fabs(a[0]-b[0])<=.01+1e-12 && std::fabs(a[1]-b[1])<=.01+1e-12;
  }
  static bool same(const Decision& a,const Decision& b) {
    return near(a.robot,b.robot)&&near(a.candidate,b.candidate)&&sameContext(a.context,b.context);
  }
  bool continuing(const Decision& current) const {
    return tracking_valid_ && same(tracking_anchor_,current);
  }
  bool cyclic(const Decision& current) const {
    const size_t current_segment=continuing(current)?segment_:segment_+1;
    for(const auto& record:history_) {
      const auto& old=record.decision;
      if(record.segment==current_segment || current.step-old.step<12 || !same(old,current))continue;
      // Spatial neighborhoods can overlap during genuine route progress.
      // Compare against the SAME old candidate, not a newly drifting goal.
      const double old_distance=std::hypot(old.robot[0]-old.candidate[0],old.robot[1]-old.candidate[1]);
      const double current_distance=std::hypot(current.robot[0]-old.candidate[0],current.robot[1]-old.candidate[1]);
      if(current_distance<old_distance-1e-9)continue;
      return true;
    }
    return false;
  }
  void stage(const Decision& d) { pending_=d;pending_valid_=true; }
  bool confirm(int previous_step,int executed_action) {
    bool committed=pending_valid_ && pending_.step==previous_step &&
        executed_action>0 && pending_.action==executed_action;
    if(committed) {
      if(!continuing(pending_)) {tracking_anchor_=pending_;tracking_valid_=true;++segment_;}
      history_.push_back({pending_,segment_});if(history_.size()>512)history_.pop_front();
    }
    pending_valid_=false;return committed;
  }
  void clearPending() { pending_valid_=false; }
  const Decision& pending() const { return pending_; }
  size_t size() const { return history_.size(); }
private:
  struct Record { Decision decision;size_t segment; };
  std::deque<Record> history_;
  Decision pending_{};
  Decision tracking_anchor_{};
  size_t segment_{0};
  bool tracking_valid_{false};
  bool pending_valid_{false};
};
}
