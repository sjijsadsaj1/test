# Modifications to ApexNav

This fork is based on upstream ApexNav commit `049971b` and keeps the upstream
Git history, copyright notices, citations, and GNU GPL v3 license.

The experimental changes prepared on 2026-09-05 include:

- a process-isolated `avic_bridge` for optional SVC/Qwen visual imagination;
- a versioned HTTP protocol, fail-open controller, real-observation anchors,
  per-episode frontier caching, and bounded temporary score bonuses;
- ROS/C++ frontier candidate publication and acknowledgement support;
- observation-evidence support in the value-map interface;
- exact episode-manifest execution and isolated result directories;
- portable launcher configuration, tests, and research-status documentation;
- compatibility fixes required by the validated local ROS/Habitat environment.

The AVIC path is disabled by default (`--avic-mode apexnav`). It remains an
experimental research feature and has not yet demonstrated an improvement over
the upstream ApexNav baseline. See `docs/CURRENT_RESULTS.zh-CN.md`.
