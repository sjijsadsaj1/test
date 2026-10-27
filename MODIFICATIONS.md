# Research-fork modifications

This fork retains ApexNav attribution and licensing. The current research path uses real frontier observations and independent object-presence probabilities to assist exploration. Stable SR improvement has not been demonstrated.

- [Current architecture and retained historical work](docs/CORE_AND_HISTORY.zh-CN.md)
- [Project progress and next steps](docs/PROJECT_PLAN.zh-CN.md)
- [Code changes and measured experiment results](docs/CHANGELOG.zh-CN.md)
- [2026-10-27 cleanup inventory](docs/CLEANUP_MANIFEST.json)

The working planner includes synchronous observations, bounded path search, TSP recovery and search-point confirmation. It is not equivalent to pristine ApexNav when the optional VLM is disabled. Official baseline experiments use the separately downloaded, verified `Apexnav_1/runtime` workspace.

Historical image/video experiment entrypoints were removed from the active tree after a local source backup. Shared compatibility modules remain where deleting them would change runtime imports or existing planning behavior. Datasets, checkpoints, credentials and generated experiment artifacts are not distributed here.
