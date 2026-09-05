# Third-party code and data

This repository is derived from
[Robotics-STAR-Lab/ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) and
retains its GNU GPL v3 license and upstream attribution.

The AVIC integration expects several external projects and models at runtime,
including Stable Virtual Camera, Qwen2.5-VL, Habitat, GroundingDINO, MobileSAM,
YOLOv7, and BLIP-2/LAVIS. Their source code, checkpoints, model caches, and
datasets are not redistributed by this repository. Obtain them from their
official sources and follow each project's license and dataset access terms.

In particular, `AVIC_ROOT` points to a separately obtained
[Adaptive-Visual-Imagination-Control](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control)
checkout. Do not copy that checkout or its weights into this repository without
first confirming redistribution rights.
