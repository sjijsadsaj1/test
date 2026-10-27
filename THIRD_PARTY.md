# Third-party code and data

This repository is derived from
[Robotics-STAR-Lab/ApexNav](https://github.com/Robotics-STAR-Lab/ApexNav) and
retains its GNU GPL v3 license and upstream attribution.

The current probability-prediction experiments use an external Qwen3-VL model
and the ApexNav perception/simulation stack, including Habitat, GroundingDINO,
MobileSAM, YOLOv7, and BLIP-2/LAVIS. Retained legacy imagination backends also
support Stable Virtual Camera, Qwen2.5-VL, and Qwen-Image. Their source code, checkpoints, model caches, and
datasets are not redistributed by this repository. Obtain them from their
official sources and follow each project's license and dataset access terms.

In particular, `AVIC_ROOT` points to a separately obtained
[Adaptive-Visual-Imagination-Control](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control)
checkout. Do not copy that checkout or its weights into this repository without
first confirming redistribution rights.
