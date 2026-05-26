"""UAV Navigation package — VIO with semantic landmark-matching.

This package implements a complete pipeline for UAV navigation using
Visual-Inertial Odometry (VIO) combined with semantic landmark matching
for GNSS-denied environments.

Modules:
    data:        Dataset loaders and calibration utilities.
    perception:  YOLO segmentation, embedding, and feature extraction.
    memory:      Topological Invariant Landmark Map (TILM) and matching.
    estimation:  IMU pre-integration and EKF-based VIO.
    planning:    Path following, adaptive corridor, and mission modes.
    runtime:     Pipeline orchestration, simulation bridges, monitoring.
    deployment:  Model export (ONNX/NCNN) and Pi5-specific runtime.
    eval:        Trajectory evaluation, embedding eval, ablation studies.
"""

__version__ = "0.1.0"
__author__ = "UAV Nav Research"
