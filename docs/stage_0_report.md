# Stage 0 Report: Project Structure and Scaffold

**Date:** 2026-05-23
**Status:** Complete

---

## 1. Summary

Stage 0 successfully created the complete project scaffold for the UAV navigation
dissertation. All directories, module stubs, configuration files, and tooling
have been created. The project is ready for Stage 1 implementation work.

---

## 2. Files Created

### Package Source (`uav_nav/`)

**Root**
- `uav_nav/__init__.py` — Package metadata and docstring.

**`uav_nav/data/`** (3 modules)
- `__init__.py` — Public API exports.
- `calibration.py` — `CameraCalibration`, `StereoCalibration` dataclasses with `.K` property and `from_midair_defaults()`.
- `midair_loader.py` — `MidAirLoader` iterable, `MidAirSample`, `IMUSample` dataclasses.
- `dataset_builder.py` — `DatasetBuilder`, `SplitConfig`; build_all/build_split stubs.

**`uav_nav/perception/`** (5 modules)
- `__init__.py` — Public API exports.
- `yolo_segmenter.py` — `YOLOSegmenter`, `SegmentationResult`; predict/predict_batch/warmup stubs.
- `embedding_head.py` — `EmbeddingHead`, `EmbeddingConfig`; forward/train_epoch/evaluate/save/load stubs.
- `feature_extractor.py` — `FeatureExtractor`, `KeypointSet`; extract/match/track stubs.
- `stereo_depth.py` — `StereoDepthEstimator`, `DepthResult`, `StereoBackend`; estimate stub.
- `landmark_extractor.py` — `LandmarkExtractor`, `Landmark`; extract/_crop_and_embed/_unproject stubs.

**`uav_nav/memory/`** (4 modules)
- `__init__.py` — Public API exports.
- `tilm.py` — `TILM`, `TILMNode`, `TILMEdge`; add_node/add_edge/nearest_node/shortest_path/save/load stubs.
- `tilm_builder.py` — `TILMBuilder`, `BuilderConfig`; process_frame/finalise stubs.
- `temporal_matcher.py` — `TemporalMatcher`, `MatchResult`; match/retrieve_candidates/geometric_verify stubs.
- `place_descriptor.py` — `PlaceDescriptor`, `PlaceQuery`, `AggregationMethod`; compute/similarity implemented.

**`uav_nav/estimation/`** (3 modules)
- `__init__.py` — Public API exports.
- `imu_preintegrator.py` — `IMUPreintegrator`, `PreintegratedState`; integrate/reset/predict_pose stubs.
- `ekf_vio.py` — `EKFVIO`, `EKFState`; propagate/update_visual/update_landmark/update_gnss stubs.
- `pose_utils.py` — `skew`, `so3_exp`, `quat_to_rot`, `ned_to_enu`, `compute_relative_pose` **implemented**; `so3_log`, `se3_exp`, `se3_log`, `rot_to_quat`, `interpolate_poses` as stubs.

**`uav_nav/planning/`** (4 modules)
- `__init__.py` — Public API exports.
- `path_follower.py` — `PathFollower`, `PathFollowerConfig` (D_min=5, D_max_cap=50, dt_window=5, THRESHOLD=0.7); `set_path`, `is_path_complete`, `state` property implemented; `step`, `_compute_look_ahead`, `_carrot_point`, `cross_track_error` as stubs.
- `adaptive_corridor.py` — `AdaptiveCorridor`, `CorridorState`; update/is_inside stubs.
- `mission_modes.py` — `MissionMode` enum (NOMINAL, GNSS_DEGRADED, GNSS_LOST, REJOIN, EMERGENCY_LAND, HOLD), `MissionManager`, `MissionStatus`, `TransitionRule`; update/force_mode stubs.
- `rejoin_planner.py` — `RejoinPlanner`, `RejoinResult`; plan/_select_merge_point/_generate_bezier stubs.

**`uav_nav/runtime/`** (5 modules)
- `__init__.py` — Public API exports.
- `pipeline.py` — `NavigationPipeline`, `PipelineConfig`; initialise/process_frame/shutdown stubs.
- `pseudo_simulator.py` — `PseudoSimulator`, `GNSSOutageConfig`, `SimulationResult`; run/_process_sample stubs.
- `airsim_bridge.py` — `AirSimBridge`, `AirSimConfig`; connect/disconnect/run stubs.
- `monitor.py` — `SystemMonitor`, `MonitorMetrics`; `record`, `mean_latency_ms`, `fps`, `reset` implemented; `summary`, `_append_csv`, `_push_rerun` as stubs.
- `logger.py` — `setup_logger` **fully implemented** with console + file + JSON sinks; `get_logger` implemented; Rerun sink stub.

**`uav_nav/deployment/`** (4 modules)
- `__init__.py` — Public API exports.
- `export_onnx.py` — `export_to_onnx`, `ONNXExportConfig`, `verify_onnx_outputs`, `benchmark_onnx` stubs.
- `export_ncnn.py` — `export_to_ncnn`, `NCNNExportConfig`, `verify_ncnn_outputs`, `benchmark_ncnn` stubs.
- `benchmark_pi5.py` — `Pi5Benchmark`, `BenchmarkResult` with `from_latencies` factory implemented; target latencies table defined; `benchmark_callable`/`run_all`/`report`/`save_csv` stubs.
- `pi5_runtime.py` — `Pi5Runtime`, `Pi5RuntimeConfig`; setup/run/shutdown stubs.

**`uav_nav/eval/`** (5 modules)
- `__init__.py` — Public API exports.
- `trajectory_eval.py` — `TrajectoryEvaluator`, `TrajectoryMetrics`; `to_dict` implemented; evaluate/evaluate_batch/save_report/plot_trajectories stubs.
- `embedding_eval.py` — `EmbeddingEvaluator`, `EmbeddingMetrics`; evaluate/recall_at_k/compute_map/plot_tsne stubs.
- `matching_eval.py` — `MatchingEvaluator`, `MatchingMetrics`; evaluate/precision_recall_curve stubs.
- `scenario_runner.py` — `ScenarioRunner`, `ScenarioConfig`, `ScenarioResult`; run_scenario/run_all/save_summary stubs.
- `ablation.py` — `AblationStudy`, `AblationConfig`, `AblationResult`; run_variant/run_all/delta_table/save_table stubs.

### Configurations (`uav_nav/configs/`)
- `default.yaml` — Full system config: camera (MidAir intrinsics), IMU noise, EKF params, TILM params, path follower (D_min=5, D_max_cap=50, dt_window=5, THRESHOLD=0.7), embedding, YOLO, runtime, eval.
- `pi5.yaml` — Inherits from default; overrides: NCNN backend, 320×240 resolution, 8 FPS target, 2 worker threads, BM stereo.
- `label_map.json` — MidAir class IDs → {background, tree, rock, river} with colors.
- `experiments/5_3_cross_weather.yaml` — Cross-weather experiment config.
- `experiments/5_5_tilm_correction.yaml` — TILM correction ablation config.

### Scripts (`uav_nav/scripts/`)
- `01_build_dataset.py` — Hydra entry point for dataset construction.
- `02_train_yolo.py` — Hydra entry point for YOLO training.
- `03_train_embedding.py` — Hydra entry point for embedding training.
- `04_build_tilm.py` — Hydra entry point for TILM construction.
- `05_simulate_gnss_loss.py` — Hydra entry point for GNSS loss simulation.
- `06_return_to_path_demo.py` — Hydra entry point for re-join demo.
- `07_benchmark_pi5.py` — Hydra entry point for Pi 5 benchmark.
- `08_run_all_experiments.py` — Hydra entry point for all experiments.

### Tests (`uav_nav/tests/`)
- `__init__.py`, `conftest.py` — 9 fixtures: `camera_calibration`, `stereo_calibration`, `synthetic_rgb_frame`, `synthetic_depth_frame`, `synthetic_segmentation_frame`, `synthetic_imu_batch`, `imu_preintegrator`, `initial_ekf_state`, `synthetic_landmarks`, `minimal_tilm`, `ned_waypoints`.
- `test_data/__init__.py`
- `test_estimation/__init__.py`, `test_estimation/test_pose_utils.py` — 14 passing tests for `skew`, `so3_exp`, `quat_to_rot`, NED↔ENU, `compute_relative_pose`.
- `test_memory/__init__.py`, `test_memory/test_place_descriptor.py` — 8 tests for `PlaceDescriptor.similarity`.
- `test_planning/__init__.py`, `test_planning/test_path_follower_config.py` — 8 tests for config defaults and PathFollower init.

### Project Root Files
- `pyproject.toml` — Build system, all dependencies, pytest config, black/ruff config.
- `requirements.txt` — Pinned dependency list with install instructions.
- `.pre-commit-config.yaml` — black (line-length 100) + ruff (E, F, I) + standard hooks.
- `Makefile` — Targets: `train_yolo`, `train_embedding`, `build_tilm`, `run_experiment`, `benchmark_pi5`, `export_onnx`, `export_ncnn`, `test`, `test-fast`, `lint`, `format`, `check`, `clean`, `clean-all`.

### Documentation (`docs/`)
- `docs/stage_0_plan.md` — Full task decomposition, module interfaces, acceptance criteria.
- `docs/stage_0_report.md` — This document.

---

## 3. Immediately Testable Code (No Implementation Required)

The following functions are fully implemented and have passing unit tests:

| Module                     | Implemented Functions                          |
|----------------------------|------------------------------------------------|
| `estimation.pose_utils`    | `skew`, `so3_exp`, `quat_to_rot`, `ned_to_enu`, `enu_to_ned`, `compute_relative_pose` |
| `memory.place_descriptor`  | `PlaceDescriptor.similarity`                   |
| `planning.path_follower`   | `PathFollower.set_path`, `is_path_complete`, `state` property |
| `runtime.logger`           | `setup_logger` (console + file + JSON sinks)   |
| `runtime.monitor`          | `SystemMonitor.record`, `mean_latency_ms`, `fps`, `reset` |
| `deployment.benchmark_pi5` | `BenchmarkResult.from_latencies`               |
| `eval.trajectory_eval`     | `TrajectoryMetrics.to_dict`                    |
| `data.calibration`         | `CameraCalibration.K`, `from_midair_defaults`, `StereoCalibration.baseline` |

---

## 4. Next Steps (Stage 1)

1. Implement `MidAirLoader._discover_sequences` and `_load_sequence` using `h5py`.
2. Implement `DatasetBuilder._undistort_image` and `_remap_labels`.
3. Implement `CameraCalibration.from_yaml` and `StereoCalibration.from_yaml`.
4. Add integration tests tagged `requires_midair` that run against a local dataset copy.
5. Create `data/midair_yolo.yaml` for YOLO training.
