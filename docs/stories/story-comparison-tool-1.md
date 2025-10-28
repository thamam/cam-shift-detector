# Story: Core Comparison Infrastructure

**Status:** ready-for-dev
**Epic:** Validation Comparison Tool (`comparison-tool`)
**Story Points:** 5
**Time Estimate:** 3-5 days
**Story ID:** story-comparison-tool-1

---

## User Story

**As a** developer,
**I want** dual detector orchestration and comparison metrics infrastructure,
**so that** I can measure agreement between ChArUco 6-DOF pose estimation and cam-shift detector outputs systematically.

---

## Acceptance Criteria

**AC-1: DualDetectorRunner Orchestration**
- ✅ DualDetectorRunner class initializes both ChArUco and cam-shift detectors
- ✅ set_baseline() configures baselines for both detectors simultaneously
- ✅ process_frame() runs both detectors and returns DualDetectionResult with:
  - ChArUco displacement (2D pixels from 3D pose projection)
  - Cam-shift displacement (2D pixels from feature matching)
  - Comparison metric (||d1-d2||_2)
  - Agreement status (GREEN/RED)
- ✅ Handles ChArUco detection failures gracefully (no crash)

**AC-2: Comparison Metrics Calculations**
- ✅ calculate_displacement_difference() computes L2 norm: abs(d1 - d2)
- ✅ calculate_threshold() uses 3% of min(width, height)
- ✅ classify_agreement() returns "GREEN" if diff ≤ threshold, else "RED"
- ✅ calculate_charuco_displacement_2d() converts 3D translation to 2D pixels using camera matrix projection
- ✅ calculate_mse() computes Mean Squared Error over sequence

**AC-3: Comparison Logger**
- ✅ ComparisonLogger logs DualDetectionResult to JSON file per frame
- ✅ calculate_mse() returns MSE across all logged frames
- ✅ get_worst_matches(n) retrieves top N frames with largest displacement_diff
- ✅ generate_mse_graph() creates matplotlib plot with:
  - X-axis: frame_idx
  - Y-axis: displacement_diff
  - Threshold line (horizontal at threshold value)
  - Worst matches highlighted in red
- ✅ save_log() persists results to structured JSON

**AC-4: Unit Test Coverage**
- ✅ Unit tests for DualDetectorRunner (initialization, baseline, processing)
- ✅ Unit tests for all comparison_metrics functions
- ✅ Unit tests for ComparisonLogger (logging, MSE, worst matches, graph)
- ✅ Overall test coverage ≥80%
- ✅ All tests pass on session_001 data

---

## Tasks / Subtasks

### Task 1: Create DualDetectorRunner Module (AC: #1)
- [ ] Create `validation/utilities/dual_detector_runner.py`
- [ ] Define DualDetectionResult dataclass with fields:
  - frame_idx, timestamp_ns
  - charuco_detected, charuco_displacement_px, charuco_confidence
  - camshift_status, camshift_displacement_px, camshift_confidence
  - displacement_diff, agreement_status, threshold_px
- [ ] Implement DualDetectorRunner.__init__():
  - Initialize ChArUco detector (make_charuco_board, read_yaml_camera)
  - Initialize CameraMovementDetector with config_path
  - Calculate threshold_px from image dimensions
- [ ] Implement DualDetectorRunner.set_baseline():
  - Store baseline ChArUco pose (tvec_baseline)
  - Call camshift_detector.set_baseline()
- [ ] Implement DualDetectorRunner.process_frame():
  - Run estimate_pose_charuco() → rvec, tvec, inliers
  - Run camshift_detector.process_frame() → status, displacement, confidence
  - Calculate charuco_displacement_2d() using projection
  - Calculate comparison metric ||d1-d2||_2
  - Classify agreement status
  - Return DualDetectionResult

### Task 2: Create Comparison Metrics Module (AC: #2)
- [ ] Create `validation/utilities/comparison_metrics.py`
- [ ] Implement calculate_displacement_difference(d1, d2) → L2 norm
- [ ] Implement calculate_threshold(width, height, percent=0.03) → threshold_px
- [ ] Implement classify_agreement(diff, threshold) → "GREEN"/"RED"
- [ ] Implement calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, z_distance=1.15):
  - Extract fx, fy from camera matrix K
  - Calculate delta_x, delta_y from translation vectors
  - Project: dx_px = (delta_x * fx) / z_distance
  - Project: dy_px = (delta_y * fy) / z_distance
  - Return: sqrt(dx_px^2 + dy_px^2)
- [ ] Implement calculate_mse(charuco_list, camshift_list) → MSE

### Task 3: Create Comparison Logger Module (AC: #3)
- [ ] Create `validation/utilities/comparison_logger.py`
- [ ] Define ComparisonLogger class with __init__(output_dir, session_name)
- [ ] Implement log_frame(result: DualDetectionResult) → append to results list
- [ ] Implement save_log() → write JSON with structured format:
  - session_name, timestamp, threshold_px
  - results array with all DualDetectionResult fields
- [ ] Implement calculate_mse() → call comparison_metrics.calculate_mse()
- [ ] Implement get_worst_matches(n=10) → sort by displacement_diff descending, return top N
- [ ] Implement generate_mse_graph(output_path):
  - Create matplotlib figure with displacement_diff over frame_idx
  - Add horizontal threshold line
  - Highlight worst 10 matches in red
  - Add axis labels, legend, title
  - Save to output_path

### Task 4: Unit Testing (AC: #4)
- [ ] Create `tests/validation/test_dual_detector_runner.py`:
  - Test initialization with valid configs
  - Test set_baseline() with mock image
  - Test process_frame() with ChArUco detected
  - Test process_frame() with ChArUco not detected (graceful handling)
  - Test displacement calculation accuracy
- [ ] Create `tests/validation/test_comparison_metrics.py`:
  - Test calculate_displacement_difference() with various inputs
  - Test calculate_threshold() with different image sizes
  - Test classify_agreement() boundary conditions
  - Test calculate_charuco_displacement_2d() projection math
  - Test calculate_mse() with known sequences
- [ ] Create `tests/validation/test_comparison_logger.py`:
  - Test log_frame() appends correctly
  - Test save_log() creates valid JSON
  - Test calculate_mse() returns correct value
  - Test get_worst_matches() sorting and limit
  - Test generate_mse_graph() creates file
- [ ] Run coverage report → verify ≥80%

---

## Dev Notes

### Technical Summary

**Architecture:**
- DualDetectorRunner orchestrates parallel execution of ChArUco pose estimator and cam-shift detector
- ComparisonMetrics provides pure calculation functions for displacement comparison
- ComparisonLogger handles persistence and analysis (JSON logs, MSE, worst matches)

**Key Technical Decisions:**
1. **3D-to-2D Projection**: Use camera matrix focal lengths (fx, fy) to project 3D ChArUco displacement onto 2D image plane
2. **Threshold Calculation**: 3% of minimum image dimension aligns with Stage 2 handoff success criteria
3. **Comparison Metric**: L2 norm (Euclidean distance) provides intuitive measure of agreement
4. **Failure Handling**: ChArUco detection failures return NaN, logger gracefully handles missing data

**Dependencies:**
- matplotlib==3.9.2 (new) for MSE graph generation
- Existing: opencv-python, numpy, pandas, src.camera_movement_detector, tools.aruco.camshift_annotator

### Files to Create

```
validation/utilities/
├── dual_detector_runner.py     # ~250 lines (DualDetectorRunner class, DualDetectionResult dataclass)
├── comparison_metrics.py        # ~80 lines (5 pure calculation functions)
└── comparison_logger.py         # ~180 lines (ComparisonLogger class with JSON, MSE, graph)

tests/validation/
├── test_dual_detector_runner.py   # ~200 lines (initialization, baseline, processing tests)
├── test_comparison_metrics.py     # ~150 lines (calculation function tests)
└── test_comparison_logger.py      # ~180 lines (logging, MSE, worst match, graph tests)
```

**Total Code:** ~1,040 lines (510 implementation + 530 tests)

### Test Locations

**Unit Tests:**
- `tests/validation/test_dual_detector_runner.py`
- `tests/validation/test_comparison_metrics.py`
- `tests/validation/test_comparison_logger.py`

**Test Data:**
- Use existing `session_001/poses.csv` (203 frames, 158 detected)
- Use existing `session_001/frames/` for integration test data
- Use existing `camera.yaml` for camera intrinsics

**Coverage Target:**
- Overall: ≥80%
- DualDetectorRunner: ≥85%
- ComparisonMetrics: ≥90% (pure functions easier to test)
- ComparisonLogger: ≥80%

### Architecture References

**Tech-Spec Sections:**
- Section 5.1: DualDetectorRunner implementation details
- Section 5.2: ComparisonMetrics implementation details
- Section 5.3: ComparisonLogger implementation details
- Section 4.3: Configuration file structure (comparison_config.json)

**Existing Code References:**
- `validation/core/stage2_charuco_validation.py` - CharucoValidator for reference patterns
- `tools/aruco/camshift_annotator.py` - estimate_pose_charuco(), 3D-to-2D conversion
- `src/camera_movement_detector.py` - CameraMovementDetector.process_frame() API

---

## Dev Agent Record

### Context Generation (SM Agent)
**Context Reference:**
- `docs/stories/story-comparison-tool-1.context.md` - Complete story context with artifacts, constraints, interfaces, and testing guidance

**Analysis:**
- ✅ Project structure analysis complete
- ✅ Dependency mapping complete (matplotlib==3.9.2 NEW + 5 existing packages)
- ✅ Integration points identified (ChArUco utilities, CameraMovementDetector API)

**Design Decisions:**
- ✅ Class structure finalized (DualDetectorRunner, ComparisonLogger, DualDetectionResult dataclass)
- ✅ Data flow documented (dual detector orchestration → comparison metrics → logging/analysis)
- ✅ Error handling strategy (ChArUco failures return NaN, logger graceful handling)

**Constraints & Dependencies:**
- ✅ matplotlib dependency confirmed (matplotlib==3.9.2 - must be added to pyproject.toml)
- ✅ Camera calibration file requirement (camera.yaml with K matrix, distortion coefficients)
- ✅ ChArUco board configuration (7×5, 35mm squares, 26mm markers, DICT_4X4_50)

### Implementation (DEV Agent)
*(To be populated during dev-story workflow)*

**Progress Log:**
- [ ] Task 1 complete: DualDetectorRunner
- [ ] Task 2 complete: ComparisonMetrics
- [ ] Task 3 complete: ComparisonLogger
- [ ] Task 4 complete: Unit testing

**Blockers & Resolutions:**
- [ ] None yet

**Code Review Notes:**
- [ ] Test coverage verified
- [ ] Documentation complete
- [ ] Integration tested

---

**Story Created:** 2025-10-27
**Last Updated:** 2025-10-27
**Ready for:** Implementation (DEV Agent)
