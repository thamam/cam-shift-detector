<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>comparison-tool</epicId>
    <storyId>story-comparison-tool-1</storyId>
    <title>Core Comparison Infrastructure</title>
    <status>Draft</status>
    <generatedAt>2025-10-27</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>docs/stories/story-comparison-tool-1.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>developer</asA>
    <iWant>dual detector orchestration and comparison metrics infrastructure</iWant>
    <soThat>I can measure agreement between ChArUco 6-DOF pose estimation and cam-shift detector outputs systematically</soThat>
    <tasks>
### Task 1: Create DualDetectorRunner Module (AC: #1)
- Create `validation/utilities/dual_detector_runner.py`
- Define DualDetectionResult dataclass with fields
- Implement DualDetectorRunner.__init__() - Initialize ChArUco and CameraMovementDetector
- Implement DualDetectorRunner.set_baseline() - Store baseline poses
- Implement DualDetectorRunner.process_frame() - Run both detectors, calculate comparison metric

### Task 2: Create Comparison Metrics Module (AC: #2)
- Create `validation/utilities/comparison_metrics.py`
- Implement calculate_displacement_difference() - L2 norm calculation
- Implement calculate_threshold() - 3% of min(width, height)
- Implement classify_agreement() - GREEN/RED classification
- Implement calculate_charuco_displacement_2d() - 3D to 2D projection
- Implement calculate_mse() - Mean Squared Error over sequence

### Task 3: Create Comparison Logger Module (AC: #3)
- Create `validation/utilities/comparison_logger.py`
- Define ComparisonLogger class
- Implement log_frame() - Per-frame JSON logging
- Implement save_log() - Persist structured JSON
- Implement calculate_mse() - MSE calculation
- Implement get_worst_matches(n) - Top N worst agreements
- Implement generate_mse_graph() - Matplotlib visualization

### Task 4: Unit Testing (AC: #4)
- Create `tests/validation/test_dual_detector_runner.py`
- Create `tests/validation/test_comparison_metrics.py`
- Create `tests/validation/test_comparison_logger.py`
- Achieve ≥80% overall test coverage
- All tests pass on session_001 data
</tasks>
  </story>

  <acceptanceCriteria>
**AC-1: DualDetectorRunner Orchestration**
- DualDetectorRunner class initializes both ChArUco and cam-shift detectors
- set_baseline() configures baselines for both detectors simultaneously
- process_frame() runs both detectors and returns DualDetectionResult with ChArUco displacement, Cam-shift displacement, comparison metric (||d1-d2||_2), and agreement status (GREEN/RED)
- Handles ChArUco detection failures gracefully (no crash)

**AC-2: Comparison Metrics Calculations**
- calculate_displacement_difference() computes L2 norm: abs(d1 - d2)
- calculate_threshold() uses 3% of min(width, height)
- classify_agreement() returns "GREEN" if diff ≤ threshold, else "RED"
- calculate_charuco_displacement_2d() converts 3D translation to 2D pixels using camera matrix projection
- calculate_mse() computes Mean Squared Error over sequence

**AC-3: Comparison Logger**
- ComparisonLogger logs DualDetectionResult to JSON file per frame
- calculate_mse() returns MSE across all logged frames
- get_worst_matches(n) retrieves top N frames with largest displacement_diff
- generate_mse_graph() creates matplotlib plot with proper axes, threshold line, and highlighted worst matches
- save_log() persists results to structured JSON

**AC-4: Unit Test Coverage**
- Unit tests for DualDetectorRunner (initialization, baseline, processing)
- Unit tests for all comparison_metrics functions
- Unit tests for ComparisonLogger (logging, MSE, worst matches, graph)
- Overall test coverage ≥80%
- All tests pass on session_001 data
</acceptanceCriteria>

  <artifacts>
    <docs>
      <!-- Primary Technical Specification -->
      <doc>
        <path>tech-spec.md</path>
        <title>Technical Specification: ChArUco vs Cam-Shift Validation Comparison Tool</title>
        <section>Section 5: Technical Details</section>
        <snippet>Definitive technical specification covering DualDetectorRunner (orchestration), ComparisonMetrics (calculations), and ComparisonLogger (persistence/analysis). Includes exact implementation details for 3D-to-2D projection, comparison metric formula, and MSE graph generation.</snippet>
      </doc>

      <!-- Validation System Documentation -->
      <doc>
        <path>validation/README.md</path>
        <title>Validation System Overview</title>
        <section>Utilities Section</section>
        <snippet>Describes existing validation/utilities/ structure with real_data_loader, performance_profiler, and report_generator. New comparison modules integrate into this existing framework.</snippet>
      </doc>

      <!-- Epic Overview -->
      <doc>
        <path>epics.md</path>
        <title>Epic: Validation Comparison Tool</title>
        <section>Success Criteria and Story Map</section>
        <snippet>Defines epic goal, success criteria (functional, performance, quality), and dependency on Story 2 for tool integration. Total 8 story points across 2 stories.</snippet>
      </doc>
    </docs>

    <code>
      <!-- Reference ChArUco Validation Implementation -->
      <code>
        <path>validation/core/stage2_charuco_validation.py</path>
        <kind>validation module</kind>
        <symbol>CharucoValidator</symbol>
        <lines>134-614</lines>
        <reason>Reference implementation showing ChArUco pose estimation, 3D-to-2D conversion pattern (lines 219-282), ground truth comparison, and metrics calculation. Demonstrates how to use estimate_pose_charuco() and project 3D displacement to 2D pixels.</reason>
      </code>

      <!-- ChArUco Pose Estimation Utilities -->
      <code>
        <path>tools/aruco/camshift_annotator.py</path>
        <kind>utility module</kind>
        <symbol>estimate_pose_charuco</symbol>
        <lines>297-322</lines>
        <reason>Core ChArUco pose estimation function. Returns rvec, tvec, and corner count. Must be wrapped in DualDetectorRunner to pair with cam-shift detector. Shows ChArUco detector initialization pattern.</reason>
      </code>
      <code>
        <path>tools/aruco/camshift_annotator.py</path>
        <kind>utility module</kind>
        <symbol>make_charuco_board</symbol>
        <lines>83-89</lines>
        <reason>ChArUco board initialization (7×5 squares, 0.035m square size, 0.026m marker size, DICT_4X4_50). Required for DualDetectorRunner initialization.</reason>
      </code>
      <code>
        <path>tools/aruco/camshift_annotator.py</path>
        <kind>utility module</kind>
        <symbol>read_yaml_camera</symbol>
        <lines>62-71</lines>
        <reason>Load camera intrinsics from YAML (K matrix, distortion, image size). Required for 3D-to-2D projection in calculate_charuco_displacement_2d().</reason>
      </code>

      <!-- Cam-Shift Detector API -->
      <code>
        <path>src/camera_movement_detector.py</path>
        <kind>detector API</kind>
        <symbol>CameraMovementDetector</symbol>
        <lines>19-352</lines>
        <reason>Main cam-shift detector API. Use set_baseline() and process_frame() in DualDetectorRunner. Returns dict with status, translation_displacement, confidence.</reason>
      </code>

      <!-- Existing Validation Utilities (Patterns) -->
      <code>
        <path>validation/utilities/performance_profiler.py</path>
        <kind>utility module</kind>
        <symbol>PerformanceProfiler</symbol>
        <lines>20-269</lines>
        <reason>Reference pattern for profiling and metrics collection. Shows dataclass usage, timing measurement, and result aggregation patterns to follow.</reason>
      </code>
      <code>
        <path>validation/utilities/report_generator.py</path>
        <kind>utility module</kind>
        <symbol>JSONReportGenerator</symbol>
        <lines>26-234</lines>
        <reason>Reference pattern for JSON report generation. Shows how to structure output data and save to file. ComparisonLogger should follow similar patterns.</reason>
      </code>

      <!-- Test Infrastructure -->
      <code>
        <path>tests/validation/conftest.py</path>
        <kind>test fixture</kind>
        <symbol>pytest fixtures</symbol>
        <lines>1-100</lines>
        <reason>Shared test fixtures for validation tests. Includes cv2 mocking and common test data setup. Use for new comparison module tests.</reason>
      </code>
    </code>

    <dependencies>
      <python version=">=3.11">
        <package>opencv-python==4.10.0.84</package>
        <package>opencv-contrib-python>=4.12.0.88</package>
        <package>numpy==1.26.4</package>
        <package>pandas==2.2.3</package>
        <package>matplotlib==3.9.2 (NEW - add to dependencies)</package>
        <package>psutil>=5.9.0</package>
      </python>
    </dependencies>
  </artifacts>

  <constraints>
    <!-- Technical Constraints -->
    - **3D-to-2D Projection Accuracy**: Conversion depends on camera calibration quality (camera.yaml must exist and contain accurate K matrix and distortion coefficients)
    - **ChArUco Detection Failure Handling**: DualDetectorRunner must handle ChArUco detection failures gracefully (return NaN displacement, no crashes)
    - **Comparison Metric Formula**: MUST use ||d1-d2||_2 (Euclidean L2 norm) with threshold = 3% of min(image_width, image_height)
    - **Matplotlib Dependency**: New dependency on matplotlib==3.9.2 for MSE graph generation (must be added to pyproject.toml)

    <!-- Testing Constraints -->
    - **Test Coverage**: Overall coverage ≥80%, DualDetectorRunner ≥85%, ComparisonMetrics ≥90%, ComparisonLogger ≥80%
    - **Test Data**: Use existing session_001 data (203 frames, 158 detected) for integration tests
    - **Test Framework**: pytest with coverage plugin

    <!-- Code Organization Constraints -->
    - **Module Location**: New modules MUST go in validation/utilities/ (not tools/ or validation/core/)
    - **Dataclass Usage**: Use dataclasses for DualDetectionResult (follow existing patterns in project)
    - **Project-Relative Paths**: All file paths in code and context must be project-relative, not absolute

    <!-- Implementation Constraints from Tech-Spec -->
    - **Default Z Distance**: Use 1.15m as default camera-to-board distance for 3D-to-2D projection (based on session_001 typical distance)
    - **Timestamp Format**: Use time.time_ns() for nanosecond precision timestamps (consistent with camshift_annotator.py)
    - **JSON Structure**: Log files must be structured JSON with session metadata and per-frame results array
  </constraints>

  <interfaces>
    <!-- CameraMovementDetector API -->
    <interface>
      <name>CameraMovementDetector.process_frame</name>
      <kind>method</kind>
      <signature>process_frame(image_array: np.ndarray, frame_id: Optional[str] = None) -> Dict</signature>
      <path>src/camera_movement_detector.py:192-258</path>
      <returns>{"status": "VALID"|"INVALID", "translation_displacement": float, "confidence": float, "frame_id": str, "timestamp": str}</returns>
    </interface>

    <!-- ChArUco Pose Estimation API -->
    <interface>
      <name>estimate_pose_charuco</name>
      <kind>function</kind>
      <signature>estimate_pose_charuco(frame_gray, detector, charuco_board, K, dist) -> Optional[Tuple[rvec, tvec, n_corners]]</signature>
      <path>tools/aruco/camshift_annotator.py:297-322</path>
      <returns>Tuple of (rvec: np.ndarray, tvec: np.ndarray, n_corners: int) or None if detection fails</returns>
    </interface>

    <!-- Camera Intrinsics Loading -->
    <interface>
      <name>read_yaml_camera</name>
      <kind>function</kind>
      <signature>read_yaml_camera(path: str) -> Tuple[K, dist, (width, height)]</signature>
      <path>tools/aruco/camshift_annotator.py:62-71</path>
      <returns>Tuple of (K: np.ndarray 3×3, dist: np.ndarray, image_size: Tuple[int, int])</returns>
    </interface>

    <!-- Comparison Metrics API (to be implemented) -->
    <interface>
      <name>calculate_displacement_difference</name>
      <kind>function</kind>
      <signature>calculate_displacement_difference(d1: float, d2: float) -> float</signature>
      <path>validation/utilities/comparison_metrics.py (NEW)</path>
      <returns>L2 norm: abs(d1 - d2)</returns>
    </interface>
    <interface>
      <name>calculate_charuco_displacement_2d</name>
      <kind>function</kind>
      <signature>calculate_charuco_displacement_2d(tvec_current: np.ndarray, tvec_baseline: np.ndarray, K: np.ndarray, z_distance_m: float = 1.15) -> float</signature>
      <path>validation/utilities/comparison_metrics.py (NEW)</path>
      <returns>2D pixel displacement from 3D translation vectors: sqrt((delta_x * fx / z)^2 + (delta_y * fy / z)^2)</returns>
    </interface>
  </interfaces>

  <tests>
    <standards>
**Testing Framework**: pytest with coverage plugin (pytest-cov)

**Test Organization**:
- Unit tests: tests/validation/test_*.py (one test file per module)
- Shared fixtures: tests/validation/conftest.py (includes cv2 mocking)
- Test data: session_001/ directory (203 frames, 158 with ChArUco detection)

**Naming Conventions**:
- Test files: test_{module_name}.py (e.g., test_dual_detector_runner.py)
- Test classes: Test{ClassName} (e.g., TestDualDetectorRunner)
- Test methods: test_{function_name}_{scenario} (e.g., test_process_frame_charuco_detected)

**Coverage Requirements**:
- Overall: ≥80% coverage
- DualDetectorRunner: ≥85% (complex orchestration logic)
- ComparisonMetrics: ≥90% (pure functions, easier to test)
- ComparisonLogger: ≥80% (I/O operations)

**Mocking Strategy**:
- Mock cv2 operations (use conftest.py fixtures)
- Mock file I/O for logger tests
- Use real session_001 data for integration tests (validate end-to-end flow)
    </standards>

    <locations>
tests/validation/test_dual_detector_runner.py
tests/validation/test_comparison_metrics.py
tests/validation/test_comparison_logger.py
    </locations>

    <ideas>
**AC-1: DualDetectorRunner Orchestration**
- Test initialization with valid configs (ChArUco board, camera YAML, detector config)
- Test initialization with missing/invalid configs (should raise appropriate errors)
- Test set_baseline() stores both ChArUco pose and cam-shift baseline
- Test process_frame() with ChArUco detected (returns valid DualDetectionResult)
- Test process_frame() with ChArUco NOT detected (graceful handling, no crash)
- Test displacement calculations match expected values
- Test agreement classification (GREEN when diff ≤ threshold, RED when > threshold)

**AC-2: Comparison Metrics Calculations**
- Test calculate_displacement_difference() with d1=d2 (should return 0.0)
- Test calculate_displacement_difference() with known values (e.g., d1=18.5, d2=15.2 → 3.3)
- Test calculate_threshold() for 640×480 image (should return 14.4px = 0.03 * 480)
- Test classify_agreement() at boundary (diff = threshold should be GREEN)
- Test calculate_charuco_displacement_2d() projection math with known camera matrix
- Test calculate_mse() with known sequence (e.g., [10, 12] vs [11, 13] → MSE = 1.0)

**AC-3: Comparison Logger**
- Test log_frame() appends result to internal list
- Test save_log() creates valid JSON file with expected structure
- Test calculate_mse() returns correct value for logged sequence
- Test get_worst_matches(n=5) returns top 5 by displacement_diff (descending order)
- Test get_worst_matches() handles edge case (n > total frames)
- Test generate_mse_graph() creates PNG file at specified path
- Test generate_mse_graph() includes threshold line and worst match highlights

**AC-4: Integration Tests**
- End-to-end test with session_001 frames (load first frame as baseline, process 10 frames)
- Verify JSON log structure matches expected format
- Verify MSE graph file exists and is valid PNG
- Verify worst matches are correctly identified
    </ideas>
  </tests>
</story-context>
