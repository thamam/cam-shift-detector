<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>comparison-tool</epicId>
    <storyId>story-comparison-tool-2</storyId>
    <title>Tool Integration & Testing</title>
    <status>Draft</status>
    <generatedAt>2025-10-27</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>docs/stories/story-comparison-tool-2.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>QA personnel</asA>
    <iWant>a standalone comparison tool with dual display windows and multiple operation modes</iWant>
    <soThat>I can visually validate detector agreement in both offline validation and online debugging scenarios</soThat>
    <tasks>
### Task 1: Create Main Executable (AC: #1)
- Create `tools/validation/comparison_tool.py` with argparse CLI
- Define CLI arguments (--mode, --input-dir, --camera-id, --camera-yaml, --charuco-config, --camshift-config, --output-dir, --show-features)
- Implement argument validation with helpful error messages
- Create main() entry point routing to run_offline_comparison() or run_online_comparison()

### Task 2: Implement Offline Mode (AC: #2)
- Load comparison_config.json and camera.yaml
- Initialize DualDetectorRunner
- Glob images from input_dir (sorted)
- Set baseline from first image
- Process each frame and update dual display windows
- Log results and generate MSE graph + worst matches report

### Task 3: Implement Online Mode (AC: #3)
- Load configurations and initialize DualDetectorRunner
- Open cv2.VideoCapture(camera_id)
- Display instructions and wait for 's' keypress for baseline
- Continuous loop with frame processing and dual display updates
- Clean exit on 'q' with graph and report generation

### Task 4: Create Display Windows (AC: #4)
- Implement create_display_windows() with ChArUco and Cam-shift frame annotation
- Implement draw_comparison_status_bar() with colored bar and metrics
- Combine frames horizontally with status bar below

### Task 5: Configuration and Documentation (AC: #5)
- Create comparison_config.json with ChArUco board and display settings
- Create tools/validation/README.md with installation, usage, and troubleshooting

### Task 6: Integration Testing (AC: #6)
- Create test_comparison_tool_integration.py for offline mode end-to-end testing
- Manual online mode test protocol
- Performance benchmark verification
</tasks>
  </story>

  <acceptanceCriteria>
**AC-1: Main Executable CLI**
- comparison_tool.py accepts --mode [offline|online] argument
- Offline mode accepts: --input-dir, --camera-yaml, --charuco-config, --camshift-config, --output-dir
- Online mode accepts: --camera-id, --camera-yaml, --charuco-config, --camshift-config, --output-dir
- Validates all required arguments before execution
- Displays helpful error messages for missing/invalid arguments

**AC-2: Offline Mode Implementation**
- Loads all images from --input-dir directory (sorted by filename)
- Sets baseline from first image
- Processes all frames sequentially
- Updates dual display windows for each frame (ChArUco left, Cam-shift right)
- Displays frame counter, FPS, and comparison status bar
- Logs all results to JSON in --output-dir
- Generates MSE graph on completion
- Generates worst matches report on completion
- Processes 157 frames in <30 seconds (~5 FPS minimum)

**AC-3: Online Mode Implementation**
- Opens camera capture from --camera-id (default: 0)
- Displays live preview with instructions: "Press 's' to set baseline, 'q' to quit"
- User presses 's' → captures baseline for both detectors
- Continuously processes frames and updates dual display windows
- Maintains 15-20 FPS on live feed
- Logs results to JSON in real-time
- User presses 'q' → generates MSE graph and worst matches report
- Cleanly closes camera and windows on exit

**AC-4: Display Windows**
- Two OpenCV windows side-by-side: "ChArUco Detector" and "Cam-Shift Detector"
- ChArUco window shows: frame with pose axes (if detected), displacement overlay, status overlay, confidence overlay
- Cam-shift window shows: frame with ORB features (if --show-features enabled), displacement overlay, status overlay, confidence overlay
- Status bar below windows shows: comparison metric, threshold, frame info, FPS
- Green status bar when agreement ≤ threshold, red when > threshold
- Window synchronization latency <100ms

**AC-5: Configuration and Documentation**
- comparison_config.json created with default ChArUco board parameters
- tools/validation/README.md documents installation, offline/online usage, output structure, troubleshooting

**AC-6: Integration Testing**
- End-to-end offline test with session_001 data (157 frames)
- End-to-end online test with webcam (manual verification)
- JSON log structure validation
- MSE graph generation validation
- Worst matches retrieval accuracy
- Performance benchmarks met (FPS requirements)
</acceptanceCriteria>

  <artifacts>
    <docs>
      <!-- Primary Technical Specification -->
      <doc>
        <path>tech-spec.md</path>
        <title>Technical Specification: ChArUco vs Cam-Shift Validation Comparison Tool</title>
        <section>Section 5.4: comparison_tool.py Implementation</section>
        <snippet>Definitive technical specification for main executable with CLI, offline/online modes, dual display windows, and integration with Story 1 modules. Includes UI design, execution flow, and performance requirements.</snippet>
      </doc>

      <!-- Epic Overview -->
      <doc>
        <path>epics.md</path>
        <title>Epic: Validation Comparison Tool</title>
        <section>Story 2 Summary and Dependencies</section>
        <snippet>Defines Story 2 dependencies on Story 1, key deliverables (comparison_tool.py, README.md, comparison_config.json), and integration testing requirements. Total 3 story points, 2-3 days estimate.</snippet>
      </doc>

      <!-- Story 1 Context (completed infrastructure) -->
      <doc>
        <path>docs/stories/story-comparison-tool-1.context.md</path>
        <title>Story 1 Context: Core Comparison Infrastructure</title>
        <section>Implementation Artifacts</section>
        <snippet>Complete context for Story 1 infrastructure modules (DualDetectorRunner, ComparisonMetrics, ComparisonLogger) that Story 2 integrates. Provides API interfaces and usage patterns.</snippet>
      </doc>
    </docs>

    <code>
      <!-- Story 1 Infrastructure (completed) -->
      <code>
        <path>validation/utilities/dual_detector_runner.py</path>
        <kind>orchestration module</kind>
        <symbol>DualDetectorRunner</symbol>
        <lines>1-268</lines>
        <reason>Core infrastructure from Story 1. Main executable imports and uses this for dual detector orchestration. API: __init__, set_baseline(), process_frame() returning DualDetectionResult.</reason>
      </code>
      <code>
        <path>validation/utilities/comparison_logger.py</path>
        <kind>logging module</kind>
        <symbol>ComparisonLogger</symbol>
        <lines>1-244</lines>
        <reason>Logging and analysis from Story 1. Main executable uses for log_frame(), save_log(), calculate_mse(), get_worst_matches(), generate_mse_graph().</reason>
      </code>

      <!-- Display and Camera I/O Patterns -->
      <code>
        <path>validation/core/stage2_charuco_validation.py</path>
        <kind>validation module</kind>
        <symbol>run_detector_validation</symbol>
        <lines>283-478</lines>
        <reason>Reference implementation for frame processing loop pattern. Shows how to iterate frames, process with detector, calculate FPS, and display results. Pattern for offline mode implementation.</reason>
      </code>
      <code>
        <path>tools/aruco/camshift_annotator.py</path>
        <kind>utility module</kind>
        <symbol>main</symbol>
        <lines>324-515</lines>
        <reason>Reference implementation for camera capture and OpenCV display pattern. Shows cv2.VideoCapture usage, live preview, keypress handling, and window management. Pattern for online mode implementation.</reason>
      </code>
      <code>
        <path>tools/aruco/camshift_annotator.py</path>
        <kind>utility module</kind>
        <symbol>draw_axes</symbol>
        <lines>92-105</lines>
        <reason>Shows how to draw ChArUco pose axes on frame using cv2.drawFrameAxes(). Required for ChArUco window visualization in AC-4.</reason>
      </code>

      <!-- Performance Measurement Pattern -->
      <code>
        <path>validation/utilities/performance_profiler.py</path>
        <kind>utility module</kind>
        <symbol>PerformanceProfiler</symbol>
        <lines>20-269</lines>
        <reason>Reference pattern for FPS measurement and timing. Shows how to track frame timing, calculate FPS metrics, and display performance info. Pattern for status bar FPS display.</reason>
      </code>

      <!-- JSON Configuration Pattern -->
      <code>
        <path>validation/utilities/report_generator.py</path>
        <kind>utility module</kind>
        <symbol>JSONReportGenerator</symbol>
        <lines>26-234</lines>
        <reason>Reference pattern for JSON configuration and report structure. Shows how to load/save structured JSON data. Pattern for comparison_config.json structure.</reason>
      </code>
    </code>

    <dependencies>
      <python version=">=3.11">
        <package>opencv-python==4.10.0.84</package>
        <package>opencv-contrib-python>=4.12.0.88</package>
        <package>numpy==1.26.4</package>
        <package>matplotlib==3.9.2</package>
        <package>psutil>=5.9.0</package>
      </python>
    </dependencies>
  </artifacts>

  <constraints>
    <!-- Technical Constraints -->
    - **Story 1 Dependency**: MUST complete story-comparison-tool-1 before implementation (DualDetectorRunner, ComparisonMetrics, ComparisonLogger required)
    - **Display Window Layout**: Horizontal split (ChArUco left, Cam-shift right) with status bar below. Window synchronization latency <100ms.
    - **Baseline Capture UX**: Offline mode uses first frame automatically, online mode waits for user 's' keypress
    - **Frame Synchronization**: Process both detectors sequentially (not parallel) to ensure same frame
    - **Configuration File**: Centralized comparison_config.json with ChArUco board params, display settings, logging settings

    <!-- Performance Constraints -->
    - **Offline Mode Performance**: Process 157 frames in <30 seconds (≥5.2 FPS minimum)
    - **Online Mode Performance**: Maintain 15-20 FPS on live camera feed
    - **Display Latency**: Window updates must synchronize within 100ms

    <!-- Testing Constraints -->
    - **Integration Test Data**: Use session_001/frames/ (203 frames, 158 with ChArUco detection) for offline mode testing
    - **Manual Testing**: Online mode requires webcam hardware and manual verification protocol
    - **Performance Benchmarks**: Offline FPS, online FPS, display latency measurements required

    <!-- Code Organization Constraints -->
    - **Module Location**: Main executable in tools/validation/ (not validation/core/ or src/)
    - **Configuration Location**: comparison_config.json in project root
    - **Documentation Location**: tools/validation/README.md with usage examples
    - **Project-Relative Paths**: All file paths in code must be project-relative, not absolute
  </constraints>

  <interfaces>
    <!-- Story 1 Infrastructure APIs (already implemented) -->
    <interface>
      <name>DualDetectorRunner.__init__</name>
      <kind>constructor</kind>
      <signature>__init__(camera_yaml_path: str, camshift_config_path: str, charuco_squares_x: int = 7, charuco_squares_y: int = 5, charuco_square_len_m: float = 0.035, charuco_marker_len_m: float = 0.026, charuco_dict_name: str = "DICT_4X4_50", z_distance_m: float = 1.15)</signature>
      <path>validation/utilities/dual_detector_runner.py:48-127</path>
      <returns>DualDetectorRunner instance</returns>
    </interface>

    <interface>
      <name>DualDetectorRunner.set_baseline</name>
      <kind>method</kind>
      <signature>set_baseline(image: np.ndarray) -> bool</signature>
      <path>validation/utilities/dual_detector_runner.py:129-167</path>
      <returns>True if baselines set successfully (ChArUco detected), False otherwise</returns>
    </interface>

    <interface>
      <name>DualDetectorRunner.process_frame</name>
      <kind>method</kind>
      <signature>process_frame(image: np.ndarray, frame_id: Optional[str] = None) -> DualDetectionResult</signature>
      <path>validation/utilities/dual_detector_runner.py:169-268</path>
      <returns>DualDetectionResult with both detector outputs and comparison metrics</returns>
    </interface>

    <interface>
      <name>ComparisonLogger.__init__</name>
      <kind>constructor</kind>
      <signature>__init__(output_dir: str, session_name: str)</signature>
      <path>validation/utilities/comparison_logger.py:29-49</path>
      <returns>ComparisonLogger instance</returns>
    </interface>

    <interface>
      <name>ComparisonLogger.log_frame</name>
      <kind>method</kind>
      <signature>log_frame(result: DualDetectionResult) -> None</signature>
      <path>validation/utilities/comparison_logger.py:51-59</path>
      <returns>None (logs to internal list)</returns>
    </interface>

    <interface>
      <name>ComparisonLogger.save_log</name>
      <kind>method</kind>
      <signature>save_log(filename: Optional[str] = None) -> Path</signature>
      <path>validation/utilities/comparison_logger.py:61-119</path>
      <returns>Path to saved JSON file</returns>
    </interface>

    <interface>
      <name>ComparisonLogger.generate_mse_graph</name>
      <kind>method</kind>
      <signature>generate_mse_graph(output_path: Optional[str] = None, highlight_worst_n: int = 10) -> Path</signature>
      <path>validation/utilities/comparison_logger.py:158-244</path>
      <returns>Path to saved PNG file</returns>
    </interface>

    <!-- OpenCV Display APIs -->
    <interface>
      <name>cv2.VideoCapture</name>
      <kind>class</kind>
      <signature>cv2.VideoCapture(camera_id: int)</signature>
      <path>opencv-python (external library)</path>
      <returns>VideoCapture object for camera I/O</returns>
    </interface>

    <interface>
      <name>cv2.imshow</name>
      <kind>function</kind>
      <signature>cv2.imshow(window_name: str, image: np.ndarray) -> None</signature>
      <path>opencv-python (external library)</path>
      <returns>None (displays image in window)</returns>
    </interface>

    <interface>
      <name>cv2.waitKey</name>
      <kind>function</kind>
      <signature>cv2.waitKey(delay: int) -> int</signature>
      <path>opencv-python (external library)</path>
      <returns>ASCII code of pressed key (or -1 if no key pressed)</returns>
    </interface>

    <interface>
      <name>cv2.drawFrameAxes</name>
      <kind>function</kind>
      <signature>cv2.drawFrameAxes(image, K, dist, rvec, tvec, length)</signature>
      <path>opencv-python (external library)</path>
      <returns>Image with drawn axes</returns>
    </interface>
  </interfaces>

  <tests>
    <standards>
**Testing Framework**: pytest with coverage plugin (pytest-cov)

**Test Organization**:
- Integration tests: tests/validation/test_comparison_tool_integration.py
- Manual test protocol: Documented in tools/validation/README.md for online mode
- Test data: session_001/frames/ (203 frames for offline mode testing)

**Naming Conventions**:
- Test files: test_{module_name}_integration.py
- Test classes: Test{FeatureName}Integration
- Test methods: test_{mode}_{scenario} (e.g., test_offline_mode_with_session_001_data)

**Testing Strategy**:
- Offline mode: Full end-to-end automated testing with session_001 data
- Online mode: Manual test protocol with webcam (requires human verification)
- Performance: Benchmark measurements for FPS and display latency

**Mocking Strategy**:
- Mock cv2.VideoCapture for offline mode tests (no camera required)
- Use real session_001 data for offline integration tests
- Manual testing required for online mode (camera hardware dependency)
    </standards>

    <locations>
tests/validation/test_comparison_tool_integration.py
tools/validation/README.md (Manual test protocol for online mode)
    </locations>

    <ideas>
**AC-1: Main Executable CLI**
- Test CLI argument parsing with valid arguments (both offline and online modes)
- Test argument validation with missing required arguments (should display helpful error)
- Test argument validation with invalid mode (should reject)
- Test help message display (--help flag)

**AC-2: Offline Mode Implementation**
- Test offline mode end-to-end with session_001 data (157 frames)
- Verify JSON log created with correct structure and all 157 entries
- Verify MSE graph PNG file created at expected path
- Verify worst matches correctly identified and logged
- Verify processing time <30 seconds (FPS ≥5.2)
- Test offline mode with empty directory (should display error)
- Test offline mode with missing baseline detection (graceful handling)

**AC-3: Online Mode Implementation**
- Manual test: Open webcam and verify live preview displays
- Manual test: Press 's' to set baseline and verify confirmation message
- Manual test: Verify dual windows update in real-time
- Manual test: Measure FPS (should be 15-20 FPS)
- Manual test: Press 'q' to exit and verify graph/report generation
- Manual test: Verify clean camera/window shutdown

**AC-4: Display Windows**
- Test dual window creation and layout (side-by-side)
- Verify ChArUco window shows pose axes when detected
- Verify Cam-shift window shows ORB features when --show-features enabled
- Verify status bar color (green when agreement ≤ threshold, red when > threshold)
- Verify status bar text includes all required metrics (comparison, threshold, frame info, FPS)
- Test window synchronization latency <100ms

**AC-5: Configuration and Documentation**
- Verify comparison_config.json created with correct JSON structure
- Verify ChArUco board parameters match tech-spec (7×5, 0.035m, 0.026m, DICT_4X4_50)
- Verify tools/validation/README.md exists and contains all required sections
- Test loading comparison_config.json and validating structure

**AC-6: Integration Testing**
- End-to-end offline test: session_001 → JSON log + MSE graph + worst matches
- Verify JSON log structure matches expected format from ComparisonLogger
- Verify MSE graph is valid PNG with correct dimensions
- Verify worst matches are sorted descending by displacement_diff
- Performance benchmark: Measure offline FPS, online FPS, display latency
    </ideas>
  </tests>
</story-context>
