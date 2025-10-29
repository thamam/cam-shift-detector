# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-10-29

### Added

**Epic 3: Dual Detector Comparison Tool**
- Side-by-side comparison tool for ChArUco vs cam-shift detectors
- Dual OpenCV display windows with synchronized output
- Offline mode: Process recorded image sequences from directories
- Online mode: Real-time comparison with live camera feeds
- Comparison metrics with 3% agreement threshold (||d1-d2||_2)
- JSON logging for post-analysis and trend detection
- MSE (Mean Squared Error) graph generation
- Worst matches identification and retrieval
- Performance: 8.4 FPS offline processing (exceeds 5 FPS requirement)
- Comprehensive documentation: `tools/validation/README.md`
- Test coverage: 100% integration tests (14 tests passing)

**Epic 4: Interactive Debugging Tools (Stage 4)**
- **Mode A - 4-Quadrant Comparison Tool** (`tools/validation/comparison_tool.py`)
  - Simultaneous ChArUco and cam-shift detection visualization
  - 4-image layout: baseline/current × charuco/csd
  - Feature overlay: ChArUco corners (cyan) + ORB features (yellow)
  - Manual frame stepping (FWD/BKWD arrows)
  - Enhanced metrics display (Δdx, Δdy, error magnitude)
  - CSV export with comprehensive detection data
  - PNG snapshot capability

- **Mode B - Baseline Correspondence Tool** (`tools/validation/baseline_correspondence_tool.py`)
  - Motion vector visualization with arrow overlays
  - Inlier/outlier coloring (green/red) based on RANSAC mask
  - Baseline pinning mechanism for drift analysis
  - Match quality metrics (RMSE, inlier percentage)
  - Difference heatmap overlay for visual shift verification
  - Diagnostic mode with keypoint density visualization
  - 29 comprehensive unit tests (100% coverage)

- **Mode C - Enhanced Alpha Blending Tool** (`tools/validation/alpha_blending_tool.py`)
  - Transform computation using CSD homography estimation
  - Pre-warp toggle (W key) for visual alignment verification
  - Blink mode (Space key) with 500ms A/B alternation
  - Frame selector for arbitrary frame pair comparison
  - Alpha blending with adjustable transparency (↑/↓ keys)
  - 10×10 alignment grid overlay (G key, cyan with 50% transparency)
  - Descriptive file naming: frameA_frameB_alpha_prewarp_timestamp.png
  - CSV export with full transform parameters
  - 34 comprehensive unit tests (59% coverage for non-GUI code)

**API Extensions (Minimal Surface Area)**
- `MovementDetector.get_last_matches()` - Retrieve matched feature coordinates for visualization
- `MovementDetector.get_last_homography()` - Retrieve transformation matrix from last detection
- `CameraMovementDetector.get_last_homography()` - Delegation method for transform access

### Improved

**Testing Coverage**
- Added 63 new tests for Stage 4 tools (Mode B: 29, Mode C: 34)
- Full test suite: 509 tests passing (99.6% success rate)
- Integration coverage for all three interactive modes
- Edge case handling: CSD failures, invalid frames, missing data

**Documentation**
- Enhanced `tools/validation/README.md` with Mode A/B/C usage examples
- Added keyboard shortcut reference for all interactive tools
- Comprehensive troubleshooting guide for common issues
- Performance optimization tips and best practices

### Technical Details

**Epic 4 Performance:**
- Tool initialization: <100ms for 200 frame sequences
- Frame stepping: Real-time (<50ms latency)
- Transform computation: <500ms (AC requirement met)
- Blink timing accuracy: ±50ms over 10 cycles

**Epic 4 Architecture:**
- Pure Python + OpenCV (no web UI overhead)
- Standalone tool pattern (one tool per mode)
- Reused existing infrastructure (comparison_tool, annotator patterns)
- Clean API extensions via delegation pattern
- No breaking changes to existing API

**Known Limitations:**
- Interactive tools require X11 display (no headless mode)
- Manual testing required for GUI behavior (automated tests cover non-GUI logic)
- Mode A integration tests less comprehensive than Mode B/C unit tests

For Epic 4 technical specifications, see [docs/epic-stage4-debug-tools.md](docs/epic-stage4-debug-tools.md).

---

## [0.1.0] - 2025-10-26

### Added

**Epic 1: Core Camera Shift Detection (100% detection on synthetic data)**
- Camera shift detection module using ORB feature matching with homography estimation
- Affine transformation model for improved detection accuracy
- Static ROI (Region of Interest) configuration and selection tool
- Manual recalibration capability for baseline reset
- Result history buffer maintaining last 100 detection results
- Configuration-driven threshold system (default: 2.0 pixels)
- JSON-based configuration file support
- Comprehensive unit test suite (109 tests, 93% coverage)
- Black-box API with 5 public methods:
  - `__init__(config_path)` - Initialize detector with configuration
  - `set_baseline(image_array)` - Capture reference baseline features
  - `process_frame(image_array, frame_id)` - Detect camera movement in frame
  - `recalibrate(image_array)` - Reset baseline features manually
  - `get_history(frame_id, limit)` - Query detection history

**Epic 2: Production Validation Framework**
- Real DAF image data loader with 50 sample images from 3 production sites
  - OF_JERUSALEM: 23 images (site: 9bc4603f-0d21-4f60-afea-b6343d372034)
  - CARMIT: 17 images (site: e2336087-0143-4895-bc12-59b8b8f97790)
  - GAD: 10 images (site: f10f17d4-ac26-4c28-b601-06c64b8a22a4)
- Ground truth annotation system with JSON-based validation data
- Stage 3 validation test harness with automated testing
- Performance profiling capabilities:
  - FPS (frames per second) measurement
  - Memory usage tracking (psutil integration)
  - CPU utilization monitoring
- Automated validation runner (`run_stage3_validation.py`)
- Dual-format reporting system:
  - JSON report (machine-readable metrics)
  - Markdown report (human-readable analysis)
- Go/no-go production readiness assessment with gate criteria:
  - Detection accuracy ≥95%
  - False positive rate ≤5%
  - Processing speed ≥0.0167 FPS (1 frame per 60 seconds)
  - Memory usage ≤500 MB

**Release Preparation**
- Comprehensive integration documentation (`documentation/integration-guide.md`)
- Step-by-step installation guide (`documentation/installation.md`)
- Integration cheat sheet for quick reference (`documentation/integration-cheat-sheet.md`)
- Stub implementation for parallel integration development
- Professional package metadata (pyproject.toml with PEP 621 compliance)
- MIT License for open source distribution
- Complete CHANGELOG following "Keep a Changelog" format

### Known Limitations

- **Single Camera Support**: Multi-camera scenarios not supported in v0.1.0 (deferred to future releases)
- **Manual Ground Truth**: Requires manual annotation for validation (no automatic ground truth generation)
- **Manual Recalibration Only**: No automatic recalibration on lighting changes (operator must trigger manually)
- **Static ROI**: ROI cannot be adjusted at runtime (requires configuration file update and restart)
- **CPU-Only Processing**: No GPU acceleration (sufficient for 1 frame/60s requirement)

### Technical Details

**Detection Algorithm:**
- ORB (Oriented FAST and Rotated BRIEF) feature detection
- FLANN-based feature matching with ratio test
- RANSAC-based affine transformation estimation
- Displacement calculation from transformation matrix

**Performance Targets:**
- Detection accuracy: ≥95% on real DAF imagery
- Processing speed: ≥0.0167 FPS (1 frame per 60 seconds)
- Memory footprint: ≤500 MB
- False positive rate: ≤5%

**Testing Coverage:**
- Unit tests: 109 tests, 93% code coverage
- Integration tests: Stage 3 validation framework
- Real-world validation: 50 DAF production images

For full technical specifications, see [docs/tech-spec-epic-MVP-001.md](docs/tech-spec-epic-MVP-001.md).

For integration guidance, see [documentation/integration-guide.md](documentation/integration-guide.md).

---

## Version History

- **0.2.0** (2025-10-29) - Added dual detector comparison tool (Epic 3) and interactive debugging tools (Epic 4: Modes A/B/C)
- **0.1.0** (2025-10-26) - Initial release with core detection and validation framework
