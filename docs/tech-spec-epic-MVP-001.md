# Technical Specification: Camera Movement Detection Module

Date: 2025-10-18
Author: Tomer
Epic ID: MVP-001
Status: Draft

---

## Overview

The Camera Movement Detection Module is a black-box computer vision component that monitors camera position in DAF (Dissolved Air Flotation) water quality monitoring systems. When cameras shift position, the neural network's Region of Interest (ROI) becomes misaligned, causing inaccurate turbidity and flow measurements. This module detects camera displacement exceeding 2 pixels and signals the parent DAF system to halt data collection until manual correction is performed.

The solution implements ORB feature matching on static image regions (tank walls, pipes, equipment), compares current frames to a baseline reference, and returns immediate validation status via a simple Python API. The MVP focuses on proving detection accuracy with minimal complexity: no automatic recalibration, no real-time monitoring, no UI—just reliable movement detection integrated via direct function calls.

## Objectives and Scope

**Primary Objective:** Prevent use of corrupted water quality measurements by detecting camera position changes ≥2 pixels with >95% accuracy.

**In Scope:**
- Static ROI definition tool (OpenCV GUI for manual region selection)
- ORB feature extraction from static regions (tank walls, pipes, equipment)
- Homography-based movement detection (simple method initially, RANSAC optional based on testing)
- Threshold-based validation (2.0 pixel displacement threshold)
- Direct API integration (`process_frame()` returns status dict)
- Manual recalibration capability
- In-memory history buffer (last 100 detection results)
- Basic validation testing (simulated transforms, real footage, live deployment)
- Single camera support only

**Out of Scope (Deferred to Post-MVP):**
- Multi-camera support
- Automatic recalibration or drift detection
- Real-time 1Hz monitoring (MVP uses 5-10 minute intervals)
- REST API endpoints (parent system reads direct return values)
- Rich UI (Qt5/Flask interfaces) - headless module only
- Persistent logging (SQLite, file-based logs)
- Scene quality monitoring (inlier ratio diagnostics)
- Comprehensive test suites (bubble/water dynamics testing)
- Full documentation suite (user/developer guides)

## System Architecture Alignment

**Integration Point:** The module integrates with the existing DAF system's camera interface, which already provides image arrays (NumPy format, H×W×3, uint8, BGR). No new camera capture infrastructure required.

**Architectural Constraints:**
- **Black-box design:** Module exposes only `CameraMovementDetector` class with 3 public methods
- **Stateless operation:** Each `process_frame()` call is independent; history buffer provides queryability without requiring state persistence
- **Synchronous execution:** Parent system calls detector synchronously every 5-10 minutes during existing measurement intervals
- **No side effects:** Returns status directly; no flag files, no external REST calls, no database writes
- **Configuration-driven:** ROI coordinates and thresholds defined in `config.json`, version-controlled with codebase

**Component Dependencies:**
- Static Region Manager → Config file (`config.json`)
- Feature Extractor → OpenCV ORB implementation
- Movement Detector → Homography estimation (simple, no RANSAC initially)
- Result Manager → In-memory FIFO buffer (no persistence layer)

**Data Flow:** Image array (input) → Generate static mask → Extract features (masked) → Match to baseline → Calculate displacement → Build result dict → Store in history → Return result (output)

## Detailed Design

### Services and Modules

| Module | File | Responsibility | Inputs | Outputs | Owner |
|--------|------|----------------|--------|---------|-------|
| **CameraMovementDetector** | `src/camera_movement_detector.py` | Main black-box API, orchestrates all components | `image_array`, `frame_id` | Result dict with status, confidence | Core |
| **StaticRegionManager** | `src/static_region_manager.py` | Load ROI config, generate binary masks for static region | Config file, image_shape | Binary mask (H×W, uint8: 255=static, 0=dynamic) | Core |
| **FeatureExtractor** | `src/feature_extractor.py` | Extract/manage ORB features, store baseline | Full image, binary mask | Keypoints, descriptors | Core |
| **MovementDetector** | `src/movement_detector.py` | Compare features via homography, calculate displacement & confidence | Baseline features, current features | `(moved: bool, displacement: float, confidence: float)` | Core |
| **ResultManager** | `src/result_manager.py` | Build result dicts, maintain FIFO history buffer | Displacement, confidence, frame_id | Result dict, history queries | Core |
| **ROI Selection Tool** | `tools/select_roi.py` | Interactive GUI for defining static regions | Image (camera/file) | `config.json` with ROI coordinates | Utility |
| **Recalibration Script** | `tools/recalibrate.py` | Manual baseline reset helper | TBD (define during story creation) | Success/failure status | Utility |

### Data Models and Contracts

**Configuration Schema** (`config.json`):
```json
{
  "roi": {
    "x": 100,              // ROI top-left X coordinate (pixels)
    "y": 50,               // ROI top-left Y coordinate (pixels)
    "width": 400,          // ROI width (pixels)
    "height": 300          // ROI height (pixels)
  },
  "threshold_pixels": 2.0,         // Displacement threshold (float)
  "history_buffer_size": 100,      // FIFO buffer size (int)
  "min_features_required": 50      // Minimum ORB features in ROI (int)
}
```

**Detection Result Schema**:
```python
{
  "status": str,          # "VALID" | "INVALID"
  "displacement": float,  # Magnitude in pixels (rounded to 2 decimals)
  "confidence": float,    # Confidence score [0.0, 1.0] based on inlier ratio
  "frame_id": str,        # Identifier from caller or auto-generated
  "timestamp": str        # ISO 8601 UTC (e.g., "2025-10-18T14:32:18.456Z")
}
```

**Confidence Score Calculation:**
- Based on inlier ratio: `confidence = num_inliers / total_matches`
- Range: [0.0, 1.0] where 1.0 = all matched points agree on transformation
- Low confidence (< 0.5) suggests scene changes, lighting shifts, or ambiguous features
- Allows DAF system to make nuanced decisions even when `status="VALID"`

**Feature Data Structures**:
```python
# ORB Keypoints (cv2.KeyPoint objects)
keypoints: List[cv2.KeyPoint]

# ORB Descriptors (NumPy array, shape: (n_features, 32), dtype: uint8)
descriptors: np.ndarray

# Baseline Features Tuple
baseline_features: Tuple[List[cv2.KeyPoint], np.ndarray]
```

**History Buffer** (in-memory only):
```python
# FIFO deque with maxlen=100
history: deque[Dict]  # Each entry is a Detection Result dict
```

### APIs and Interfaces

**CameraMovementDetector Public API**:

```python
class CameraMovementDetector:
    """Black-box interface for DAF system integration"""

    def __init__(self, config_path: str = 'config.json') -> None:
        """
        Initialize detector with configuration.

        Args:
            config_path: Path to JSON config file with ROI and parameters

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config validation fails
        """

    def process_frame(self, image_array: np.ndarray, frame_id: str = None) -> Dict:
        """
        Detect camera movement in single frame.

        Args:
            image_array: NumPy array (H × W × 3, uint8, BGR format)
            frame_id: Optional identifier for tracking (auto-generated if None)

        Returns:
            {
                "status": "VALID" | "INVALID",
                "displacement": float,  # pixels
                "confidence": float,    # [0.0, 1.0] inlier ratio
                "frame_id": str,
                "timestamp": str  # ISO 8601 UTC
            }

        Raises:
            RuntimeError: If baseline not set (call set_baseline() first)
            ValueError: If image_array invalid format
        """

    def get_history(self, frame_id: str = None, limit: int = None) -> List[Dict]:
        """
        Query detection history buffer.

        Args:
            frame_id: Return results for specific frame_id (optional)
            limit: Return last N results (optional)

        Returns:
            List of detection result dicts (empty list if no matches)
        """

    def recalibrate(self, image_array: np.ndarray = None) -> bool:
        """
        Manually reset baseline features.

        Args:
            image_array: New reference image (required)

        Returns:
            True if successful, False otherwise
        """

    def set_baseline(self, image_array: np.ndarray) -> None:
        """
        Capture initial baseline features (setup phase).

        Args:
            image_array: Reference image for baseline

        Raises:
            ValueError: If insufficient features detected (<50)
        """
```

**Integration Example**:
```python
# DAF System Integration
from src.camera_movement_detector import CameraMovementDetector

# Initialize once at startup
detector = CameraMovementDetector('config.json')
detector.set_baseline(initial_camera_frame)

# Runtime: Called every 5-10 minutes during measurement cycle
def measurement_cycle():
    image = camera_interface.get_frame()  # Your existing method
    result = detector.process_frame(image, frame_id=f"frame_{timestamp}")

    if result['status'] == 'INVALID':
        logger.warning(f"Camera moved {result['displacement']:.2f}px - halting measurements")
        return None  # Skip this measurement
    elif result['confidence'] < 0.5:
        logger.warning(f"Low confidence {result['confidence']:.2f} - proceed with caution")
        return vision_analysis(image)  # Proceed but flag for review
    else:
        return vision_analysis(image)  # Proceed with water quality analysis
```

### Workflows and Sequencing

**Setup Workflow**:
```
1. Operator downloads recent image from cloud (or uses local camera)
2. Run: python tools/select_roi.py --source image --path site_image.jpg
3. GUI displays image → Operator clicks/drags to define static region
4. Tool validates: ≥50 ORB features detected in ROI?
   - Yes → Save config.json with ROI coordinates
   - No → Warning, request different region or larger area
5. Deploy config.json to site
6. Initialize detector: detector = CameraMovementDetector('config.json')
7. Capture baseline: detector.set_baseline(initial_image)
```

**Runtime Detection Workflow** (Every 5-10 minutes):
```
┌─────────────────────────────────────────────────────────────┐
│ DAF System Measurement Cycle                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │ Get camera frame        │
    │ image = get_frame()     │
    └──────────┬──────────────┘
               │
               ▼
    ┌─────────────────────────────────────────────┐
    │ Call detector                               │
    │ result = detector.process_frame(image, id)  │
    └──────────┬──────────────────────────────────┘
               │
               ├──► Static Region Manager: Generate binary mask
               │
               ├──► Feature Extractor: Extract ORB features
               │
               ├──► Movement Detector: Match to baseline via homography
               │                        Calculate displacement
               │
               ├──► Result Manager: displacement > 2px?
               │                     Build result dict
               │                     Store in history buffer
               │
               ▼
    ┌─────────────────────────────────────┐
    │ Return result dict to DAF system    │
    └──────────┬──────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Check status │
        └──┬───────┬───┘
           │       │
   INVALID │       │ VALID
           │       │
           ▼       ▼
    ┌──────────┐  ┌────────────────────────┐
    │ Halt     │  │ Proceed with water     │
    │ measurements│  │ quality analysis       │
    │ Log warning │  │                        │
    └──────────┘  └────────────────────────┘
```

**Manual Recalibration Workflow**:
```
Trigger: Lighting change, maintenance, operator decision

1. Operator runs: python tools/recalibrate.py
   OR calls: detector.recalibrate(current_image)

2. Module:
   - Generates binary mask for static region (using existing config)
   - Extracts new ORB features (with mask)
   - Validates ≥50 features detected
   - Replaces baseline_features in memory
   - Optionally clears history buffer

3. Returns: success (True) or failure (False)

4. Resume normal detection
```

## Non-Functional Requirements

### Performance

**Latency Requirements**:
- `process_frame()` execution time: <500ms per call (target: <200ms)
- ROI selection tool: Interactive response <100ms for GUI updates
- Baseline capture (`set_baseline()`): <2 seconds

**Throughput Requirements**:
- Detection frequency: Once every 5-10 minutes (no real-time requirement for MVP)
- History buffer queries: <10ms for `get_history()` calls

**Resource Constraints**:
- Memory footprint: <100MB for detector process
- CPU usage: <5% during idle, <50% during `process_frame()` execution
- Disk space: <10MB for codebase + dependencies (excluding Python/OpenCV)

**Scalability**:
- MVP supports single camera only
- History buffer fixed at 100 entries (configurable via config)

**Performance Targets** (PRD Section 7):
- Detection accuracy: >95% for movements ≥2 pixels (Stage 1 testing)
- False positive rate: <5% in live deployment (Stage 3 testing)
- False negative rate: 0% for real camera shifts (Stage 2 testing)

### Security

**Authentication/Authorization**:
- No authentication required (module runs within trusted DAF system environment)
- Access control managed by parent system, not module

**Data Handling**:
- Image data processed in-memory only (no persistence to disk in MVP)
- History buffer contains detection results only (no raw image data stored)
- Baseline features optionally saved to `data/baseline_features.pkl` (local filesystem only)
- Config file (`config.json`) contains only ROI coordinates and thresholds (no sensitive data)

**Input Validation**:
- Validate image_array format: Must be NumPy array, H×W×3, uint8, BGR
- Validate config.json schema on initialization
- Validate ROI coordinates are within image bounds
- Validate minimum feature count (≥50) during baseline capture

**Threat Model**:
- **Not in scope for MVP**: Network security, encrypted storage, access logging
- **Assumption**: Module operates in physically secured facility with controlled network access
- **Risk**: Malformed image_array could cause OpenCV crashes → Mitigation: Input validation

**Data Retention**:
- History buffer: In-memory only, cleared on process restart
- Baseline features: Persist across restarts if saved to disk (optional)
- No PII or sensitive operational data collected

### Reliability/Availability

**Uptime Requirements**:
- Target: 99% uptime during DAF operational hours
- Acceptable downtime: Planned maintenance windows only
- MVP Success Criterion (PRD Section 7): Stable operation for 1 week continuous without crashes

**Error Handling**:
- **Insufficient features (<50)**: Raise `ValueError` during baseline capture, require operator intervention
- **Homography estimation failure**: Return `status="INVALID"` with `displacement=inf`, log warning
- **Invalid image format**: Raise `ValueError` with clear message
- **Config file missing/invalid**: Raise appropriate exception at initialization

**Failure Modes**:
| Failure Scenario | Behavior | Recovery |
|------------------|----------|----------|
| Baseline not set | Raise `RuntimeError` on `process_frame()` | Call `set_baseline()` |
| Too few feature matches (<10) | Return `status="INVALID"`, `displacement=inf` | Manual recalibration needed |
| Config file corrupt | Initialization fails with exception | Fix config.json, restart |
| OpenCV crash | Process terminates, no graceful degradation | Restart detector |

**Degradation Strategy**:
- MVP has no graceful degradation (fail-fast design)
- On critical error: Return `status="INVALID"` or raise exception
- Parent DAF system responsible for handling detector failures

**Recovery**:
- Restart detector process (stateless design enables fast recovery)
- Re-initialize with existing config.json
- Recapture baseline if needed

### Observability

**Logging Requirements** (Minimal for MVP):
- **No structured logging system** (SQLite, file-based logs out of scope for MVP)
- **Ad-hoc logging only**: Print statements or basic Python logging if needed during development/debugging
- **Log Levels** (if implemented):
  - ERROR: Critical failures (e.g., baseline not set, config invalid)
  - WARNING: Homography failures, insufficient matches
  - INFO: Baseline capture, recalibration events (optional)
  - DEBUG: Feature counts, displacement calculations (optional)

**Metrics** (Not collected in MVP):
- Detection latency, false positive rate, false negative rate tracked manually via testing
- No automated metrics collection or monitoring dashboard

**Tracing**:
- Frame IDs enable tracing individual detections through history buffer
- No distributed tracing (single-process module)

**Debugging Aids**:
- History buffer provides queryable record of recent detections
- `get_history()` method for post-hoc analysis
- Config file version-controlled for reproducibility

**Monitoring Strategy**:
- MVP relies on parent DAF system to monitor detector health
- Detection: DAF system observes `status="INVALID"` returns
- Availability: DAF system detects if detector process crashes (via exception handling)

## Dependencies and Integrations

**External Dependencies** (`requirements.txt`):
```
opencv-python>=4.8.0,<5.0.0      # Computer vision library (ORB, homography)
numpy>=1.24.0,<2.0.0             # Array operations, image manipulation
```

**Python Version**:
- Python 3.8+ (PRD Section 6)
- No Python 2 compatibility required

**System Dependencies**:
- None (OpenCV bundles all required binaries)
- GUI support for ROI selection tool (X11/Wayland on Linux, native on Windows/macOS)

**Integration Points**:

| Integration | Type | Direction | Protocol | Notes |
|-------------|------|-----------|----------|-------|
| DAF Camera Interface | Internal | Input | Direct function call | Existing interface provides image arrays |
| Config File | External | Input | JSON file read | `config.json` with ROI coordinates |
| History Buffer | Internal | Storage | In-memory (deque) | No external persistence |

**Existing DAF System Dependencies**:
- Camera interface method: `get_frame()` or equivalent (assumed to exist per PRD)
- Image format: NumPy arrays (H×W×3, uint8, BGR)
- Measurement cycle: 5-10 minute intervals

**No External Services**:
- No REST API calls
- No database connections
- No cloud service integrations
- No message queues

## Acceptance Criteria (Authoritative)

These criteria are derived from PRD Section 7 (Success Criteria) and must ALL be met for MVP acceptance:

1. **AC-001: Detection Accuracy** - System detects camera movements ≥2 pixels with >95% accuracy in Stage 1 testing (simulated transforms)

2. **AC-002: Zero False Negatives** - System detects 100% of real camera shifts in Stage 2 testing (recorded footage with known movements)

3. **AC-003: Low False Positive Rate** - False positive rate <5% during Stage 3 testing (1-week live deployment monitoring)

4. **AC-004: API Integration** - DAF system successfully integrates via `process_frame()` API, receives status dict, and halts measurements when `status="INVALID"`

5. **AC-005: Manual Recalibration** - Operator can complete manual recalibration in <2 minutes using provided tool or API method

6. **AC-006: System Stability** - Detector runs stable for 1 week continuous operation without crashes or memory leaks

7. **AC-007: History Buffer** - System maintains last 100 detection results in memory; `get_history()` queries work correctly by frame_id or limit parameter

8. **AC-008: ROI Selection Tool** - Tool enables operator to define static region, validates ≥50 features, saves valid `config.json`

9. **AC-009: Baseline Capture** - `set_baseline()` successfully captures baseline features with validation (≥50 features required)

10. **AC-010: Error Handling** - System raises appropriate exceptions for invalid inputs, missing baseline, or config errors with clear messages

11. **AC-011: Confidence Score** - System returns confidence score [0.0, 1.0] in result dict based on feature match quality (inlier ratio or match count)

## Traceability Mapping

| AC | Spec Section(s) | Component(s)/API(s) | Test Idea |
|----|----------------|---------------------|-----------|
| AC-001 | Detailed Design → Movement Detector | `MovementDetector.detect_movement()` | Stage 1: 20-30 images with known 2px, 5px, 10px shifts; verify accuracy >95% |
| AC-002 | Detailed Design → Movement Detector | `MovementDetector.detect_movement()` | Stage 2: Run on recordings with documented camera movements; verify 0 missed |
| AC-003 | Detailed Design → Result Manager | `ResultManager.create_result()` | Stage 3: Deploy to site, monitor 1 week, manually verify each alert; measure false positives <5% |
| AC-004 | APIs and Interfaces | `CameraMovementDetector.process_frame()` | Integration test: Mock DAF system calls API, verifies status handling logic |
| AC-005 | Workflows and Sequencing → Recalibration | `CameraMovementDetector.recalibrate()`, `tools/recalibrate.py` | Time operator workflow from start to completion; verify <2 min |
| AC-006 | NFR → Reliability/Availability | All components | Continuous run test: Deploy detector, monitor for 1 week, check for crashes/memory growth |
| AC-007 | Detailed Design → Result Manager | `ResultManager.get_history()`, `CameraMovementDetector.get_history()` | Unit test: Fill buffer with 150 results, query by frame_id, verify FIFO behavior |
| AC-008 | Detailed Design → Static Region Manager | `tools/select_roi.py` | Manual test: Run tool on sample image, define region, verify config.json valid and feature count ≥50 |
| AC-009 | Detailed Design → Feature Extractor | `FeatureExtractor.set_baseline()`, `CameraMovementDetector.set_baseline()` | Unit test: Baseline capture on various images, verify ≥50 features or ValueError |
| AC-010 | APIs and Interfaces, NFR → Reliability | All API methods | Unit tests: Invalid inputs (wrong dtype, missing baseline, corrupt config), verify correct exceptions |
| AC-011 | Detailed Design → Movement Detector, Result Manager | `MovementDetector.detect_movement()`, `ResultManager.create_result()` | Unit test: Verify confidence in [0.0, 1.0], varies with match quality |

## Risks, Assumptions, Open Questions

**Risks**:

| ID | Risk | Likelihood | Impact | Mitigation |
|----|------|------------|--------|------------|
| R-001 | False positives from lighting changes may exceed 5% threshold | Medium | High | Start testing quickly; if FP>5%, add RANSAC to homography estimation |
| R-002 | Static regions may have insufficient features (<50) at some sites | Low | Medium | ROI selection tool validates feature count; operator selects different region if needed |
| R-003 | Simple homography may be unreliable for large camera movements | Low | Low | MVP focuses on 2px threshold; large movements are obvious failures |
| R-004 | DAF system integration may require API changes | Low | Medium | Black-box design minimizes coupling; simple dict return value |
| R-005 | Manual recalibration may be forgotten after lighting changes | Medium | Medium | Documented in README; future: add automatic drift detection (post-MVP) |

**Assumptions**:

| ID | Assumption | Validation | Impact if Wrong |
|----|-----------|------------|-----------------|
| A-001 | Existing DAF camera interface provides NumPy image arrays | Verify with DAF system docs/code | Minor: Add conversion layer if different format |
| A-002 | Static regions (tank walls, pipes) are truly static (no movement) | Validate during ROI selection | Critical: If regions move, detection fails; select better ROI |
| A-003 | 5-10 minute detection frequency is sufficient | Confirm with operations team | Low: Increase frequency if needed (simple config change) |
| A-004 | Camera shifts ≥2 pixels significantly impact water quality measurements | Verify with neural network team | Low: Adjust threshold if needed |
| A-005 | Lighting changes are gradual enough for manual recalibration | Monitor during Stage 3 testing | High: May need automatic recalibration sooner than planned |

**Open Questions**:

| ID | Question | Owner | Deadline | Resolution |
|----|----------|-------|----------|------------|
| Q-001 | What's the actual false positive rate with simple homography in real DAF environments? | QA Team | After Stage 1 | TBD - testing will reveal |
| Q-002 | How many features are typically detectable in static regions across different sites? | Engineer | During ROI setup | TBD - measure at first site |
| Q-003 | Should history buffer persist across restarts or start fresh? | PM | Before implementation | Deferred for MVP: start fresh |
| Q-004 | Should there be a "confidence score" in addition to VALID/INVALID? | PM | Before implementation | Deferred for MVP: binary only |
| Q-005 | What's the optimal history buffer size (100 is arbitrary)? | Engineer | During testing | May tune based on usage patterns |

## Test Strategy Summary

**Testing Framework**:
- **Unit Tests**: Python `pytest` for individual component testing
- **Integration Tests**: Mock DAF system integration via pytest fixtures
- **Manual Validation**: Operator-driven ROI selection and recalibration workflows
- **Live Testing**: 1-week deployment monitoring at pilot site

**Test Levels**:

| Level | Scope | Framework | Coverage Target | Timeline |
|-------|-------|-----------|-----------------|----------|
| Unit | Individual components (StaticRegionManager, FeatureExtractor, MovementDetector, ResultManager) | pytest | >80% code coverage | Week 1 |
| Integration | API contract (CameraMovementDetector), DAF system integration | pytest with mocks | All API methods, error paths | Week 1-2 |
| Stage 1 Validation | Simulated camera transforms (2px, 5px, 10px shifts) | Test harness with known transforms | 100 test images, >95% accuracy | Week 2 |
| Stage 2 Validation | Real recorded camera shifts | Manual review with labeled footage | All known real movements detected | Week 2 |
| Stage 3 Validation | Live deployment monitoring | Manual alert verification | 1 week continuous, <5% false positives | Week 3 |

**Test Data Requirements** (PRD Section 5):
- **Stage 1**: 20-30 test images with synthetic shifts (2px, 5px, 10px)
- **Stage 2**: Existing recordings where camera actually moved (if available)
- **Stage 3**: 1 pilot site, multiple times of day, various DAF operational states
- **Edge Cases**: Insufficient features (<50), invalid image formats, missing config, corrupted baseline

**Acceptance Test Plan**:

```python
# Example Unit Test
def test_movement_detector_above_threshold():
    """AC-001: Verify detection accuracy for 2px+ movements"""
    detector = MovementDetector(threshold_pixels=2.0)
    baseline_features = load_baseline("test_data/baseline.pkl")
    shifted_features = load_features("test_data/shift_2px.pkl")

    moved, displacement = detector.detect_movement(baseline_features, shifted_features)

    assert moved == True
    assert displacement >= 2.0

# Example Integration Test
def test_process_frame_returns_valid_dict():
    """AC-004: Verify API returns correct result structure"""
    detector = CameraMovementDetector('test_config.json')
    detector.set_baseline(load_image("test_data/baseline.jpg"))

    result = detector.process_frame(load_image("test_data/frame1.jpg"), frame_id="test_001")

    assert "status" in result
    assert result["status"] in ["VALID", "INVALID"]
    assert "displacement" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "frame_id" in result
    assert result["frame_id"] == "test_001"
```

**Test Environment**:
- Development: Local machines (Linux/Windows/macOS)
- Staging: Simulated DAF environment with test images
- Production: Pilot site with real DAF system integration

**Go/No-Go Criteria** (PRD Section 7):
- All 10 acceptance criteria must pass
- If any AC fails: Iterate on MVP, extend testing period
- If all ACs pass: Proceed to production deployment
