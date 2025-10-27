# Technical Specification: ChArUco vs Cam-Shift Validation Comparison Tool

**Project**: cam-shift-detector
**Feature Level**: 1 (Coherent Feature)
**Document Type**: Tech-Spec (Definitive Technical Specification)
**Created**: 2025-10-27
**Author**: PM Agent (John)

---

## 1. Overview

### 1.1 Purpose
Create a side-by-side validation comparison tool that enables real-time visual comparison of ChArUco 6-DOF pose estimation against the cam-shift detector's feature-based movement detection. This tool supports both offline validation (recorded sequences) and online testing (live camera feeds).

### 1.2 User Personas
- **Developers**: Debug detector behavior, tune parameters, identify edge cases
- **QA Personnel**: Validate detector performance against ground truth, regression testing

### 1.3 Integration Context
- **Standalone Mode**: Independent tool for QA and debugging (`tools/validation/`)
- **Validation Framework**: Integrates with existing validation system (`validation/utilities/`)
- **Offline Validation**: Supports recorded video/image sequences for systematic testing
- **Online Testing**: Critical for live feed debugging at pilot sites

### 1.4 Future Extensions
- Experimental ChArUco board deployments at DAF sites for continuous monitoring
- Integration with CI/CD pipeline for automated regression testing

---

## 2. Source Tree Structure

### 2.1 New Files

```
tools/validation/
├── comparison_tool.py           # Main comparison tool (standalone executable)
└── README.md                     # Tool usage documentation

validation/utilities/
├── dual_detector_runner.py      # Dual detector orchestration module
├── comparison_metrics.py        # Comparison metric calculations
└── comparison_logger.py         # Results logging and retrieval

validation/results/
└── comparison_tool/              # Comparison tool output directory
    ├── logs/                     # Per-session JSON logs
    └── analysis/                 # Generated MSE graphs and reports
```

### 2.2 Modified Files

**None** - This is a new feature with no modifications to existing components.

### 2.3 Dependencies

**Existing Modules**:
- `src.camera_movement_detector` - Cam-shift detector API
- `tools.aruco.camshift_annotator` - ChArUco pose estimation utilities
- `validation.utilities.performance_profiler` - FPS and latency profiling

**External Libraries** (already in project):
- `opencv-python==4.10.0.84` - Display, ChArUco detection, camera I/O
- `numpy==1.26.4` - Numerical operations
- `pandas==2.2.3` - CSV logging and analysis
- `matplotlib==3.9.2` - MSE graph generation (add to dependencies)

---

## 3. Technical Approach

### 3.1 Architecture

**Component Diagram**:
```
┌─────────────────────────────────────────────────────────┐
│          Comparison Tool (comparison_tool.py)           │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Input Handler                                   │   │
│  │  - Directory reader (offline)                    │   │
│  │  - Camera capture (online)                       │   │
│  └──────────────────┬───────────────────────────────┘   │
│                     │                                    │
│  ┌──────────────────▼───────────────────────────────┐   │
│  │  DualDetectorRunner (dual_detector_runner.py)    │   │
│  │  ┌─────────────┐         ┌──────────────┐        │   │
│  │  │ ChArUco     │         │ Cam-Shift    │        │   │
│  │  │ Detector    │         │ Detector     │        │   │
│  │  └─────┬───────┘         └───────┬──────┘        │   │
│  │        │                         │               │   │
│  │        └────────┬────────────────┘               │   │
│  └─────────────────┼────────────────────────────────┘   │
│                    │                                     │
│  ┌─────────────────▼────────────────────────────────┐   │
│  │  ComparisonMetrics (comparison_metrics.py)       │   │
│  │  - ||d1-d2||_2 calculation                       │   │
│  │  - Threshold-based classification                │   │
│  │  - Red/green flagging                            │   │
│  └─────────────────┬────────────────────────────────┘   │
│                    │                                     │
│  ┌─────────────────▼────────────────────────────────┐   │
│  │  Display Windows (OpenCV)                        │   │
│  │  ┌─────────────┐         ┌──────────────┐        │   │
│  │  │ ChArUco     │         │ Cam-Shift    │        │   │
│  │  │ Window      │         │ Window       │        │   │
│  │  │ - Pose axes │         │ - Features   │        │   │
│  │  │ - Disp: Xpx │         │ - Disp: Ypx  │        │   │
│  │  │ - Status    │         │ - Status     │        │   │
│  │  └─────────────┘         └──────────────┘        │   │
│  └─────────────────┬────────────────────────────────┘   │
│                    │                                     │
│  ┌─────────────────▼────────────────────────────────┐   │
│  │  ComparisonLogger (comparison_logger.py)         │   │
│  │  - Per-frame JSON logging                        │   │
│  │  - MSE calculation over sequence                 │   │
│  │  - Worst match retrieval                         │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Execution Flow

**Offline Mode** (Directory Input):
1. Load calibration (camera.yaml) and detector configs
2. Set baseline from first frame
3. For each frame in directory:
   - Load image
   - Run ChArUco pose estimation → displacement d1
   - Run cam-shift detector → displacement d2
   - Calculate comparison metric: ||d1-d2||_2
   - Update display windows with green/red status
   - Log results to JSON
4. Generate MSE graph and save worst matches

**Online Mode** (Live Feed):
1. Open camera capture
2. User presses 's' to set baseline
3. Continuous processing:
   - Capture frame
   - Run both detectors (d1, d2)
   - Calculate comparison metric
   - Update display windows
   - Log results
4. User presses 'q' to stop and generate analysis

### 3.3 Comparison Metric

**Formula**:
```
displacement_diff = ||d1 - d2||_2
threshold = 0.03 * min(image_width, image_height)

status = GREEN if displacement_diff <= threshold else RED
```

**Where**:
- `d1` = ChArUco 2D displacement (pixels) from baseline
- `d2` = Cam-shift detector displacement (pixels) from baseline
- `||·||_2` = Euclidean L2 norm

**Example** (640×480 image):
- `threshold = 0.03 * min(640, 480) = 0.03 * 480 = 14.4px`
- If `d1 = 18.5px` and `d2 = 15.2px`, then `||18.5 - 15.2||_2 = 3.3px` → **GREEN** (good agreement)
- If `d1 = 22.1px` and `d2 = 5.8px`, then `||22.1 - 5.8||_2 = 16.3px` → **RED** (poor agreement)

### 3.4 UI Design

**Layout**:
```
┌─────────────────────────────────────────────────────────┐
│  ChArUco Detector           Cam-Shift Detector          │
│  ┌────────────────┐         ┌────────────────┐          │
│  │                │         │                │          │
│  │  [Frame with   │         │  [Frame with   │          │
│  │   pose axes]   │         │   ORB features]│          │
│  │                │         │                │          │
│  │                │         │                │          │
│  │                │         │                │          │
│  ├────────────────┤         ├────────────────┤          │
│  │ Disp: 18.5px   │         │ Disp: 15.2px   │          │
│  │ Status: MOVED  │         │ Status: INVALID│          │
│  │ Confidence: 95%│         │ Confidence: 87%│          │
│  └────────────────┘         └────────────────┘          │
│                                                          │
│  Comparison: ||d1-d2||_2 = 3.3px  [GREEN]               │
│  Threshold: 14.4px (3% of 480px)                        │
│  Frame: 0042/0157  |  FPS: 18.3                         │
│  Press 's' to set baseline, 'q' to quit                 │
└─────────────────────────────────────────────────────────┘
```

**Color Coding**:
- **Green status bar**: Agreement within threshold (||d1-d2||_2 ≤ threshold)
- **Red status bar**: Disagreement exceeds threshold (||d1-d2||_2 > threshold)

---

## 4. Implementation Stack

### 4.1 Language & Runtime
- **Python 3.11** (project standard)
- **OpenCV 4.10.0.84** (cv2 module)

### 4.2 Core Libraries
```python
# Detector Integration
from src.camera_movement_detector import CameraMovementDetector
from tools.aruco.camshift_annotator import (
    make_charuco_board,
    estimate_pose_charuco,
    read_yaml_camera
)

# UI & Display
import cv2 as cv  # OpenCV windows, drawing, camera I/O

# Numerical Operations
import numpy as np  # Displacement calculations, Euclidean norm

# Logging & Analysis
import pandas as pd  # CSV logging for per-frame results
import json  # JSON output for structured logging
import matplotlib.pyplot as plt  # MSE graph generation

# Profiling
from validation.utilities.performance_profiler import PerformanceProfiler
```

### 4.3 Configuration Files
- **camera.yaml** - Camera intrinsics (K matrix, distortion coefficients)
- **config.json** - Cam-shift detector ROI and thresholds
- **comparison_config.json** - Comparison tool settings (NEW)

**comparison_config.json structure**:
```json
{
  "charuco_board": {
    "squares_x": 7,
    "squares_y": 5,
    "square_m": 0.035,
    "marker_m": 0.026,
    "dict": "DICT_4X4_50"
  },
  "comparison": {
    "threshold_percent": 0.03,
    "default_z_distance_m": 1.15
  },
  "display": {
    "window_width": 640,
    "window_height": 480,
    "show_axes": true,
    "show_features": true
  },
  "logging": {
    "output_dir": "validation/results/comparison_tool",
    "log_format": "json",
    "save_frames": false
  }
}
```

---

## 5. Technical Details

### 5.1 Module: dual_detector_runner.py

**Purpose**: Orchestrate parallel execution of ChArUco and cam-shift detectors.

**Key Classes**:

```python
@dataclass
class DualDetectionResult:
    """Results from both detectors for a single frame"""
    frame_idx: int
    timestamp_ns: int

    # ChArUco results
    charuco_detected: bool
    charuco_displacement_px: float
    charuco_confidence: float

    # Cam-shift results
    camshift_status: str  # "VALID" or "INVALID"
    camshift_displacement_px: float
    camshift_confidence: float

    # Comparison
    displacement_diff: float  # ||d1-d2||_2
    agreement_status: str  # "GREEN" or "RED"
    threshold_px: float

class DualDetectorRunner:
    """Runs both detectors in parallel on frames"""

    def __init__(
        self,
        charuco_config: Dict,
        camshift_config_path: str,
        camera_yaml_path: str
    ):
        """Initialize both detectors"""
        # Create ChArUco detector components
        self.dictionary, self.board, self.detector = make_charuco_board(...)
        self.K, self.dist, self.image_size = read_yaml_camera(camera_yaml_path)

        # Create cam-shift detector
        self.camshift_detector = CameraMovementDetector(camshift_config_path)

        self.baseline_set = False
        self.threshold_px = None

    def set_baseline(self, image: np.ndarray) -> None:
        """Set baseline for both detectors"""
        # ChArUco: store first detected pose
        # Cam-shift: call detector.set_baseline()

    def process_frame(
        self,
        image: np.ndarray,
        frame_idx: int
    ) -> DualDetectionResult:
        """Run both detectors and compare results"""
        # 1. Run ChArUco pose estimation
        # 2. Run cam-shift detector
        # 3. Calculate comparison metric
        # 4. Return DualDetectionResult
```

**ChArUco Displacement Calculation**:
```python
def calculate_charuco_displacement_2d(
    tvec_current: np.ndarray,
    tvec_baseline: np.ndarray,
    K: np.ndarray,
    z_distance_m: float = 1.15
) -> float:
    """
    Convert 3D translation vector to 2D pixel displacement

    Approach: Project 3D displacement onto image plane
    dx_px = (delta_x_m * fx) / z_m
    dy_px = (delta_y_m * fy) / z_m
    displacement_px = sqrt(dx_px^2 + dy_px^2)
    """
    delta_x = tvec_current[0] - tvec_baseline[0]
    delta_y = tvec_current[1] - tvec_baseline[1]

    fx = K[0, 0]  # Focal length X
    fy = K[1, 1]  # Focal length Y

    dx_px = (delta_x * fx) / z_distance_m
    dy_px = (delta_y * fy) / z_distance_m

    return np.sqrt(dx_px**2 + dy_px**2)
```

### 5.2 Module: comparison_metrics.py

**Purpose**: Calculate comparison metrics and classifications.

**Key Functions**:

```python
def calculate_displacement_difference(
    d1: float,
    d2: float
) -> float:
    """Calculate L2 norm of displacement difference"""
    return abs(d1 - d2)

def calculate_threshold(
    image_width: int,
    image_height: int,
    threshold_percent: float = 0.03
) -> float:
    """Calculate threshold as 3% of minimum dimension"""
    return threshold_percent * min(image_width, image_height)

def classify_agreement(
    displacement_diff: float,
    threshold: float
) -> str:
    """Classify as GREEN (agree) or RED (disagree)"""
    return "GREEN" if displacement_diff <= threshold else "RED"

def calculate_mse(
    charuco_displacements: List[float],
    camshift_displacements: List[float]
) -> float:
    """Calculate Mean Squared Error between detector outputs"""
    diffs = np.array(charuco_displacements) - np.array(camshift_displacements)
    return np.mean(diffs ** 2)
```

### 5.3 Module: comparison_logger.py

**Purpose**: Log results and support analysis.

**Key Classes**:

```python
class ComparisonLogger:
    """Logs comparison results for later analysis"""

    def __init__(self, output_dir: str, session_name: str):
        """Initialize logger with output directory"""
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        self.log_file = self.output_dir / "logs" / f"{session_name}.json"
        self.results: List[DualDetectionResult] = []

    def log_frame(self, result: DualDetectionResult) -> None:
        """Log single frame result"""
        self.results.append(result)

    def save_log(self) -> None:
        """Save results to JSON file"""
        # Convert dataclasses to dict and write JSON

    def calculate_mse(self) -> float:
        """Calculate MSE over entire sequence"""
        # Extract displacements and call comparison_metrics.calculate_mse()

    def get_worst_matches(self, n: int = 10) -> List[DualDetectionResult]:
        """Get N frames with largest displacement differences"""
        sorted_results = sorted(
            self.results,
            key=lambda r: r.displacement_diff,
            reverse=True
        )
        return sorted_results[:n]

    def generate_mse_graph(self, output_path: str) -> None:
        """Generate MSE graph using matplotlib"""
        # Plot displacement_diff over time
        # Highlight worst matches in red
        # Save to output_path
```

### 5.4 Module: comparison_tool.py (Main Executable)

**Purpose**: Standalone tool for running comparisons.

**Command-Line Interface**:
```bash
# Offline mode (directory input)
python tools/validation/comparison_tool.py \
  --mode offline \
  --input-dir session_001/frames \
  --camera-yaml camera.yaml \
  --charuco-config comparison_config.json \
  --camshift-config config.json \
  --output-dir validation/results/comparison_tool/session_001

# Online mode (live feed)
python tools/validation/comparison_tool.py \
  --mode online \
  --camera-id 0 \
  --camera-yaml camera.yaml \
  --charuco-config comparison_config.json \
  --camshift-config config.json \
  --output-dir validation/results/comparison_tool/live_$(date +%Y%m%d_%H%M%S)
```

**Key Functions**:
```python
def run_offline_comparison(args) -> None:
    """Run comparison on directory of images"""
    # 1. Initialize DualDetectorRunner
    # 2. Load images from directory
    # 3. Set baseline (first image)
    # 4. For each frame:
    #    - Run detectors
    #    - Update display windows
    #    - Log results
    # 5. Generate MSE graph and worst matches report

def run_online_comparison(args) -> None:
    """Run comparison on live camera feed"""
    # 1. Initialize DualDetectorRunner
    # 2. Open camera capture
    # 3. Wait for user to press 's' to set baseline
    # 4. Continuous loop:
    #    - Capture frame
    #    - Run detectors
    #    - Update display windows
    #    - Log results
    #    - Check for 'q' to quit
    # 5. Generate analysis on exit

def create_display_windows(
    charuco_frame: np.ndarray,
    camshift_frame: np.ndarray,
    result: DualDetectionResult,
    config: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Create annotated display frames"""
    # ChArUco window:
    # - Draw pose axes if detected
    # - Overlay displacement text
    # - Overlay status and confidence

    # Cam-shift window:
    # - Draw detected features if configured
    # - Overlay displacement text
    # - Overlay status and confidence

    return charuco_annotated, camshift_annotated

def draw_comparison_status_bar(
    width: int,
    result: DualDetectionResult
) -> np.ndarray:
    """Create status bar showing comparison result"""
    # Green or red background based on agreement
    # Text: "||d1-d2||_2 = 3.3px [GREEN]"
    # Text: "Threshold: 14.4px (3% of 480px)"
```

---

## 6. Testing Approach

### 6.1 Unit Testing

**Test Coverage**:
- `test_dual_detector_runner.py` - DualDetectorRunner initialization and execution
- `test_comparison_metrics.py` - Metric calculations (displacement_diff, threshold, MSE)
- `test_comparison_logger.py` - Logging, worst match retrieval, MSE graph generation

**Test Data**:
- Use existing `session_001` ChArUco data (203 frames, 158 detected)
- Synthetic edge cases: d1=d2 (perfect agreement), d1>>d2 (large disagreement)

### 6.2 Integration Testing

**Scenarios**:
1. **Offline Mode**: Run on `session_001/frames/`, verify JSON log structure
2. **Online Mode**: Run with webcam, verify display updates at ~20 FPS
3. **MSE Graph**: Verify graph generation with correct axis labels and worst match highlights
4. **Worst Matches**: Verify retrieval of top 10 frames with largest displacement differences

### 6.3 Validation Criteria

**Functional Requirements**:
- ✅ Both detector windows display simultaneously
- ✅ Displacement values match detector outputs
- ✅ Green/red status updates correctly based on threshold
- ✅ JSON log contains all required fields
- ✅ MSE graph generated successfully
- ✅ Worst matches retrieval returns correct frames

**Performance Requirements**:
- ✅ Offline mode: Process 157 frames in < 30 seconds (~5 FPS minimum)
- ✅ Online mode: Maintain 15-20 FPS on live feed
- ✅ Display latency: < 100ms between detectors

---

## 7. Deployment Strategy

### 7.1 Installation

**Add matplotlib to dependencies**:
```bash
# Update requirements.txt or pyproject.toml
matplotlib==3.9.2
```

**Install**:
```bash
pip install matplotlib==3.9.2
# or
uv pip install matplotlib==3.9.2
```

### 7.2 File Deployment

**Create directories**:
```bash
mkdir -p tools/validation
mkdir -p validation/utilities
mkdir -p validation/results/comparison_tool/{logs,analysis}
```

**Copy/create files**:
1. `tools/validation/comparison_tool.py` (main executable)
2. `tools/validation/README.md` (usage documentation)
3. `validation/utilities/dual_detector_runner.py`
4. `validation/utilities/comparison_metrics.py`
5. `validation/utilities/comparison_logger.py`
6. `comparison_config.json` (project root)

### 7.3 Configuration

**Create default comparison_config.json**:
```bash
cat > comparison_config.json <<EOF
{
  "charuco_board": {
    "squares_x": 7,
    "squares_y": 5,
    "square_m": 0.035,
    "marker_m": 0.026,
    "dict": "DICT_4X4_50"
  },
  "comparison": {
    "threshold_percent": 0.03,
    "default_z_distance_m": 1.15
  },
  "display": {
    "window_width": 640,
    "window_height": 480,
    "show_axes": true,
    "show_features": true
  },
  "logging": {
    "output_dir": "validation/results/comparison_tool",
    "log_format": "json",
    "save_frames": false
  }
}
EOF
```

### 7.4 Verification

**Run offline test**:
```bash
python tools/validation/comparison_tool.py \
  --mode offline \
  --input-dir session_001/frames \
  --camera-yaml camera.yaml \
  --charuco-config comparison_config.json \
  --camshift-config config.json \
  --output-dir validation/results/comparison_tool/test_run
```

**Expected outputs**:
- `validation/results/comparison_tool/test_run/logs/offline_session.json` (per-frame results)
- `validation/results/comparison_tool/test_run/analysis/mse_graph.png` (MSE plot)
- `validation/results/comparison_tool/test_run/analysis/worst_matches.txt` (top 10 disagreements)
- Console output with MSE and worst match summary

---

## 8. Future Enhancements

### 8.1 Immediate Extensions (Post-MVP)
- **3D Displacement Overlay**: Show ChArUco 3D translation vectors alongside 2D projection
- **Recording Mode**: Save comparison video for presentations
- **Parameter Tuning**: Interactive sliders for threshold adjustment

### 8.2 Long-Term Roadmap
- **Multi-Session Analysis**: Compare multiple sessions and generate aggregate reports
- **Automated Regression Testing**: Integration with CI/CD pipeline
- **Site Deployment**: Continuous monitoring at DAF pilot sites with ChArUco boards

---

## 9. Success Metrics

### 9.1 Acceptance Criteria
- ✅ Tool runs in both offline and online modes without errors
- ✅ Displays update at ≥15 FPS in online mode
- ✅ JSON logs contain all required fields and are valid JSON
- ✅ MSE graph generated with correct data and labels
- ✅ Worst matches retrieval accurate (top 10 by displacement_diff)
- ✅ Documentation complete in `tools/validation/README.md`

### 9.2 Quality Metrics
- **Code Coverage**: ≥80% for all new modules
- **Performance**: Process 157 frames in <30s (offline), maintain 15-20 FPS (online)
- **Usability**: Single command execution for both modes

---

## 10. Dependencies & Constraints

### 10.1 External Dependencies
- **Existing**: `opencv-python==4.10.0.84`, `numpy==1.26.4`, `pandas==2.2.3`
- **New**: `matplotlib==3.9.2` (MSE graph generation)

### 10.2 Hardware Requirements
- **Camera**: Any OpenCV-compatible camera for online mode
- **Display**: Dual-window display support (1280×480 minimum resolution)

### 10.3 Constraints
- **ChArUco Limitation**: Requires visible ChArUco board for ground truth in online mode
- **Calibration Dependency**: Requires `camera.yaml` for 3D-to-2D conversion accuracy
- **Frame Rate**: Limited by slower detector (typically cam-shift at 15-20 FPS)

---

## 11. Risks & Mitigations

### 11.1 Technical Risks

**Risk**: ChArUco detection failure in low-light conditions
**Impact**: No ground truth, comparison fails
**Mitigation**: Graceful degradation - show cam-shift detector only, log missing ground truth

**Risk**: Detector synchronization lag (>100ms)
**Impact**: Misleading comparison due to temporal mismatch
**Mitigation**: Timestamp both detector calls, flag if lag >50ms

**Risk**: 3D-to-2D conversion inaccuracy
**Impact**: Threshold calibration errors
**Mitigation**: Make Z distance configurable, add calibration wizard

### 11.2 Operational Risks

**Risk**: High FPR/FNR in field deployment
**Impact**: Tool reports false disagreements
**Mitigation**: User-adjustable threshold, calibration against known sequences

---

## 12. Document Changelog

| Version | Date       | Author   | Changes                          |
|---------|------------|----------|----------------------------------|
| 1.0     | 2025-10-27 | PM Agent | Initial definitive specification |

---

**END OF TECHNICAL SPECIFICATION**
