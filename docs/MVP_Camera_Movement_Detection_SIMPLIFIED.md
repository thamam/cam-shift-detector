# MVP: Camera Movement Detection for DAF Systems
## Simplified Product Requirements Document

**Version:** 1.0-SIMPLIFIED
**Date:** 2025-10-17
**Status:** Ready for Implementation
**Target Timeline:** 2 weeks (1 engineer)

---

## 1. Problem Statement

**The Problem:**
Camera movement causes ROI misalignment → Neural network measures wrong region → Inaccurate water flow and turbidity data → Bad treatment decisions

**The Solution:**
Detect camera movement beyond threshold → Alert system → Trigger manual correction (reposition camera OR recalibrate)

**Success Criteria:**
System correctly identifies camera shifts and prevents use of corrupted measurement data.

**Important Distinction - Detection Only, No Alignment:**
This system performs DETECTION only - it identifies when camera movement has occurred and flags the data as invalid. It does NOT perform automatic alignment correction or image registration. The response to detected movement is:
1. Flag measurements as INVALID to prevent corrupted data usage
2. Alert operators for manual intervention (physical camera repositioning or recalibration)
3. System remains in invalid state until manual recalibration is performed

---

## 2. MVP Scope

### ✅ What's IN Scope

**Core Functionality:**
1. Static region definition tool (ROI selection)
2. ORB feature extraction from static region
3. Homography/Affine-based translation detection (TRANSLATION ONLY - see limitation below)
4. Threshold-based alerting (> 2 pixels translation = movement)
5. Flag file output (VALID/INVALID status)
6. Manual recalibration capability
7. Basic validation testing

**Critical Architectural Limitation - Translation Detection Only:**
The MVP implementation measures ONLY the translation displacement (tx, ty) extracted from the transformation matrix (homography or affine). It does NOT detect:
- Pure rotation (camera rotates without translating)
- Scale changes (zoom in/out)
- Perspective distortion/shear

**Why this matters:** A camera could rotate significantly (corrupting ROI alignment) while showing <2px translation, resulting in a false "no movement" detection. This is an acceptable MVP constraint because:
1. Physical camera mounts typically fail via translation (loosening, vibration) rather than pure rotation
2. Pure rotation without translation is geometrically rare in fixed camera scenarios
3. The cost of false negatives for pure rotation is lower than the complexity of full 6-DOF analysis

**Post-MVP Enhancement:** Full homography decomposition with separate rotation/scale/shear thresholds can be added if field data shows pure rotation events are occurring.

**Key Constraints:**
- Single camera only
- Manual recalibration (no automatic drift handling)
- Periodic checks every 5-10 minutes (not real-time)
- Simple homography first (RANSAC only if needed)
- Minimal/no logging (add ad-hoc if needed)
- No UI required (headless operation)

### ❌ What's OUT of Scope

**Explicitly Deferred or Cut:**
- Multi-camera support
- Automatic recalibration / drift detection
- Real-time 1Hz monitoring
- REST API integration (other system handles)
- Rich UI (Qt5/Flask interfaces)
- SQLite event logging
- Scene quality monitoring
- Comprehensive test suites (bubbles/water dynamics)
- Full documentation suite

**Philosophy:** Ship simple working version, add features based on real performance data (YAGNI)

---

## 3. Module Interface Contract

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**
**│                 CAM-SHIFT-DETECTOR MODULE INTERFACE              │**
**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

Black-box module for detecting camera movement in DAF water quality monitoring systems.

### API

**Initialization:**
```python
detector = CameraMovementDetector(config_path='config.json')
# config.json contains: ROI coordinates, detection threshold (default: 2.0 pixels)
```

**Core Methods:**
```python
# Process frame and get immediate result
result = detector.process_frame(image_array, frame_id=None)
# Returns: {"status": "VALID"|"INVALID", "displacement": float, "frame_id": str}

# Query historical results
history = detector.get_history(frame_id=None, limit=100)
# Returns: List of recent detection results (buffered in memory)

# Manual recalibration
success = detector.recalibrate(image_array=None)
# Returns: bool - True if recalibration successful
```

### Data Types

**Input:**
- `image_array`: NumPy array (H × W × 3, uint8, BGR)
- `frame_id`: Optional string identifier for tracking (e.g., timestamp or sequence number)

**Output:**
```python
{
  "status": "VALID",           # "VALID" | "INVALID"
  "displacement": 0.8,         # float - pixels moved
  "frame_id": "frame_12345",   # echoed from input or auto-generated
  "timestamp": "2025-10-17T14:32:18.456Z"
}
```

**History Buffer:**
- Stores last 100 detection results (configurable)
- FIFO - oldest results dropped when buffer full
- Query by `frame_id` or retrieve recent N results

### Configuration (`config.json`)
```json
{
  "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
  "threshold_pixels": 2.0,
  "history_buffer_size": 100
}
```

### Typical Usage
```python
# Setup
detector = CameraMovementDetector('config.json')

# Runtime (parent DAF system calls every 5-10 minutes)
image = get_camera_frame()  # Your existing interface
result = detector.process_frame(image, frame_id="2025-10-17_14:32:18")

if result['status'] == 'INVALID':
    # Halt water quality measurements
    logger.warning(f"Camera moved {result['displacement']:.2f}px - data invalid")
else:
    # Safe to use vision data
    perform_water_quality_analysis(image)

# Query recent history if needed
recent = detector.get_history(limit=10)
```

**━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━**

---

## 4. Architecture & Design

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│              EXISTING DAF SYSTEM                        │
│   Camera → Image Array Interface (already exists)      │
└────────────────────┬────────────────────────────────────┘
                     │ process_frame(image_array, frame_id)
┌────────────────────▼────────────────────────────────────┐
│      CAMERA MOVEMENT DETECTOR (Black Box Module)       │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  1. Static Region Manager                        │  │
│  │     - Load ROI coordinates from config           │  │
│  │     - Crop image to static region                │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                   │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  2. Feature Extractor                            │  │
│  │     - Extract ORB features from cropped region   │  │
│  │     - Store baseline features (reference)        │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                   │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  3. Movement Detector                            │  │
│  │     - Match current features to baseline         │  │
│  │     - Estimate homography transformation         │  │
│  │     - Calculate displacement magnitude           │  │
│  └──────────────────┬───────────────────────────────┘  │
│                     │                                   │
│  ┌──────────────────▼───────────────────────────────┐  │
│  │  4. Result Manager & History Buffer              │  │
│  │     - Check: displacement > 2 pixels?            │  │
│  │     - Build result dict with status              │  │
│  │     - Store in FIFO history buffer (last 100)    │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────┬───────────────────────────────────┘
                      │ Return: {"status": "VALID"|"INVALID",
                      │          "displacement": float, ...}
┌─────────────────────▼───────────────────────────────────┐
│           EXTERNAL DAF SYSTEM (existing)                │
│   If result['status'] == 'INVALID': Halt measurements  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

**Setup Phase:**
1. Download recent image from cloud (or use local camera)
2. Run ROI selection tool (OpenCV GUI)
3. Click/draw static region (tank walls, pipes - NOT water)
4. Save ROI coordinates to `config.json`
5. Capture baseline features from static region
6. Deploy config to site

**Runtime Phase (Every 5-10 minutes):**
1. DAF system calls: `result = detector.process_frame(image_array, frame_id)`
2. Module crops image to static region (using config coordinates)
3. Extract ORB features from cropped region
4. Match features to baseline using homography estimation
5. Calculate displacement magnitude
6. Build result dict:
   - If displacement > 2 pixels: `status = "INVALID"`
   - Else: `status = "VALID"`
7. Store result in history buffer (FIFO, last 100)
8. Return result to caller
9. DAF system checks `result['status']` before using vision data

**Recalibration Phase (Manual trigger):**
1. Operator calls: `detector.recalibrate(image_array)` or runs script
2. Module extracts new baseline features from static region
3. Replaces old baseline in memory
4. Clears history buffer (optional)
5. Returns success/failure status
6. Resumes monitoring

---

## 4. Implementation Details

### 4.1 Static Region Manager

**Purpose:** Define and manage the static region (ROI) where features are extracted

**Components:**

**A. ROI Selection Tool** (`tools/select_roi.py`)
```python
# Simple OpenCV window for ROI selection
# Supports two modes:

# Mode 1: Local laptop with camera
python select_roi.py --source camera

# Mode 2: Remote via cloud image
python select_roi.py --source image --path downloaded_image.jpg

# Output: config.json with ROI coordinates
{
  "roi": {
    "x": 100,
    "y": 50,
    "width": 400,
    "height": 300
  }
}
```

**B. Static Region Cropper** (`src/static_region_manager.py`)
```python
class StaticRegionManager:
    def __init__(self, config_path):
        self.roi = load_roi_from_config(config_path)

    def crop_to_static_region(self, image):
        x, y, w, h = self.roi['x'], self.roi['y'],
                     self.roi['width'], self.roi['height']
        return image[y:y+h, x:x+w]
```

**Requirements:**
- ROI must contain ≥50 trackable features (validate during setup)
- ROI should exclude water surface, bubble zones, moving elements
- Coordinates stored in `config.json`, version controlled

---

### 4.2 Feature Extractor

**Purpose:** Extract ORB features from static region and manage baseline

**Implementation:** `src/feature_extractor.py`

```python
class FeatureExtractor:
    def __init__(self):
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=8
        )
        self.baseline_features = None

    def extract_features(self, image):
        """Extract ORB keypoints and descriptors"""
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        return keypoints, descriptors

    def set_baseline(self, image):
        """Capture baseline features for comparison"""
        kp, desc = self.extract_features(image)
        if len(kp) < 50:
            raise ValueError(f"Insufficient features: {len(kp)} < 50")
        self.baseline_features = (kp, desc)

    def get_baseline(self):
        return self.baseline_features
```

**Parameters:**
- ORB features: 500 max
- Minimum required: 50 features in static region
- Scale factor: 1.2
- Pyramid levels: 8

**Baseline Management:**
- Stored in memory during runtime
- Persisted to disk: `data/baseline_features.pkl` (optional)
- Replaced during manual recalibration

---

### 4.3 Movement Detector

**Purpose:** Detect camera movement by comparing current features to baseline

**Implementation:** `src/movement_detector.py`

```python
class MovementDetector:
    def __init__(self, threshold_pixels=2.0):
        self.threshold = threshold_pixels
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_movement(self, baseline_features, current_features):
        """
        Compare current features to baseline using homography
        Returns: (movement_detected, displacement_magnitude)
        """
        baseline_kp, baseline_desc = baseline_features
        current_kp, current_desc = current_features

        # Match features
        matches = self.matcher.match(baseline_desc, current_desc)

        if len(matches) < 10:
            return True, float('inf')  # Too few matches = problem

        # Extract matched point coordinates
        src_pts = np.float32([baseline_kp[m.queryIdx].pt
                             for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([current_kp[m.trainIdx].pt
                             for m in matches]).reshape(-1, 1, 2)

        # Estimate homography (simple method, no RANSAC initially)
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)

        if H is None:
            return True, float('inf')

        # Calculate displacement from homography TRANSLATION COMPONENTS ONLY
        # NOTE: H[0,2] and H[1,2] are the translation (tx, ty) components
        # This will be ZERO for pure rotation - rotation is NOT detected (MVP limitation)
        # See "Critical Architectural Limitation" in Section 2 for details
        tx = H[0, 2]
        ty = H[1, 2]
        displacement = np.sqrt(tx**2 + ty**2)

        movement_detected = displacement > self.threshold
        return movement_detected, displacement
```

**Detection Logic:**
- Use simple homography estimation (no RANSAC initially)
- Displacement = magnitude of TRANSLATION VECTOR ONLY (tx, ty) from homography
  - `displacement = sqrt(H[0,2]^2 + H[1,2]^2)`
  - Does NOT measure rotation, scale, or perspective distortion
  - Pure camera rotation will show ~0 displacement (false negative)
- Threshold: 2.0 pixels translation (configurable)

**Actual Implementation Notes:**
- Current production version uses affine transformation model (`use_affine_model=True`)
- Affine model is more stable for pure translations and vertical movements
- Both homography and affine extract translation components the same way
- RANSAC was NOT needed - false positive rate <5% with simple method

**When to Enhance Detection:**
- If field data shows pure rotation events causing measurement corruption
- Add rotation angle extraction from transformation matrix
- Implement separate thresholds: translation_threshold AND rotation_threshold
- Use full 6-DOF homography decomposition (see OpenCV `decomposeHomographyMat`)

---

### 4.4 Result Manager & History Buffer

**Purpose:** Build detection results and maintain history buffer for querying

**Implementation:** `src/result_manager.py`

```python
from collections import deque
from datetime import datetime

class ResultManager:
    def __init__(self, threshold_pixels=2.0, buffer_size=100):
        self.threshold = threshold_pixels
        self.history = deque(maxlen=buffer_size)  # FIFO buffer

    def create_result(self, displacement, frame_id=None):
        """Create detection result dict"""
        status = "INVALID" if displacement > self.threshold else "VALID"

        result = {
            "status": status,
            "displacement": round(displacement, 2),
            "frame_id": frame_id or self._generate_frame_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Store in history buffer (FIFO - oldest dropped when full)
        self.history.append(result)

        return result

    def get_history(self, frame_id=None, limit=None):
        """Query history buffer"""
        if frame_id:
            # Find specific frame_id
            return [r for r in self.history if r['frame_id'] == frame_id]
        elif limit:
            # Return last N results
            return list(self.history)[-limit:]
        else:
            # Return all history
            return list(self.history)

    def clear_history(self):
        """Clear history buffer (e.g., after recalibration)"""
        self.history.clear()

    def _generate_frame_id(self):
        """Auto-generate frame_id if not provided"""
        return f"frame_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
```

**Result Dict Structure:**
```python
{
  "status": "VALID",                        # "VALID" | "INVALID"
  "displacement": 0.8,                     # float - pixels
  "frame_id": "2025-10-17_14:32:18",       # string identifier
  "timestamp": "2025-10-17T14:32:18.456Z"  # ISO 8601 UTC
}
```

**History Buffer Behavior:**
- FIFO queue with configurable size (default: 100)
- Oldest results automatically dropped when buffer full
- Query by `frame_id` or retrieve last N results
- In-memory only (no persistence for MVP)
- Optional: Clear on recalibration

---

### 4.5 Complete Detector Class

**Implementation:** `src/camera_movement_detector.py`

```python
class CameraMovementDetector:
    """Main detector class - black-box interface for DAF system"""

    def __init__(self, config_path='config.json'):
        config = load_config(config_path)

        # Initialize all components
        self.region_mgr = StaticRegionManager(config)
        self.feature_ext = FeatureExtractor()
        self.movement_det = MovementDetector(
            threshold_pixels=config['threshold_pixels']
        )
        self.result_mgr = ResultManager(
            threshold_pixels=config['threshold_pixels'],
            buffer_size=config.get('history_buffer_size', 100)
        )

        self._baseline_set = False

    def process_frame(self, image_array, frame_id=None):
        """
        Process single frame and return detection result

        Args:
            image_array: NumPy array (H x W x 3, uint8, BGR)
            frame_id: Optional string identifier for tracking

        Returns:
            dict: {"status": "VALID"|"INVALID", "displacement": float,
                   "frame_id": str, "timestamp": str}
        """
        if not self._baseline_set:
            raise RuntimeError("Baseline not set. Call set_baseline() first.")

        # Crop to static region
        cropped = self.region_mgr.crop_to_static_region(image_array)

        # Extract features
        current_features = self.feature_ext.extract_features(cropped)
        baseline_features = self.feature_ext.get_baseline()

        # Detect movement
        moved, displacement = self.movement_det.detect_movement(
            baseline_features, current_features
        )

        # Create result and store in history
        result = self.result_mgr.create_result(displacement, frame_id)

        return result

    def set_baseline(self, image_array):
        """Capture baseline features from image (initial setup)"""
        cropped = self.region_mgr.crop_to_static_region(image_array)
        self.feature_ext.set_baseline(cropped)
        self._baseline_set = True

    def recalibrate(self, image_array=None):
        """
        Manual recalibration - reset baseline

        Args:
            image_array: New reference image (if None, caller must provide)

        Returns:
            bool: True if successful
        """
        if image_array is None:
            return False

        try:
            self.set_baseline(image_array)
            self.result_mgr.clear_history()  # Optional
            return True
        except Exception as e:
            print(f"Recalibration failed: {e}")
            return False

    def get_history(self, frame_id=None, limit=None):
        """Query detection history buffer"""
        return self.result_mgr.get_history(frame_id, limit)

    def get_status(self):
        """Get most recent detection status"""
        history = self.result_mgr.get_history(limit=1)
        return history[0] if history else {"status": "UNKNOWN"}
```

**Configuration:** `config.json`
```json
{
  "roi": {
    "x": 100,
    "y": 50,
    "width": 400,
    "height": 300
  },
  "threshold_pixels": 2.0,
  "history_buffer_size": 100,
  "min_features_required": 50
}
```

---

## 5. Testing & Validation

### Validation Strategy

**Three-Stage Testing:**

**Stage 1: Simulated Camera Transforms**
- Take reference image
- Apply known TRANSLATION transformations (2px, 5px, 10px shifts in x/y)
- Verify detection accuracy:
  - 0-1.9px translation: No detection (below threshold)
  - 2-10px translation: Detection triggered
  - Target: >95% accuracy
- **Important:** Pure rotation testing NOT included in Stage 1 (known limitation - rotation not detected)

**Stage 2: Real Recorded Camera Shifts**
- Use existing recordings where camera actually moved
- Verify system detects these real movements
- Measure false negative rate (missed movements)
- Target: 0% missed movements

**Stage 3: Live Video Testing**
- Deploy to site(s)
- Monitor multiple times of day
- Various DAF process stages
- Log all detections
- Manually verify each alert (true positive vs. false positive)
- Target: <5% false positive rate

**Test Data Requirements:**
- At least 100 test images per stage
- Multiple sites if possible
- Different lighting conditions (day/night if outdoor)
- Different DAF operational states

---

## 6. Deployment

### Installation

**Prerequisites:**
- Python 3.8+
- OpenCV 4.8+
- NumPy
- Access to existing camera image interface

**Setup Steps:**
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Define ROI using selection tool
4. Integrate detector into DAF system:
   ```python
   from src.camera_movement_detector import CameraMovementDetector

   detector = CameraMovementDetector('config.json')
   detector.set_baseline(initial_image)
   ```
5. Call `process_frame()` from DAF system every 5-10 minutes
6. Check returned status before using vision data

**File Structure:**
```
cam-shift-detector/
├── config.json                      # ROI and detection parameters
├── requirements.txt                 # Python dependencies
├── README.md                       # Setup instructions only
├── src/
│   ├── camera_movement_detector.py  # Main detector class (black-box API)
│   ├── static_region_manager.py
│   ├── feature_extractor.py
│   ├── movement_detector.py
│   └── result_manager.py           # Result dict and history buffer
├── tools/
│   ├── select_roi.py               # ROI selection tool
│   └── recalibrate.py              # Manual recalibration script
├── data/
│   └── baseline_features.pkl       # Saved baseline (optional)
└── tests/
    └── test_detection.py           # Basic validation tests
```

---

## 7. Success Criteria

### MVP is Successful if:

1. ✅ **Detects 2+ pixel TRANSLATION movements** with >95% accuracy (Stage 1 testing)
   - Note: Pure rotation detection is explicitly out of scope for MVP
2. ✅ **Zero missed real camera shifts** in recorded footage (Stage 2 testing)
3. ✅ **False positive rate <5%** in live deployment (Stage 3 testing)
4. ✅ **API integration works** - DAF system receives status and halts on INVALID correctly
5. ✅ **Manual recalibration completes** in <2 minutes
6. ✅ **System runs stable** for 1 week continuous operation without crashes
7. ✅ **History buffer functions** - Can query past results by frame_id or retrieve recent N

### Go/No-Go Decision:

- **GO:** All 7 criteria met → Deploy to production, consider future features
- **NO-GO:** Any critical criterion fails → Iterate on MVP, extend testing

---

## 8. Future Considerations (Post-MVP)

**Only consider AFTER MVP proves itself:**

1. Add RANSAC if false positives > 5%
2. Explore automatic recalibration strategies (if lighting drift becomes issue)
3. Optimize check frequency (1-second intervals if 5-10 min insufficient)
4. Multi-site generalization (auto-detect static regions)
5. Comprehensive logging and diagnostics to file/database
6. Persistent history storage (save to disk, not just in-memory)

**Decision Rule:** Add features based on real performance data and actual problems encountered, not theoretical concerns.

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-SIMPLIFIED | 2025-10-17 | Initial simplified MVP PRD based on brainstorming session |

---

**End of Document**
