# Camera Shift Detector - Integration Cheat Sheet

**Version:** 0.1.0 | **Meeting Date:** 2025-10-27

---

## Quick Start (30 seconds)

```python
from src.camera_movement_detector import CameraMovementDetector

# 1. Initialize
detector = CameraMovementDetector('config.json')

# 2. Set baseline
detector.set_baseline(initial_frame)

# 3. Check for movement
result = detector.process_frame(current_frame)

# 4. Handle result
if result['status'] == 'INVALID':
    halt_measurements()  # Camera moved!
```

---

## API Quick Reference

| Method | Purpose | Returns |
|--------|---------|---------|
| `__init__(config_path)` | Initialize detector | None |
| `set_baseline(image)` | Capture reference | None |
| `process_frame(image, id)` | Detect movement | Result dict |
| `recalibrate(image)` | Reset baseline | bool |
| `get_history(id, limit)` | Query history | List[dict] |

---

## Result Format

```python
{
    "status": "VALID" | "INVALID",      # Camera position status
    "displacement": 1.23,                # Pixels moved (float)
    "confidence": 0.95,                  # Detection confidence [0.0-1.0]
    "frame_id": "frame_001",             # Your identifier
    "timestamp": "2025-10-26T14:32:18Z"  # ISO 8601 UTC
}
```

**Key Decision Logic:**
- `status == "INVALID"` → Camera moved ≥2px → **HALT measurements**
- `status == "VALID"` + `confidence < 0.5` → **WARNING** (lighting change?)
- `status == "VALID"` + `confidence ≥ 0.5` → **PROCEED** with measurements

---

## Installation (1 command)

```bash
cd cam-shift-detector && pip install -e .
```

**Dependencies:** Automatic (opencv-python, opencv-contrib-python, numpy)

**Requirements:** Python 3.11+, Linux/macOS/Windows

---

## Configuration File (config.json)

```json
{
  "roi": {
    "x": 100, "y": 50,
    "width": 400, "height": 300
  },
  "threshold_pixels": 2.0,
  "history_buffer_size": 100,
  "min_features_required": 50
}
```

**Generate automatically:** `python tools/select_roi.py --source image --path baseline.jpg`

---

## Integration Pattern (Copy-Paste Ready)

```python
import cv2
import logging
from src.camera_movement_detector import CameraMovementDetector

logger = logging.getLogger(__name__)

class DAFCameraMonitor:
    def __init__(self, config_path='config.json'):
        # Initialize detector
        self.detector = CameraMovementDetector(config_path)
        self.baseline_set = False

    def setup(self, baseline_image):
        """Call once at startup"""
        self.detector.set_baseline(baseline_image)
        self.baseline_set = True
        logger.info("Camera monitor ready")

    def check_before_measurement(self, current_image, measurement_id):
        """Call before each water quality measurement"""
        if not self.baseline_set:
            raise RuntimeError("Baseline not set - call setup() first")

        # Detect camera movement
        result = self.detector.process_frame(
            current_image,
            frame_id=measurement_id
        )

        # Log result
        logger.info(
            f"Camera check: {result['status']} "
            f"(displacement={result['displacement']:.2f}px, "
            f"confidence={result['confidence']:.2f})"
        )

        # Decision logic
        if result['status'] == 'INVALID':
            logger.warning(
                f"Camera moved {result['displacement']:.2f}px - "
                "halting measurements until manual correction"
            )
            return False  # Don't proceed

        elif result['confidence'] < 0.5:
            logger.warning(
                f"Low confidence {result['confidence']:.2f} - "
                "proceed with caution (lighting change?)"
            )
            return True  # Proceed but flagged

        else:
            return True  # Proceed normally

    def recalibrate_baseline(self, new_baseline_image):
        """Call after lighting changes or maintenance"""
        success = self.detector.recalibrate(new_baseline_image)
        if success:
            logger.info("Recalibration successful")
        else:
            logger.error("Recalibration failed")
        return success

# Usage example:
monitor = DAFCameraMonitor('config.json')
monitor.setup(camera.get_frame())

# In your measurement cycle:
if monitor.check_before_measurement(camera.get_frame(), "measurement_123"):
    # Proceed with water quality analysis
    perform_turbidity_measurement()
else:
    # Skip this measurement
    alert_operator("Camera shift detected")
```

---

## Common Integration Flows

### Flow 1: Periodic Check (Every 5-10 minutes)

```
Measurement Cycle Start
  ↓
Get camera frame
  ↓
Call detector.process_frame()
  ↓
Check result['status']
  ├─ INVALID → Halt + Alert
  └─ VALID → Proceed
```

### Flow 2: Error Handling

```python
try:
    result = detector.process_frame(frame)
except RuntimeError:
    # Baseline not set
    detector.set_baseline(frame)
except ValueError:
    # Invalid image format
    logger.error("Invalid frame format")
```

### Flow 3: Recalibration

```
Trigger (operator/scheduled)
  ↓
Pause monitoring
  ↓
Capture new frame
  ↓
Call detector.recalibrate(frame)
  ↓
Resume monitoring
```

---

## Key Talking Points for Meeting

### 1. **Simple Integration**
   - 5 methods, 1 result dict
   - No complex state management
   - Synchronous, no async complexity

### 2. **Black-Box Design**
   - Internal algorithms abstracted
   - Clear input/output contract
   - Easy to test and validate

### 3. **Production Ready**
   - 100% detection on synthetic data (Epic 1)
   - Validation framework complete (Epic 2)
   - 109 tests passing, 93% coverage

### 4. **Integration Options**
   - **Option A:** Periodic checks (5-10 min intervals)
   - **Option B:** Continuous monitoring
   - **Option C:** Hybrid (periodic + on-demand)

### 5. **Operational Flexibility**
   - Manual recalibration when needed
   - History buffer for debugging
   - Confidence scoring for nuanced decisions

### 6. **Parallel Development Possible**
   - Stub implementation available
   - Integrate now, swap real module later
   - No waiting for final implementation

---

## Stub Implementation (Optional)

For immediate integration testing without waiting for production module:

```python
# src/camera_movement_detector_stub.py
import numpy as np
from datetime import datetime
from typing import Dict, List

class CameraMovementDetector:
    """Stub implementation for integration testing"""

    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.baseline_set = False

    def set_baseline(self, image_array: np.ndarray) -> None:
        self.baseline_set = True

    def process_frame(self, image_array: np.ndarray, frame_id: str = None) -> Dict:
        if not self.baseline_set:
            raise RuntimeError("Baseline not set")

        return {
            "status": "VALID",
            "displacement": 0.5,
            "confidence": 0.95,
            "frame_id": frame_id or f"stub_{datetime.utcnow().timestamp()}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

    def recalibrate(self, image_array: np.ndarray) -> bool:
        self.baseline_set = True
        return True

    def get_history(self, frame_id: str = None, limit: int = None) -> List[Dict]:
        return []
```

**Swap later:**
```python
# Change from:
from src.camera_movement_detector_stub import CameraMovementDetector

# To:
from src.camera_movement_detector import CameraMovementDetector
```

---

## Questions to Discuss

1. **Integration Timing:** When will you integrate? (now with stub vs. after validation)
2. **Check Frequency:** How often to check? (5-10 min periodic vs. continuous)
3. **Alert Strategy:** How to handle INVALID status? (halt + alert vs. log only)
4. **Recalibration:** Manual only or scheduled automatic?
5. **Confidence Thresholds:** Default 0.5 acceptable or adjust?
6. **History Usage:** Need historical data for debugging?
7. **Deployment:** Timeline for production deployment?

---

## Next Steps After Meeting

1. **Immediate:** Install package (`pip install -e .`)
2. **Day 1:** Test with stub implementation
3. **Day 2-3:** Run Stage 3 validation
4. **Week 1:** Integrate with real module
5. **Week 2:** Pilot deployment at first site

---

## Documentation Links

- **Full Integration Guide:** `docs/integration-guide.md`
- **Installation Guide:** `docs/installation.md`
- **Technical Spec:** `docs/tech-spec-epic-MVP-001.md`
- **API Reference:** See tech-spec Section "APIs and Interfaces"

---

## Contact

**Questions during integration?**
- Check integration guide first
- Review code examples
- Reach out to development team

---

**Cheat Sheet Version:** 0.1.0 | **Prepared for:** 2025-10-27 Stakeholder Meeting
