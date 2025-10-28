# Camera Shift Detector - Integration Guide

**Version:** 0.1.0
**Last Updated:** 2025-10-26
**Target Audience:** System Integrators, DAF System Developers

---

## Table of Contents

1. [Introduction](#introduction)
2. [Black-Box API Overview](#black-box-api-overview)
3. [Integration Patterns](#integration-patterns)
4. [Flow Diagrams](#flow-diagrams)
5. [Code Examples](#code-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Introduction

The Camera Shift Detector is a black-box computer vision module designed to monitor camera position in DAF (Dissolved Air Flotation) water quality monitoring systems. This guide provides comprehensive integration documentation for system integrators.

### Purpose

When cameras shift position, the neural network's Region of Interest (ROI) becomes misaligned, causing inaccurate turbidity and flow measurements. This module detects camera displacement exceeding 2 pixels and signals the parent DAF system to halt data collection until manual correction is performed.

### Key Features

- **Simple API**: 5 methods, 1 result dict format
- **Black-Box Design**: No internal state exposure
- **Synchronous Execution**: Direct function calls, no async complexity
- **History Buffer**: Query last 100 detection results
- **Manual Recalibration**: Reset baseline when needed

---

## Black-Box API Overview

### CameraMovementDetector Class

The `CameraMovementDetector` class is the single public interface for integration.

```python
from src.camera_movement_detector import CameraMovementDetector
```

### API Methods

#### 1. `__init__(config_path='config.json')`

Initialize the detector with configuration.

**Parameters:**
- `config_path` (str): Path to JSON config file with ROI and parameters

**Raises:**
- `FileNotFoundError`: If config file not found
- `ValueError`: If config validation fails

**Example:**
```python
detector = CameraMovementDetector('config.json')
```

---

#### 2. `set_baseline(image_array)`

Capture initial baseline features during setup phase.

**Parameters:**
- `image_array` (np.ndarray): Reference image (H × W × 3, uint8, BGR format)

**Raises:**
- `ValueError`: If insufficient features detected (<50)

**Example:**
```python
import cv2
initial_frame = cv2.imread('baseline.jpg')
detector.set_baseline(initial_frame)
```

---

#### 3. `process_frame(image_array, frame_id=None)`

Detect camera movement in a single frame.

**Parameters:**
- `image_array` (np.ndarray): Current image (H × W × 3, uint8, BGR format)
- `frame_id` (str, optional): Identifier for tracking (auto-generated if None)

**Returns:**
```python
{
    "status": "VALID" | "INVALID",
    "displacement": float,  # pixels
    "confidence": float,    # [0.0, 1.0] inlier ratio
    "frame_id": str,
    "timestamp": str  # ISO 8601 UTC
}
```

**Raises:**
- `RuntimeError`: If baseline not set (call set_baseline() first)
- `ValueError`: If image_array invalid format

**Example:**
```python
current_frame = camera.get_frame()
result = detector.process_frame(current_frame, frame_id="frame_001")

if result['status'] == 'INVALID':
    print(f"Camera moved {result['displacement']:.2f} pixels!")
```

---

#### 4. `recalibrate(image_array)`

Manually reset baseline features (e.g., after lighting changes).

**Parameters:**
- `image_array` (np.ndarray): New reference image (required)

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
new_baseline = camera.get_frame()
success = detector.recalibrate(new_baseline)
if success:
    print("Recalibration successful!")
```

---

#### 5. `get_history(frame_id=None, limit=None)`

Query detection history buffer.

**Parameters:**
- `frame_id` (str, optional): Return results for specific frame_id
- `limit` (int, optional): Return last N results

**Returns:**
- `List[Dict]`: List of detection result dicts (empty list if no matches)

**Example:**
```python
# Get last 10 detection results
recent_results = detector.get_history(limit=10)

# Get result for specific frame
frame_result = detector.get_history(frame_id="frame_001")
```

---

### Configuration File Format

The detector requires a `config.json` file with the following structure:

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

**Field Descriptions:**
- `roi`: Region of Interest coordinates (static image region)
- `threshold_pixels`: Displacement threshold for INVALID status
- `history_buffer_size`: Maximum history buffer entries
- `min_features_required`: Minimum ORB features required in ROI

---

## Integration Patterns

### Pattern 1: Simple Periodic Check

**Use Case:** Check camera position every 5-10 minutes during measurement cycles.

**Flow:**
1. Initialize detector once at startup
2. Set baseline with initial camera frame
3. During each measurement cycle:
   - Get current camera frame
   - Call `process_frame()`
   - Check `status` field
   - Proceed or halt based on result

**When to Use:**
- Standard DAF monitoring with periodic measurements
- Low-frequency checking (5-10 min intervals)
- Synchronous integration with existing measurement loops

---

### Pattern 2: Continuous Monitoring

**Use Case:** Monitor camera position continuously during operations.

**Flow:**
1. Initialize detector once at startup
2. Set baseline with initial camera frame
3. In main monitoring loop:
   - Get current camera frame
   - Call `process_frame()`
   - Log results to monitoring system
   - Alert if `status == INVALID`

**When to Use:**
- High-criticality applications requiring immediate detection
- Systems with continuous camera frame availability
- Integration with real-time monitoring dashboards

---

### Pattern 3: Error Handling and Recovery

**Use Case:** Robust integration with comprehensive error handling.

**Flow:**
1. Try initialization with error handling
2. Validate baseline capture
3. Wrap `process_frame()` in try-except
4. Handle specific exceptions appropriately
5. Implement retry logic for transient failures

**When to Use:**
- Production deployments requiring high reliability
- Systems with unpredictable camera feed quality
- Integration with fault-tolerant architectures

---

### Pattern 4: Manual Recalibration Workflow

**Use Case:** Allow operators to reset baseline after maintenance or lighting changes.

**Flow:**
1. Detect recalibration trigger (operator command, scheduled event)
2. Capture new baseline frame
3. Call `recalibrate()` with new frame
4. Verify success
5. Resume normal monitoring

**When to Use:**
- Sites with frequent lighting changes
- After camera maintenance or repositioning
- Scheduled baseline refreshes

---

## Flow Diagrams

### 1. Setup Flow

```
┌─────────────────────────────────────┐
│ System Startup                      │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ Load config.json         │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Initialize CameraMovementDetector│
    │ detector = CMD('config.json')    │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Get initial camera frame         │
    │ initial_frame = camera.get()     │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Set baseline                     │
    │ detector.set_baseline(frame)     │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Ready for monitoring             │
    └──────────────────────────────────┘
```

---

### 2. Runtime Detection Flow

```
┌─────────────────────────────────────┐
│ Measurement Cycle Start             │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Get current camera frame         │
    │ frame = camera.get_frame()       │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Detect camera movement           │
    │ result = detector.process_frame()│
    └──────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Check status │
        └──┬───────┬───┘
           │       │
   INVALID │       │ VALID
           │       │
           ▼       ▼
    ┌──────────┐  ┌────────────────────┐
    │ Log      │  │ Check confidence   │
    │ warning  │  └────┬───────────────┘
    │ Halt     │       │
    │ measure  │       ▼
    └──────────┘  ┌────────────────────┐
                  │ Proceed with       │
                  │ measurements       │
                  └────────────────────┘
```

---

### 3. Error Handling Flow

```
┌─────────────────────────────────────┐
│ Call process_frame()                │
└──────────────┬──────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Try          │
        └──┬───────────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │ Execute process_frame()          │
    └──┬────────────┬──────────────────┘
       │            │
   Success          Exception
       │            │
       ▼            ▼
    ┌──────────┐  ┌────────────────────┐
    │ Return   │  │ Exception type?    │
    │ result   │  └──┬─────────────┬───┘
    └──────────┘     │             │
                     │             │
              RuntimeError    ValueError
                     │             │
                     ▼             ▼
              ┌──────────┐  ┌────────────┐
              │ Baseline │  │ Invalid    │
              │ not set  │  │ image      │
              │ → Reset  │  │ → Retry    │
              └──────────┘  └────────────┘
```

---

### 4. Recalibration Flow

```
┌─────────────────────────────────────┐
│ Recalibration Trigger               │
│ (operator/scheduled)                │
└──────────────┬──────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Pause monitoring                 │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Capture new baseline frame       │
    │ new_frame = camera.get_frame()   │
    └──────────┬───────────────────────┘
               │
               ▼
    ┌──────────────────────────────────┐
    │ Recalibrate detector             │
    │ success = detector.recalibrate() │
    └──────────┬───────────────────────┘
               │
               ▼
        ┌──────────────┐
        │ Check result │
        └──┬───────┬───┘
           │       │
       Success   Failure
           │       │
           ▼       ▼
    ┌──────────┐  ┌────────────────────┐
    │ Resume   │  │ Log error          │
    │ monitor  │  │ Retry or alert     │
    └──────────┘  └────────────────────┘
```

---

## Code Examples

### Scenario 1: Simple Periodic Check

**Use Case:** Basic integration with periodic camera checks every 5-10 minutes.

```python
import cv2
from src.camera_movement_detector import CameraMovementDetector

# Initialize once at startup
detector = CameraMovementDetector('config.json')

# Capture initial baseline
initial_frame = cv2.imread('baseline.jpg')  # or camera.get_frame()
detector.set_baseline(initial_frame)

# Measurement cycle (called every 5-10 minutes)
def measurement_cycle():
    # Get current camera frame
    current_frame = camera.get_frame()

    # Detect camera movement
    result = detector.process_frame(current_frame, frame_id=f"frame_{timestamp}")

    # Check status
    if result['status'] == 'INVALID':
        logger.warning(
            f"Camera moved {result['displacement']:.2f}px - halting measurements"
        )
        return None  # Skip this measurement

    # Proceed with water quality analysis
    return vision_analysis(current_frame)
```

---

### Scenario 2: Continuous Monitoring

**Use Case:** Continuous monitoring with real-time alerting.

```python
import cv2
import time
from src.camera_movement_detector import CameraMovementDetector

# Initialize detector
detector = CameraMovementDetector('config.json')
detector.set_baseline(camera.get_frame())

# Continuous monitoring loop
def continuous_monitor():
    while monitoring_active:
        # Get current frame
        frame = camera.get_frame()

        # Detect movement
        result = detector.process_frame(frame, frame_id=generate_frame_id())

        # Log result to monitoring system
        monitoring_system.log(result)

        # Alert if camera shifted
        if result['status'] == 'INVALID':
            alert_system.send_alert(
                severity='HIGH',
                message=f"Camera shift detected: {result['displacement']:.2f}px",
                confidence=result['confidence']
            )

        # Check confidence for warnings
        elif result['confidence'] < 0.5:
            alert_system.send_alert(
                severity='MEDIUM',
                message=f"Low detection confidence: {result['confidence']:.2f}",
                displacement=result['displacement']
            )

        # Wait before next check
        time.sleep(check_interval)
```

---

### Scenario 3: Error Handling and Recovery

**Use Case:** Production integration with comprehensive error handling.

```python
import cv2
import logging
from src.camera_movement_detector import CameraMovementDetector

logger = logging.getLogger(__name__)

class RobustCameraMonitor:
    def __init__(self, config_path='config.json'):
        try:
            self.detector = CameraMovementDetector(config_path)
            logger.info("Detector initialized successfully")
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except ValueError as e:
            logger.error(f"Invalid config: {e}")
            raise

    def initialize_baseline(self, max_retries=3):
        """Initialize baseline with retry logic"""
        for attempt in range(max_retries):
            try:
                frame = camera.get_frame()
                self.detector.set_baseline(frame)
                logger.info("Baseline set successfully")
                return True
            except ValueError as e:
                logger.warning(
                    f"Baseline capture failed (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error("Failed to set baseline after all retries")
                    return False

    def check_camera_position(self, frame_id=None):
        """Check camera position with error handling"""
        try:
            frame = camera.get_frame()
            result = self.detector.process_frame(frame, frame_id=frame_id)
            return result

        except RuntimeError as e:
            logger.error(f"Baseline not set: {e}")
            # Attempt to reinitialize
            if self.initialize_baseline():
                return self.check_camera_position(frame_id)
            else:
                return None

        except ValueError as e:
            logger.error(f"Invalid frame format: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def safe_measurement_cycle(self):
        """Measurement cycle with error handling"""
        result = self.check_camera_position(frame_id=generate_frame_id())

        if result is None:
            logger.error("Camera check failed - skipping measurement")
            return None

        if result['status'] == 'INVALID':
            logger.warning(
                f"Camera moved {result['displacement']:.2f}px - halting measurements"
            )
            return None

        # Proceed with analysis
        return vision_analysis(camera.get_frame())
```

---

### Scenario 4: Manual Recalibration Workflow

**Use Case:** Allow operators to recalibrate the detector.

```python
import cv2
from src.camera_movement_detector import CameraMovementDetector

class CameraMonitorWithRecalibration:
    def __init__(self, config_path='config.json'):
        self.detector = CameraMovementDetector(config_path)
        self.baseline_set = False

    def set_initial_baseline(self):
        """Set initial baseline during setup"""
        frame = camera.get_frame()
        self.detector.set_baseline(frame)
        self.baseline_set = True
        logger.info("Initial baseline set")

    def recalibrate(self):
        """Manually recalibrate the detector"""
        logger.info("Starting recalibration...")

        # Pause monitoring
        self.pause_monitoring()

        # Wait for stable frame (optional)
        time.sleep(2)

        # Capture new baseline
        new_frame = camera.get_frame()

        # Recalibrate
        success = self.detector.recalibrate(new_frame)

        if success:
            logger.info("Recalibration successful")
            self.baseline_set = True
            # Clear history (optional)
            # Resume monitoring
            self.resume_monitoring()
            return True
        else:
            logger.error("Recalibration failed")
            return False

    def auto_recalibrate_on_schedule(self, interval_hours=24):
        """Automatically recalibrate every N hours"""
        while monitoring_active:
            time.sleep(interval_hours * 3600)
            logger.info("Scheduled recalibration triggered")
            self.recalibrate()

    def handle_recalibration_command(self, command):
        """Handle operator recalibration command"""
        if command == "RECALIBRATE":
            success = self.recalibrate()
            return {
                "status": "SUCCESS" if success else "FAILED",
                "message": "Recalibration completed" if success else "Recalibration failed",
                "timestamp": datetime.utcnow().isoformat()
            }
```

---

## Best Practices

### 1. Initialization

✅ **DO:**
- Initialize detector once at system startup
- Set baseline immediately after initialization
- Validate baseline capture success

❌ **DON'T:**
- Reinitialize detector for each frame
- Skip baseline capture
- Ignore baseline validation errors

---

### 2. Frame Processing

✅ **DO:**
- Use meaningful frame IDs for tracking
- Check both `status` and `confidence` fields
- Log all detection results for debugging

❌ **DON'T:**
- Ignore low confidence warnings
- Process frames without baseline set
- Skip error handling

---

### 3. Error Handling

✅ **DO:**
- Wrap API calls in try-except blocks
- Handle specific exceptions appropriately
- Implement retry logic for transient failures

❌ **DON'T:**
- Catch all exceptions without specific handling
- Continue processing after critical errors
- Ignore RuntimeError (baseline not set)

---

### 4. Recalibration

✅ **DO:**
- Recalibrate after lighting changes
- Pause monitoring during recalibration
- Verify recalibration success

❌ **DON'T:**
- Recalibrate during measurement cycles
- Skip recalibration after camera maintenance
- Ignore recalibration failures

---

### 5. Performance

✅ **DO:**
- Use appropriate check intervals (5-10 min for periodic)
- Monitor processing time for performance issues
- Keep history buffer size reasonable (default: 100)

❌ **DON'T:**
- Check every frame unnecessarily (performance impact)
- Ignore processing time warnings
- Set history buffer too large (memory impact)

---

## Troubleshooting

### Common Issues

#### 1. RuntimeError: Baseline not set

**Cause:** Attempting to call `process_frame()` before setting baseline.

**Solution:**
```python
# Always set baseline first
detector.set_baseline(initial_frame)
```

---

#### 2. ValueError: Insufficient features detected

**Cause:** ROI region has too few detectable features (<50).

**Solution:**
- Choose a different ROI with more texture (tank walls, pipes)
- Increase ROI size in `config.json`
- Use ROI selection tool to validate feature count

---

#### 3. ValueError: Invalid image format

**Cause:** Image array not in correct format (H × W × 3, uint8, BGR).

**Solution:**
```python
# Ensure correct format
frame = cv2.imread('image.jpg')  # Loads as BGR, uint8
# or
frame = camera.get_frame()  # Ensure camera returns BGR format
```

---

#### 4. Low confidence warnings (confidence < 0.5)

**Cause:** Lighting changes, scene changes, or ambiguous features.

**Solution:**
- Recalibrate if lighting changed
- Check if camera was bumped/moved
- Verify ROI still contains static features

---

#### 5. False positives (INVALID status when camera didn't move)

**Cause:** Lighting changes, water bubbles in ROI, environmental factors.

**Solution:**
- Recalibrate to account for lighting
- Adjust `threshold_pixels` in config (increase)
- Choose better ROI with truly static features

---

### Getting Help

For integration support:

1. Check this guide and API documentation
2. Review code examples matching your use case
3. Verify configuration and baseline setup
4. Check logs for specific error messages
5. Contact development team with:
   - Error message
   - Configuration file
   - Sample images (if possible)
   - Integration code snippet

---

**Document Version:** 0.1.0
**Last Updated:** 2025-10-26
