# Story 1.5: Camera Movement Detector (Main API)

Status: Done

## Story

As a **DAF water quality monitoring system**,
I want to **integrate camera movement detection via a simple black-box API that processes frames and returns validation status**,
so that **I can halt measurements when camera displacement is detected and prevent use of corrupted data**.

## Acceptance Criteria

1. **AC-1.5.1: Initialization & Config Loading** - CameraMovementDetector loads configuration from JSON file, validates schema, initializes all internal components (StaticRegionManager, FeatureExtractor, MovementDetector, ResultManager)

2. **AC-1.5.2: Baseline Capture** - `set_baseline(image_array)` captures initial reference features with validation (≥50 features required), raises ValueError if insufficient

3. **AC-1.5.3: Frame Processing** - `process_frame(image_array, frame_id)` orchestrates full detection pipeline (mask → extract → detect → build result → store history) and returns standardized result dict

4. **AC-1.5.4: Runtime Error Handling** - Raises RuntimeError if `process_frame()` called before baseline set; raises ValueError for invalid image formats; returns appropriate status on detection failures

5. **AC-1.5.5: History Query Interface** - `get_history(frame_id, limit)` queries ResultManager history buffer, supports filtering by frame_id or limiting to last N results

6. **AC-1.5.6: Manual Recalibration** - `recalibrate(image_array)` resets baseline features, validates minimum feature count, returns success/failure boolean

7. **AC-1.5.7: Config Validation** - Validates config.json schema at initialization: required fields (roi, threshold_pixels, history_buffer_size, min_features_required), correct data types, reasonable value ranges

8. **AC-1.5.8: Integration Testing** - System processes frames end-to-end with all components integrated, handles edge cases (insufficient matches, homography failures), maintains thread-safe operation if needed

## Tasks / Subtasks

- [x] **Task 1: Create CameraMovementDetector class skeleton** (AC: #1.5.1, #1.5.7)
  - [x] 1.1: Define `__init__(self, config_path='config.json')` - load and validate config
  - [x] 1.2: Implement config schema validation (roi, threshold_pixels, history_buffer_size, min_features_required)
  - [x] 1.3: Initialize all component instances (StaticRegionManager, FeatureExtractor, MovementDetector, ResultManager)
  - [x] 1.4: Handle FileNotFoundError for missing config
  - [x] 1.5: Raise ValueError with descriptive message for invalid config schema
  - [x] 1.6: Store config values as instance variables

- [x] **Task 2: Implement baseline capture** (AC: #1.5.2)
  - [x] 2.1: Define `set_baseline(self, image_array: np.ndarray) -> None`
  - [x] 2.2: Validate image_array format (H×W×3, uint8, NumPy array)
  - [x] 2.3: Generate static mask via StaticRegionManager
  - [x] 2.4: Call FeatureExtractor.set_baseline(image, mask)
  - [x] 2.5: Validate ≥min_features_required detected
  - [x] 2.6: Raise ValueError if insufficient features
  - [x] 2.7: Set internal flag baseline_set = True

- [x] **Task 3: Implement frame processing pipeline** (AC: #1.5.3, #1.5.4)
  - [x] 3.1: Define `process_frame(self, image_array: np.ndarray, frame_id: Optional[str] = None) -> Dict`
  - [x] 3.2: Check baseline_set flag, raise RuntimeError if False
  - [x] 3.3: Validate image_array format (same as baseline validation)
  - [x] 3.4: Generate static mask via StaticRegionManager
  - [x] 3.5: Extract current features via FeatureExtractor
  - [x] 3.6: Get baseline features via FeatureExtractor.get_baseline()
  - [x] 3.7: Detect movement via MovementDetector.detect_movement()
  - [x] 3.8: Build result dict via ResultManager.create_result()
  - [x] 3.9: Store in history via ResultManager.add_to_history()
  - [x] 3.10: Return result dict to caller

- [x] **Task 4: Implement history query interface** (AC: #1.5.5)
  - [x] 4.1: Define `get_history(self, frame_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]`
  - [x] 4.2: If frame_id provided: call ResultManager.get_by_frame_id(frame_id), return as list
  - [x] 4.3: If limit provided: call ResultManager.get_last_n(limit)
  - [x] 4.4: If neither: call ResultManager.get_history() for all results
  - [x] 4.5: Handle edge cases (frame_id not found → empty list)

- [x] **Task 5: Implement manual recalibration** (AC: #1.5.6)
  - [x] 5.1: Define `recalibrate(self, image_array: np.ndarray) -> bool`
  - [x] 5.2: Validate image_array format
  - [x] 5.3: Call set_baseline(image_array) internally
  - [x] 5.4: Return True on success, False on failure (catch ValueError)
  - [x] 5.5: Optionally clear history buffer on recalibration

- [x] **Task 6: Error handling and edge cases** (AC: #1.5.4)
  - [x] 6.1: Handle insufficient feature matches (< min matches) → return status="INVALID", displacement=inf
  - [x] 6.2: Handle homography estimation failure → return status="INVALID", displacement=inf, log warning
  - [x] 6.3: Validate image dimensions match baseline dimensions
  - [x] 6.4: Add type hints throughout for clarity
  - [x] 6.5: Add docstrings matching tech-spec API documentation

- [x] **Task 7: Integration tests** (AC: #1.5.8)
  - [x] 7.1: Test full pipeline: init → set_baseline → process_frame → get_history
  - [x] 7.2: Test with real sample images from sample_images/
  - [x] 7.3: Test baseline not set error
  - [x] 7.4: Test invalid config file scenarios
  - [x] 7.5: Test recalibration workflow
  - [x] 7.6: Test history queries (by frame_id, by limit)
  - [x] 7.7: Test edge cases (insufficient features, homography failure)
  - [x] 7.8: Performance test: process_frame < 500ms target

## Dev Notes

### Architecture & Design Patterns

**Module Responsibility** (Tech-Spec Section: Services and Modules):
- Black-box API orchestrator integrating all 4 core components
- Config file loading and validation
- Component lifecycle management
- Error handling and exception propagation
- Public interface: init, set_baseline, process_frame, get_history, recalibrate

**Component Integration Flow**:
```python
# Initialization
detector = CameraMovementDetector('config.json')  # Loads config, creates components

# Setup phase
detector.set_baseline(reference_image)  # Captures baseline features

# Runtime detection (every 5-10 minutes)
result = detector.process_frame(current_image, frame_id="frame_001")
# → StaticRegionManager.get_static_mask()
# → FeatureExtractor.extract_features(image, mask)
# → MovementDetector.detect_movement(baseline, current)
# → ResultManager.create_result(displacement, confidence, frame_id)
# → ResultManager.add_to_history(result)
# → return result

# Query history
history = detector.get_history(limit=10)

# Recalibration
success = detector.recalibrate(new_reference_image)
```

**Integration with Previous Stories**:
- **Story 1.1 (StaticRegionManager)**: Provides binary masks for ROI
- **Story 1.2 (FeatureExtractor)**: Baseline capture, feature extraction
- **Story 1.3 (MovementDetector)**: Displacement calculation, confidence scoring
- **Story 1.4 (ResultManager)**: Result dict building, history management

### Implementation Guidance

**Class Structure** (Tech-Spec Section: APIs and Interfaces):
```python
import numpy as np
from typing import Dict, List, Optional
from src.static_region_manager import StaticRegionManager
from src.feature_extractor import FeatureExtractor
from src.movement_detector import MovementDetector
from src.result_manager import ResultManager
import json

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
        # Load config.json
        # Validate schema (required fields, types, ranges)
        # Initialize components:
        # - self.region_manager = StaticRegionManager(config)
        # - self.feature_extractor = FeatureExtractor(min_features=config['min_features_required'])
        # - self.movement_detector = MovementDetector(threshold=config['threshold_pixels'])
        # - self.result_manager = ResultManager(threshold=config['threshold_pixels'],
        #                                       buffer_size=config['history_buffer_size'])
        # self.baseline_set = False

    def set_baseline(self, image_array: np.ndarray) -> None:
        """
        Capture initial baseline features (setup phase).

        Args:
            image_array: Reference image for baseline (H×W×3, uint8, BGR)

        Raises:
            ValueError: If insufficient features detected (<min_features_required)
            ValueError: If image_array invalid format
        """
        # Validate image format
        # Generate mask: mask = self.region_manager.get_static_mask(image_array.shape[:2])
        # Set baseline: self.feature_extractor.set_baseline(image_array, mask)
        # Validate feature count (FeatureExtractor raises ValueError if insufficient)
        # self.baseline_set = True

    def process_frame(self, image_array: np.ndarray, frame_id: str = None) -> Dict:
        """
        Detect camera movement in single frame.

        Args:
            image_array: NumPy array (H × W × 3, uint8, BGR format)
            frame_id: Optional identifier for tracking (auto-generated if None)

        Returns:
            {
                "status": "VALID" | "INVALID",
                "displacement": float,  # pixels (translation only)
                "confidence": float,    # [0.0, 1.0] inlier ratio
                "frame_id": str,
                "timestamp": str  # ISO 8601 UTC
            }

        Raises:
            RuntimeError: If baseline not set (call set_baseline() first)
            ValueError: If image_array invalid format
        """
        # Check self.baseline_set, raise RuntimeError if False
        # Validate image format
        # mask = self.region_manager.get_static_mask(image_array.shape[:2])
        # baseline = self.feature_extractor.get_baseline()
        # current = self.feature_extractor.extract_features(image_array, mask)
        # moved, displacement, confidence = self.movement_detector.detect_movement(baseline, current)
        # result = self.result_manager.create_result(displacement, confidence, frame_id)
        # self.result_manager.add_to_history(result)
        # return result

    def get_history(self, frame_id: str = None, limit: int = None) -> List[Dict]:
        """
        Query detection history buffer.

        Args:
            frame_id: Return results for specific frame_id (optional)
            limit: Return last N results (optional)

        Returns:
            List of detection result dicts (empty list if no matches)
        """
        # Delegate to ResultManager
        # if frame_id: return [self.result_manager.get_by_frame_id(frame_id)] or []
        # elif limit: return self.result_manager.get_last_n(limit)
        # else: return self.result_manager.get_history()

    def recalibrate(self, image_array: np.ndarray) -> bool:
        """
        Manually reset baseline features.

        Args:
            image_array: New reference image (required)

        Returns:
            True if successful, False otherwise
        """
        # try: self.set_baseline(image_array); return True
        # except ValueError: return False
```

**Config Schema Validation**:
```python
# config.json schema
required_fields = {
    "roi": dict,                        # ROI coordinates
    "threshold_pixels": (int, float),   # Displacement threshold
    "history_buffer_size": int,         # Buffer size (positive integer)
    "min_features_required": int        # Min features (positive integer)
}

# Validation logic
def _validate_config(config: dict) -> None:
    # Check all required fields present
    # Check roi has: x, y, width, height (all positive numbers)
    # Check threshold_pixels > 0
    # Check history_buffer_size > 0 and integer
    # Check min_features_required > 0 and integer
    # Raise ValueError with descriptive message on any failure
```

**Error Messages**:
- Baseline not set: `"Baseline not set. Call set_baseline() before process_frame()"`
- Invalid image format: `"image_array must be NumPy array with shape (H, W, 3) and dtype uint8, got {actual}"`
- Config file missing: `"Config file not found: {config_path}"`
- Invalid config schema: `"Invalid config: missing required field '{field}'" or "Invalid config: {field} must be {type}, got {actual}"`
- Insufficient features: `"Insufficient features detected: {count} < {min_required}"`

### Project Structure Notes

**File Location**: `src/camera_movement_detector.py`

**Directory Structure**:
```
cam-shift-detector/
├── src/
│   ├── static_region_manager.py      ← Story 1.1 (Done)
│   ├── feature_extractor.py          ← Story 1.2 (Done)
│   ├── movement_detector.py          ← Story 1.3 (Done)
│   ├── result_manager.py             ← Story 1.4 (Done)
│   └── camera_movement_detector.py   ← This story (Main API)
├── tests/
│   ├── test_static_region_manager.py
│   ├── test_feature_extractor.py
│   ├── test_movement_detector.py
│   ├── test_result_manager.py
│   └── test_camera_movement_detector.py  ← Integration tests
├── sample_images/                    ← Real test images
│   ├── of_jerusalem/
│   ├── carmit/
│   └── gad/
└── config.json                       ← Configuration file
```

**Dependencies**:
- Stories 1.1-1.4 (all core components)
- Python stdlib: json
- OpenCV (via FeatureExtractor)
- NumPy (image array handling)

**Config File** (`config.json`):
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

### Testing Standards

**Test Framework**: pytest

**Coverage Target**: >80% for this module (focus on integration paths)

**Test Categories**:
1. **Initialization**: Config loading, validation, component creation
2. **Baseline capture**: Valid/invalid images, feature count validation
3. **Frame processing**: Full pipeline, error handling, result accuracy
4. **History queries**: Filter by frame_id, limit, empty buffer
5. **Recalibration**: Success/failure scenarios
6. **Error cases**: Baseline not set, invalid formats, config errors
7. **Integration**: End-to-end workflows with real sample images

**Sample Integration Test**:
```python
def test_full_detection_workflow():
    """Test complete workflow: init → baseline → process → history"""
    # Initialize with test config
    detector = CameraMovementDetector('tests/fixtures/test_config.json')

    # Load sample images
    baseline = cv2.imread('sample_images/of_jerusalem/001.jpg')
    current = cv2.imread('sample_images/of_jerusalem/002.jpg')

    # Set baseline
    detector.set_baseline(baseline)

    # Process frame
    result = detector.process_frame(current, frame_id="test_001")

    # Validate result structure
    assert result["status"] in ["VALID", "INVALID"]
    assert "displacement" in result
    assert "confidence" in result
    assert result["frame_id"] == "test_001"
    assert "timestamp" in result

    # Query history
    history = detector.get_history()
    assert len(history) == 1
    assert history[0]["frame_id"] == "test_001"

def test_baseline_not_set_error():
    """Test RuntimeError when processing frame before baseline set"""
    detector = CameraMovementDetector('tests/fixtures/test_config.json')
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="Baseline not set"):
        detector.process_frame(image)
```

**Performance Test**:
```python
import time

def test_process_frame_performance():
    """Test process_frame executes in < 500ms (target: <200ms)"""
    detector = CameraMovementDetector('config.json')
    baseline = cv2.imread('sample_images/of_jerusalem/001.jpg')
    current = cv2.imread('sample_images/of_jerusalem/002.jpg')

    detector.set_baseline(baseline)

    start = time.time()
    result = detector.process_frame(current)
    elapsed = time.time() - start

    assert elapsed < 0.5, f"process_frame took {elapsed:.3f}s (> 500ms limit)"
```

### References

- [Source: tech-spec-epic-MVP-001.md#Services and Modules] - CameraMovementDetector module specifications, black-box design
- [Source: tech-spec-epic-MVP-001.md#APIs and Interfaces] - Complete API documentation with signatures and examples
- [Source: tech-spec-epic-MVP-001.md#Workflows and Sequencing] - Runtime detection workflow, integration examples
- [Source: tech-spec-epic-MVP-001.md#Data Models and Contracts] - Config schema, result schema
- [Source: tech-spec-epic-MVP-001.md#Acceptance Criteria] - AC-004 (API Integration), AC-009 (Baseline Capture), AC-010 (Error Handling)
- [Source: story-1.1.md] - StaticRegionManager integration
- [Source: story-1.2.md] - FeatureExtractor integration
- [Source: story-1.3.md] - MovementDetector integration
- [Source: story-1.4.md] - ResultManager integration

## Dev Agent Record

### Context Reference

- `docs/stories/story-context-1.1.5.xml` - Comprehensive implementation context generated 2025-10-23

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

Implementation executed Tasks 1-7 in single continuous session (2025-10-23):
1. Created CameraMovementDetector class with complete config validation and component initialization
2. Implemented all 5 public methods (set_baseline, process_frame, get_history, recalibrate) plus private helpers
3. Added comprehensive error handling including try/except for insufficient matches and homography failures
4. Created 34 integration tests covering all 8 acceptance criteria
5. Fixed edge case handling for ValueError/RuntimeError in process_frame (returns INVALID status with inf displacement)
6. Verified 156/156 tests passing (122 existing + 34 new) with no regressions

### Completion Notes List

**Completed:** 2025-10-23
**Definition of Done:** All acceptance criteria met, code reviewed, tests passing (156/156), no regressions

**2025-10-23**: Implemented CameraMovementDetector main API with complete integration of all 4 core components. All 8 acceptance criteria satisfied:

- **AC-1.5.1 ✓**: Initialization loads config from JSON, validates schema (roi, threshold_pixels, history_buffer_size, min_features_required), initializes all components (StaticRegionManager, FeatureExtractor, MovementDetector, ResultManager)
- **AC-1.5.2 ✓**: set_baseline() captures baseline features with validation (≥50 features), raises ValueError if insufficient
- **AC-1.5.3 ✓**: process_frame() orchestrates full pipeline (mask → extract → detect → build result → store history), returns standardized result dict
- **AC-1.5.4 ✓**: Runtime error handling raises RuntimeError if baseline not set, ValueError for invalid formats, returns INVALID status on detection failures (insufficient matches, homography failures)
- **AC-1.5.5 ✓**: get_history() queries ResultManager with filtering by frame_id or limit
- **AC-1.5.6 ✓**: recalibrate() resets baseline, returns True on success, False on failure
- **AC-1.5.7 ✓**: Config validation checks all required fields, types (int/float), value ranges (positive, non-negative)
- **AC-1.5.8 ✓**: Integration tests verify end-to-end workflow with real sample images, edge cases, performance (<500ms target met)

**Test Results**: 156/156 tests passing (34 new integration tests + 122 existing component tests). Zero regressions. Coverage includes all acceptance criteria with comprehensive edge case handling.

**Key Implementation Details**:
- Black-box design: Only 5 public methods exposed (init, set_baseline, process_frame, get_history, recalibrate)
- Stateless operation: Each process_frame() call is independent with history buffer for queryability
- Synchronous execution: No threading, straightforward sequential pipeline
- Exception handling: ValueError for invalid inputs, RuntimeError for baseline not set, graceful degradation for detection failures
- Type hints throughout for API clarity
- Comprehensive docstrings matching tech-spec documentation

### File List

- `src/camera_movement_detector.py` (new, 347 lines) - Main CameraMovementDetector class with all 5 public methods
- `tests/test_camera_movement_detector.py` (new, 527 lines) - Comprehensive integration tests with 34 test cases
