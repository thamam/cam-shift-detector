# Story 1.3: Movement Detector

Status: Done

## Story

As a **camera movement detection system**,
I want to **compare current frame features to baseline features using homography and calculate camera displacement**,
so that **I can determine if the camera has moved beyond the 2-pixel threshold and return movement status with confidence scores**.

## Acceptance Criteria

1. **AC-1.3.1: Feature Matching** - MovementDetector matches current features to baseline features using BFMatcher with NORM_HAMMING distance for ORB descriptors

2. **AC-1.3.2: Homography Estimation** - Calculates homography transformation matrix from matched keypoint pairs; handles cases with insufficient matches (<10) by returning appropriate status

3. **AC-1.3.3: Displacement Calculation** - Computes displacement magnitude from homography matrix; extracts translation vector and calculates Euclidean distance in pixels (rounded to 2 decimals)

4. **AC-1.3.4: Threshold Validation** - Compares displacement to configured threshold (default 2.0 pixels); returns `moved=True` if displacement ≥ threshold, `moved=False` otherwise

5. **AC-1.3.5: Confidence Score** - Calculates confidence score [0.0, 1.0] based on inlier ratio from homography estimation; returns confidence with detection result

6. **AC-1.3.6: Error Handling** - Raises appropriate exceptions for:
   - Invalid feature inputs (not correct format for keypoints/descriptors)
   - Insufficient feature matches (<10 matches)
   - Homography estimation failure (singular matrix, degenerate configuration)

## Tasks / Subtasks

- [x] **Task 1: Create MovementDetector class** (AC: #1.3.1, #1.3.2)
  - [x] 1.1: Define `__init__(self, threshold_pixels=2.0)` - initialize matcher and threshold
  - [x] 1.2: Initialize BFMatcher with NORM_HAMMING for ORB descriptors
  - [x] 1.3: Implement `detect_movement(baseline_features, current_features)` - main detection method
  - [x] 1.4: Implement feature matching using BFMatcher
  - [x] 1.5: Validate minimum match count (≥10 matches required)

- [x] **Task 2: Implement homography estimation** (AC: #1.3.2, #1.3.3)
  - [x] 2.1: Extract matched keypoint coordinates from baseline and current features
  - [x] 2.2: Call cv2.findHomography() with matched point pairs
  - [x] 2.3: Handle insufficient matches (<10) - return special status
  - [x] 2.4: Extract translation vector from homography matrix
  - [x] 2.5: Calculate displacement magnitude (Euclidean distance)
  - [x] 2.6: Round displacement to 2 decimal places

- [x] **Task 3: Implement threshold validation and confidence** (AC: #1.3.4, #1.3.5)
  - [x] 3.1: Compare displacement to threshold_pixels
  - [x] 3.2: Set moved flag: True if displacement ≥ threshold, False otherwise
  - [x] 3.3: Calculate confidence score from inlier ratio (num_inliers / total_matches)
  - [x] 3.4: Ensure confidence in range [0.0, 1.0]
  - [x] 3.5: Return tuple: (moved, displacement, confidence)

- [x] **Task 4: Implement error handling** (AC: #1.3.6)
  - [x] 4.1: Validate baseline_features format (tuple of keypoints and descriptors)
  - [x] 4.2: Validate current_features format (tuple of keypoints and descriptors)
  - [x] 4.3: Handle homography failure (None return from cv2.findHomography)
  - [x] 4.4: Raise descriptive errors for invalid inputs

- [x] **Task 5: Unit tests** (AC: All)
  - [x] 5.1: Test successful detection with no movement (displacement <2px)
  - [x] 5.2: Test successful detection with movement (displacement ≥2px)
  - [x] 5.3: Test insufficient matches (<10) handling
  - [x] 5.4: Test confidence score calculation (various inlier ratios)
  - [x] 5.5: Test homography failure scenarios
  - [x] 5.6: Test error cases (invalid feature format, None inputs)
  - [x] 5.7: Test edge cases (exactly 2.0px displacement, high/low confidence)

## Dev Notes

### Architecture & Design Patterns

**Module Responsibility** (Tech-Spec Section: Services and Modules):
- Match current frame features to baseline features using BFMatcher
- Estimate homography transformation between matched keypoint pairs
- Calculate camera displacement magnitude from homography matrix
- Compare displacement to threshold (2.0 pixels default)
- Calculate confidence score based on inlier ratio
- Return movement status tuple: (moved, displacement, confidence)
- **No feature extraction or mask generation** - pure comparison/detection logic

**Integration with Previous Stories**:
```python
# Story 1.1: StaticRegionManager provides masks
# Story 1.2: FeatureExtractor provides features
from src.static_region_manager import StaticRegionManager
from src.feature_extractor import FeatureExtractor
from src.movement_detector import MovementDetector

# Initialize components
region_manager = StaticRegionManager('config.json')
feature_extractor = FeatureExtractor(min_features_required=50)
movement_detector = MovementDetector(threshold_pixels=2.0)

# Setup baseline
mask = region_manager.get_static_mask(image.shape[:2])
feature_extractor.set_baseline(baseline_image, mask)

# Runtime detection
baseline_features = feature_extractor.get_baseline()
current_keypoints, current_descriptors = feature_extractor.extract_features(current_image, mask)
moved, displacement, confidence = movement_detector.detect_movement(
    baseline_features,
    (current_keypoints, current_descriptors)
)
```

**Data Structures** (Tech-Spec Section: Data Models):
```python
# Input Features (from FeatureExtractor)
baseline_features: Tuple[List[cv2.KeyPoint], np.ndarray]
current_features: Tuple[List[cv2.KeyPoint], np.ndarray]

# Feature Matches (cv2.DMatch objects)
matches: List[cv2.DMatch]

# Homography Matrix (3x3 transformation)
H: np.ndarray  # Shape (3, 3), dtype float64

# Output Tuple
(moved: bool, displacement: float, confidence: float)
```

**Homography and Displacement Calculation**:
```python
# Homography matrix H represents transformation from baseline to current
# H = [h11  h12  tx ]
#     [h21  h22  ty ]
#     [h31  h32  1  ]
#
# Translation vector: (tx, ty)
# Translation Displacement = sqrt(tx^2 + ty^2)
#
# ARCHITECTURAL LIMITATION (MVP Constraint):
# This implementation only extracts TRANSLATION displacement from homography.
# It does NOT detect:
#   - Rotation: Camera pan/tilt without translation
#   - Scale: Zoom in/out
#   - Shear/Perspective: Distortion from camera angle changes
#
# Impact: A camera could rotate significantly (corrupting measurements) while
# showing <2.0px translation, resulting in false "no movement" detection.
#
# Mitigation: Documented as MVP constraint. Full homography decomposition
# planned for Story 1.5 with separate thresholds for rotation, scale, translation.
```

**Confidence Score Calculation** (Tech-Spec Section: Data Models):
- Confidence = num_inliers / total_matches
- Inliers: Points that agree with homography transformation (cv2.findHomography returns mask)
- Range: [0.0, 1.0] where 1.0 = perfect agreement
- Low confidence (<0.5) suggests scene changes or ambiguous features

### Implementation Guidance

**Class Structure**:
```python
class MovementDetector:
    def __init__(self, threshold_pixels: float = 2.0):
        """Initialize movement detector with threshold"""
        # Create BFMatcher with NORM_HAMMING
        # Store threshold_pixels

    def detect_movement(
        self,
        baseline_features: Tuple[List, np.ndarray],
        current_features: Tuple[List, np.ndarray]
    ) -> Tuple[bool, float, float]:
        """
        Detect camera movement between baseline and current features.

        Returns:
            (moved, displacement, confidence)
            - moved: True if displacement ≥ threshold
            - displacement: Magnitude in pixels (rounded to 2 decimals)
            - confidence: Inlier ratio [0.0, 1.0]

        Raises:
            ValueError: If features invalid or insufficient matches
        """
        # Validate inputs
        # Match features using BFMatcher
        # Check minimum match count (≥10)
        # Estimate homography
        # Calculate displacement
        # Calculate confidence
        # Compare to threshold
        # Return (moved, displacement, confidence)
```

**BFMatcher Configuration**:
```python
# Use Brute Force Matcher with Hamming distance for ORB
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = matcher.match(baseline_descriptors, current_descriptors)

# Sort matches by distance (quality)
matches = sorted(matches, key=lambda x: x.distance)
```

**Homography Estimation**:
```python
# Extract matched point coordinates
src_pts = np.float32([baseline_keypoints[m.queryIdx].pt for m in matches])
dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in matches])

# Find homography (simple method initially, RANSAC optional)
H, mask = cv2.findHomography(src_pts, dst_pts, method=0)  # method=0: simple
# Or with RANSAC: cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Extract translation
tx = H[0, 2]
ty = H[1, 2]
displacement = np.sqrt(tx**2 + ty**2)

# Calculate confidence
inliers = np.sum(mask)
confidence = inliers / len(matches)
```

**Error Messages**:
- Invalid features: `"Invalid features format: expected tuple of (keypoints, descriptors)"`
- Insufficient matches: `"Insufficient feature matches: found {count} < 10 required"`
- Homography failure: `"Homography estimation failed: singular matrix or degenerate configuration"`

### Project Structure Notes

**File Location**: `src/movement_detector.py`

**Directory Structure**:
```
cam-shift-detector/
├── src/
│   ├── static_region_manager.py  ← Story 1.1 (Done)
│   ├── feature_extractor.py      ← Story 1.2 (Done)
│   └── movement_detector.py      ← This story
├── tests/
│   ├── test_static_region_manager.py
│   ├── test_feature_extractor.py
│   └── test_movement_detector.py ← Unit tests for this story
└── config.json
```

**Dependencies**:
- `opencv-python` - for BFMatcher, homography estimation (cv2.BFMatcher, cv2.findHomography)
- `numpy` - for array operations, coordinate extraction
- Story 1.1 (`StaticRegionManager`) - provides masks (integration)
- Story 1.2 (`FeatureExtractor`) - provides features (direct dependency)

**Config Integration**:
```json
{
  "roi": {
    "x": 100,
    "y": 50,
    "width": 400,
    "height": 300
  },
  "threshold_pixels": 2.0,  ← Used by MovementDetector
  "history_buffer_size": 100,
  "min_features_required": 50
}
```

### Testing Standards

**Test Framework**: pytest

**Coverage Target**: >80% for this module

**Test Categories**:
1. **Happy path**: Valid features, successful matching, displacement above/below threshold
2. **Boundary conditions**: Exactly 10 matches, exactly 2.0px displacement, confidence edge cases
3. **Error cases**: Insufficient matches, invalid feature formats, homography failure
4. **Edge cases**: Zero displacement, very high displacement, various confidence scores

**Example Test**:
```python
def test_detect_movement_above_threshold():
    """Test detection with displacement above threshold"""
    detector = MovementDetector(threshold_pixels=2.0)

    # Create baseline features
    baseline_kp = [cv2.KeyPoint(x=100, y=100, size=10) for _ in range(50)]
    baseline_desc = np.random.randint(0, 255, (50, 32), dtype=np.uint8)

    # Create shifted features (3 pixel shift)
    current_kp = [cv2.KeyPoint(x=103, y=100, size=10) for _ in range(50)]
    current_desc = baseline_desc.copy()  # Same descriptors for matching

    moved, displacement, confidence = detector.detect_movement(
        (baseline_kp, baseline_desc),
        (current_kp, current_desc)
    )

    assert moved == True
    assert displacement >= 2.0
    assert 0.0 <= confidence <= 1.0

def test_detect_movement_insufficient_matches():
    """Test handling of insufficient feature matches"""
    detector = MovementDetector(threshold_pixels=2.0)

    # Create features with very different descriptors (no matches)
    baseline_kp = [cv2.KeyPoint(x=100, y=100, size=10) for _ in range(20)]
    baseline_desc = np.zeros((20, 32), dtype=np.uint8)

    current_kp = [cv2.KeyPoint(x=100, y=100, size=10) for _ in range(20)]
    current_desc = np.ones((20, 32), dtype=np.uint8) * 255

    with pytest.raises(ValueError, match="Insufficient feature matches"):
        detector.detect_movement(
            (baseline_kp, baseline_desc),
            (current_kp, current_desc)
        )
```

### References

- [Source: tech-spec-epic-MVP-001.md#Services and Modules] - Module table row for MovementDetector
- [Source: tech-spec-epic-MVP-001.md#Data Models and Contracts] - Confidence score calculation
- [Source: tech-spec-epic-MVP-001.md#Workflows and Sequencing] - Runtime detection workflow
- [Source: tech-spec-epic-MVP-001.md#Acceptance Criteria] - AC-001: Detection accuracy requirements
- [Source: story-1.1.md] - StaticRegionManager integration (mask generation)
- [Source: story-1.2.md] - FeatureExtractor integration (feature input format)

## Dev Agent Record

### Context Reference

- `docs/stories/story-context-1.3.xml` - Comprehensive implementation context generated 2025-10-21

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

Implementation executed Tasks 1-5 sequentially:
1. Created MovementDetector class with BFMatcher (NORM_HAMMING, crossCheck=True)
2. Implemented homography estimation with translation extraction (H[0,2], H[1,2])
3. Added threshold validation and confidence calculation (inlier ratio)
4. Comprehensive input validation in _validate_features() method
5. Created 29 unit tests covering all 6 ACs with 98% coverage

Test failures resolved: Fixed degenerate configuration issue by distributing keypoints in grid pattern instead of single point.

### Completion Notes List

**2025-10-21**: Implemented MovementDetector with complete feature matching, homography estimation, and movement detection. All acceptance criteria satisfied:
- AC-1.3.1 ✓: BFMatcher with NORM_HAMMING for ORB descriptors
- AC-1.3.2 ✓: Homography estimation with insufficient match handling
- AC-1.3.3 ✓: Displacement calculation from translation vector, rounded to 2 decimals
- AC-1.3.4 ✓: Threshold validation (default 2.0px)
- AC-1.3.5 ✓: Confidence score [0.0, 1.0] from inlier ratio
- AC-1.3.6 ✓: Comprehensive error handling for invalid inputs

Test Results: 82/82 tests passing (29 new tests for MovementDetector + 23 existing tests for FeatureExtractor + 30 for StaticRegionManager). Coverage: 98% for MovementDetector, 99% overall.

**Architectural Limitation Identified (2025-10-21)**: Current implementation only measures translation displacement (tx, ty) from homography matrix. Does not detect rotation, scale, or shear components. A camera could rotate significantly while showing <2.0px translation, resulting in false "no movement" detection. **Accepted as MVP constraint** - full homography decomposition with rotation/scale thresholds planned for Story 1.5 enhancement.

### File List

- `src/movement_detector.py` (new, 163 lines) - MovementDetector class implementation
- `tests/test_movement_detector.py` (new, 425 lines) - Comprehensive test suite with 29 tests
