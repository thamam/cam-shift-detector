# Story 1.2: Feature Extractor

Status: Done

## Story

As a **camera movement detection system**,
I want to **extract ORB features from camera images using binary masks and store baseline features**,
so that **movement detection can compare current frames to a reference baseline and identify camera displacement**.

## Acceptance Criteria

1. **AC-1.2.1: ORB Feature Extraction** - FeatureExtractor extracts ORB keypoints and descriptors from full images using binary masks (mask parameter) where 255=detect features, 0=ignore

2. **AC-1.2.2: Baseline Storage** - FeatureExtractor stores baseline features (keypoints + descriptors) in memory for comparison; provides `set_baseline(image, mask)` and `get_baseline()` methods

3. **AC-1.2.3: Feature Count Validation** - Validates that ≥50 features are detected during baseline capture; raises `ValueError` if insufficient features (prevents unreliable detection)

4. **AC-1.2.4: Current Features Extraction** - Provides `extract_features(image, mask)` method that returns (keypoints, descriptors) tuple for current frame analysis

5. **AC-1.2.5: Error Handling** - Raises appropriate exceptions for:
   - Invalid image format (not NumPy array, wrong shape/dtype)
   - Invalid mask format (not 2D uint8 array matching image dimensions)
   - Baseline not set when calling `get_baseline()`
   - Insufficient features (<50) during baseline capture

## Tasks / Subtasks

- [x] **Task 1: Create FeatureExtractor class** (AC: #1.2.1, #1.2.4)
  - [x] 1.1: Define `__init__(self, min_features_required=50)` - initialize ORB detector
  - [x] 1.2: Initialize OpenCV ORB with default parameters (or from config)
  - [x] 1.3: Implement `extract_features(image, mask)` - extract ORB keypoints and descriptors
  - [x] 1.4: Validate image format (H×W×3, uint8, NumPy array)
  - [x] 1.5: Validate mask format (H×W, uint8, NumPy array, matches image dimensions)

- [x] **Task 2: Implement baseline management** (AC: #1.2.2, #1.2.3)
  - [x] 2.1: Implement `set_baseline(image, mask)` - capture and store baseline features
  - [x] 2.2: Validate feature count ≥ min_features_required (default 50)
  - [x] 2.3: Raise `ValueError` if insufficient features detected
  - [x] 2.4: Implement `get_baseline()` - return stored baseline features
  - [x] 2.5: Raise `RuntimeError` if baseline not set when `get_baseline()` called
  - [x] 2.6: Store baseline as tuple: (keypoints, descriptors)

- [x] **Task 3: Implement error handling** (AC: #1.2.5)
  - [x] 3.1: Validate image is NumPy array with shape (H, W, 3) and dtype uint8
  - [x] 3.2: Validate mask is NumPy array with shape (H, W) and dtype uint8
  - [x] 3.3: Validate mask dimensions match image dimensions (height, width)
  - [x] 3.4: Raise descriptive `ValueError` for format violations

- [x] **Task 4: Unit tests** (AC: All)
  - [x] 4.1: Test successful feature extraction with valid image and mask
  - [x] 4.2: Test baseline capture with sufficient features (≥50)
  - [x] 4.3: Test baseline capture failure with insufficient features (<50)
  - [x] 4.4: Test get_baseline() returns stored baseline correctly
  - [x] 4.5: Test get_baseline() raises error when baseline not set
  - [x] 4.6: Test error cases (invalid image format, invalid mask format, dimension mismatch)
  - [x] 4.7: Test edge cases (minimal features, maximum features, various mask patterns)

## Dev Notes

### Architecture & Design Patterns

**Module Responsibility** (Tech-Spec Section: Services and Modules):
- Extract ORB features from full images using binary masks
- Store baseline features (keypoints + descriptors) for comparison
- Validate minimum feature count (≥50) to ensure reliable detection
- Provide clean interface for baseline management and current feature extraction
- **No movement detection** - pure feature extraction utility

**Integration with StaticRegionManager** (Story 1.1):
```python
# Usage pattern from Story 1.1 integration
from src.static_region_manager import StaticRegionManager
from src.feature_extractor import FeatureExtractor

# Initialize
region_manager = StaticRegionManager('config.json')
feature_extractor = FeatureExtractor(min_features_required=50)

# Get mask from Story 1.1
mask = region_manager.get_static_mask(image.shape[:2])

# Extract features with mask (Story 1.2)
keypoints, descriptors = feature_extractor.extract_features(image, mask)

# Set baseline for movement detection
feature_extractor.set_baseline(baseline_image, mask)
```

**Data Structures** (Tech-Spec Section: Data Models):
```python
# ORB Keypoints (cv2.KeyPoint objects)
keypoints: List[cv2.KeyPoint]

# ORB Descriptors (NumPy array, shape: (n_features, 32), dtype: uint8)
descriptors: np.ndarray

# Baseline Features Tuple
baseline_features: Tuple[List[cv2.KeyPoint], np.ndarray]
```

**OpenCV ORB Configuration**:
- Default ORB parameters (can be tuned later if needed)
- `cv2.ORB_create()` with default settings
- `orb.detectAndCompute(image, mask=mask)` for masked feature detection
- Returns: (keypoints, descriptors)

### Implementation Guidance

**Class Structure**:
```python
class FeatureExtractor:
    def __init__(self, min_features_required: int = 50):
        """Initialize ORB feature detector"""
        # Create ORB detector with default parameters
        # Store min_features_required threshold
        # Initialize baseline_features to None

    def extract_features(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract ORB features from image using mask"""
        # Validate image format (H×W×3, uint8)
        # Validate mask format (H×W, uint8)
        # Validate mask dimensions match image
        # Call orb.detectAndCompute(image, mask=mask)
        # Return (keypoints, descriptors)

    def set_baseline(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Capture and store baseline features"""
        # Extract features using extract_features()
        # Validate feature count ≥ min_features_required
        # If insufficient, raise ValueError with count
        # Store as self.baseline_features = (keypoints, descriptors)

    def get_baseline(self) -> Tuple[List, np.ndarray]:
        """Get stored baseline features"""
        # Check if baseline is set
        # If not set, raise RuntimeError
        # Return self.baseline_features
```

**Input Validation**:
- Image must be NumPy array with shape (H, W, 3) and dtype uint8
- Mask must be NumPy array with shape (H, W) and dtype uint8
- Mask height must equal image height
- Mask width must equal image width
- Feature count must be ≥ min_features_required (default 50)

**Error Messages**:
- Invalid image: `"Invalid image format: expected numpy.ndarray (H×W×3, uint8), got ..."`
- Invalid mask: `"Invalid mask format: expected numpy.ndarray (H×W, uint8), got ..."`
- Dimension mismatch: `"Mask dimensions (H, W) must match image dimensions (H, W)"`
- Insufficient features: `"Insufficient features detected: {count} < {min_required}. Try different ROI or larger region."`
- Baseline not set: `"Baseline features not set. Call set_baseline() first."`

### Project Structure Notes

**File Location**: `src/feature_extractor.py`

**Directory Structure**:
```
cam-shift-detector/
├── src/
│   ├── static_region_manager.py  ← Story 1.1 (Done)
│   └── feature_extractor.py      ← This story
├── tests/
│   ├── test_static_region_manager.py  ← Story 1.1 tests
│   └── test_feature_extractor.py      ← Unit tests for this story
└── config.json  ← Includes min_features_required parameter
```

**Dependencies**:
- `opencv-python` - for ORB feature detection (`cv2.ORB_create()`, `detectAndCompute()`)
- `numpy` - for array operations
- Story 1.1 (`StaticRegionManager`) - provides binary masks for feature extraction

**Config Integration**:
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
  "min_features_required": 50  ← Used by FeatureExtractor
}
```

### Testing Standards

**Test Framework**: pytest

**Coverage Target**: >80% for this module

**Test Categories**:
1. **Happy path**: Valid image + mask, successful feature extraction, baseline capture with ≥50 features
2. **Boundary conditions**: Exactly 50 features, mask with small regions, various image sizes
3. **Error cases**: Invalid image/mask formats, dimension mismatches, insufficient features
4. **Edge cases**: Baseline not set, multiple baseline captures (overwrite), empty mask (no features)

**Example Test**:
```python
def test_extract_features_success():
    """Test successful feature extraction with valid inputs"""
    extractor = FeatureExtractor(min_features_required=50)

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:400, 100:500] = 255  # Define static region

    keypoints, descriptors = extractor.extract_features(image, mask)

    assert isinstance(keypoints, list)
    assert isinstance(descriptors, np.ndarray)
    assert descriptors.dtype == np.uint8
    # Note: Feature count varies by image content

def test_set_baseline_insufficient_features():
    """Test baseline capture fails with insufficient features"""
    extractor = FeatureExtractor(min_features_required=50)

    # Create image with very few features
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Uniform gray
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:150, 100:150] = 255  # Small region

    with pytest.raises(ValueError, match="Insufficient features"):
        extractor.set_baseline(image, mask)
```

### References

- [Source: tech-spec-epic-MVP-001.md#Services and Modules] - Module table row for FeatureExtractor
- [Source: tech-spec-epic-MVP-001.md#Data Models and Contracts] - Feature data structures
- [Source: tech-spec-epic-MVP-001.md#Workflows and Sequencing] - Baseline capture workflow
- [Source: tech-spec-epic-MVP-001.md#Acceptance Criteria] - AC-009: Baseline capture validation
- [Source: story-1.1.md] - StaticRegionManager integration (mask generation)

## Dev Agent Record

### Context Reference

- `docs/stories/story-context-1.2.xml` - Comprehensive implementation context generated 2025-10-21

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

### Completion Notes

**Completed:** 2025-10-21
**Definition of Done:** All acceptance criteria met, code reviewed, tests passing, deployed

### Completion Notes List

**2025-10-21**: Implemented FeatureExtractor with comprehensive test coverage.
- Created black-box module with clean API: `__init__`, `extract_features`, `set_baseline`, `get_baseline`
- Implemented ORB feature extraction with binary mask support (OpenCV ORB.detectAndCompute with mask parameter)
- Added strict validation: image format (H×W×3, uint8), mask format (H×W, uint8), dimension matching
- Implemented baseline management with minimum feature count validation (≥50 features required)
- Comprehensive error handling with descriptive error messages for all invalid inputs
- Created 23 unit tests with pytest covering all 5 acceptance criteria
- All tests passing (23/23), achieving 100% code coverage (40 statements, 0 missed)
- No regressions: Story 1.1 tests still passing (30/30)
- Integration pattern validated: mask-based architecture works seamlessly with StaticRegionManager

### File List

**Created:**
- `src/feature_extractor.py` - FeatureExtractor class implementation (168 lines)
- `tests/test_feature_extractor.py` - Comprehensive unit test suite (23 tests, 500+ lines)

**Dependencies:**
- opencv-python ≥4.8.0,<5.0.0 (already in requirements.txt)
- numpy ≥1.24.0,<2.0.0 (already in requirements.txt)
- pytest ≥7.0.0 (already in requirements.txt)
- pytest-cov ≥4.0.0 (already in requirements.txt)
