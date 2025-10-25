# Story 1.1: Static Region Manager

Status: Done

## Story

As a **camera movement detection system**,
I want to **load ROI coordinates from configuration and generate binary masks for the static region**,
so that **feature extraction operates only on stable areas (tank walls, pipes) and avoids dynamic elements (water, bubbles)**.

## Acceptance Criteria

1. **AC-1.1.1: Config Loading** - StaticRegionManager loads ROI coordinates from `config.json` with schema validation (x, y, width, height must be positive integers)

2. **AC-1.1.2: Binary Mask Generation** - Given image dimensions (height, width) and ROI coordinates, returns binary mask (H×W, uint8) where 255=static region, 0=dynamic region

3. **AC-1.1.3: Boundary Validation** - Validates that ROI coordinates are within image bounds; raises `ValueError` if ROI exceeds image dimensions

4. **AC-1.1.4: Error Handling** - Raises appropriate exceptions for:
   - Missing config file (`FileNotFoundError`)
   - Invalid JSON schema (`ValueError` with descriptive message)
   - Invalid image_shape parameter (not tuple, wrong length, non-integers, non-positive dimensions)

## Tasks / Subtasks

- [x] **Task 1: Create StaticRegionManager class** (AC: #1.1.1, #1.1.2)
  - [x] 1.1: Define `__init__(self, config_path)` - load and validate config
  - [x] 1.2: Implement config schema validation (x, y, width, height)
  - [x] 1.3: Define `get_static_mask(self, image_shape)` - generate binary mask from ROI
  - [x] 1.4: Store ROI as instance variable for efficient access
  - [x] 1.5: Implement `_generate_rectangular_mask()` private helper for mask generation

- [x] **Task 2: Implement boundary validation** (AC: #1.1.3)
  - [x] 2.1: Add validation in `get_static_mask()` to check ROI vs image bounds
  - [x] 2.2: Raise `ValueError` with clear message if out of bounds

- [x] **Task 3: Implement error handling** (AC: #1.1.4)
  - [x] 3.1: Handle missing config file with `FileNotFoundError`
  - [x] 3.2: Validate JSON structure and raise `ValueError` for schema violations
  - [x] 3.3: Validate image_shape parameter (tuple, 2 elements, integers, positive values)

- [x] **Task 4: Unit tests** (AC: All)
  - [x] 4.1: Test successful config loading with valid `config.json`
  - [x] 4.2: Test successful mask generation with various image sizes
  - [x] 4.3: Test boundary validation (ROI within bounds, ROI out of bounds)
  - [x] 4.4: Test error cases (missing file, invalid JSON, invalid image_shape parameter)
  - [x] 4.5: Test edge cases (ROI at image boundaries, 1×1 ROI, full image ROI)
  - [x] 4.6: Test mask values (binary 0/255, correct ROI region marked as static)

## Dev Notes

### Architecture & Design Patterns

**Module Responsibility** (Tech-Spec Section: Services and Modules):
- Load ROI coordinates from `config.json`
- Generate binary masks for static region (255=static, 0=dynamic)
- Validate ROI bounds against image dimensions
- **No feature extraction** - pure mask generation utility
- Supports integration with OpenCV masked feature detection (e.g., ORB.detectAndCompute with mask parameter)

**Configuration Contract** (Tech-Spec Section: Data Models):
```json
{
  "roi": {
    "x": 100,       // Top-left X coordinate
    "y": 50,        // Top-left Y coordinate
    "width": 400,   // ROI width in pixels
    "height": 300   // ROI height in pixels
  }
}
```

**Data Flow** (Tech-Spec Section: System Architecture):
```
Image dimensions (H, W) → StaticRegionManager.get_static_mask((H, W))
                        → Binary mask (H×W, uint8: 255=static, 0=dynamic)
                        → Used with ORB.detectAndCompute(image, mask=mask)
```

### Implementation Guidance

**Class Structure**:
```python
class StaticRegionManager:
    def __init__(self, config_path: str):
        """Load and validate ROI config from JSON file"""
        # Load config.json
        # Extract ROI dict
        # Validate x, y, width, height are positive

    def get_static_mask(self, image_shape: tuple) -> np.ndarray:
        """Generate binary mask for static region"""
        # Validate image_shape (tuple of 2 positive integers)
        # Validate ROI within image bounds
        # Generate binary mask: 255 inside ROI, 0 outside
        # Return mask (H×W, uint8)

    def _generate_rectangular_mask(self, image_shape: tuple) -> np.ndarray:
        """Private helper to generate rectangular mask from ROI coordinates"""
        # Create zero-filled mask (all dynamic)
        # Set ROI region to 255 (static)
        # Return mask
```

**Input Validation**:
- Config must have `roi` key with nested `x`, `y`, `width`, `height`
- ROI values: x,y must be non-negative integers (≥0); width,height must be positive integers (>0)
- image_shape must be tuple of exactly 2 positive integers (height, width)

**Error Messages**:
- Missing file: `"Config file not found: {config_path}"`
- Invalid schema: `"Invalid config schema: missing required field '{field}'"`
- Out of bounds: `"ROI exceeds image bounds: ROI({x}, {y}, {w}, {h}) vs Image({H}, {W})"`

### Project Structure Notes

**File Location**: `src/static_region_manager.py`

**Directory Structure**:
```
cam-shift-detector/
├── src/
│   └── static_region_manager.py  ← This story
├── tests/
│   └── test_static_region_manager.py  ← Unit tests
└── config.json  ← Sample configuration
```

**Dependencies**:
- `numpy` - for array operations
- `json` - for config loading (stdlib)
- `pathlib` or `os.path` - for file operations (stdlib)

### Testing Standards

**Test Framework**: pytest

**Coverage Target**: >90% for this module

**Test Categories**:
1. **Happy path**: Valid config, valid images, successful cropping
2. **Boundary conditions**: ROI at edges, minimal/maximal sizes
3. **Error cases**: Missing file, invalid JSON, invalid image, out-of-bounds ROI
4. **Edge cases**: 1×1 ROI, full-image ROI, non-square ROIs

**Example Test**:
```python
def test_get_static_mask_success():
    """Test successful mask generation with valid ROI"""
    config = create_temp_config(roi={"x": 10, "y": 20, "width": 100, "height": 80})
    manager = StaticRegionManager(config)

    mask = manager.get_static_mask((480, 640))

    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    assert np.all(mask[20:100, 10:110] == 255)  # ROI region is static
    assert np.all(mask[0:20, :] == 0)  # Outside ROI is dynamic
```

### References

- [Source: tech-spec-epic-MVP-001.md#Services and Modules] - Module table row for StaticRegionManager
- [Source: tech-spec-epic-MVP-001.md#Data Models and Contracts] - Configuration schema
- [Source: tech-spec-epic-MVP-001.md#System Architecture Alignment] - Component dependencies
- [Source: MVP_Camera_Movement_Detection_SIMPLIFIED.md#4.1 Static Region Manager] - Implementation details

## Dev Agent Record

### Context Reference

- `docs/stories/story-context-1.1.xml` - Comprehensive implementation context generated 2025-10-18

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

### Completion Notes List

**2025-10-18**: Implemented StaticRegionManager with comprehensive error handling and validation (initial crop-based implementation).
- Created black-box module with clean API: `__init__` and `crop_to_static_region`
- Implemented strict schema validation: x,y non-negative (≥0), width,height positive (>0)
- Added robust boundary checking to prevent out-of-bounds ROI access
- Created 30 unit tests with pytest covering all 4 acceptance criteria
- All tests pass (30/30), achieving comprehensive coverage
- Fixed initial validation bug: x=0,y=0 are valid (top-left corner) - changed from "positive" to "non-negative"

**2025-10-20**: Refactored to mask-based architecture (major architectural improvement).
- **Rationale**: User identified that crop-based approach would create architectural limitations:
  - Downstream modules (FeatureExtractor, MovementDetector) would work with cropped images
  - Switching to masks later would require refactoring all 4 modules (Stories 1.2-1.5)
  - OpenCV ORB natively supports masks via `detectAndCompute(image, mask=mask)`
- **Changes**:
  - Replaced `crop_to_static_region(image: np.ndarray) -> np.ndarray` with `get_static_mask(image_shape: tuple) -> np.ndarray`
  - Added `_generate_rectangular_mask()` private helper method
  - Changed interface from cropping to mask generation (255=static, 0=dynamic)
  - Mask-based approach is simpler: no coordinate transformations, works in absolute coordinates
- **Updated tests** (all 30 tests still passing):
  - Config loading (12 tests): unchanged
  - Binary mask generation (5 tests): updated from cropping to mask generation
  - Boundary validation (5 tests): updated to test mask bounds
  - Error handling (5 tests): updated to test image_shape parameter validation
  - Edge cases (3 tests): updated to test mask edge cases
- **Updated documentation**:
  - Updated tech-spec (3 references to cropping → masking)
  - Updated story-1.1.md (acceptance criteria, tasks, examples)
- **Impact**: Provides correct architectural foundation for Stories 1.2-1.5 without refactoring
- **Cost**: +1 day now, saves 5 days later (avoids refactoring 4 downstream modules)

**2025-10-20**: Story approved and marked Done.
- **Definition of Done**: All acceptance criteria met, code reviewed, tests passing (30/30), architecture validated
- **Final deliverables**: Mask-based StaticRegionManager with comprehensive test coverage
- **Architectural decision**: Mask-first approach confirmed as correct foundation for downstream modules

### File List

**Created:**
- `src/static_region_manager.py` - Main StaticRegionManager class implementation (145 lines)
- `src/__init__.py` - Python package marker
- `tests/test_static_region_manager.py` - Comprehensive unit test suite (30 tests, 500+ lines)
- `tests/__init__.py` - Python package marker
- `requirements.txt` - Project dependencies (numpy, opencv-python, pytest)
- `config.json` - Sample configuration file with ROI definition
