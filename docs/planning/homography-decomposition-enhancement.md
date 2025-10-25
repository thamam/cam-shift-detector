# Homography Decomposition Enhancement - Planning Notes

**Status**: Planning / Future Enhancement
**Created**: 2025-10-21
**Context**: Addresses architectural limitation identified in Story 1.3
**Target**: Post-MVP or Story 1.5 integration

---

## Current Limitation (MVP Constraint)

The current MovementDetector implementation (Story 1.3) only extracts **translation displacement** (tx, ty) from the homography matrix. It does NOT detect:

- **Rotation**: Camera pan/tilt changes
- **Scale**: Zoom in/out
- **Shear/Perspective**: Distortion from camera angle changes

**Impact**: A camera could rotate significantly (corrupting water quality measurements) while showing <2.0px translation, resulting in false "VALID" status despite actual movement.

**Current Threshold**: Single threshold (2.0 pixels) for translation only

---

## Proposed Enhancement: Full Homography Decomposition

### Overview

Decompose the 3x3 homography matrix **H** into its constituent transformation components:

```python
H = [h11  h12  tx]
    [h21  h22  ty]
    [h31  h32  1 ]
```

Extractable components:
1. **Translation**: (tx, ty) - Already implemented
2. **Rotation**: Angle θ from rotation component
3. **Scale**: Scale factors (sx, sy)
4. **Shear**: Shear coefficients

### Technical Approach

#### Option 1: Direct Decomposition (Recommended for MVP+)

```python
import cv2
import numpy as np

def decompose_homography(H: np.ndarray) -> Dict[str, float]:
    """
    Decompose homography into translation, rotation, and scale components.

    Args:
        H: 3x3 homography matrix from cv2.findHomography()

    Returns:
        {
            'translation_x': float,      # tx in pixels
            'translation_y': float,      # ty in pixels
            'rotation_deg': float,       # rotation angle in degrees
            'scale_x': float,           # horizontal scale factor
            'scale_y': float,           # vertical scale factor
            'translation_magnitude': float,  # sqrt(tx^2 + ty^2)
            'rotation_magnitude': float,     # abs(rotation_deg)
            'scale_change': float           # max(abs(sx-1), abs(sy-1))
        }
    """
    # Extract translation
    tx = H[0, 2]
    ty = H[1, 2]
    translation_magnitude = np.sqrt(tx**2 + ty**2)

    # Extract rotation and scale from upper-left 2x2 submatrix
    # [h11  h12]  =  [sx*cos(θ)  -sy*sin(θ)]
    # [h21  h22]     [sx*sin(θ)   sy*cos(θ)]

    # Calculate scale factors
    sx = np.sqrt(H[0, 0]**2 + H[1, 0]**2)
    sy = np.sqrt(H[0, 1]**2 + H[1, 1]**2)

    # Calculate rotation angle (use atan2 for proper quadrant)
    rotation_rad = np.arctan2(H[1, 0] / sx, H[0, 0] / sx)
    rotation_deg = np.degrees(rotation_rad)

    return {
        'translation_x': round(tx, 2),
        'translation_y': round(ty, 2),
        'rotation_deg': round(rotation_deg, 2),
        'scale_x': round(sx, 3),
        'scale_y': round(sy, 3),
        'translation_magnitude': round(translation_magnitude, 2),
        'rotation_magnitude': round(abs(rotation_deg), 2),
        'scale_change': round(max(abs(sx - 1.0), abs(sy - 1.0)), 3)
    }
```

#### Option 2: OpenCV cv2.decomposeHomographyMat (More Robust)

```python
def decompose_homography_opencv(H: np.ndarray, K: np.ndarray) -> Dict:
    """
    Use OpenCV's built-in decomposition (requires camera intrinsic matrix K).

    Args:
        H: 3x3 homography matrix
        K: 3x3 camera intrinsic matrix (if available)

    Returns:
        Multiple possible decomposition solutions with rotation, translation, normals
    """
    # cv2.decomposeHomographyMat returns multiple solutions
    retval, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    # Select most plausible solution (e.g., smallest rotation magnitude)
    # Return decomposed components
    pass
```

**Note**: Option 2 requires camera calibration matrix K, which may not be available for MVP. Option 1 is simpler and sufficient for detecting rotation/scale changes.

---

## Enhanced Detection Logic

### Multi-Threshold Approach

Instead of single threshold (2.0 pixels), use **separate thresholds** for each component:

```python
class EnhancedMovementDetector:
    def __init__(
        self,
        translation_threshold_px: float = 2.0,    # Translation threshold
        rotation_threshold_deg: float = 1.0,      # Rotation threshold (degrees)
        scale_threshold_pct: float = 0.05         # Scale change threshold (5%)
    ):
        self.translation_threshold = translation_threshold_px
        self.rotation_threshold = rotation_threshold_deg
        self.scale_threshold = scale_threshold_pct
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_movement(
        self,
        baseline_features: Tuple,
        current_features: Tuple
    ) -> Tuple[bool, Dict[str, float], float]:
        """
        Enhanced detection with rotation and scale awareness.

        Returns:
            (moved, components_dict, confidence)
            - moved: True if ANY component exceeds threshold
            - components_dict: {
                'translation': float,
                'rotation': float,
                'scale_change': float,
                'exceeded_thresholds': List[str]  # Which thresholds exceeded
              }
            - confidence: Inlier ratio [0.0, 1.0]
        """
        # Match features and estimate homography (same as MVP)
        baseline_keypoints, baseline_descriptors = baseline_features
        current_keypoints, current_descriptors = current_features
        matches = self.matcher.match(baseline_descriptors, current_descriptors)

        if len(matches) < 10:
            raise ValueError(f"Insufficient matches: {len(matches)} < 10")

        src_pts = np.float32([baseline_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([current_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, method=0)

        if H is None:
            raise RuntimeError("Homography estimation failed")

        # Decompose homography into components
        components = self.decompose_homography(H)

        # Calculate confidence (same as MVP)
        inliers = np.sum(mask)
        confidence = round(float(np.clip(inliers / len(matches), 0.0, 1.0)), 2)

        # Check each component against thresholds
        exceeded = []
        translation_exceeded = components['translation_magnitude'] >= self.translation_threshold
        rotation_exceeded = components['rotation_magnitude'] >= self.rotation_threshold
        scale_exceeded = components['scale_change'] >= self.scale_threshold

        if translation_exceeded:
            exceeded.append('translation')
        if rotation_exceeded:
            exceeded.append('rotation')
        if scale_exceeded:
            exceeded.append('scale')

        # Movement detected if ANY component exceeds threshold
        moved = bool(len(exceeded) > 0)

        result_components = {
            'translation': components['translation_magnitude'],
            'rotation': components['rotation_magnitude'],
            'scale_change': components['scale_change'],
            'exceeded_thresholds': exceeded
        }

        return (moved, result_components, confidence)

    def decompose_homography(self, H: np.ndarray) -> Dict[str, float]:
        """Decompose homography matrix (see Option 1 above)"""
        # Implementation from Option 1
        pass
```

---

## Integration with ResultManager

Update ResultManager (Story 1.4) to handle enhanced movement components:

### Enhanced Result Dictionary Schema

```python
{
  "status": "VALID" | "INVALID",
  "components": {
    "translation_displacement": float,     # Magnitude in pixels
    "rotation_degrees": float,             # Rotation angle
    "scale_change_percent": float,         # Max scale change
    "exceeded_thresholds": List[str]       # ["translation", "rotation", "scale"]
  },
  "confidence": float,                     # [0.0, 1.0]
  "frame_id": str,
  "timestamp": str                         # ISO 8601 UTC
}
```

### Backward Compatibility

To maintain backward compatibility with MVP API:

```python
# Option A: Dual API (recommended)
result = {
    "status": "VALID" | "INVALID",
    "translation_displacement": float,     # Keep flat for backward compatibility
    "confidence": float,
    "frame_id": str,
    "timestamp": str,

    # New enhanced fields (optional for backward compatibility)
    "components": {                        # Enhanced breakdown
        "translation": float,
        "rotation": float,
        "scale_change": float,
        "exceeded_thresholds": List[str]
    }
}

# Option B: Feature flag in config.json
{
  "detection_mode": "simple" | "enhanced",   # Toggle between MVP and enhanced
  "thresholds": {
    "translation_pixels": 2.0,
    "rotation_degrees": 1.0,                 # Only used if mode == "enhanced"
    "scale_percent": 0.05                    # Only used if mode == "enhanced"
  }
}
```

---

## Configuration Updates

Enhanced `config.json` schema:

```json
{
  "roi": {
    "x": 100,
    "y": 50,
    "width": 400,
    "height": 300
  },
  "detection_mode": "enhanced",              // "simple" | "enhanced"
  "thresholds": {
    "translation_pixels": 2.0,               // Translation threshold
    "rotation_degrees": 1.0,                 // Rotation threshold (enhanced mode only)
    "scale_percent": 0.05                    // Scale threshold (enhanced mode only)
  },
  "history_buffer_size": 100,
  "min_features_required": 50
}
```

---

## Testing Strategy

### Unit Tests for Decomposition

```python
def test_decompose_translation_only():
    """Test decomposition with pure translation (no rotation/scale)"""
    # H with tx=3.0, ty=0, no rotation, no scale
    H = np.array([
        [1.0, 0.0, 3.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    components = decompose_homography(H)

    assert components['translation_magnitude'] == 3.0
    assert abs(components['rotation_deg']) < 0.01
    assert abs(components['scale_x'] - 1.0) < 0.01
    assert abs(components['scale_y'] - 1.0) < 0.01

def test_decompose_rotation_only():
    """Test decomposition with pure rotation (no translation/scale)"""
    # H with 5° rotation, no translation, no scale
    theta = np.radians(5.0)
    H = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])

    components = decompose_homography(H)

    assert components['translation_magnitude'] < 0.01
    assert abs(components['rotation_deg'] - 5.0) < 0.1
    assert abs(components['scale_x'] - 1.0) < 0.01

def test_enhanced_detector_rotation_exceeds_threshold():
    """Test detection triggers on rotation threshold"""
    detector = EnhancedMovementDetector(
        translation_threshold_px=2.0,
        rotation_threshold_deg=1.0
    )

    # Create features with 2° rotation, 0.5px translation
    # (translation below threshold, rotation above threshold)
    baseline_features = create_test_features(rotation=0)
    current_features = create_test_features(rotation=2.0, translation=0.5)

    moved, components, confidence = detector.detect_movement(
        baseline_features, current_features
    )

    assert moved == True  # Should detect movement due to rotation
    assert 'rotation' in components['exceeded_thresholds']
    assert 'translation' not in components['exceeded_thresholds']
```

### Integration Tests

```python
def test_camera_movement_detector_with_rotation():
    """Test main API with rotation movement"""
    detector = CameraMovementDetector('config_enhanced.json')
    detector.set_baseline(baseline_image)

    # Simulate 3° camera rotation
    rotated_image = simulate_rotation(baseline_image, degrees=3.0)

    result = detector.process_frame(rotated_image, frame_id="rotation_test")

    assert result["status"] == "INVALID"
    assert result["components"]["rotation"] > 1.0  # Exceeds threshold
    assert 'rotation' in result["components"]["exceeded_thresholds"]
```

---

## Implementation Roadmap

### Phase 1: Minimal Viable Enhancement (Recommended First Step)

1. **Add `decompose_homography()` function** to MovementDetector (Story 1.3 update)
   - Use Option 1 (direct decomposition) for simplicity
   - Extract translation, rotation, scale from H matrix
   - Return dict with components

2. **Update `detect_movement()` to log components** (non-breaking change)
   - Calculate components internally
   - Log to console/debug output
   - Still return original (moved, displacement, confidence) tuple

3. **Add unit tests** for decomposition accuracy
   - Test pure translation, rotation, scale transforms
   - Verify decomposition math correctness

**Benefit**: No API breaking changes, establishes decomposition foundation, enables data collection

### Phase 2: Enhanced Detection Logic

1. **Update MovementDetector API** to return components
   - Change return type: `(moved, components_dict, confidence)`
   - Add multi-threshold checking
   - Update all unit tests

2. **Update ResultManager** (Story 1.4 update)
   - Add `components` field to result dictionary
   - Keep `translation_displacement` for backward compatibility
   - Update `create_result()` signature

3. **Update config.json schema**
   - Add `detection_mode` field
   - Add rotation/scale thresholds

### Phase 3: Full Integration

1. **Update CameraMovementDetector** (Story 1.5)
   - Pass enhanced components to ResultManager
   - Update API documentation

2. **Comprehensive testing**
   - Synthetic transforms (rotation, scale, combined)
   - Real DAF footage with known camera movements
   - Regression testing for MVP behavior

---

## Threshold Recommendations

Based on water quality measurement tolerance requirements:

| Component | Recommended Threshold | Rationale |
|-----------|----------------------|-----------|
| **Translation** | 2.0 pixels | Original MVP spec; neural network ROI misalignment |
| **Rotation** | 1.0 degrees | Small rotation significantly affects flow measurements |
| **Scale** | 5% (0.05) | Zoom changes alter perspective and ROI coverage |

**Tuning**: Thresholds should be validated during live testing (Stage 3 validation) and adjusted based on false positive/negative rates.

---

## Open Questions

1. **Camera Calibration**: Do we have access to camera intrinsic matrix K for cv2.decomposeHomographyMat()?
   - **If Yes**: Option 2 (OpenCV decomposition) more robust
   - **If No**: Option 1 (direct decomposition) sufficient

2. **Threshold Tuning**: Should thresholds be site-specific or global?
   - **Recommendation**: Start global, allow per-site overrides in config

3. **False Positive Rate**: Will enhanced detection reduce false positives from lighting changes?
   - **Hypothesis**: Yes - lighting changes may affect features but shouldn't create rotation/scale
   - **Validation**: Requires Stage 1-3 testing with enhanced detector

4. **Performance Impact**: Does homography decomposition add significant latency?
   - **Estimate**: <1ms overhead (matrix operations only)
   - **Verification**: Benchmark during implementation

---

## References

- **Story 1.3**: MovementDetector implementation (current MVP)
- **Story 1.4**: ResultManager result dictionary schema
- **Tech Spec**: `docs/tech-spec-epic-MVP-001.md` - NFR-003 (performance), AC-001 (detection accuracy)
- **OpenCV Docs**:
  - `cv2.decomposeHomographyMat()`: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
  - Homography decomposition theory: https://en.wikipedia.org/wiki/Homography_(computer_vision)

---

**Status**: Ready for implementation planning
**Next Steps**:
1. Validate approach with team (rotation/scale thresholds reasonable?)
2. Decide on Option 1 vs Option 2 (depends on camera calibration availability)
3. Schedule as Story 1.5 enhancement or separate post-MVP story
4. Begin Phase 1 (minimal viable enhancement) for data collection
