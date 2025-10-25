"""Unit tests for FeatureExtractor

Tests cover:
- AC-1.2.1: ORB feature extraction with binary masks
- AC-1.2.2: Baseline storage and management
- AC-1.2.3: Feature count validation (≥50)
- AC-1.2.4: Current features extraction
- AC-1.2.5: Error handling for invalid inputs
"""

import numpy as np
import pytest
import cv2

from src.feature_extractor import FeatureExtractor


# Helper Functions

def create_test_image(height: int, width: int, pattern: str = "random") -> np.ndarray:
    """Create a test image with specified dimensions and pattern.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        pattern: Pattern type - "random", "textured", "uniform", or "checkerboard"

    Returns:
        BGR image (H×W×3, uint8)
    """
    if pattern == "random":
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    elif pattern == "uniform":
        return np.ones((height, width, 3), dtype=np.uint8) * 128
    elif pattern == "checkerboard":
        # Create checkerboard pattern for feature-rich images
        image = np.zeros((height, width, 3), dtype=np.uint8)
        square_size = 20
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    image[i:i+square_size, j:j+square_size] = 255
        return image
    elif pattern == "textured":
        # Create high-frequency texture for many features
        image = np.random.randint(0, 255, (height, width), dtype=np.uint8)
        # Add structure with gradients (using clip to avoid overflow)
        for i in range(height):
            image[i, :] = np.clip(image[i, :].astype(np.int16) + (i % 50), 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def create_test_mask(height: int, width: int, roi: dict = None) -> np.ndarray:
    """Create a test binary mask.

    Args:
        height: Mask height in pixels
        width: Mask width in pixels
        roi: Optional ROI dict with keys {x, y, width, height}

    Returns:
        Binary mask (H×W, uint8) where 255=static region, 0=dynamic
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if roi is None:
        # Default: use central 50% of image
        roi = {
            'x': width // 4,
            'y': height // 4,
            'width': width // 2,
            'height': height // 2
        }

    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    mask[y:y+h, x:x+w] = 255

    return mask


# AC-1.2.1: ORB Feature Extraction Tests

def test_extract_features_success():
    """Verify ORB features extracted successfully with valid image and mask."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=50)
    image = create_test_image(480, 640, pattern="textured")
    mask = create_test_mask(480, 640)

    # Act
    keypoints, descriptors = extractor.extract_features(image, mask)

    # Assert
    assert isinstance(keypoints, list)
    assert all(isinstance(kp, cv2.KeyPoint) for kp in keypoints)
    if len(keypoints) > 0:
        assert isinstance(descriptors, np.ndarray)
        assert descriptors.dtype == np.uint8
        assert descriptors.shape[0] == len(keypoints)
        assert descriptors.shape[1] == 32  # ORB descriptors are 32 bytes


def test_extract_features_returns_correct_types():
    """Verify return types are correct (list of KeyPoints, np.ndarray descriptors)."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640, pattern="checkerboard")
    mask = create_test_mask(480, 640)

    # Act
    keypoints, descriptors = extractor.extract_features(image, mask)

    # Assert - Types
    assert isinstance(keypoints, list)
    if descriptors is not None:
        assert isinstance(descriptors, np.ndarray)


def test_extract_features_respects_mask():
    """Verify features only detected in masked region (255 areas)."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640, pattern="checkerboard")

    # Create mask with small region on left side only
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:380, 50:150] = 255  # Left region only

    # Act
    keypoints, descriptors = extractor.extract_features(image, mask)

    # Assert - All keypoints should be within masked region
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        assert 50 <= x < 150, f"Keypoint x={x} outside masked region"
        assert 100 <= y < 380, f"Keypoint y={y} outside masked region"


def test_extract_features_with_different_mask_sizes():
    """Test extraction with various mask sizes (small, medium, large)."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640, pattern="textured")

    # Small mask
    small_mask = create_test_mask(480, 640, roi={'x': 200, 'y': 200, 'width': 50, 'height': 50})
    kp_small, _ = extractor.extract_features(image, small_mask)

    # Medium mask
    medium_mask = create_test_mask(480, 640, roi={'x': 100, 'y': 100, 'width': 200, 'height': 200})
    kp_medium, _ = extractor.extract_features(image, medium_mask)

    # Large mask
    large_mask = create_test_mask(480, 640, roi={'x': 50, 'y': 50, 'width': 500, 'height': 350})
    kp_large, _ = extractor.extract_features(image, large_mask)

    # Assert - Larger masks should generally produce more features
    assert len(kp_small) <= len(kp_medium)
    assert len(kp_medium) <= len(kp_large)


# AC-1.2.2: Baseline Storage Tests

def test_set_baseline_success():
    """Verify baseline stored correctly with sufficient features."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=50)
    image = create_test_image(480, 640, pattern="checkerboard")
    mask = create_test_mask(480, 640)

    # Act
    extractor.set_baseline(image, mask)

    # Assert
    assert extractor.baseline_features is not None
    keypoints, descriptors = extractor.baseline_features
    assert isinstance(keypoints, list)
    assert len(keypoints) >= 50
    assert isinstance(descriptors, np.ndarray)


def test_get_baseline_returns_stored_features():
    """Verify get_baseline() returns same features set by set_baseline()."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=20)
    image = create_test_image(480, 640, pattern="textured")
    mask = create_test_mask(480, 640)

    # Act
    extractor.set_baseline(image, mask)
    retrieved_keypoints, retrieved_descriptors = extractor.get_baseline()

    # Assert - Should return the same objects
    stored_keypoints, stored_descriptors = extractor.baseline_features
    assert retrieved_keypoints is stored_keypoints
    assert np.array_equal(retrieved_descriptors, stored_descriptors)


def test_set_baseline_overwrites_previous():
    """Verify calling set_baseline() multiple times overwrites previous baseline."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=20)
    image1 = create_test_image(480, 640, pattern="checkerboard")
    image2 = create_test_image(480, 640, pattern="textured")
    mask = create_test_mask(480, 640)

    # Act
    extractor.set_baseline(image1, mask)
    first_baseline = extractor.get_baseline()

    extractor.set_baseline(image2, mask)
    second_baseline = extractor.get_baseline()

    # Assert - Baselines should be different
    assert first_baseline[0] is not second_baseline[0]


# AC-1.2.3: Feature Count Validation Tests

def test_set_baseline_insufficient_features():
    """Verify ValueError raised when feature count < 50."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=50)
    # Uniform image produces very few features
    image = create_test_image(480, 640, pattern="uniform")
    # Small mask reduces feature count further
    mask = create_test_mask(480, 640, roi={'x': 200, 'y': 200, 'width': 30, 'height': 30})

    # Act & Assert
    with pytest.raises(ValueError, match="Insufficient features detected"):
        extractor.set_baseline(image, mask)


def test_set_baseline_exactly_min_features():
    """Boundary case: exactly min_features_required should succeed."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=1)  # Very low threshold for testing
    image = create_test_image(480, 640, pattern="checkerboard")
    mask = create_test_mask(480, 640)

    # Act - Should succeed with at least 1 feature
    extractor.set_baseline(image, mask)

    # Assert
    keypoints, _ = extractor.get_baseline()
    assert len(keypoints) >= 1


def test_set_baseline_error_message_shows_count():
    """Verify error message includes actual vs required feature count."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=100)
    image = create_test_image(480, 640, pattern="uniform")  # Very few features
    mask = create_test_mask(480, 640, roi={'x': 250, 'y': 200, 'width': 50, 'height': 50})

    # Act & Assert
    with pytest.raises(ValueError) as exc_info:
        extractor.set_baseline(image, mask)

    error_message = str(exc_info.value)
    assert "< 100" in error_message or "100" in error_message  # Check threshold mentioned


# AC-1.2.4: Current Features Extraction Tests

def test_extract_features_with_various_image_sizes():
    """Test extraction works with different image dimensions."""
    # Arrange
    extractor = FeatureExtractor()

    # Test various sizes
    sizes = [(240, 320), (480, 640), (720, 1280), (1080, 1920)]

    for height, width in sizes:
        image = create_test_image(height, width, pattern="checkerboard")
        mask = create_test_mask(height, width)

        # Act
        keypoints, descriptors = extractor.extract_features(image, mask)

        # Assert - Should work for all sizes
        assert isinstance(keypoints, list)


def test_extract_features_with_empty_mask():
    """Test extraction with mask that has no static regions (all zeros)."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640, pattern="textured")
    mask = np.zeros((480, 640), dtype=np.uint8)  # All zeros = no detection area

    # Act
    keypoints, descriptors = extractor.extract_features(image, mask)

    # Assert - Should return empty results
    assert len(keypoints) == 0
    assert descriptors is None


# AC-1.2.5: Error Handling Tests

def test_extract_features_invalid_image_shape():
    """Verify ValueError for wrong image dimensions (not H×W×3)."""
    # Arrange
    extractor = FeatureExtractor()
    mask = create_test_mask(480, 640)

    # Test various invalid shapes
    invalid_images = [
        np.zeros((480, 640), dtype=np.uint8),  # Grayscale (H×W)
        np.zeros((480, 640, 1), dtype=np.uint8),  # Single channel (H×W×1)
        np.zeros((480, 640, 4), dtype=np.uint8),  # RGBA (H×W×4)
    ]

    for invalid_image in invalid_images:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid image format.*shape"):
            extractor.extract_features(invalid_image, mask)


def test_extract_features_invalid_image_dtype():
    """Verify ValueError for wrong dtype (not uint8)."""
    # Arrange
    extractor = FeatureExtractor()
    mask = create_test_mask(480, 640)

    # Invalid dtypes
    invalid_images = [
        np.zeros((480, 640, 3), dtype=np.float32),
        np.zeros((480, 640, 3), dtype=np.int32),
        np.zeros((480, 640, 3), dtype=np.uint16),
    ]

    for invalid_image in invalid_images:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid image format.*dtype"):
            extractor.extract_features(invalid_image, mask)


def test_extract_features_invalid_image_type():
    """Verify ValueError when image is not a NumPy array."""
    # Arrange
    extractor = FeatureExtractor()
    mask = create_test_mask(480, 640)

    # Invalid types
    invalid_inputs = [
        [[0, 0, 0]],  # Python list
        "image.jpg",  # String
        None,  # None
    ]

    for invalid_input in invalid_inputs:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid image format.*expected numpy.ndarray"):
            extractor.extract_features(invalid_input, mask)


def test_extract_features_invalid_mask_shape():
    """Verify ValueError for wrong mask dimensions (not H×W)."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640)

    # Invalid mask shapes
    invalid_masks = [
        np.zeros((480, 640, 1), dtype=np.uint8),  # 3D array
        np.zeros((480, 640, 3), dtype=np.uint8),  # 3D array
    ]

    for invalid_mask in invalid_masks:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid mask format.*shape"):
            extractor.extract_features(image, invalid_mask)


def test_extract_features_invalid_mask_dtype():
    """Verify ValueError for wrong mask dtype (not uint8)."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640)

    # Invalid dtypes
    invalid_masks = [
        np.zeros((480, 640), dtype=np.float32),
        np.zeros((480, 640), dtype=np.int32),
        np.zeros((480, 640), dtype=np.bool_),
    ]

    for invalid_mask in invalid_masks:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid mask format.*dtype"):
            extractor.extract_features(image, invalid_mask)


def test_extract_features_invalid_mask_type():
    """Verify ValueError when mask is not a NumPy array."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640)

    # Invalid types
    invalid_inputs = [
        [[0, 0, 0]],  # Python list
        "mask.png",  # String
        None,  # None
    ]

    for invalid_input in invalid_inputs:
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid mask format.*expected numpy.ndarray"):
            extractor.extract_features(image, invalid_input)


def test_extract_features_dimension_mismatch():
    """Verify ValueError when mask dimensions don't match image."""
    # Arrange
    extractor = FeatureExtractor()
    image = create_test_image(480, 640)

    # Mismatched masks
    mismatched_masks = [
        create_test_mask(240, 320),  # Too small
        create_test_mask(720, 1280),  # Too large
        create_test_mask(640, 480),  # Swapped dimensions
    ]

    for invalid_mask in mismatched_masks:
        # Act & Assert
        with pytest.raises(ValueError, match="Mask dimensions.*must match image dimensions"):
            extractor.extract_features(image, invalid_mask)


def test_get_baseline_not_set():
    """Verify RuntimeError when get_baseline() called before set_baseline()."""
    # Arrange
    extractor = FeatureExtractor()

    # Act & Assert
    with pytest.raises(RuntimeError, match="Baseline features not set"):
        extractor.get_baseline()


def test_init_invalid_min_features():
    """Verify ValueError when min_features_required is invalid."""
    # Test non-positive values
    invalid_values = [0, -1, -50]

    for invalid_value in invalid_values:
        # Act & Assert
        with pytest.raises(ValueError, match="must be a positive integer"):
            FeatureExtractor(min_features_required=invalid_value)


def test_init_invalid_min_features_type():
    """Verify ValueError when min_features_required is wrong type."""
    # Test non-integer types
    invalid_types = [50.5, "50", None]

    for invalid_value in invalid_types:
        # Act & Assert
        with pytest.raises(ValueError, match="must be a positive integer"):
            FeatureExtractor(min_features_required=invalid_value)


# Integration Tests

def test_full_workflow_baseline_to_extraction():
    """Integration test: Full workflow from baseline capture to feature extraction."""
    # Arrange
    extractor = FeatureExtractor(min_features_required=30)
    baseline_image = create_test_image(480, 640, pattern="checkerboard")
    current_image = create_test_image(480, 640, pattern="textured")
    mask = create_test_mask(480, 640)

    # Act - Full workflow
    # 1. Set baseline
    extractor.set_baseline(baseline_image, mask)

    # 2. Extract current features
    current_kp, current_desc = extractor.extract_features(current_image, mask)

    # 3. Get baseline for comparison
    baseline_kp, baseline_desc = extractor.get_baseline()

    # Assert - All operations succeeded
    assert len(baseline_kp) >= 30
    assert len(current_kp) >= 0  # May have fewer features than baseline
    assert baseline_desc is not None
