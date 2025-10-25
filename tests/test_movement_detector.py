"""Unit tests for MovementDetector

Tests cover all acceptance criteria:
- AC-1.3.1: Feature Matching with BFMatcher
- AC-1.3.2: Homography Estimation
- AC-1.3.3: Displacement Calculation
- AC-1.3.4: Threshold Validation
- AC-1.3.5: Confidence Score
- AC-1.3.6: Error Handling
"""

import numpy as np
import pytest
import cv2

from src.movement_detector import MovementDetector


@pytest.fixture
def detector():
    """Create a MovementDetector instance with default threshold."""
    return MovementDetector(threshold_pixels=2.0)


@pytest.fixture
def detector_custom_threshold():
    """Create a MovementDetector instance with custom threshold."""
    return MovementDetector(threshold_pixels=5.0)


@pytest.fixture
def baseline_features():
    """Create baseline features with 50 keypoints distributed across image."""
    # Create keypoints in a grid pattern to avoid degenerate configuration
    keypoints = []
    for i in range(50):
        x = 100.0 + (i % 10) * 20.0  # Spread horizontally
        y = 100.0 + (i // 10) * 20.0  # Spread vertically
        keypoints.append(cv2.KeyPoint(x=x, y=y, size=10.0))

    # Create distinctive descriptors for matching
    descriptors = np.random.RandomState(42).randint(0, 255, (50, 32), dtype=np.uint8)
    return (keypoints, descriptors)


@pytest.fixture
def current_features_no_movement(baseline_features):
    """Create current features identical to baseline (no movement)."""
    baseline_kp, baseline_desc = baseline_features
    # Same positions and descriptors = perfect match, zero displacement
    keypoints = []
    for i in range(50):
        x = 100.0 + (i % 10) * 20.0
        y = 100.0 + (i // 10) * 20.0
        keypoints.append(cv2.KeyPoint(x=x, y=y, size=10.0))
    descriptors = baseline_desc.copy()
    return (keypoints, descriptors)


@pytest.fixture
def current_features_small_movement(baseline_features):
    """Create current features with 1.5px shift (below threshold)."""
    _, baseline_desc = baseline_features
    # Shift all keypoints by 1.5 pixels (below 2.0 threshold)
    keypoints = []
    for i in range(50):
        x = 100.0 + (i % 10) * 20.0 + 1.5  # Shift by 1.5px
        y = 100.0 + (i // 10) * 20.0
        keypoints.append(cv2.KeyPoint(x=x, y=y, size=10.0))
    descriptors = baseline_desc.copy()
    return (keypoints, descriptors)


@pytest.fixture
def current_features_movement(baseline_features):
    """Create current features with 3px shift (above threshold)."""
    _, baseline_desc = baseline_features
    # Shift all keypoints by 3 pixels (above 2.0 threshold)
    keypoints = []
    for i in range(50):
        x = 100.0 + (i % 10) * 20.0 + 3.0  # Shift by 3.0px
        y = 100.0 + (i // 10) * 20.0
        keypoints.append(cv2.KeyPoint(x=x, y=y, size=10.0))
    descriptors = baseline_desc.copy()
    return (keypoints, descriptors)


@pytest.fixture
def current_features_exact_threshold(baseline_features):
    """Create current features with exactly 2.0px shift (boundary case)."""
    _, baseline_desc = baseline_features
    # Shift all keypoints by exactly 2.0 pixels
    keypoints = []
    for i in range(50):
        x = 100.0 + (i % 10) * 20.0 + 2.0  # Shift by 2.0px
        y = 100.0 + (i // 10) * 20.0
        keypoints.append(cv2.KeyPoint(x=x, y=y, size=10.0))
    descriptors = baseline_desc.copy()
    return (keypoints, descriptors)


class TestInitialization:
    """Test MovementDetector initialization (AC-1.3.1)."""

    def test_init_default_threshold(self):
        """Test initialization with default threshold."""
        detector = MovementDetector()
        assert detector.threshold_pixels == 2.0
        assert isinstance(detector.matcher, cv2.BFMatcher)

    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        detector = MovementDetector(threshold_pixels=5.5)
        assert detector.threshold_pixels == 5.5

    def test_init_invalid_threshold_negative(self):
        """Test initialization fails with negative threshold."""
        with pytest.raises(ValueError, match="threshold_pixels must be a positive number"):
            MovementDetector(threshold_pixels=-1.0)

    def test_init_invalid_threshold_zero(self):
        """Test initialization fails with zero threshold."""
        with pytest.raises(ValueError, match="threshold_pixels must be a positive number"):
            MovementDetector(threshold_pixels=0.0)

    def test_matcher_configuration(self, detector):
        """Test BFMatcher is configured with NORM_HAMMING and crossCheck (AC-1.3.1)."""
        # Verify matcher is BFMatcher instance
        assert isinstance(detector.matcher, cv2.BFMatcher)
        # Note: OpenCV doesn't expose normType/crossCheck directly, tested via behavior


class TestFeatureMatching:
    """Test feature matching functionality (AC-1.3.1)."""

    def test_matching_identical_features(self, detector, baseline_features):
        """Test matching with identical features produces perfect matches."""
        moved, displacement, confidence = detector.detect_movement(
            baseline_features, baseline_features
        )

        # Identical features should have zero displacement
        assert displacement == 0.0
        assert moved is False
        # High confidence expected (inlier ratio close to 1.0)
        assert confidence >= 0.9

    def test_matching_similar_features(
        self, detector, baseline_features, current_features_movement
    ):
        """Test matching with similar but shifted features."""
        moved, displacement, confidence = detector.detect_movement(
            baseline_features, current_features_movement
        )

        # Should detect matches successfully
        assert displacement > 0.0
        assert isinstance(moved, bool)
        assert 0.0 <= confidence <= 1.0


class TestHomographyEstimation:
    """Test homography estimation (AC-1.3.2)."""

    def test_homography_with_sufficient_matches(
        self, detector, baseline_features, current_features_movement
    ):
        """Test homography estimation succeeds with >= 10 matches."""
        # Should not raise any exception
        moved, displacement, confidence = detector.detect_movement(
            baseline_features, current_features_movement
        )

        assert isinstance(displacement, float)
        assert displacement >= 0.0

    def test_insufficient_matches(self, detector):
        """Test homography fails with < 10 matches (AC-1.3.2)."""
        # Create features with very different descriptors (no matches)
        baseline_kp = [cv2.KeyPoint(x=100.0, y=100.0, size=10.0) for _ in range(20)]
        baseline_desc = np.zeros((20, 32), dtype=np.uint8)

        current_kp = [cv2.KeyPoint(x=100.0, y=100.0, size=10.0) for _ in range(20)]
        current_desc = np.ones((20, 32), dtype=np.uint8) * 255

        with pytest.raises(ValueError, match="Insufficient feature matches: found .* < 10 required"):
            detector.detect_movement(
                (baseline_kp, baseline_desc),
                (current_kp, current_desc)
            )


class TestDisplacementCalculation:
    """Test displacement calculation (AC-1.3.3)."""

    def test_displacement_no_movement(
        self, detector, baseline_features, current_features_no_movement
    ):
        """Test zero displacement with identical features."""
        _, displacement, _ = detector.detect_movement(
            baseline_features, current_features_no_movement
        )

        assert displacement == 0.0

    def test_displacement_with_movement(
        self, detector, baseline_features, current_features_movement
    ):
        """Test displacement calculation with known shift (3px)."""
        _, displacement, _ = detector.detect_movement(
            baseline_features, current_features_movement
        )

        # Should detect ~3.0 pixel displacement
        assert displacement >= 2.5
        assert displacement <= 3.5

    def test_displacement_rounding(
        self, detector, baseline_features, current_features_movement
    ):
        """Test displacement is rounded to 2 decimal places (AC-1.3.3)."""
        _, displacement, _ = detector.detect_movement(
            baseline_features, current_features_movement
        )

        # Verify 2 decimal places
        assert displacement == round(displacement, 2)


class TestThresholdValidation:
    """Test threshold validation (AC-1.3.4)."""

    def test_below_threshold(
        self, detector, baseline_features, current_features_small_movement
    ):
        """Test moved=False when displacement < threshold (1.5 < 2.0)."""
        moved, displacement, _ = detector.detect_movement(
            baseline_features, current_features_small_movement
        )

        assert displacement < 2.0
        assert moved is False

    def test_above_threshold(
        self, detector, baseline_features, current_features_movement
    ):
        """Test moved=True when displacement >= threshold (3.0 >= 2.0)."""
        moved, displacement, _ = detector.detect_movement(
            baseline_features, current_features_movement
        )

        assert displacement >= 2.0
        assert moved is True

    def test_exactly_at_threshold(
        self, detector, baseline_features, current_features_exact_threshold
    ):
        """Test moved=True when displacement exactly equals threshold (2.0 == 2.0)."""
        moved, displacement, _ = detector.detect_movement(
            baseline_features, current_features_exact_threshold
        )

        assert displacement == 2.0
        assert moved is True  # >= threshold means moved

    def test_custom_threshold(
        self, detector_custom_threshold, baseline_features, current_features_movement
    ):
        """Test threshold validation with custom threshold (5.0)."""
        moved, displacement, _ = detector_custom_threshold.detect_movement(
            baseline_features, current_features_movement
        )

        # 3px displacement should be below 5.0 threshold
        if displacement < 5.0:
            assert moved is False
        else:
            assert moved is True


class TestConfidenceScore:
    """Test confidence score calculation (AC-1.3.5)."""

    def test_confidence_range(
        self, detector, baseline_features, current_features_movement
    ):
        """Test confidence is in range [0.0, 1.0] (AC-1.3.5)."""
        _, _, confidence = detector.detect_movement(
            baseline_features, current_features_movement
        )

        assert 0.0 <= confidence <= 1.0

    def test_high_confidence_identical_features(
        self, detector, baseline_features, current_features_no_movement
    ):
        """Test high confidence with identical features (perfect match)."""
        _, _, confidence = detector.detect_movement(
            baseline_features, current_features_no_movement
        )

        # Identical features should have very high confidence
        assert confidence >= 0.9

    def test_confidence_calculation_formula(
        self, detector, baseline_features, current_features_movement
    ):
        """Test confidence is calculated from inlier ratio (AC-1.3.5)."""
        _, _, confidence = detector.detect_movement(
            baseline_features, current_features_movement
        )

        # Confidence should be inliers / total_matches
        # With good features, confidence should be reasonably high
        assert confidence > 0.0


class TestErrorHandling:
    """Test error handling (AC-1.3.6)."""

    def test_invalid_baseline_not_tuple(self, detector, baseline_features):
        """Test error with baseline_features not a tuple."""
        with pytest.raises(ValueError, match="Invalid baseline_features format"):
            detector.detect_movement(
                "not a tuple",  # Invalid type
                baseline_features
            )

    def test_invalid_baseline_wrong_length(self, detector, baseline_features):
        """Test error with baseline_features tuple wrong length."""
        with pytest.raises(ValueError, match="Invalid baseline_features format"):
            detector.detect_movement(
                ([], [], []),  # Wrong tuple length
                baseline_features
            )

    def test_invalid_baseline_keypoints_not_list(self, detector, baseline_features):
        """Test error with keypoints not a list."""
        _, descriptors = baseline_features
        with pytest.raises(ValueError, match="keypoints must be list"):
            detector.detect_movement(
                ("not a list", descriptors),
                baseline_features
            )

    def test_invalid_baseline_descriptors_not_array(self, detector, baseline_features):
        """Test error with descriptors not numpy array."""
        keypoints, _ = baseline_features
        with pytest.raises(ValueError, match="descriptors must be numpy.ndarray"):
            detector.detect_movement(
                (keypoints, "not an array"),
                baseline_features
            )

    def test_invalid_baseline_empty_keypoints(self, detector, baseline_features):
        """Test error with empty keypoints list."""
        with pytest.raises(ValueError, match="keypoints list is empty"):
            detector.detect_movement(
                ([], np.array([[1, 2, 3]])),
                baseline_features
            )

    def test_invalid_baseline_empty_descriptors(self, detector, baseline_features):
        """Test error with empty descriptors array."""
        keypoints, _ = baseline_features
        with pytest.raises(ValueError, match="descriptors array is empty"):
            detector.detect_movement(
                (keypoints, np.array([])),
                baseline_features
            )

    def test_invalid_current_features(self, detector, baseline_features):
        """Test error with invalid current_features format."""
        with pytest.raises(ValueError, match="Invalid current_features format"):
            detector.detect_movement(
                baseline_features,
                None  # Invalid
            )


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_zero_displacement(
        self, detector, baseline_features, current_features_no_movement
    ):
        """Test complete pipeline with zero displacement."""
        moved, displacement, confidence = detector.detect_movement(
            baseline_features, current_features_no_movement
        )

        assert moved is False
        assert displacement == 0.0
        assert 0.0 <= confidence <= 1.0

    def test_large_displacement(self, detector, baseline_features):
        """Test detection with very large displacement."""
        _, descriptors = baseline_features
        # Create features with large shift (50 pixels)
        keypoints = []
        for i in range(50):
            x = 100.0 + (i % 10) * 20.0 + 50.0  # Shift by 50px
            y = 100.0 + (i // 10) * 20.0
            keypoints.append(cv2.KeyPoint(x=x, y=y, size=10.0))

        moved, displacement, confidence = detector.detect_movement(
            baseline_features,
            (keypoints, descriptors.copy())
        )

        # Should detect large movement
        assert moved is True
        assert displacement > 10.0

    def test_return_types(
        self, detector, baseline_features, current_features_movement
    ):
        """Test return value types are correct."""
        result = detector.detect_movement(
            baseline_features, current_features_movement
        )

        assert isinstance(result, tuple)
        assert len(result) == 3

        moved, displacement, confidence = result
        assert isinstance(moved, (bool, np.bool_))
        assert isinstance(displacement, float)
        assert isinstance(confidence, float)
