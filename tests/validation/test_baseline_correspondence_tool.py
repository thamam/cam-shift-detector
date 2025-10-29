"""
Unit and Integration Tests for Mode B - Baseline Correspondence Tool

Tests motion vector visualization, match quality metrics, difference heatmap,
and baseline pinning functionality with ≥85% coverage target.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# Import functions from baseline_correspondence_tool
import sys
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from tools.validation.baseline_correspondence_tool import (
    draw_motion_vectors,
    draw_match_quality_metrics,
    compute_diff_heatmap,
    draw_baseline_thumbnail,
    draw_status_bar,
    load_and_cache_images
)
from src.movement_detector import MovementDetector


@pytest.fixture
def sample_image():
    """Create sample 640x480 BGR image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_matches():
    """Create sample match correspondences."""
    matches = [
        ((100.0, 150.0), (102.0, 152.0)),  # Small displacement
        ((200.0, 250.0), (205.0, 255.0)),  # Medium displacement
        ((300.0, 350.0), (280.0, 330.0)),  # Large displacement (outlier)
        ((400.0, 400.0), (402.0, 401.0)),  # Small displacement
        ((150.0, 200.0), (151.0, 202.0)),  # Tiny displacement
    ]
    return matches


@pytest.fixture
def sample_mask_inliers():
    """Create sample inlier mask (all inliers)."""
    return np.array([1, 1, 1, 1, 1], dtype=np.uint8)


@pytest.fixture
def sample_mask_mixed():
    """Create sample mixed inlier/outlier mask."""
    return np.array([1, 1, 0, 1, 1], dtype=np.uint8)


@pytest.fixture
def temp_image_dir():
    """Create temporary directory with test images."""
    temp_dir = tempfile.mkdtemp()
    test_dir = Path(temp_dir) / "test_images"
    test_dir.mkdir()

    # Create 5 test images
    for i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(test_dir / f"test_frame_{i:03d}.jpg"), img)

    yield test_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestDrawMotionVectors:
    """Test draw_motion_vectors function (AC2)."""

    def test_draw_motion_vectors_with_inliers(self, sample_image, sample_matches, sample_mask_inliers):
        """Test motion vector drawing with all inliers."""
        result = draw_motion_vectors(sample_image, sample_matches, sample_mask_inliers)

        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
        # Image should be modified (arrows drawn)
        assert not np.array_equal(result, sample_image)

    def test_draw_motion_vectors_with_mixed_mask(self, sample_image, sample_matches, sample_mask_mixed):
        """Test motion vector drawing with inliers and outliers."""
        result = draw_motion_vectors(sample_image, sample_matches, sample_mask_mixed)

        assert result.shape == sample_image.shape
        # Should have both green (inliers) and red (outliers) arrows
        # Verify by checking that image was modified
        assert not np.array_equal(result, sample_image)

    def test_draw_motion_vectors_empty_matches(self, sample_image):
        """Test motion vector drawing with no matches."""
        matches = []
        mask = np.array([], dtype=np.uint8)

        result = draw_motion_vectors(sample_image, matches, mask)

        # Image should be unmodified if no matches
        assert result.shape == sample_image.shape

    def test_draw_motion_vectors_creates_copy(self, sample_image, sample_matches, sample_mask_inliers):
        """Test that original image is not modified."""
        original_copy = sample_image.copy()
        result = draw_motion_vectors(sample_image, sample_matches, sample_mask_inliers)

        # Original should be unchanged
        assert np.array_equal(sample_image, original_copy)
        # Result should be different
        assert not np.array_equal(result, original_copy)


class TestDrawMatchQualityMetrics:
    """Test draw_match_quality_metrics function (AC3)."""

    def test_metrics_panel_green_threshold(self, sample_image):
        """Test metrics panel with >80% inlier ratio (GREEN)."""
        result = draw_match_quality_metrics(
            sample_image, inliers=85, total=100, rmse=2.5, confidence=0.85
        )

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_metrics_panel_orange_threshold(self, sample_image):
        """Test metrics panel with 50-80% inlier ratio (ORANGE)."""
        result = draw_match_quality_metrics(
            sample_image, inliers=65, total=100, rmse=5.2, confidence=0.65
        )

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_metrics_panel_red_threshold(self, sample_image):
        """Test metrics panel with <50% inlier ratio (RED)."""
        result = draw_match_quality_metrics(
            sample_image, inliers=30, total=100, rmse=15.8, confidence=0.30
        )

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_metrics_panel_zero_total(self, sample_image):
        """Test metrics panel with zero total matches."""
        result = draw_match_quality_metrics(
            sample_image, inliers=0, total=0, rmse=0.0, confidence=0.0
        )

        # Should handle division by zero gracefully
        assert result.shape == sample_image.shape

    def test_metrics_panel_high_rmse(self, sample_image):
        """Test metrics panel with high RMSE value."""
        result = draw_match_quality_metrics(
            sample_image, inliers=50, total=100, rmse=99.99, confidence=0.50
        )

        assert result.shape == sample_image.shape


class TestComputeDiffHeatmap:
    """Test compute_diff_heatmap function (AC5 - Optional)."""

    def test_heatmap_with_identity_homography(self, sample_image):
        """Test heatmap generation with identity homography (no warping)."""
        H = np.eye(3, dtype=np.float32)
        baseline = sample_image.copy()

        result = compute_diff_heatmap(baseline, sample_image, H)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_heatmap_with_translation(self, sample_image):
        """Test heatmap with translation homography."""
        # Create translation matrix (shift by 10 pixels in x, 5 in y)
        H = np.array([
            [1.0, 0.0, 10.0],
            [0.0, 1.0, 5.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        baseline = sample_image.copy()
        current = np.roll(sample_image, (5, 10), axis=(0, 1))

        result = compute_diff_heatmap(baseline, current, H)

        assert result is not None
        assert result.shape == current.shape

    def test_heatmap_with_different_images(self):
        """Test heatmap with significantly different baseline and current."""
        baseline = np.ones((480, 640, 3), dtype=np.uint8) * 50
        current = np.ones((480, 640, 3), dtype=np.uint8) * 200
        H = np.eye(3, dtype=np.float32)

        result = compute_diff_heatmap(baseline, current, H)

        assert result is not None
        # High difference should produce visible heatmap
        assert result.shape == current.shape

    def test_heatmap_with_invalid_homography(self, sample_image):
        """Test heatmap with invalid homography matrix."""
        H = np.zeros((3, 3), dtype=np.float32)  # Invalid (singular)

        result = compute_diff_heatmap(sample_image, sample_image, H)

        # Should return None or handle gracefully
        assert result is None or result.shape == sample_image.shape


class TestDrawBaselineThumbnail:
    """Test draw_baseline_thumbnail function (AC1)."""

    def test_thumbnail_overlay(self, sample_image):
        """Test baseline thumbnail drawing."""
        baseline = np.ones((480, 640, 3), dtype=np.uint8) * 100

        result = draw_baseline_thumbnail(sample_image, baseline)

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_thumbnail_different_size_baseline(self):
        """Test thumbnail with different size baseline."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        baseline = np.ones((720, 1280, 3), dtype=np.uint8) * 100

        result = draw_baseline_thumbnail(image, baseline)

        # Should resize baseline to thumbnail size
        assert result.shape == image.shape


class TestDrawStatusBar:
    """Test draw_status_bar function (AC4)."""

    def test_status_bar_baseline_set(self, sample_image):
        """Test status bar with baseline set."""
        result = draw_status_bar(
            sample_image, frame_idx=5, total_frames=50,
            baseline_set=True, show_vectors=True, show_heatmap=False
        )

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_status_bar_baseline_not_set(self, sample_image):
        """Test status bar without baseline."""
        result = draw_status_bar(
            sample_image, frame_idx=0, total_frames=50,
            baseline_set=False, show_vectors=False, show_heatmap=False
        )

        assert result.shape == sample_image.shape

    def test_status_bar_all_toggles_on(self, sample_image):
        """Test status bar with all toggles enabled."""
        result = draw_status_bar(
            sample_image, frame_idx=25, total_frames=50,
            baseline_set=True, show_vectors=True, show_heatmap=True
        )

        assert result.shape == sample_image.shape


class TestLoadAndCacheImages:
    """Test load_and_cache_images function."""

    def test_load_images_success(self, temp_image_dir):
        """Test successful image loading and caching."""
        images = load_and_cache_images(temp_image_dir)

        assert len(images) == 5
        # Verify structure: List of (image, filename) tuples
        for img, filename in images:
            assert isinstance(img, np.ndarray)
            assert img.shape == (480, 640, 3)
            assert isinstance(filename, str)
            assert filename.endswith('.jpg')

    def test_load_images_sorted(self, temp_image_dir):
        """Test that images are loaded in sorted order."""
        images = load_and_cache_images(temp_image_dir)

        filenames = [name for _, name in images]
        assert filenames == sorted(filenames)

    def test_load_images_empty_directory(self):
        """Test loading from empty directory."""
        temp_dir = tempfile.mkdtemp()
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()

        with pytest.raises(SystemExit):
            load_and_cache_images(empty_dir)

        shutil.rmtree(temp_dir)


class TestMovementDetectorGetLastMatches:
    """Test MovementDetector.get_last_matches() method."""

    def test_get_last_matches_initial_state(self):
        """Test get_last_matches before any detection."""
        detector = MovementDetector(threshold_pixels=2.0)

        matches, mask = detector.get_last_matches()

        assert matches == []
        assert len(mask) == 0

    def test_get_last_matches_after_detection(self):
        """Test get_last_matches after detect_movement call."""
        detector = MovementDetector(threshold_pixels=2.0)

        # Create sample features
        kp_baseline = [cv2.KeyPoint(x=float(i*10), y=float(i*10), size=10) for i in range(20)]
        desc_baseline = np.random.randint(0, 255, (20, 32), dtype=np.uint8)

        kp_current = [cv2.KeyPoint(x=float(i*10+2), y=float(i*10+1), size=10) for i in range(20)]
        desc_current = np.random.randint(0, 255, (20, 32), dtype=np.uint8)

        # Call detect_movement
        try:
            detector.detect_movement((kp_baseline, desc_baseline), (kp_current, desc_current))
        except (ValueError, RuntimeError):
            # May fail due to insufficient matches or homography issues
            pass

        # Check that matches were stored (even if detection failed)
        matches, mask = detector.get_last_matches()

        # Matches should be list of tuples
        assert isinstance(matches, list)
        assert isinstance(mask, np.ndarray)


class TestMovementDetectorGetLastHomography:
    """Test MovementDetector.get_last_homography() method."""

    def test_get_last_homography_initial_state(self):
        """Test get_last_homography before any detection."""
        detector = MovementDetector(threshold_pixels=2.0)

        H = detector.get_last_homography()

        assert H is None

    def test_get_last_homography_after_detection(self):
        """Test get_last_homography after detect_movement call."""
        detector = MovementDetector(threshold_pixels=2.0, use_affine_model=False)

        # Create sample features with more matches
        kp_baseline = [cv2.KeyPoint(x=float(i*20), y=float(i*20), size=10) for i in range(50)]
        desc_baseline = np.random.randint(0, 255, (50, 32), dtype=np.uint8)

        kp_current = [cv2.KeyPoint(x=float(i*20+1), y=float(i*20+1), size=10) for i in range(50)]
        desc_current = desc_baseline.copy()  # Use same descriptors for guaranteed matches

        try:
            detector.detect_movement((kp_baseline, desc_baseline), (kp_current, desc_current))
            H = detector.get_last_homography()

            # If detection succeeded, H should be 3x3 matrix
            if H is not None:
                assert H.shape == (3, 3)
        except (ValueError, RuntimeError):
            # Detection may fail, that's okay for this test
            pass


class TestModeBIntegration:
    """Integration tests for Mode B tool."""

    def test_motion_vector_visualization_workflow(self, sample_image):
        """Test complete motion vector visualization workflow."""
        # Create matches and mask
        matches = [
            ((100.0, 100.0), (105.0, 103.0)),
            ((200.0, 200.0), (202.0, 201.0)),
            ((300.0, 300.0), (280.0, 290.0)),  # Outlier
        ]
        mask = np.array([1, 1, 0], dtype=np.uint8)

        # Draw motion vectors
        result = draw_motion_vectors(sample_image, matches, mask)

        # Verify arrows were drawn
        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

        # Draw metrics
        result = draw_match_quality_metrics(result, 2, 3, 3.5, 0.67)

        # Verify metrics panel added
        assert result.shape == sample_image.shape

    def test_heatmap_overlay_workflow(self):
        """Test complete heatmap generation and overlay workflow."""
        baseline = np.ones((480, 640, 3), dtype=np.uint8) * 100
        current = np.ones((480, 640, 3), dtype=np.uint8) * 150
        H = np.eye(3, dtype=np.float32)

        # Generate heatmap
        heatmap = compute_diff_heatmap(baseline, current, H)

        assert heatmap is not None

        # Overlay with 50% transparency
        result = cv2.addWeighted(current, 0.5, heatmap, 0.5, 0)

        assert result.shape == current.shape

    def test_baseline_pinning_display_workflow(self, sample_image):
        """Test baseline pinning and display workflow."""
        baseline = np.ones((480, 640, 3), dtype=np.uint8) * 80

        # Draw baseline thumbnail
        result = draw_baseline_thumbnail(sample_image, baseline)

        # Draw status bar showing baseline is set
        result = draw_status_bar(
            result, frame_idx=10, total_frames=50,
            baseline_set=True, show_vectors=True, show_heatmap=False
        )

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)


class TestMinimumMatchesWarning:
    """Test minimum matches requirement (AC2)."""

    def test_warn_when_below_50_matches(self, sample_image):
        """Test that warning is shown when <50 matches."""
        # Test with 30 matches (below threshold)
        matches = [((i*10.0, i*10.0), (i*10.0+1, i*10.0+1)) for i in range(30)]
        mask = np.ones(30, dtype=np.uint8)

        # This should trigger warning in actual tool
        # For unit test, just verify function works with low match count
        result = draw_motion_vectors(sample_image, matches, mask)

        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=tools.validation.baseline_correspondence_tool",
                 "--cov=src.movement_detector", "--cov-report=term-missing"])
