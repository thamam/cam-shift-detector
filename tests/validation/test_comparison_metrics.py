"""
Unit Tests for Comparison Metrics Module

Tests pure calculation functions for displacement comparison between
ChArUco and cam-shift detectors.
"""

import pytest
import numpy as np

from validation.utilities.comparison_metrics import (
    calculate_displacement_difference,
    calculate_threshold,
    classify_agreement,
    calculate_charuco_displacement_2d,
    calculate_mse
)


class TestCalculateDisplacementDifference:
    """Test calculate_displacement_difference function."""

    def test_identical_displacements_returns_zero(self):
        """Test that identical displacements return 0.0."""
        result = calculate_displacement_difference(10.0, 10.0)
        assert result == 0.0

    def test_different_displacements_returns_absolute_difference(self):
        """Test that different displacements return correct L2 norm."""
        result = calculate_displacement_difference(18.5, 15.2)
        assert result == pytest.approx(3.3, abs=0.01)

    def test_negative_difference_returns_absolute_value(self):
        """Test that negative differences are handled correctly."""
        result = calculate_displacement_difference(5.0, 10.0)
        assert result == pytest.approx(5.0, abs=0.01)

    def test_large_difference(self):
        """Test with large displacement values."""
        result = calculate_displacement_difference(100.0, 50.0)
        assert result == pytest.approx(50.0, abs=0.01)

    def test_zero_displacements(self):
        """Test with both displacements at zero."""
        result = calculate_displacement_difference(0.0, 0.0)
        assert result == 0.0


class TestCalculateThreshold:
    """Test calculate_threshold function."""

    def test_standard_640x480_image(self):
        """Test threshold for standard 640x480 image."""
        result = calculate_threshold(640, 480, 0.03)
        expected = 480 * 0.03  # min(640, 480) * 0.03 = 14.4
        assert result == pytest.approx(14.4, abs=0.01)

    def test_square_image(self):
        """Test threshold for square image."""
        result = calculate_threshold(800, 800, 0.03)
        expected = 800 * 0.03  # 24.0
        assert result == pytest.approx(24.0, abs=0.01)

    def test_portrait_orientation(self):
        """Test threshold for portrait (height > width) image."""
        result = calculate_threshold(480, 640, 0.03)
        expected = 480 * 0.03  # min(480, 640) * 0.03 = 14.4
        assert result == pytest.approx(14.4, abs=0.01)

    def test_custom_percentage(self):
        """Test threshold with custom percentage."""
        result = calculate_threshold(640, 480, 0.05)
        expected = 480 * 0.05  # 24.0
        assert result == pytest.approx(24.0, abs=0.01)

    def test_small_image(self):
        """Test threshold for small image dimensions."""
        result = calculate_threshold(320, 240, 0.03)
        expected = 240 * 0.03  # 7.2
        assert result == pytest.approx(7.2, abs=0.01)


class TestClassifyAgreement:
    """Test classify_agreement function."""

    def test_diff_below_threshold_returns_green(self):
        """Test that diff < threshold returns GREEN."""
        result = classify_agreement(10.0, 14.4)
        assert result == "GREEN"

    def test_diff_above_threshold_returns_red(self):
        """Test that diff > threshold returns RED."""
        result = classify_agreement(20.0, 14.4)
        assert result == "RED"

    def test_diff_equal_threshold_returns_green(self):
        """Test boundary case: diff == threshold returns GREEN."""
        result = classify_agreement(14.4, 14.4)
        assert result == "GREEN"

    def test_zero_diff_returns_green(self):
        """Test that perfect agreement (diff=0) returns GREEN."""
        result = classify_agreement(0.0, 14.4)
        assert result == "GREEN"

    def test_large_diff_returns_red(self):
        """Test that large diff returns RED."""
        result = classify_agreement(100.0, 14.4)
        assert result == "RED"


class TestCalculateCharucoDisplacement2D:
    """Test calculate_charuco_displacement_2d function."""

    def test_zero_displacement_returns_zero(self):
        """Test that identical tvecs return zero displacement."""
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        tvec_current = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)

        result = calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, 1.15)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_x_axis_displacement_only(self):
        """Test displacement along X axis only."""
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        tvec_current = np.array([[0.01], [0.0], [1.15]], dtype=np.float32)  # 1cm X displacement

        result = calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, 1.15)
        expected = (0.01 * 800) / 1.15  # ~6.96 pixels
        assert result == pytest.approx(expected, rel=0.01)

    def test_y_axis_displacement_only(self):
        """Test displacement along Y axis only."""
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        tvec_current = np.array([[0.0], [0.02], [1.15]], dtype=np.float32)  # 2cm Y displacement

        result = calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, 1.15)
        expected = (0.02 * 800) / 1.15  # ~13.91 pixels
        assert result == pytest.approx(expected, rel=0.01)

    def test_diagonal_displacement(self):
        """Test displacement along both X and Y axes."""
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        tvec_current = np.array([[0.01], [0.02], [1.15]], dtype=np.float32)  # 1cm X, 2cm Y

        result = calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, 1.15)

        # Calculate expected
        dx_px = (0.01 * 800) / 1.15
        dy_px = (0.02 * 800) / 1.15
        expected = np.sqrt(dx_px**2 + dy_px**2)

        assert result == pytest.approx(expected, rel=0.01)

    def test_different_focal_lengths(self):
        """Test with different fx and fy focal lengths."""
        K = np.array([[1000, 0, 320], [0, 900, 240], [0, 0, 1]], dtype=np.float32)
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        tvec_current = np.array([[0.01], [0.01], [1.15]], dtype=np.float32)

        result = calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, 1.15)

        # Calculate expected with different fx, fy
        dx_px = (0.01 * 1000) / 1.15
        dy_px = (0.01 * 900) / 1.15
        expected = np.sqrt(dx_px**2 + dy_px**2)

        assert result == pytest.approx(expected, rel=0.01)

    def test_custom_z_distance(self):
        """Test with custom Z distance parameter."""
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
        tvec_baseline = np.array([[0.0], [0.0], [1.5]], dtype=np.float32)
        tvec_current = np.array([[0.01], [0.0], [1.5]], dtype=np.float32)

        # Use custom z_distance
        result = calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, z_distance_m=1.5)
        expected = (0.01 * 800) / 1.5  # ~5.33 pixels
        assert result == pytest.approx(expected, rel=0.01)


class TestCalculateMSE:
    """Test calculate_mse function."""

    def test_perfect_agreement_returns_zero(self):
        """Test that identical sequences return MSE = 0."""
        charuco_list = [5.0, 10.0, 15.0]
        camshift_list = [5.0, 10.0, 15.0]

        result = calculate_mse(charuco_list, camshift_list)
        assert result == 0.0

    def test_known_mse_calculation(self):
        """Test MSE calculation with known values."""
        charuco_list = [10.0, 12.0]
        camshift_list = [11.0, 13.0]

        # MSE = ((10-11)^2 + (12-13)^2) / 2 = (1 + 1) / 2 = 1.0
        result = calculate_mse(charuco_list, camshift_list)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_larger_sequence(self):
        """Test MSE with larger sequence."""
        charuco_list = [5.0, 10.0, 15.0, 20.0]
        camshift_list = [6.0, 11.0, 14.0, 21.0]

        # MSE = ((5-6)^2 + (10-11)^2 + (15-14)^2 + (20-21)^2) / 4
        # MSE = (1 + 1 + 1 + 1) / 4 = 1.0
        result = calculate_mse(charuco_list, camshift_list)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_different_lengths_raises_error(self):
        """Test that different length lists raise ValueError."""
        charuco_list = [10.0, 12.0, 14.0]
        camshift_list = [11.0, 13.0]

        with pytest.raises(ValueError, match="Lists must have same length"):
            calculate_mse(charuco_list, camshift_list)

    def test_empty_lists_raises_error(self):
        """Test that empty lists raise ValueError."""
        charuco_list = []
        camshift_list = []

        with pytest.raises(ValueError, match="Cannot calculate MSE for empty lists"):
            calculate_mse(charuco_list, camshift_list)

    def test_single_element_lists(self):
        """Test MSE with single element lists."""
        charuco_list = [10.0]
        camshift_list = [12.0]

        # MSE = (10-12)^2 / 1 = 4.0
        result = calculate_mse(charuco_list, camshift_list)
        assert result == pytest.approx(4.0, abs=0.01)

    def test_large_differences(self):
        """Test MSE with large displacement differences."""
        charuco_list = [10.0, 20.0, 30.0]
        camshift_list = [50.0, 60.0, 70.0]

        # MSE = ((10-50)^2 + (20-60)^2 + (30-70)^2) / 3
        # MSE = (1600 + 1600 + 1600) / 3 = 1600.0
        result = calculate_mse(charuco_list, camshift_list)
        assert result == pytest.approx(1600.0, abs=0.01)
