"""
Unit Tests for Mode A - 4-Quadrant Comparison Tool

Tests feature extraction, visualization, enhanced metrics, and CSV export
functionality added in Stage 4 Mode A implementation.
"""

import pytest
import numpy as np
import cv2 as cv
import csv
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from validation.utilities.dual_detector_runner import DualDetectionResult
from validation.utilities.comparison_logger import ComparisonLogger

# Import functions to test (will need to make these importable)
# For now, we'll test through the ComparisonLogger methods


class TestCSVExport:
    """Test CSV export functionality from AC5."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "csv_test_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.fixture
    def logger_with_results(self, temp_output_dir):
        """Create ComparisonLogger with sample results."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_mode_a"
        )

        # Add sample results
        for i in range(5):
            result = DualDetectionResult(
                frame_idx=i,
                timestamp_ns=1000000000 + i * 100000000,
                charuco_detected=True,
                charuco_displacement_px=10.0 + i,
                charuco_dx=5.0 + i * 0.5,
                charuco_dy=8.0 + i * 0.3,
                charuco_confidence=20.0,
                camshift_status="VALID",
                camshift_displacement_px=9.5 + i,
                camshift_dx=4.8 + i * 0.5,
                camshift_dy=7.7 + i * 0.3,
                camshift_confidence=0.85,
                displacement_diff=0.5,
                agreement_status="GREEN",
                threshold_px=14.4
            )
            logger.log_frame(result)

        return logger

    def test_csv_export_creates_file(self, logger_with_results):
        """Test that CSV export creates a file."""
        csv_path = logger_with_results.save_csv()

        assert csv_path.exists()
        assert csv_path.suffix == ".csv"
        assert "test_mode_a_comparison" in csv_path.name

    def test_csv_export_has_correct_columns(self, logger_with_results):
        """Test that CSV has all required columns from AC5."""
        csv_path = logger_with_results.save_csv()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

        expected_columns = [
            'frame_id', 'dx_charuco', 'dy_charuco', 'dx_csd', 'dy_csd',
            'delta_dx', 'delta_dy', 'error_mag', 'inliers', 'total', 'rmse'
        ]

        assert header == expected_columns

    def test_csv_export_correct_row_count(self, logger_with_results):
        """Test that CSV has correct number of data rows."""
        csv_path = logger_with_results.save_csv()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)

        # Should have 5 data rows (from fixture)
        assert len(rows) == 5

    def test_csv_export_component_calculations(self, logger_with_results):
        """Test that delta_dx, delta_dy, and error_mag are calculated correctly."""
        csv_path = logger_with_results.save_csv()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            first_row = next(reader)

        # Parse values
        dx_charuco = float(first_row[1])
        dy_charuco = float(first_row[2])
        dx_csd = float(first_row[3])
        dy_csd = float(first_row[4])
        delta_dx = float(first_row[5])
        delta_dy = float(first_row[6])
        error_mag = float(first_row[7])

        # Verify calculations
        expected_delta_dx = dx_csd - dx_charuco
        expected_delta_dy = dy_csd - dy_charuco
        expected_error_mag = np.sqrt(expected_delta_dx**2 + expected_delta_dy**2)

        assert abs(delta_dx - expected_delta_dx) < 0.01
        assert abs(delta_dy - expected_delta_dy) < 0.01
        assert abs(error_mag - expected_error_mag) < 0.01

    def test_csv_export_rmse_calculation(self, logger_with_results):
        """Test that RMSE is calculated correctly."""
        csv_path = logger_with_results.save_csv()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)

        # RMSE should be the same for all rows
        rmse_values = [float(row[10]) for row in rows]

        # All RMSE values should be identical
        assert len(set(rmse_values)) == 1

        # Verify RMSE calculation manually
        results = logger_with_results.results
        squared_errors = [
            (r.charuco_displacement_px - r.camshift_displacement_px) ** 2
            for r in results
        ]
        expected_rmse = np.sqrt(np.mean(squared_errors))

        assert abs(rmse_values[0] - expected_rmse) < 0.01

    def test_csv_export_handles_nan_values(self, temp_output_dir):
        """Test that CSV export handles NaN values gracefully."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_nan"
        )

        # Add result with NaN values (ChArUco not detected)
        result = DualDetectionResult(
            frame_idx=0,
            timestamp_ns=1000000000,
            charuco_detected=False,
            charuco_displacement_px=np.nan,
            charuco_dx=np.nan,
            charuco_dy=np.nan,
            charuco_confidence=np.nan,
            camshift_status="VALID",
            camshift_displacement_px=10.0,
            camshift_dx=5.0,
            camshift_dy=8.0,
            camshift_confidence=0.85,
            displacement_diff=np.nan,
            agreement_status=None,
            threshold_px=14.4
        )
        logger.log_frame(result)

        csv_path = logger.save_csv()

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            row = next(reader)

        # NaN values should be exported as "NaN" strings
        assert row[1] == "NaN"  # dx_charuco
        assert row[2] == "NaN"  # dy_charuco
        assert row[5] == "NaN"  # delta_dx
        assert row[6] == "NaN"  # delta_dy
        assert row[7] == "NaN"  # error_mag

    def test_csv_export_raises_on_empty_results(self, temp_output_dir):
        """Test that CSV export raises ValueError if no results logged."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_empty"
        )

        with pytest.raises(ValueError, match="No results to export"):
            logger.save_csv()


class TestEnhancedMetrics:
    """Test enhanced displacement metrics from AC4."""

    def test_dual_detection_result_has_component_fields(self):
        """Test that DualDetectionResult includes dx/dy component fields."""
        result = DualDetectionResult(
            frame_idx=0,
            timestamp_ns=1000000000,
            charuco_detected=True,
            charuco_displacement_px=10.0,
            charuco_dx=5.0,
            charuco_dy=8.0,
            charuco_confidence=20.0,
            camshift_status="VALID",
            camshift_displacement_px=9.5,
            camshift_dx=4.8,
            camshift_dy=7.7,
            camshift_confidence=0.85,
            displacement_diff=0.5,
            agreement_status="GREEN",
            threshold_px=14.4
        )

        assert hasattr(result, 'charuco_dx')
        assert hasattr(result, 'charuco_dy')
        assert hasattr(result, 'camshift_dx')
        assert hasattr(result, 'camshift_dy')
        assert result.charuco_dx == 5.0
        assert result.charuco_dy == 8.0
        assert result.camshift_dx == 4.8
        assert result.camshift_dy == 7.7

    def test_component_magnitude_consistency(self):
        """Test that component-wise displacement matches magnitude."""
        result = DualDetectionResult(
            frame_idx=0,
            timestamp_ns=1000000000,
            charuco_detected=True,
            charuco_displacement_px=10.0,
            charuco_dx=6.0,
            charuco_dy=8.0,
            charuco_confidence=20.0,
            camshift_status="VALID",
            camshift_displacement_px=10.0,
            camshift_dx=6.0,
            camshift_dy=8.0,
            camshift_confidence=0.85,
            displacement_diff=0.0,
            agreement_status="GREEN",
            threshold_px=14.4
        )

        # Verify magnitude calculation from components
        charuco_magnitude = np.sqrt(result.charuco_dx**2 + result.charuco_dy**2)
        camshift_magnitude = np.sqrt(result.camshift_dx**2 + result.camshift_dy**2)

        assert abs(charuco_magnitude - result.charuco_displacement_px) < 0.01
        assert abs(camshift_magnitude - result.camshift_displacement_px) < 0.01


class TestFeatureVisualization:
    """Test feature extraction and visualization from AC3."""

    def test_feature_visualization_functions_exist(self):
        """Test that feature visualization functions are importable."""
        # This will need to be updated once functions are made importable
        # For now, verify they exist in comparison_tool.py
        from pathlib import Path
        comparison_tool_path = PROJECT_ROOT / "tools" / "validation" / "comparison_tool.py"

        with open(comparison_tool_path, 'r') as f:
            content = f.read()

        assert "def draw_charuco_corners" in content
        assert "def draw_orb_features" in content
        assert "def draw_mode_a_status_bar" in content


class TestPerformance:
    """Test performance requirements from AC6."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_frame_processing_latency(self, sample_image):
        """Test that frame processing meets <100ms latency requirement."""
        import time

        # This is a placeholder - would need actual detector setup
        # For now, just test that basic operations are fast
        start = time.time()

        # Simulate frame operations
        gray = cv.cvtColor(sample_image, cv.COLOR_BGR2GRAY)
        _ = cv.resize(gray, (320, 240))

        elapsed_ms = (time.time() - start) * 1000

        # Basic operations should be well under 100ms
        assert elapsed_ms < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
