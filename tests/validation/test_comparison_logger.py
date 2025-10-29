"""
Unit Tests for Comparison Logger Module

Tests JSON logging, MSE analysis, worst match retrieval, and matplotlib
visualization for detector comparison results.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import shutil

from validation.utilities.comparison_logger import ComparisonLogger
from validation.utilities.dual_detector_runner import DualDetectionResult


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_results():
    """Create sample DualDetectionResults for testing."""
    results = []
    for i in range(10):
        result = DualDetectionResult(
            frame_idx=i,
            timestamp_ns=1234567890 + i * 1000000,
            charuco_detected=True,
            charuco_displacement_px=10.0 + i,
            charuco_dx=5.0 + i * 0.5,
            charuco_dy=8.0 + i * 0.3,
            charuco_confidence=8.0 + (i % 3),
            camshift_status="VALID",
            camshift_displacement_px=12.0 + i,
            camshift_dx=6.0 + i * 0.5,
            camshift_dy=10.0 + i * 0.3,
            camshift_confidence=0.9 + (i * 0.01),
            displacement_diff=2.0 + (i * 0.5),
            agreement_status="GREEN" if i < 5 else "RED",
            threshold_px=14.4
        )
        results.append(result)
    return results


@pytest.fixture
def results_with_charuco_failures():
    """Create results with some ChArUco detection failures."""
    results = []
    for i in range(10):
        charuco_detected = i % 3 != 0  # Fail every 3rd frame
        result = DualDetectionResult(
            frame_idx=i,
            timestamp_ns=1234567890 + i * 1000000,
            charuco_detected=charuco_detected,
            charuco_displacement_px=10.0 + i if charuco_detected else np.nan,
            charuco_dx=5.0 + i * 0.5 if charuco_detected else np.nan,
            charuco_dy=8.0 + i * 0.3 if charuco_detected else np.nan,
            charuco_confidence=8.0 if charuco_detected else np.nan,
            camshift_status="VALID",
            camshift_displacement_px=12.0 + i,
            camshift_dx=6.0 + i * 0.5,
            camshift_dy=10.0 + i * 0.3,
            camshift_confidence=0.9,
            displacement_diff=2.0 if charuco_detected else np.nan,
            agreement_status="GREEN" if charuco_detected else None,
            threshold_px=14.4
        )
        results.append(result)
    return results


class TestComparisonLoggerInitialization:
    """Test ComparisonLogger initialization."""

    def test_initialization_creates_output_dir(self, temp_output_dir):
        """Test that initialization creates output directory."""
        output_dir = temp_output_dir / "comparison_logs"
        logger = ComparisonLogger(
            output_dir=str(output_dir),
            session_name="test_session"
        )

        assert output_dir.exists()
        assert logger.session_name == "test_session"
        assert logger.output_dir == output_dir
        assert len(logger.results) == 0

    def test_initialization_with_existing_dir(self, temp_output_dir):
        """Test initialization with pre-existing directory."""
        # Create directory first
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        assert logger.output_dir.exists()


class TestComparisonLoggerLogging:
    """Test log_frame functionality."""

    def test_log_frame_appends_result(self, temp_output_dir, sample_results):
        """Test that log_frame appends results correctly."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Log first result
        logger.log_frame(sample_results[0])
        assert len(logger.results) == 1

        # Log second result
        logger.log_frame(sample_results[1])
        assert len(logger.results) == 2

    def test_log_multiple_frames(self, temp_output_dir, sample_results):
        """Test logging multiple frames."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Log all results
        for result in sample_results:
            logger.log_frame(result)

        assert len(logger.results) == len(sample_results)


class TestComparisonLoggerSaveLog:
    """Test save_log functionality."""

    def test_save_log_creates_json_file(self, temp_output_dir, sample_results):
        """Test that save_log creates valid JSON file."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Log results
        for result in sample_results:
            logger.log_frame(result)

        # Save log
        output_path = logger.save_log()

        # Verify file exists
        assert output_path.exists()
        assert output_path.name == "test_session_comparison.json"

    def test_save_log_json_structure(self, temp_output_dir, sample_results):
        """Test that saved JSON has correct structure."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Log results
        for result in sample_results:
            logger.log_frame(result)

        # Save and read back
        output_path = logger.save_log()
        with open(output_path, 'r') as f:
            data = json.load(f)

        # Verify structure
        assert "session_name" in data
        assert "timestamp" in data
        assert "threshold_px" in data
        assert "total_frames" in data
        assert "results" in data

        assert data["session_name"] == "test_session"
        assert data["threshold_px"] == 14.4
        assert data["total_frames"] == len(sample_results)
        assert len(data["results"]) == len(sample_results)

    def test_save_log_with_custom_filename(self, temp_output_dir, sample_results):
        """Test saving log with custom filename."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results:
            logger.log_frame(result)

        # Save with custom filename
        output_path = logger.save_log(filename="custom_log.json")

        assert output_path.exists()
        assert output_path.name == "custom_log.json"


class TestComparisonLoggerCalculateMSE:
    """Test calculate_mse functionality."""

    def test_calculate_mse_with_valid_data(self, temp_output_dir, sample_results):
        """Test MSE calculation with valid comparison data."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Log results
        for result in sample_results:
            logger.log_frame(result)

        # Calculate MSE
        mse = logger.calculate_mse()

        # MSE should be positive
        assert mse >= 0
        assert isinstance(mse, float)

    def test_calculate_mse_matches_expected_value(self, temp_output_dir):
        """Test that MSE calculation matches expected value."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Create results with known displacements
        # ChArUco: [10, 12], Cam-shift: [11, 13]
        # MSE = ((10-11)^2 + (12-13)^2) / 2 = 1.0
        result1 = DualDetectionResult(
            frame_idx=0, timestamp_ns=1000, charuco_detected=True,
            charuco_displacement_px=10.0, charuco_dx=6.0, charuco_dy=8.0,
            charuco_confidence=8.0, camshift_status="VALID",
            camshift_displacement_px=11.0, camshift_dx=6.6, camshift_dy=8.8,
            camshift_confidence=0.9, displacement_diff=1.0,
            agreement_status="GREEN", threshold_px=14.4
        )
        result2 = DualDetectionResult(
            frame_idx=1, timestamp_ns=2000, charuco_detected=True,
            charuco_displacement_px=12.0, charuco_dx=7.0, charuco_dy=9.0,
            charuco_confidence=8.0, camshift_status="VALID",
            camshift_displacement_px=13.0, camshift_dx=7.6, camshift_dy=10.0,
            camshift_confidence=0.9, displacement_diff=1.0,
            agreement_status="GREEN", threshold_px=14.4
        )

        logger.log_frame(result1)
        logger.log_frame(result2)

        mse = logger.calculate_mse()
        assert mse == pytest.approx(1.0, abs=0.01)

    def test_calculate_mse_with_charuco_failures_raises_error(
        self,
        temp_output_dir
    ):
        """Test that MSE calculation with all ChArUco failures raises error."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Create results with all ChArUco failures
        for i in range(5):
            result = DualDetectionResult(
                frame_idx=i, timestamp_ns=1000 + i,
                charuco_detected=False,
                charuco_displacement_px=np.nan,
                charuco_dx=np.nan,
                charuco_dy=np.nan,
                charuco_confidence=np.nan,
                camshift_status="VALID",
                camshift_displacement_px=10.0,
                camshift_dx=6.0,
                camshift_dy=8.0,
                camshift_confidence=0.9,
                displacement_diff=np.nan,
                agreement_status=None,
                threshold_px=14.4
            )
            logger.log_frame(result)

        # Should raise ValueError
        with pytest.raises(ValueError, match="No valid comparison data"):
            logger.calculate_mse()

    def test_calculate_mse_ignores_charuco_failures(
        self,
        temp_output_dir,
        results_with_charuco_failures
    ):
        """Test that MSE calculation ignores ChArUco failures."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in results_with_charuco_failures:
            logger.log_frame(result)

        # Should calculate MSE only for valid frames
        mse = logger.calculate_mse()
        assert mse >= 0
        assert isinstance(mse, float)


class TestComparisonLoggerGetWorstMatches:
    """Test get_worst_matches functionality."""

    def test_get_worst_matches_returns_correct_count(self, temp_output_dir, sample_results):
        """Test that get_worst_matches returns requested number."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results:
            logger.log_frame(result)

        worst_matches = logger.get_worst_matches(n=5)
        assert len(worst_matches) == 5

    def test_get_worst_matches_sorted_descending(self, temp_output_dir, sample_results):
        """Test that worst matches are sorted by displacement_diff descending."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results:
            logger.log_frame(result)

        worst_matches = logger.get_worst_matches(n=5)

        # Verify descending order
        for i in range(len(worst_matches) - 1):
            assert worst_matches[i].displacement_diff >= worst_matches[i + 1].displacement_diff

    def test_get_worst_matches_when_n_exceeds_available(self, temp_output_dir, sample_results):
        """Test get_worst_matches when n > available frames."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results[:3]:  # Only log 3 results
            logger.log_frame(result)

        worst_matches = logger.get_worst_matches(n=10)
        assert len(worst_matches) == 3  # Returns all available

    def test_get_worst_matches_ignores_charuco_failures(
        self,
        temp_output_dir,
        results_with_charuco_failures
    ):
        """Test that get_worst_matches ignores ChArUco failures."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in results_with_charuco_failures:
            logger.log_frame(result)

        worst_matches = logger.get_worst_matches(n=10)

        # Should only include frames with valid ChArUco detection
        for match in worst_matches:
            assert match.charuco_detected is True
            assert not np.isnan(match.displacement_diff)


class TestComparisonLoggerGenerateMSEGraph:
    """Test generate_mse_graph functionality."""

    def test_generate_mse_graph_creates_file(self, temp_output_dir, sample_results):
        """Test that graph generation creates PNG file."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results:
            logger.log_frame(result)

        output_path = logger.generate_mse_graph()

        # Verify file exists
        assert output_path.exists()
        assert output_path.suffix == ".png"
        assert output_path.name == "test_session_mse_graph.png"

    def test_generate_mse_graph_with_custom_path(self, temp_output_dir, sample_results):
        """Test graph generation with custom output path."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results:
            logger.log_frame(result)

        custom_path = temp_output_dir / "custom_graph.png"
        output_path = logger.generate_mse_graph(output_path=str(custom_path))

        assert output_path.exists()
        assert output_path.name == "custom_graph.png"

    def test_generate_mse_graph_with_no_valid_data_raises_error(
        self,
        temp_output_dir
    ):
        """Test that graph generation with no valid data raises error."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        # Create results with all ChArUco failures
        for i in range(3):
            result = DualDetectionResult(
                frame_idx=i, timestamp_ns=1000 + i,
                charuco_detected=False,
                charuco_displacement_px=np.nan,
                charuco_dx=np.nan,
                charuco_dy=np.nan,
                charuco_confidence=np.nan,
                camshift_status="VALID",
                camshift_displacement_px=10.0,
                camshift_dx=6.0,
                camshift_dy=8.0,
                camshift_confidence=0.9,
                displacement_diff=np.nan,
                agreement_status=None,
                threshold_px=14.4
            )
            logger.log_frame(result)

        with pytest.raises(ValueError, match="No valid comparison data"):
            logger.generate_mse_graph()

    def test_generate_mse_graph_highlights_worst_matches(
        self,
        temp_output_dir,
        sample_results
    ):
        """Test that graph generation completes with worst match highlighting."""
        logger = ComparisonLogger(
            output_dir=str(temp_output_dir),
            session_name="test_session"
        )

        for result in sample_results:
            logger.log_frame(result)

        # Should not raise exception
        output_path = logger.generate_mse_graph(highlight_worst_n=3)
        assert output_path.exists()
