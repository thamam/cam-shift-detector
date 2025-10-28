"""
Unit Tests for Dual Detector Runner Module

Tests orchestration of ChArUco and cam-shift detectors for comparison.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from validation.utilities.dual_detector_runner import (
    DualDetectorRunner,
    DualDetectionResult
)


@pytest.fixture
def mock_camera_yaml():
    """Mock camera calibration data."""
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
    dist = np.array([0.1, -0.2, 0.0, 0.0, 0.3], dtype=np.float32)
    image_size = (640, 480)
    return K, dist, image_size


@pytest.fixture
def mock_charuco_components():
    """Mock ChArUco board components."""
    dictionary = MagicMock()
    board = MagicMock()
    detector = MagicMock()
    return dictionary, board, detector


@pytest.fixture
def test_image():
    """Create a test grayscale image."""
    return np.zeros((480, 640), dtype=np.uint8)


class TestDualDetectorRunnerInitialization:
    """Test DualDetectorRunner initialization."""

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    def test_initialization_with_valid_configs(
        self,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components
    ):
        """Test successful initialization with valid configurations."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components

        # Initialize runner
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )

        # Assertions
        assert runner.image_width == 640
        assert runner.image_height == 480
        assert runner.threshold_px == pytest.approx(14.4, abs=0.01)  # 3% of 480
        assert runner.z_distance_m == 1.15
        assert not runner.baseline_set
        assert runner.tvec_baseline is None

    @patch('validation.utilities.dual_detector_runner.Path')
    def test_initialization_with_missing_camera_yaml_raises_error(self, mock_path):
        """Test that missing camera YAML raises FileNotFoundError."""
        # Mock camera.yaml as not existing
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        with pytest.raises(FileNotFoundError, match="Camera YAML not found"):
            DualDetectorRunner(
                camera_yaml_path="missing.yaml",
                camshift_config_path="config.json"
            )

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    def test_initialization_with_invalid_camera_yaml_raises_error(
        self,
        mock_read_yaml,
        mock_path
    ):
        """Test that invalid camera YAML raises ValueError."""
        # Mock paths as existing
        mock_path.return_value.exists.return_value = True

        # Mock read_yaml_camera returning None (invalid file)
        mock_read_yaml.return_value = None

        with pytest.raises(ValueError, match="Failed to load camera calibration"):
            DualDetectorRunner(
                camera_yaml_path="camera.yaml",
                camshift_config_path="config.json"
            )

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    def test_custom_charuco_parameters(
        self,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components
    ):
        """Test initialization with custom ChArUco parameters."""
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components

        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json",
            charuco_squares_x=10,
            charuco_squares_y=8,
            z_distance_m=2.0
        )

        # Verify custom parameters were passed
        mock_make_board.assert_called_once_with(
            10, 8, 0.035, 0.026, "DICT_4X4_50"
        )
        assert runner.z_distance_m == 2.0


class TestDualDetectorRunnerBaseline:
    """Test baseline configuration."""

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    @patch('validation.utilities.dual_detector_runner.estimate_pose_charuco')
    def test_set_baseline_with_successful_charuco_detection(
        self,
        mock_estimate_pose,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components,
        test_image
    ):
        """Test baseline setting with successful ChArUco detection."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components

        # Mock successful ChArUco detection
        tvec = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        mock_estimate_pose.return_value = (rvec, tvec, 8)

        # Initialize and set baseline
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )
        success = runner.set_baseline(test_image)

        # Assertions
        assert success is True
        assert runner.baseline_set is True
        assert runner.tvec_baseline is not None
        np.testing.assert_array_equal(runner.tvec_baseline, tvec)
        assert runner.frame_counter == 0

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    @patch('validation.utilities.dual_detector_runner.estimate_pose_charuco')
    def test_set_baseline_with_failed_charuco_detection(
        self,
        mock_estimate_pose,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components,
        test_image
    ):
        """Test baseline setting with failed ChArUco detection."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components

        # Mock failed ChArUco detection
        mock_estimate_pose.return_value = None

        # Initialize and attempt baseline
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )
        success = runner.set_baseline(test_image)

        # Assertions - baseline not set but no crash
        assert success is False
        assert runner.baseline_set is False
        assert runner.tvec_baseline is None


class TestDualDetectorRunnerProcessFrame:
    """Test frame processing."""

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    @patch('validation.utilities.dual_detector_runner.estimate_pose_charuco')
    @patch('validation.utilities.dual_detector_runner.time.time_ns')
    def test_process_frame_with_charuco_detected(
        self,
        mock_time_ns,
        mock_estimate_pose,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components,
        test_image
    ):
        """Test frame processing with successful ChArUco detection."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components
        mock_time_ns.return_value = 1234567890

        # Mock baseline detection
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        rvec_baseline = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        mock_estimate_pose.return_value = (rvec_baseline, tvec_baseline, 8)

        # Initialize and set baseline
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )
        runner.set_baseline(test_image)

        # Mock current frame detection with displacement
        tvec_current = np.array([[0.01], [0.02], [1.15]], dtype=np.float32)  # 1cm X, 2cm Y
        rvec_current = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        mock_estimate_pose.return_value = (rvec_current, tvec_current, 10)

        # Mock cam-shift result
        mock_detector_instance = mock_detector_class.return_value
        mock_detector_instance.process_frame.return_value = {
            "status": "VALID",
            "translation_displacement": 15.0,
            "confidence": 0.95,
            "frame_id": "test_frame",
            "timestamp": "2025-10-27T12:00:00"
        }

        # Process frame
        result = runner.process_frame(test_image, frame_id="test_frame")

        # Assertions
        assert isinstance(result, DualDetectionResult)
        assert result.frame_idx == 0
        assert result.timestamp_ns == 1234567890
        assert result.charuco_detected is True
        assert result.charuco_displacement_px > 0  # Should have some displacement
        assert result.charuco_confidence == 10.0
        assert result.camshift_status == "VALID"
        assert result.camshift_displacement_px == 15.0
        assert result.camshift_confidence == 0.95
        assert not np.isnan(result.displacement_diff)
        assert result.agreement_status in ["GREEN", "RED"]
        assert result.threshold_px == pytest.approx(14.4, abs=0.01)

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    @patch('validation.utilities.dual_detector_runner.estimate_pose_charuco')
    @patch('validation.utilities.dual_detector_runner.time.time_ns')
    def test_process_frame_with_charuco_not_detected(
        self,
        mock_time_ns,
        mock_estimate_pose,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components,
        test_image
    ):
        """Test frame processing with ChArUco detection failure - graceful handling."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components
        mock_time_ns.return_value = 1234567890

        # Mock baseline detection (successful)
        tvec_baseline = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        rvec_baseline = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        mock_estimate_pose.return_value = (rvec_baseline, tvec_baseline, 8)

        # Initialize and set baseline
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )
        runner.set_baseline(test_image)

        # Mock current frame detection failure
        mock_estimate_pose.return_value = None  # ChArUco not detected

        # Mock cam-shift result (still works)
        mock_detector_instance = mock_detector_class.return_value
        mock_detector_instance.process_frame.return_value = {
            "status": "VALID",
            "translation_displacement": 15.0,
            "confidence": 0.95,
            "frame_id": "test_frame",
            "timestamp": "2025-10-27T12:00:00"
        }

        # Process frame - should not crash
        result = runner.process_frame(test_image, frame_id="test_frame")

        # Assertions - graceful handling
        assert isinstance(result, DualDetectionResult)
        assert result.charuco_detected is False
        assert np.isnan(result.charuco_displacement_px)
        assert np.isnan(result.charuco_confidence)
        assert result.camshift_status == "VALID"
        assert result.camshift_displacement_px == 15.0
        assert np.isnan(result.displacement_diff)  # Cannot compare
        assert result.agreement_status is None  # Cannot classify

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    def test_process_frame_without_baseline_raises_error(
        self,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components,
        test_image
    ):
        """Test that processing without baseline raises RuntimeError."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components

        # Initialize without setting baseline
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )

        # Attempt to process without baseline
        with pytest.raises(RuntimeError, match="Baseline must be set"):
            runner.process_frame(test_image)

    @patch('validation.utilities.dual_detector_runner.Path')
    @patch('validation.utilities.dual_detector_runner.read_yaml_camera')
    @patch('validation.utilities.dual_detector_runner.make_charuco_board')
    @patch('validation.utilities.dual_detector_runner.CameraMovementDetector')
    @patch('validation.utilities.dual_detector_runner.estimate_pose_charuco')
    @patch('validation.utilities.dual_detector_runner.time.time_ns')
    def test_frame_counter_increments(
        self,
        mock_time_ns,
        mock_estimate_pose,
        mock_detector_class,
        mock_make_board,
        mock_read_yaml,
        mock_path,
        mock_camera_yaml,
        mock_charuco_components,
        test_image
    ):
        """Test that frame counter increments correctly."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_read_yaml.return_value = mock_camera_yaml
        mock_make_board.return_value = mock_charuco_components
        mock_time_ns.return_value = 1234567890

        # Mock detection
        tvec = np.array([[0.0], [0.0], [1.15]], dtype=np.float32)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        mock_estimate_pose.return_value = (rvec, tvec, 8)

        # Mock cam-shift
        mock_detector_instance = mock_detector_class.return_value
        mock_detector_instance.process_frame.return_value = {
            "status": "VALID",
            "translation_displacement": 10.0,
            "confidence": 0.9,
            "frame_id": "frame",
            "timestamp": "2025-10-27T12:00:00"
        }

        # Initialize and set baseline
        runner = DualDetectorRunner(
            camera_yaml_path="camera.yaml",
            camshift_config_path="config.json"
        )
        runner.set_baseline(test_image)

        # Process multiple frames
        result1 = runner.process_frame(test_image)
        result2 = runner.process_frame(test_image)
        result3 = runner.process_frame(test_image)

        # Verify frame indices
        assert result1.frame_idx == 0
        assert result2.frame_idx == 1
        assert result3.frame_idx == 2
