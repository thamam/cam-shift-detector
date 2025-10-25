"""Integration tests for CameraMovementDetector main API.

Tests cover all 8 acceptance criteria for Story 1.5:
- AC-1.5.1: Initialization & Config Loading
- AC-1.5.2: Baseline Capture
- AC-1.5.3: Frame Processing
- AC-1.5.4: Runtime Error Handling
- AC-1.5.5: History Query Interface
- AC-1.5.6: Manual Recalibration
- AC-1.5.7: Config Validation
- AC-1.5.8: Integration Testing
"""

import pytest
import numpy as np
import cv2
import json
import os
import tempfile
import time
from pathlib import Path

from src.camera_movement_detector import CameraMovementDetector


# Test fixtures and helpers

@pytest.fixture
def test_config_dict():
    """Valid test configuration dictionary."""
    return {
        "roi": {
            "x": 100,
            "y": 50,
            "width": 400,
            "height": 300
        },
        "threshold_pixels": 2.0,
        "history_buffer_size": 100,
        "min_features_required": 50
    }


@pytest.fixture
def test_config_file(test_config_dict, tmp_path):
    """Create temporary test config file."""
    config_path = tmp_path / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)
    return str(config_path)


@pytest.fixture
def sample_image():
    """Generate valid test image (640x480, BGR, uint8)."""
    # Create image with some texture for feature detection
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # Add some structured content (rectangles) for better features
    cv2.rectangle(image, (150, 100), (250, 200), (255, 255, 255), -1)
    cv2.rectangle(image, (350, 150), (450, 250), (0, 0, 0), -1)
    cv2.rectangle(image, (200, 300), (300, 400), (128, 128, 128), -1)
    return image


@pytest.fixture
def real_sample_image():
    """Load real sample image from sample_images directory."""
    # Try to load a real image for integration tests
    sample_dir = Path("sample_images/of_jerusalem")
    if sample_dir.exists():
        images = list(sample_dir.glob("*.jpg"))
        if images:
            return cv2.imread(str(images[0]))
    # Fall back to generated image if real samples not available
    return None


# AC-1.5.1: Initialization & Config Loading

def test_init_loads_valid_config(test_config_file):
    """AC-1.5.1: Test successful initialization with valid config."""
    detector = CameraMovementDetector(test_config_file)

    assert detector.config_path == test_config_file
    assert detector.config is not None
    assert detector.region_manager is not None
    assert detector.feature_extractor is not None
    assert detector.movement_detector is not None
    assert detector.result_manager is not None
    assert detector.baseline_set is False


def test_init_raises_file_not_found():
    """AC-1.5.1: Test FileNotFoundError for missing config file."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        CameraMovementDetector("nonexistent_config.json")


def test_init_raises_on_invalid_json(tmp_path):
    """AC-1.5.1: Test ValueError for invalid JSON syntax."""
    config_path = tmp_path / "invalid.json"
    config_path.write_text("{invalid json syntax")

    with pytest.raises(ValueError, match="Invalid JSON"):
        CameraMovementDetector(str(config_path))


# AC-1.5.7: Config Validation

def test_config_validation_missing_required_field(tmp_path):
    """AC-1.5.7: Test validation catches missing required fields."""
    config = {"roi": {"x": 100, "y": 50, "width": 400, "height": 300}}
    config_path = tmp_path / "incomplete.json"
    with open(config_path, 'w') as f:
        json.dump(config, f)

    with pytest.raises(ValueError, match="missing required field"):
        CameraMovementDetector(str(config_path))


def test_config_validation_missing_roi_field(tmp_path, test_config_dict):
    """AC-1.5.7: Test validation catches missing ROI fields."""
    test_config_dict["roi"] = {"x": 100, "y": 50, "width": 400}  # Missing height
    config_path = tmp_path / "bad_roi.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    with pytest.raises(ValueError, match="missing required ROI field"):
        CameraMovementDetector(str(config_path))


def test_config_validation_invalid_roi_type(tmp_path, test_config_dict):
    """AC-1.5.7: Test validation catches invalid ROI types."""
    test_config_dict["roi"] = "not a dict"
    config_path = tmp_path / "bad_roi_type.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    with pytest.raises(ValueError, match="roi must be dict"):
        CameraMovementDetector(str(config_path))


def test_config_validation_negative_roi_value(tmp_path, test_config_dict):
    """AC-1.5.7: Test validation catches negative ROI values."""
    test_config_dict["roi"]["x"] = -10
    config_path = tmp_path / "negative_roi.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    with pytest.raises(ValueError, match="must be non-negative"):
        CameraMovementDetector(str(config_path))


def test_config_validation_negative_threshold(tmp_path, test_config_dict):
    """AC-1.5.7: Test validation catches negative threshold."""
    test_config_dict["threshold_pixels"] = -1.0
    config_path = tmp_path / "negative_threshold.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    with pytest.raises(ValueError, match="threshold_pixels must be positive"):
        CameraMovementDetector(str(config_path))


def test_config_validation_non_integer_buffer_size(tmp_path, test_config_dict):
    """AC-1.5.7: Test validation catches non-integer buffer size."""
    test_config_dict["history_buffer_size"] = 100.5
    config_path = tmp_path / "float_buffer.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    with pytest.raises(ValueError, match="history_buffer_size must be int"):
        CameraMovementDetector(str(config_path))


def test_config_validation_non_integer_min_features(tmp_path, test_config_dict):
    """AC-1.5.7: Test validation catches non-integer min features."""
    test_config_dict["min_features_required"] = 50.5
    config_path = tmp_path / "float_features.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    with pytest.raises(ValueError, match="min_features_required must be int"):
        CameraMovementDetector(str(config_path))


# AC-1.5.2: Baseline Capture

def test_set_baseline_success(test_config_file, sample_image):
    """AC-1.5.2: Test successful baseline capture with valid image."""
    detector = CameraMovementDetector(test_config_file)

    detector.set_baseline(sample_image)

    assert detector.baseline_set is True


def test_set_baseline_invalid_image_type(test_config_file):
    """AC-1.5.2: Test ValueError on invalid image type."""
    detector = CameraMovementDetector(test_config_file)

    with pytest.raises(ValueError, match="image_array must be NumPy array"):
        detector.set_baseline("not an array")


def test_set_baseline_wrong_dimensions(test_config_file):
    """AC-1.5.2: Test ValueError on wrong image dimensions."""
    detector = CameraMovementDetector(test_config_file)
    wrong_shape = np.zeros((480, 640), dtype=np.uint8)  # 2D instead of 3D

    with pytest.raises(ValueError, match="must have shape"):
        detector.set_baseline(wrong_shape)


def test_set_baseline_wrong_channels(test_config_file):
    """AC-1.5.2: Test ValueError on wrong number of channels."""
    detector = CameraMovementDetector(test_config_file)
    wrong_channels = np.zeros((480, 640, 4), dtype=np.uint8)  # 4 channels

    with pytest.raises(ValueError, match="must have 3 channels"):
        detector.set_baseline(wrong_channels)


def test_set_baseline_wrong_dtype(test_config_file):
    """AC-1.5.2: Test ValueError on wrong dtype."""
    detector = CameraMovementDetector(test_config_file)
    wrong_dtype = np.zeros((480, 640, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="must have dtype uint8"):
        detector.set_baseline(wrong_dtype)


def test_set_baseline_insufficient_features(test_config_file):
    """AC-1.5.2: Test ValueError when insufficient features detected."""
    detector = CameraMovementDetector(test_config_file)
    # Completely black image - no features
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="Insufficient features"):
        detector.set_baseline(blank_image)


# AC-1.5.3: Frame Processing

def test_process_frame_returns_correct_structure(test_config_file, sample_image):
    """AC-1.5.3: Test process_frame returns correct result dict structure."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    result = detector.process_frame(sample_image, frame_id="test_001")

    assert "status" in result
    assert result["status"] in ["VALID", "INVALID"]
    assert "translation_displacement" in result
    assert isinstance(result["translation_displacement"], float)
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "frame_id" in result
    assert result["frame_id"] == "test_001"
    assert "timestamp" in result
    assert result["timestamp"].endswith("Z")  # ISO 8601 UTC


def test_process_frame_auto_generates_frame_id(test_config_file, sample_image):
    """AC-1.5.3: Test auto-generation of frame_id when not provided."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    result = detector.process_frame(sample_image)

    assert "frame_id" in result
    assert result["frame_id"] is not None
    assert len(result["frame_id"]) > 0


def test_process_frame_with_real_images(test_config_file, real_sample_image):
    """AC-1.5.3: Test with real sample images if available."""
    if real_sample_image is None:
        pytest.skip("Real sample images not available")

    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(real_sample_image)

    result = detector.process_frame(real_sample_image)

    assert result["status"] in ["VALID", "INVALID"]
    assert "translation_displacement" in result
    assert "confidence" in result


# AC-1.5.4: Runtime Error Handling

def test_process_frame_raises_without_baseline(test_config_file, sample_image):
    """AC-1.5.4: Test RuntimeError when process_frame called before set_baseline."""
    detector = CameraMovementDetector(test_config_file)

    with pytest.raises(RuntimeError, match="Baseline not set"):
        detector.process_frame(sample_image)


def test_process_frame_validates_image_format(test_config_file, sample_image):
    """AC-1.5.4: Test ValueError on invalid image format in process_frame."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    wrong_dtype = np.zeros((480, 640, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must have dtype uint8"):
        detector.process_frame(wrong_dtype)


# AC-1.5.5: History Query Interface

def test_get_history_returns_all_results(test_config_file, sample_image):
    """AC-1.5.5: Test get_history() returns all results."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Process multiple frames
    detector.process_frame(sample_image, frame_id="frame_001")
    detector.process_frame(sample_image, frame_id="frame_002")
    detector.process_frame(sample_image, frame_id="frame_003")

    history = detector.get_history()

    assert len(history) == 3
    assert history[0]["frame_id"] == "frame_001"
    assert history[1]["frame_id"] == "frame_002"
    assert history[2]["frame_id"] == "frame_003"


def test_get_history_filters_by_frame_id(test_config_file, sample_image):
    """AC-1.5.5: Test get_history(frame_id) filters correctly."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    detector.process_frame(sample_image, frame_id="frame_001")
    detector.process_frame(sample_image, frame_id="frame_002")

    result = detector.get_history(frame_id="frame_001")

    assert len(result) == 1
    assert result[0]["frame_id"] == "frame_001"


def test_get_history_returns_empty_for_nonexistent_frame_id(test_config_file, sample_image):
    """AC-1.5.5: Test get_history returns empty list for nonexistent frame_id."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    detector.process_frame(sample_image, frame_id="frame_001")

    result = detector.get_history(frame_id="nonexistent")

    assert result == []


def test_get_history_limits_to_last_n(test_config_file, sample_image):
    """AC-1.5.5: Test get_history(limit=N) returns last N results."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Process 5 frames
    for i in range(5):
        detector.process_frame(sample_image, frame_id=f"frame_{i:03d}")

    result = detector.get_history(limit=3)

    assert len(result) == 3
    # Should be the last 3: frame_002, frame_003, frame_004
    assert result[0]["frame_id"] == "frame_002"
    assert result[1]["frame_id"] == "frame_003"
    assert result[2]["frame_id"] == "frame_004"


# AC-1.5.6: Manual Recalibration

def test_recalibrate_success(test_config_file, sample_image):
    """AC-1.5.6: Test successful recalibration returns True."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Create new reference image
    new_reference = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(new_reference, (100, 100), (200, 200), (255, 255, 255), -1)

    success = detector.recalibrate(new_reference)

    assert success is True
    assert detector.baseline_set is True


def test_recalibrate_failure_insufficient_features(test_config_file, sample_image):
    """AC-1.5.6: Test recalibration with insufficient features returns False."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Black image - no features
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

    success = detector.recalibrate(blank_image)

    assert success is False


def test_recalibrate_failure_invalid_format(test_config_file, sample_image):
    """AC-1.5.6: Test recalibration with invalid format returns False."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Wrong dtype
    wrong_dtype = np.zeros((480, 640, 3), dtype=np.float32)

    success = detector.recalibrate(wrong_dtype)

    assert success is False


# AC-1.5.8: Integration Testing

def test_full_workflow_integration(test_config_file, sample_image):
    """AC-1.5.8: Test complete workflow: init → baseline → process → history."""
    # Initialize detector
    detector = CameraMovementDetector(test_config_file)
    assert detector.baseline_set is False

    # Set baseline
    detector.set_baseline(sample_image)
    assert detector.baseline_set is True

    # Process frame
    result = detector.process_frame(sample_image, frame_id="test_001")
    assert result["status"] in ["VALID", "INVALID"]
    assert result["frame_id"] == "test_001"

    # Query history
    history = detector.get_history()
    assert len(history) == 1
    assert history[0]["frame_id"] == "test_001"

    # Recalibrate
    success = detector.recalibrate(sample_image)
    assert success is True


def test_process_frame_performance(test_config_file, sample_image):
    """AC-1.5.8: Test process_frame executes within performance target (<500ms)."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Warm up (first call may be slower)
    detector.process_frame(sample_image)

    # Measure performance
    start = time.time()
    result = detector.process_frame(sample_image)
    elapsed = time.time() - start

    assert elapsed < 0.5, f"process_frame took {elapsed:.3f}s (> 500ms limit)"


def test_multiple_frames_with_history_buffer(test_config_file, sample_image):
    """AC-1.5.8: Test processing multiple frames with history management."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Process 10 frames
    for i in range(10):
        result = detector.process_frame(sample_image, frame_id=f"frame_{i:03d}")
        assert result["frame_id"] == f"frame_{i:03d}"

    # Verify all in history
    history = detector.get_history()
    assert len(history) == 10

    # Verify FIFO ordering
    for i, result in enumerate(history):
        assert result["frame_id"] == f"frame_{i:03d}"


def test_real_images_end_to_end(test_config_file, real_sample_image):
    """AC-1.5.8: Integration test with real sample images if available."""
    if real_sample_image is None:
        pytest.skip("Real sample images not available")

    detector = CameraMovementDetector(test_config_file)

    # Set baseline with real image
    detector.set_baseline(real_sample_image)

    # Process the same image (should be VALID with low displacement)
    result = detector.process_frame(real_sample_image, frame_id="real_001")

    assert result["status"] == "VALID"
    assert result["translation_displacement"] < 1.0  # Same image, minimal displacement
    assert result["confidence"] > 0.8  # High confidence for same image

    # Verify history
    history = detector.get_history()
    assert len(history) == 1


def test_edge_case_insufficient_matches_returns_invalid(test_config_file, sample_image):
    """AC-1.5.8: Test edge case where insufficient feature matches occur."""
    detector = CameraMovementDetector(test_config_file)
    detector.set_baseline(sample_image)

    # Create completely different image (should have few/no matches)
    different_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(different_image, (320, 240), 100, (255, 255, 255), -1)

    result = detector.process_frame(different_image, frame_id="different")

    # Should return INVALID status with low confidence or inf displacement
    assert result["status"] == "INVALID"


# Additional edge cases

def test_history_buffer_respects_configured_size(tmp_path, test_config_dict, sample_image):
    """Test that history buffer respects configured size limit."""
    # Set small buffer size
    test_config_dict["history_buffer_size"] = 5
    config_path = tmp_path / "small_buffer.json"
    with open(config_path, 'w') as f:
        json.dump(test_config_dict, f)

    detector = CameraMovementDetector(str(config_path))
    detector.set_baseline(sample_image)

    # Process 10 frames (more than buffer size)
    for i in range(10):
        detector.process_frame(sample_image, frame_id=f"frame_{i:03d}")

    history = detector.get_history()

    # Should only keep last 5
    assert len(history) == 5
    assert history[0]["frame_id"] == "frame_005"
    assert history[4]["frame_id"] == "frame_009"
