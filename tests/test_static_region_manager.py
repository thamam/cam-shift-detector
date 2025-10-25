"""Unit tests for StaticRegionManager

Tests cover:
- AC-1.1.1: Config loading and validation
- AC-1.1.2: Binary mask generation functionality
- AC-1.1.3: Boundary validation
- AC-1.1.4: Error handling for invalid inputs
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.static_region_manager import StaticRegionManager


# Helper Functions

def create_temp_config(roi: dict, additional_fields: dict = None) -> str:
    """Create a temporary config file for testing.

    Args:
        roi: Dictionary with x, y, width, height fields
        additional_fields: Optional additional config fields

    Returns:
        Path to temporary config file
    """
    config = {'roi': roi}
    if additional_fields:
        config.update(additional_fields)

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_file)
    temp_file.close()
    return temp_file.name


def create_test_image(height: int, width: int) -> np.ndarray:
    """Create a test image with specified dimensions.

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Random BGR image (H×W×3, uint8)
    """
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


# AC-1.1.1: Config Loading Tests

def test_load_valid_config():
    """Verify StaticRegionManager loads valid config.json successfully."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'width': 400, 'height': 300}
    config_path = create_temp_config(roi)

    # Act
    manager = StaticRegionManager(config_path)

    # Assert
    assert manager.roi == roi
    Path(config_path).unlink()  # Cleanup


def test_load_missing_config():
    """Verify FileNotFoundError raised when config missing."""
    # Arrange
    config_path = '/nonexistent/path/config.json'

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        StaticRegionManager(config_path)


def test_load_invalid_json():
    """Verify ValueError raised for malformed JSON."""
    # Arrange
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.write("{invalid json content")
    temp_file.close()

    # Act & Assert
    with pytest.raises(ValueError, match="Invalid JSON"):
        StaticRegionManager(temp_file.name)

    Path(temp_file.name).unlink()  # Cleanup


def test_load_invalid_schema_missing_roi():
    """Verify ValueError when 'roi' key missing."""
    # Arrange
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump({'other_field': 'value'}, temp_file)
    temp_file.close()

    # Act & Assert
    with pytest.raises(ValueError, match="missing required field 'roi'"):
        StaticRegionManager(temp_file.name)

    Path(temp_file.name).unlink()  # Cleanup


def test_load_invalid_schema_missing_x():
    """Verify ValueError when 'x' field missing."""
    # Arrange
    roi = {'y': 50, 'width': 400, 'height': 300}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="missing required field 'roi.x'"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_invalid_schema_missing_y():
    """Verify ValueError when 'y' field missing."""
    # Arrange
    roi = {'x': 100, 'width': 400, 'height': 300}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="missing required field 'roi.y'"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_invalid_schema_missing_width():
    """Verify ValueError when 'width' field missing."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'height': 300}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="missing required field 'roi.width'"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_invalid_schema_missing_height():
    """Verify ValueError when 'height' field missing."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'width': 400}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="missing required field 'roi.height'"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_negative_x():
    """Verify ValueError when x is negative."""
    # Arrange
    roi = {'x': -10, 'y': 50, 'width': 400, 'height': 300}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="'roi.x' must be non-negative"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_negative_y():
    """Verify ValueError when y is negative."""
    # Arrange
    roi = {'x': 100, 'y': -5, 'width': 400, 'height': 300}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="'roi.y' must be non-negative"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_zero_width():
    """Verify ValueError when width is zero."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'width': 0, 'height': 300}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="'roi.width' must be positive"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


def test_load_zero_height():
    """Verify ValueError when height is zero."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'width': 400, 'height': 0}
    config_path = create_temp_config(roi)

    # Act & Assert
    with pytest.raises(ValueError, match="'roi.height' must be positive"):
        StaticRegionManager(config_path)

    Path(config_path).unlink()  # Cleanup


# AC-1.1.2: Binary Mask Generation Tests

def test_mask_generation_640x480():
    """Verify mask generation with standard VGA image size."""
    # Arrange
    roi = {'x': 10, 'y': 20, 'width': 100, 'height': 80}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert mask.dtype == np.uint8
    Path(config_path).unlink()  # Cleanup


def test_mask_generation_1920x1080():
    """Verify mask generation with HD image size."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'width': 400, 'height': 300}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((1080, 1920))

    # Assert
    assert mask.shape == (1080, 1920)
    assert mask.dtype == np.uint8
    Path(config_path).unlink()  # Cleanup


def test_mask_returns_correct_shape():
    """Verify mask has expected dimensions (height, width)."""
    # Arrange
    roi = {'x': 50, 'y': 30, 'width': 200, 'height': 150}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert mask.ndim == 2  # 2D array
    Path(config_path).unlink()  # Cleanup


def test_mask_values_binary():
    """Verify mask contains only 0 and 255 values."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.dtype == np.uint8
    unique_values = np.unique(mask)
    assert len(unique_values) <= 2
    assert all(val in [0, 255] for val in unique_values)
    Path(config_path).unlink()  # Cleanup


def test_mask_roi_region_is_static():
    """Verify ROI region is marked as static (255) and outside is dynamic (0)."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert - ROI region should be 255
    assert np.all(mask[10:60, 10:60] == 255)
    # Assert - Outside ROI should be 0
    assert np.all(mask[0:10, :] == 0)  # Top region
    assert np.all(mask[:, 0:10] == 0)  # Left region
    assert np.all(mask[60:, :] == 0)   # Bottom region
    assert np.all(mask[:, 60:] == 0)   # Right region
    Path(config_path).unlink()  # Cleanup


# AC-1.1.3: Boundary Validation Tests

def test_roi_within_bounds():
    """Verify mask generation succeeds when ROI fully inside image."""
    # Arrange
    roi = {'x': 100, 'y': 50, 'width': 200, 'height': 150}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert np.all(mask[50:200, 100:300] == 255)  # ROI region is static
    Path(config_path).unlink()  # Cleanup


def test_roi_at_boundaries():
    """Verify mask generation succeeds when ROI touches image edges."""
    # Arrange
    roi = {'x': 0, 'y': 0, 'width': 640, 'height': 480}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert np.all(mask == 255)  # Entire image is static region
    Path(config_path).unlink()  # Cleanup


def test_roi_exceeds_width():
    """Verify ValueError when ROI extends beyond image width."""
    # Arrange
    roi = {'x': 500, 'y': 50, 'width': 200, 'height': 150}  # 500 + 200 = 700 > 640
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act & Assert
    with pytest.raises(ValueError, match="ROI exceeds image bounds"):
        manager.get_static_mask((480, 640))

    Path(config_path).unlink()  # Cleanup


def test_roi_exceeds_height():
    """Verify ValueError when ROI extends beyond image height."""
    # Arrange
    roi = {'x': 100, 'y': 400, 'width': 200, 'height': 150}  # 400 + 150 = 550 > 480
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act & Assert
    with pytest.raises(ValueError, match="ROI exceeds image bounds"):
        manager.get_static_mask((480, 640))

    Path(config_path).unlink()  # Cleanup


def test_roi_exceeds_both():
    """Verify ValueError when ROI exceeds both dimensions."""
    # Arrange
    roi = {'x': 500, 'y': 400, 'width': 200, 'height': 150}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act & Assert
    with pytest.raises(ValueError, match="ROI exceeds image bounds"):
        manager.get_static_mask((480, 640))

    Path(config_path).unlink()  # Cleanup


# AC-1.1.4: Error Handling Tests

def test_invalid_image_shape_not_tuple():
    """Verify ValueError when image_shape is not a tuple."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)
    invalid_shape = [480, 640]  # List instead of tuple

    # Act & Assert
    with pytest.raises(ValueError, match="expected tuple"):
        manager.get_static_mask(invalid_shape)

    Path(config_path).unlink()  # Cleanup


def test_invalid_image_shape_wrong_length():
    """Verify ValueError when image_shape has wrong number of elements."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)
    wrong_length = (480, 640, 3)  # 3 elements instead of 2

    # Act & Assert
    with pytest.raises(ValueError, match="expected \\(height, width\\)"):
        manager.get_static_mask(wrong_length)

    Path(config_path).unlink()  # Cleanup


def test_invalid_image_shape_non_integers():
    """Verify ValueError when dimensions are not integers."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)
    float_dimensions = (480.5, 640.0)  # Float instead of int

    # Act & Assert
    with pytest.raises(ValueError, match="dimensions must be integers"):
        manager.get_static_mask(float_dimensions)

    Path(config_path).unlink()  # Cleanup


def test_invalid_image_shape_zero_height():
    """Verify ValueError when height is zero."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)
    zero_height = (0, 640)

    # Act & Assert
    with pytest.raises(ValueError, match="dimensions must be positive"):
        manager.get_static_mask(zero_height)

    Path(config_path).unlink()  # Cleanup


def test_invalid_image_shape_negative_width():
    """Verify ValueError when width is negative."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 50, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)
    negative_width = (480, -640)

    # Act & Assert
    with pytest.raises(ValueError, match="dimensions must be positive"):
        manager.get_static_mask(negative_width)

    Path(config_path).unlink()  # Cleanup


# Edge Cases

def test_edge_case_1x1_roi():
    """Verify handling of minimal 1×1 ROI."""
    # Arrange
    roi = {'x': 100, 'y': 100, 'width': 1, 'height': 1}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert mask[100, 100] == 255  # Single pixel is static
    assert np.sum(mask == 255) == 1  # Only 1 pixel is static
    Path(config_path).unlink()  # Cleanup


def test_edge_case_full_image_roi():
    """Verify ROI can equal full image dimensions."""
    # Arrange
    roi = {'x': 0, 'y': 0, 'width': 640, 'height': 480}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert np.all(mask == 255)  # Entire image is static
    Path(config_path).unlink()  # Cleanup


def test_edge_case_non_square_roi():
    """Verify non-square aspect ratios (e.g., 100×50)."""
    # Arrange
    roi = {'x': 10, 'y': 10, 'width': 100, 'height': 50}
    config_path = create_temp_config(roi)
    manager = StaticRegionManager(config_path)

    # Act
    mask = manager.get_static_mask((480, 640))

    # Assert
    assert mask.shape == (480, 640)
    assert np.all(mask[10:60, 10:110] == 255)  # ROI region is static
    assert np.sum(mask == 255) == 100 * 50  # 5000 static pixels
    Path(config_path).unlink()  # Cleanup
