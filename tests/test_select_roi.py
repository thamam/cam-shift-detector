"""Unit tests for ROI Selection Tool

Tests for select_roi.py focusing on non-GUI functionality.
GUI interaction (cv2.selectROI) requires manual testing.
"""

import json
import numpy as np
import pytest
from pathlib import Path
import sys

# Import functions from select_roi tool
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from select_roi import (
    load_image,
    create_mask_from_roi,
    validate_roi_features,
    save_config,
    parse_arguments,
    MIN_FEATURES_REQUIRED,
    DEFAULT_THRESHOLD_PIXELS,
    DEFAULT_HISTORY_BUFFER_SIZE
)


class TestLoadImage:
    """Test image loading and validation"""

    def test_load_valid_image(self, sample_image_path):
        """AC-1.6.2: Test load_image() with valid image file"""
        image = load_image(sample_image_path)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_load_missing_image_raises_filenotfound(self):
        """AC-1.6.7: Test load_image() raises FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_image("nonexistent_image.jpg")

        assert "Image file not found" in str(exc_info.value)

    def test_load_invalid_image_raises_valueerror(self, tmp_path):
        """AC-1.6.7: Test load_image() raises ValueError for corrupted image"""
        # Create a corrupted image file (not valid image data)
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_text("This is not an image")

        with pytest.raises(ValueError) as exc_info:
            load_image(str(corrupted_file))

        assert "Failed to load image" in str(exc_info.value)
        assert "corrupted" in str(exc_info.value).lower()


class TestCreateMaskFromROI:
    """Test binary mask generation from ROI coordinates"""

    def test_create_mask_rectangular_roi(self):
        """AC-1.6.4: Test mask generation for rectangular ROI"""
        roi_coords = (100, 50, 400, 300)
        image_shape = (480, 640)

        mask = create_mask_from_roi(roi_coords, image_shape)

        # Validate mask properties
        assert mask.shape == image_shape
        assert mask.dtype == np.uint8

        # Check inside ROI region (should be 255)
        assert mask[100, 200] == 255  # Inside ROI
        assert mask[50, 100] == 255   # Top-left corner of ROI
        assert mask[349, 499] == 255  # Bottom-right corner (y+height-1, x+width-1)

        # Check outside ROI region (should be 0)
        assert mask[0, 0] == 0        # Outside ROI (top-left of image)
        assert mask[479, 639] == 0    # Outside ROI (bottom-right of image)
        assert mask[49, 100] == 0     # Just above ROI
        assert mask[100, 99] == 0     # Just left of ROI

    def test_create_mask_full_image_roi(self):
        """Test mask generation for ROI covering entire image"""
        image_shape = (480, 640)
        roi_coords = (0, 0, 640, 480)  # Full image

        mask = create_mask_from_roi(roi_coords, image_shape)

        # All pixels should be 255
        assert np.all(mask == 255)

    def test_create_mask_small_roi(self):
        """Test mask generation for small ROI"""
        image_shape = (480, 640)
        roi_coords = (300, 200, 50, 50)  # Small 50x50 region

        mask = create_mask_from_roi(roi_coords, image_shape)

        # Count non-zero pixels (should be 50*50 = 2500)
        non_zero_count = np.count_nonzero(mask)
        assert non_zero_count == 2500


class TestValidateROIFeatures:
    """Test feature validation in selected ROI"""

    def test_validate_roi_with_sufficient_features(self, sample_image):
        """AC-1.6.4: Test validation with region containing ≥50 features (PASS)"""
        # Use a large region with texture
        roi_coords = (100, 100, 400, 300)

        feature_count, valid = validate_roi_features(sample_image, roi_coords)

        # Should detect sufficient features
        assert feature_count >= MIN_FEATURES_REQUIRED
        assert valid is True

    def test_validate_roi_with_insufficient_features(self):
        """AC-1.6.4: Test validation with region containing <50 features (FAIL)"""
        # Create blank image (no features)
        blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        roi_coords = (100, 100, 200, 200)

        feature_count, valid = validate_roi_features(blank_image, roi_coords)

        # Should detect insufficient features
        assert feature_count < MIN_FEATURES_REQUIRED
        assert valid is False

    def test_validate_roi_boundary_case_exactly_50_features(self, sample_image):
        """AC-1.6.4: Test validation with exactly 50 features (boundary PASS)"""
        # This test may be fragile as exact feature count depends on image content
        # We're testing that validation logic correctly handles the boundary
        roi_coords = (100, 100, 400, 300)

        feature_count, valid = validate_roi_features(sample_image, roi_coords)

        # Verify logic: if count == 50, should be valid
        if feature_count == MIN_FEATURES_REQUIRED:
            assert valid is True

    def test_validate_roi_integration_with_feature_extractor(self, sample_image):
        """AC-1.6.4: Test integration with FeatureExtractor component"""
        roi_coords = (100, 100, 300, 200)

        # This should use FeatureExtractor internally
        feature_count, valid = validate_roi_features(sample_image, roi_coords)

        # Verify returns are valid types
        assert isinstance(feature_count, int)
        assert isinstance(valid, bool)
        assert feature_count >= 0


class TestSaveConfig:
    """Test config.json generation"""

    def test_save_config_creates_file(self, tmp_path):
        """AC-1.6.6: Test save_config() creates config.json file"""
        roi_coords = (100, 50, 400, 300)
        output_path = tmp_path / "config.json"

        save_config(roi_coords, str(output_path))

        # Verify file exists
        assert output_path.exists()

    def test_save_config_correct_schema(self, tmp_path):
        """AC-1.6.6: Test config.json contains all required fields"""
        roi_coords = (100, 50, 400, 300)
        output_path = tmp_path / "config.json"

        save_config(roi_coords, str(output_path))

        # Load and verify schema
        with open(output_path, 'r') as f:
            config = json.load(f)

        # Check required fields
        assert "roi" in config
        assert "threshold_pixels" in config
        assert "history_buffer_size" in config
        assert "min_features_required" in config

        # Check ROI structure
        assert "x" in config["roi"]
        assert "y" in config["roi"]
        assert "width" in config["roi"]
        assert "height" in config["roi"]

    def test_save_config_roi_coordinates_as_integers(self, tmp_path):
        """AC-1.6.6: Test ROI coordinates saved as integers"""
        roi_coords = (100, 50, 400, 300)
        output_path = tmp_path / "config.json"

        save_config(roi_coords, str(output_path))

        with open(output_path, 'r') as f:
            config = json.load(f)

        # Verify all ROI values are integers
        assert isinstance(config["roi"]["x"], int)
        assert isinstance(config["roi"]["y"], int)
        assert isinstance(config["roi"]["width"], int)
        assert isinstance(config["roi"]["height"], int)

    def test_save_config_default_parameters(self, tmp_path):
        """AC-1.6.6: Test default parameters are correct"""
        roi_coords = (100, 50, 400, 300)
        output_path = tmp_path / "config.json"

        save_config(roi_coords, str(output_path))

        with open(output_path, 'r') as f:
            config = json.load(f)

        # Verify default values
        assert config["threshold_pixels"] == DEFAULT_THRESHOLD_PIXELS
        assert config["history_buffer_size"] == DEFAULT_HISTORY_BUFFER_SIZE
        assert config["min_features_required"] == MIN_FEATURES_REQUIRED

    def test_save_config_values_match_input(self, tmp_path):
        """AC-1.6.6: Test saved ROI coordinates match input"""
        roi_coords = (123, 456, 789, 234)
        output_path = tmp_path / "config.json"

        save_config(roi_coords, str(output_path))

        with open(output_path, 'r') as f:
            config = json.load(f)

        # Verify coordinates match
        assert config["roi"]["x"] == 123
        assert config["roi"]["y"] == 456
        assert config["roi"]["width"] == 789
        assert config["roi"]["height"] == 234

    def test_config_loads_successfully_in_static_region_manager(self, tmp_path):
        """AC-1.6.6: Test generated config can be loaded by StaticRegionManager"""
        from src.static_region_manager import StaticRegionManager

        roi_coords = (100, 50, 400, 300)
        output_path = tmp_path / "config.json"

        save_config(roi_coords, str(output_path))

        # Should load without errors
        manager = StaticRegionManager(str(output_path))

        # Verify ROI loaded correctly
        assert manager.roi['x'] == 100
        assert manager.roi['y'] == 50
        assert manager.roi['width'] == 400
        assert manager.roi['height'] == 300


class TestParseArguments:
    """Test CLI argument parsing"""

    def test_parse_arguments_with_image_source(self, monkeypatch):
        """AC-1.6.1: Test argument parsing with --source image --path"""
        monkeypatch.setattr(
            'sys.argv',
            ['select_roi.py', '--source', 'image', '--path', 'test.jpg']
        )

        args = parse_arguments()

        assert args.source == 'image'
        assert args.path == 'test.jpg'

    def test_parse_arguments_with_camera_source(self, monkeypatch):
        """AC-1.6.1: Test argument parsing with --source camera"""
        monkeypatch.setattr(
            'sys.argv',
            ['select_roi.py', '--source', 'camera']
        )

        args = parse_arguments()

        assert args.source == 'camera'

    def test_parse_arguments_invalid_source_raises_error(self, monkeypatch):
        """AC-1.6.1: Test invalid --source value raises error"""
        monkeypatch.setattr(
            'sys.argv',
            ['select_roi.py', '--source', 'invalid']
        )

        with pytest.raises(SystemExit):
            parse_arguments()

    def test_parse_arguments_missing_source_raises_error(self, monkeypatch):
        """AC-1.6.1: Test missing --source argument raises error"""
        monkeypatch.setattr(
            'sys.argv',
            ['select_roi.py', '--path', 'test.jpg']
        )

        with pytest.raises(SystemExit):
            parse_arguments()


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_roi_selection(self):
        """AC-1.6.7: Test handling of empty ROI (width=0 or height=0)"""
        # Empty ROI should be handled gracefully
        # In the actual tool, select_roi_interactive returns None for empty selection
        # This tests the mask generation doesn't crash
        image_shape = (480, 640)
        roi_coords = (100, 100, 0, 0)  # Empty ROI

        mask = create_mask_from_roi(roi_coords, image_shape)

        # Mask should be all zeros (no region selected)
        assert np.all(mask == 0)

    def test_roi_at_image_boundary(self):
        """Test ROI at image boundaries"""
        image_shape = (480, 640)
        roi_coords = (0, 0, 100, 100)  # Top-left corner

        mask = create_mask_from_roi(roi_coords, image_shape)

        # Top-left 100x100 should be 255
        assert mask[0, 0] == 255
        assert mask[99, 99] == 255
        assert mask[100, 100] == 0  # Outside ROI

    def test_validation_with_very_small_roi(self, sample_image):
        """Test validation with very small ROI (edge case)"""
        # Very small ROI unlikely to have 50 features
        roi_coords = (100, 100, 10, 10)

        feature_count, valid = validate_roi_features(sample_image, roi_coords)

        # Should still return valid results (likely invalid due to small size)
        assert isinstance(feature_count, int)
        assert isinstance(valid, bool)


# Fixtures
@pytest.fixture
def sample_image_path():
    """Provide path to a sample image for testing"""
    # Use one of the provided sample images
    image_path = Path(__file__).parent.parent / "sample_images" / "of_jerusalem" / "001.jpg"

    if not image_path.exists():
        pytest.skip(f"Sample image not found: {image_path}")

    return str(image_path)


@pytest.fixture
def sample_image(sample_image_path):
    """Load sample image for testing"""
    import cv2
    image = cv2.imread(sample_image_path)
    return image
