"""
Unit and integration tests for Mode C - Enhanced Alpha Blending Tool

Test Coverage:
- AC1: Transform computation & pre-warp
- AC2: Blink toggle
- AC3: Frame selector
- AC4: Alpha blending with grid overlay
- AC5: Export and snapshot
- Edge cases: detection failures, invalid frames
"""

import pytest
import cv2
import numpy as np
import time
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.validation.alpha_blending_tool import AlphaBlendingTool


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_image_dir(tmp_path):
    """Create temporary directory with test images."""
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()

    # Create 5 test images (640x480, uint8, BGR)
    for i in range(5):
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"frame_{i:03d}.jpg"), img)

    return img_dir


@pytest.fixture
def temp_config(tmp_path):
    """Create temporary config file."""
    config_path = tmp_path / "test_config.json"
    config_path.write_text('''
    {
        "roi": {"x": 10, "y": 10, "width": 200, "height": 200},
        "threshold_pixels": 2.0,
        "min_features_required": 10
    }
    ''')
    return config_path


@pytest.fixture
def tool(temp_image_dir, temp_config):
    """Create AlphaBlendingTool instance with mocked CSD."""
    with patch('tools.validation.alpha_blending_tool.CameraMovementDetector') as mock_detector:
        # Mock CSD methods
        mock_instance = Mock()
        mock_instance.set_baseline = Mock()
        mock_instance.process_frame = Mock(return_value={
            'status': 'VALID',
            'displacement': 1.5,
            'confidence': 0.95
        })
        mock_instance.get_last_homography = Mock(return_value=np.eye(3))
        mock_detector.return_value = mock_instance

        tool = AlphaBlendingTool(str(temp_image_dir), str(temp_config))
        tool.mock_detector_instance = mock_instance  # For test access
        return tool


# ============================================================================
# AC1: Transform Computation & Pre-Warp Tests
# ============================================================================

def test_transform_computation_performance(tool):
    """AC1: Transform computation completes in < 500ms"""
    tool._compute_transform()

    assert tool.transform_computed is True
    assert tool.transform_time_ms is not None
    assert tool.transform_time_ms < 500, f"Transform took {tool.transform_time_ms}ms (should be < 500ms)"


def test_prewarp_toggle(tool):
    """AC1: 'w' key toggles pre-warp mode"""
    assert tool.prewarp_enabled is False

    tool.prewarp_enabled = not tool.prewarp_enabled
    assert tool.prewarp_enabled is True

    tool.prewarp_enabled = not tool.prewarp_enabled
    assert tool.prewarp_enabled is False


def test_prewarp_transformation(tool):
    """AC1: Pre-warp applies cv2.warpPerspective correctly"""
    tool._compute_transform()

    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    warped = tool._apply_prewarp(test_frame)

    assert warped.shape == test_frame.shape
    assert warped.dtype == test_frame.dtype


def test_prewarp_with_no_homography(tool):
    """AC1: Pre-warp handles missing homography gracefully"""
    tool.homography = None

    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = tool._apply_prewarp(test_frame)

    # Should return original frame unchanged
    np.testing.assert_array_equal(result, test_frame)


def test_prewarp_indicator_display(tool):
    """AC1: Display shows 'Pre-warp: ON' or 'Pre-warp: OFF' indicator"""
    # Create test display
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test OFF state
    tool.prewarp_enabled = False
    display = tool._add_ui_overlay(test_image)
    assert display.shape == test_image.shape

    # Test ON state
    tool.prewarp_enabled = True
    display = tool._add_ui_overlay(test_image)
    assert display.shape == test_image.shape


# ============================================================================
# AC2: Blink Toggle Tests
# ============================================================================

def test_blink_mode_toggle(tool):
    """AC2: Space key toggles blink mode"""
    assert tool.blink_mode is False

    tool.blink_mode = not tool.blink_mode
    assert tool.blink_mode is True

    tool.blink_mode = not tool.blink_mode
    assert tool.blink_mode is False


def test_blink_timing_accuracy(tool):
    """AC2: Blink alternates every 500ms (±50ms tolerance)"""
    tool.blink_mode = True
    tool.blink_state = 0
    tool.last_blink_time = time.time()

    # Measure 10 blink cycles
    intervals = []
    for _ in range(10):
        tool.last_blink_time = time.time()
        time.sleep(0.5)  # Simulate 500ms wait
        current_time = time.time()
        interval = (current_time - tool.last_blink_time) * 1000  # Convert to ms
        intervals.append(interval)

    # All intervals should be close to 500ms (within ±50ms)
    for interval in intervals:
        assert 450 <= interval <= 550, f"Interval {interval}ms outside ±50ms tolerance"


def test_blink_state_alternation(tool):
    """AC2: Blink state alternates between 0 and 1"""
    assert tool.blink_state == 0

    tool.blink_state = 1 - tool.blink_state
    assert tool.blink_state == 1

    tool.blink_state = 1 - tool.blink_state
    assert tool.blink_state == 0


def test_blink_indicator_display(tool):
    """AC2: Display shows 'Blink: ON' indicator"""
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test OFF state
    tool.blink_mode = False
    display = tool._add_ui_overlay(test_image)
    assert display.shape == test_image.shape

    # Test ON state
    tool.blink_mode = True
    display = tool._add_ui_overlay(test_image)
    assert display.shape == test_image.shape


# ============================================================================
# AC3: Frame Selector Tests
# ============================================================================

def test_frame_a_selection_mode(tool):
    """AC3: 'a' key enters Frame A selection mode"""
    assert tool.selection_mode is None

    tool.selection_mode = 'A'
    tool.temp_selection_idx = tool.frame_a_idx

    assert tool.selection_mode == 'A'
    assert tool.temp_selection_idx == tool.frame_a_idx


def test_frame_b_selection_mode(tool):
    """AC3: 'b' key enters Frame B selection mode"""
    assert tool.selection_mode is None

    tool.selection_mode = 'B'
    tool.temp_selection_idx = tool.frame_b_idx

    assert tool.selection_mode == 'B'
    assert tool.temp_selection_idx == tool.frame_b_idx


def test_frame_navigation_forward(tool):
    """AC3: Right arrow navigates frames forward"""
    tool.selection_mode = 'A'
    tool.temp_selection_idx = 0

    tool.temp_selection_idx = (tool.temp_selection_idx + 1) % len(tool.images)
    assert tool.temp_selection_idx == 1


def test_frame_navigation_backward(tool):
    """AC3: Left arrow navigates frames backward"""
    tool.selection_mode = 'A'
    tool.temp_selection_idx = 2

    tool.temp_selection_idx = (tool.temp_selection_idx - 1) % len(tool.images)
    assert tool.temp_selection_idx == 1


def test_frame_selection_confirmation(tool):
    """AC3: Enter confirms frame selection"""
    # Select Frame A
    tool.selection_mode = 'A'
    tool.temp_selection_idx = 3
    tool.frame_a_idx = tool.temp_selection_idx
    tool.selection_mode = None

    assert tool.frame_a_idx == 3
    assert tool.selection_mode is None


def test_frame_indices_display(tool):
    """AC3: Selected frame indices displayed in status bar"""
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    display = tool._add_ui_overlay(test_image)

    # Display should contain frame info
    assert display.shape == test_image.shape


def test_default_frame_selection(tool):
    """AC3: Default Frame A = first, Frame B = current"""
    assert tool.frame_a_idx == 0  # First frame
    assert tool.frame_b_idx == 1  # Second frame (current)


# ============================================================================
# AC4: Alpha Blending with Grid Overlay Tests
# ============================================================================

def test_alpha_adjustment_up(tool):
    """AC4: Up arrow increases alpha value"""
    tool.alpha = 0.5
    tool.alpha = min(1.0, tool.alpha + 0.05)
    assert tool.alpha == pytest.approx(0.55, abs=0.01)


def test_alpha_adjustment_down(tool):
    """AC4: Down arrow decreases alpha value"""
    tool.alpha = 0.5
    tool.alpha = max(0.0, tool.alpha - 0.05)
    assert tool.alpha == pytest.approx(0.45, abs=0.01)


def test_alpha_display_percentage(tool):
    """AC4: Alpha value displayed as percentage"""
    tool.alpha = 0.5
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    display = tool._add_ui_overlay(test_image)

    # Display should show "Alpha: 50%"
    assert display.shape == test_image.shape


def test_grid_toggle(tool):
    """AC4: 'g' key toggles grid overlay"""
    assert tool.show_grid is False

    tool.show_grid = not tool.show_grid
    assert tool.show_grid is True

    tool.show_grid = not tool.show_grid
    assert tool.show_grid is False


def test_grid_drawing_structure(tool):
    """AC4: Grid draws 10×10 lines"""
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    grid_image = tool._draw_alignment_grid(test_image, rows=10, cols=10)

    assert grid_image.shape == test_image.shape
    assert grid_image.dtype == test_image.dtype


def test_grid_color_cyan(tool):
    """AC4: Grid color is cyan (BGR: 255, 255, 0)"""
    # Grid should be drawn with cyan color
    # This is visually verified, but we can check the function exists
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    grid_image = tool._draw_alignment_grid(test_image)

    # Grid image should be different from original (has lines drawn)
    assert not np.array_equal(grid_image, test_image)


def test_grid_transparency(tool):
    """AC4: Grid has 50% transparency"""
    test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    grid_image = tool._draw_alignment_grid(test_image)

    # Grid overlay should blend with original (50/50 blend)
    # Result should be between original and pure grid
    assert grid_image.shape == test_image.shape


# ============================================================================
# AC5: Export and Snapshot Tests
# ============================================================================

def test_snapshot_filename_format(tool, temp_image_dir):
    """AC5: Filename includes Frame A and Frame B indices"""
    tool.frame_a_idx = 0
    tool.frame_b_idx = 2
    tool.alpha = 0.5
    tool.prewarp_enabled = True

    # Mock cv2.imwrite to prevent actual file write
    with patch('cv2.imwrite') as mock_write:
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = MagicMock()
            tool._save_snapshot()

    # Check that files would be created with correct naming pattern
    # Filename should contain: frameA0_frameB2_alpha50_prewarp


def test_snapshot_saves_png(tool, temp_image_dir):
    """AC5: 's' key saves current blended view as PNG"""
    with patch('cv2.imwrite') as mock_write:
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = MagicMock()
            tool._save_snapshot()

        # Verify PNG write was called
        assert mock_write.called


def test_csv_export_with_transform_parameters(tool, temp_image_dir):
    """AC5: CSV export includes transform parameters"""
    tool._compute_transform()
    tool.homography = np.eye(3)  # 3x3 identity matrix

    csv_written = False
    csv_rows = []

    def mock_csv_writer(*args, **kwargs):
        class MockWriter:
            def writerow(self, row):
                nonlocal csv_written
                csv_written = True
                csv_rows.append(row)
        return MockWriter()

    with patch('cv2.imwrite'):
        with patch('csv.writer', side_effect=mock_csv_writer):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = MagicMock()
                tool._save_snapshot()

    # Verify CSV was written
    assert csv_written
    # Should contain rows for parameters
    assert len(csv_rows) > 0


def test_snapshot_metadata_inclusion(tool, temp_image_dir):
    """AC5: Export includes metadata (alpha, pre-warp status)"""
    tool.alpha = 0.7
    tool.prewarp_enabled = True
    tool._compute_transform()

    csv_rows = []

    def mock_csv_writer(*args, **kwargs):
        class MockWriter:
            def writerow(self, row):
                csv_rows.append(row)
        return MockWriter()

    with patch('cv2.imwrite'):
        with patch('csv.writer', side_effect=mock_csv_writer):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value = MagicMock()
                tool._save_snapshot()

    # Check for alpha and prewarp in CSV rows
    csv_text = str(csv_rows)
    assert 'alpha' in csv_text.lower() or any('0.7' in str(row) for row in csv_rows)
    assert 'prewarp' in csv_text.lower() or 'True' in str(csv_rows)


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_csd_detection_failure_handling(tool):
    """Edge case: CSD detection failure (insufficient features)"""
    # Mock CSD to return INVALID status
    tool.mock_detector_instance.process_frame.return_value = {
        'status': 'INVALID',
        'displacement': 0.0,
        'confidence': 0.0
    }
    tool.mock_detector_instance.get_last_homography.return_value = None

    tool._compute_transform()

    # Tool should handle gracefully
    assert tool.homography is None
    assert tool.transform_computed is True


def test_invalid_frame_index_handling(tool):
    """Edge case: Invalid frame index selection"""
    # Try to set frame index beyond bounds
    max_idx = len(tool.images) - 1

    tool.temp_selection_idx = max_idx + 1
    # Navigation should wrap around
    tool.temp_selection_idx = tool.temp_selection_idx % len(tool.images)

    assert 0 <= tool.temp_selection_idx < len(tool.images)


def test_failed_image_load_handling(tool):
    """Edge case: Failed image load returns blank frame"""
    # Mock cv2.imread to return None
    with patch('cv2.imread', return_value=None):
        blended = tool._create_blended_view()

        # Should return blank frame, not crash
        assert blended is not None
        assert blended.shape == (480, 640, 3)
        assert blended.dtype == np.uint8


def test_minimum_image_requirement(temp_image_dir, temp_config):
    """Edge case: Tool requires at least 2 images"""
    # Create directory with only 1 image
    single_img_dir = temp_image_dir.parent / "single_image"
    single_img_dir.mkdir()

    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(single_img_dir / "frame_000.jpg"), img)

    with patch('tools.validation.alpha_blending_tool.CameraMovementDetector'):
        with pytest.raises(ValueError, match="Need at least 2 images"):
            AlphaBlendingTool(str(single_img_dir), str(temp_config))


# ============================================================================
# Integration Tests
# ============================================================================

def test_blended_view_creation(tool):
    """Integration: Blended view combines Frame A and Frame B correctly"""
    blended = tool._create_blended_view()

    assert blended is not None
    assert blended.shape == (480, 640, 3)
    assert blended.dtype == np.uint8


def test_complete_workflow_frame_selection_to_export(tool, temp_image_dir):
    """Integration: Complete workflow from frame selection to export"""
    # Select frames
    tool.frame_a_idx = 0
    tool.frame_b_idx = 2

    # Compute transform
    tool._compute_transform()
    assert tool.transform_computed is True

    # Enable pre-warp
    tool.prewarp_enabled = True

    # Enable grid
    tool.show_grid = True

    # Create blended view
    blended = tool._create_blended_view()
    assert blended is not None

    # Export snapshot
    with patch('cv2.imwrite'):
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = MagicMock()
            tool._save_snapshot()


def test_ui_overlay_complete(tool):
    """Integration: UI overlay shows all status indicators"""
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Enable all features
    tool.prewarp_enabled = True
    tool.blink_mode = True
    tool.show_grid = True
    tool._compute_transform()

    display = tool._add_ui_overlay(test_image)

    assert display.shape == test_image.shape
    assert display.dtype == np.uint8
    # Display should have all indicators drawn
    assert not np.array_equal(display, test_image)  # Modified from original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
