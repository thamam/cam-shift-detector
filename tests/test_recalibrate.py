"""Unit tests for Recalibration Script

Tests for recalibrate.py focusing on CLI interface, delegation to intrinsic method,
and error handling. Tests verify thin wrapper pattern with NO validation logic in script.
"""

import json
import numpy as np
import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Import functions from recalibrate tool
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))
from recalibrate import load_image, parse_arguments


class TestCLIInterface:
    """Test CLI argument parsing and validation (AC-1.7.1)"""

    def test_parse_arguments_with_required_args(self):
        """AC-1.7.1: Test --config and --image arguments parsed correctly"""
        test_args = ['--config', 'config.json', '--image', 'frame.jpg']

        with patch('sys.argv', ['recalibrate.py'] + test_args):
            args = parse_arguments()

        assert args.config == 'config.json'
        assert args.image == 'frame.jpg'
        assert args.clear_history is False  # default

    def test_parse_arguments_with_clear_history_flag(self):
        """AC-1.7.1, AC-1.7.8: Test --clear-history flag optional and defaults to False"""
        test_args = ['--config', 'config.json', '--image', 'frame.jpg', '--clear-history']

        with patch('sys.argv', ['recalibrate.py'] + test_args):
            args = parse_arguments()

        assert args.clear_history is True

    def test_parse_arguments_missing_config_exits(self):
        """AC-1.7.1: Test missing --config argument raises SystemExit"""
        test_args = ['--image', 'frame.jpg']

        with patch('sys.argv', ['recalibrate.py'] + test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_arguments_missing_image_exits(self):
        """AC-1.7.1: Test missing --image argument raises SystemExit"""
        test_args = ['--config', 'config.json']

        with patch('sys.argv', ['recalibrate.py'] + test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_help_message_displays(self, capsys):
        """AC-1.7.1: Test --help displays usage instructions"""
        with patch('sys.argv', ['recalibrate.py', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                parse_arguments()

        assert exc_info.value.code == 0  # --help exits with 0
        captured = capsys.readouterr()
        assert 'Manual recalibration tool' in captured.out
        assert '--config' in captured.out
        assert '--image' in captured.out
        assert '--clear-history' in captured.out


class TestImageLoading:
    """Test image loading and validation (AC-1.7.3)"""

    def test_load_valid_image(self, sample_image_path):
        """AC-1.7.3: Test load_image() loads valid image successfully"""
        image = load_image(sample_image_path)

        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8

    def test_load_missing_image_raises_filenotfound(self):
        """AC-1.7.7: Test load_image() raises FileNotFoundError for missing file"""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_image("nonexistent_image.jpg")

        assert "Image file not found" in str(exc_info.value)

    def test_load_corrupted_image_raises_valueerror(self, tmp_path):
        """AC-1.7.7: Test load_image() raises ValueError for corrupted image"""
        # Create a corrupted image file (not valid image data)
        corrupted_file = tmp_path / "corrupted.jpg"
        corrupted_file.write_text("This is not an image")

        with pytest.raises(ValueError) as exc_info:
            load_image(str(corrupted_file))

        assert "Failed to load image" in str(exc_info.value)
        assert "corrupted" in str(exc_info.value)


class TestRecalibrationSuccess:
    """Test successful recalibration workflow (AC-1.7.4, AC-1.7.5, AC-1.7.6)"""

    @patch('recalibrate.CameraMovementDetector')
    def test_recalibrate_success_with_valid_image(self, mock_detector_class, sample_image_path, tmp_path, capsys):
        """AC-1.7.4, AC-1.7.5, AC-1.7.6: Test successful recalibration with ≥50 features"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }))

        # Mock detector instance that returns success
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = True
        mock_detector.result_manager.history = MagicMock()
        mock_detector_class.return_value = mock_detector

        # Run recalibrate script
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Verify exit code 0 on success
        assert exc_info.value.code == 0

        # Verify detector.recalibrate() was called
        mock_detector.recalibrate.assert_called_once()

        # Verify success message printed
        captured = capsys.readouterr()
        assert "✓ Recalibration successful" in captured.out
        # Timestamp present (ISO format with timezone)
        assert "+" in captured.out or "Z" in captured.out

    @patch('recalibrate.CameraMovementDetector')
    def test_exit_code_zero_on_success(self, mock_detector_class, sample_image_path, tmp_path):
        """AC-1.7.6: Test exit code 0 on successful recalibration"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector that returns success
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = True
        mock_detector.result_manager.history = MagicMock()
        mock_detector_class.return_value = mock_detector

        # Run script
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        assert exc_info.value.code == 0


class TestRecalibrationFailure:
    """Test recalibration failure with insufficient features (AC-1.7.5, AC-1.7.6)"""

    @patch('recalibrate.CameraMovementDetector')
    def test_recalibrate_failure_insufficient_features(self, mock_detector_class, sample_image_path, tmp_path, capsys):
        """AC-1.7.5, AC-1.7.6: Test recalibration failure with <50 features"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector that returns failure
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = False
        mock_detector_class.return_value = mock_detector

        # Run script
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Verify exit code 1 on failure
        assert exc_info.value.code == 1

        # Verify failure message with reason
        captured = capsys.readouterr()
        assert "✗ Recalibration failed" in captured.out
        assert "Insufficient features" in captured.out
        assert "<50 required" in captured.out
        assert "Try a different image" in captured.out

    @patch('recalibrate.CameraMovementDetector')
    def test_exit_code_one_on_failure(self, mock_detector_class, sample_image_path, tmp_path):
        """AC-1.7.6: Test exit code 1 on recalibration failure"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector that returns failure
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = False
        mock_detector_class.return_value = mock_detector

        # Run script
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        assert exc_info.value.code == 1


class TestHistoryClearing:
    """Test optional history buffer clearing functionality (AC-1.7.8)"""

    @patch('recalibrate.CameraMovementDetector')
    def test_clear_history_flag_clears_buffer_on_success(self, mock_detector_class, sample_image_path, tmp_path, capsys):
        """AC-1.7.8: Test --clear-history flag clears buffer after successful recalibration"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector with history buffer
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = True
        mock_history = MagicMock()
        mock_detector.result_manager.history = mock_history
        mock_detector_class.return_value = mock_detector

        # Run script with --clear-history flag
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path, '--clear-history']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                from recalibrate import main
                main()

        # Verify history.clear() was called
        mock_history.clear.assert_called_once()

        # Verify clear message printed
        captured = capsys.readouterr()
        assert "✓ Detection history buffer cleared" in captured.out

    @patch('recalibrate.CameraMovementDetector')
    def test_history_not_cleared_without_flag(self, mock_detector_class, sample_image_path, tmp_path):
        """AC-1.7.8: Test history NOT cleared if flag not provided"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector with history buffer
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = True
        mock_history = MagicMock()
        mock_detector.result_manager.history = mock_history
        mock_detector_class.return_value = mock_detector

        # Run script WITHOUT --clear-history flag
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                from recalibrate import main
                main()

        # Verify history.clear() was NOT called
        mock_history.clear.assert_not_called()

    @patch('recalibrate.CameraMovementDetector')
    def test_history_not_cleared_on_failure(self, mock_detector_class, sample_image_path, tmp_path):
        """AC-1.7.8: Test history NOT cleared on recalibration failure"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector that returns failure
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = False
        mock_history = MagicMock()
        mock_detector.result_manager.history = mock_history
        mock_detector_class.return_value = mock_detector

        # Run script with --clear-history flag (but recalibration fails)
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path, '--clear-history']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                from recalibrate import main
                main()

        # Verify history.clear() was NOT called on failure
        mock_history.clear.assert_not_called()


class TestErrorHandling:
    """Test error handling for missing files and invalid inputs (AC-1.7.7)"""

    def test_missing_config_file_error(self, sample_image_path, capsys):
        """AC-1.7.7: Test missing config file raises clear error with exit code 1"""
        test_args = ['recalibrate.py', '--config', 'nonexistent_config.json', '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Verify exit code 1
        assert exc_info.value.code == 1

        # Verify clear error message
        captured = capsys.readouterr()
        assert "Error: Config file not found" in captured.out
        assert "nonexistent_config.json" in captured.out

    def test_missing_image_file_error(self, tmp_path, capsys):
        """AC-1.7.7: Test missing image file raises clear error with exit code 1"""
        # Create valid config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        test_args = ['recalibrate.py', '--config', str(config_path), '--image', 'nonexistent_image.jpg']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Verify exit code 1
        assert exc_info.value.code == 1

        # Verify clear error message
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "Image file not found" in captured.out

    def test_corrupted_image_file_error(self, tmp_path, capsys):
        """AC-1.7.7: Test corrupted image file raises clear error"""
        # Create valid config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Create corrupted image
        corrupted_image = tmp_path / "corrupted.jpg"
        corrupted_image.write_text("Not an image")

        test_args = ['recalibrate.py', '--config', str(config_path), '--image', str(corrupted_image)]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Verify exit code 1
        assert exc_info.value.code == 1

        # Verify error message mentions corruption
        captured = capsys.readouterr()
        assert "Error:" in captured.out
        assert "corrupted" in captured.out

    @patch('recalibrate.CameraMovementDetector')
    def test_detector_initialization_error(self, mock_detector_class, sample_image_path, tmp_path, capsys):
        """AC-1.7.7: Test detector initialization errors handled gracefully"""
        # Create valid config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector initialization that raises exception
        mock_detector_class.side_effect = Exception("Invalid config schema")

        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Verify exit code 1
        assert exc_info.value.code == 1

        # Verify error message
        captured = capsys.readouterr()
        assert "Error: Failed to initialize detector" in captured.out


class TestDelegationToIntrinsicMethod:
    """Test proper delegation to detector.recalibrate() with NO validation in script (AC-1.7.4)"""

    @patch('recalibrate.CameraMovementDetector')
    def test_script_calls_detector_recalibrate(self, mock_detector_class, sample_image_path, tmp_path):
        """AC-1.7.4: Test script calls detector.recalibrate() with loaded image"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Mock detector
        mock_detector = Mock()
        mock_detector.recalibrate.return_value = True
        mock_detector.result_manager.history = MagicMock()
        mock_detector_class.return_value = mock_detector

        # Run script
        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                from recalibrate import main
                main()

        # Verify detector.recalibrate() was called exactly once
        assert mock_detector.recalibrate.call_count == 1

        # Verify it was called with an image array (numpy array)
        call_args = mock_detector.recalibrate.call_args[0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], np.ndarray)

    @patch('recalibrate.CameraMovementDetector')
    def test_script_respects_boolean_return_value(self, mock_detector_class, sample_image_path, tmp_path):
        """AC-1.7.4: Test script respects boolean return value from method"""
        # Create mock config
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300}
        }))

        # Test success path
        mock_detector_success = Mock()
        mock_detector_success.recalibrate.return_value = True
        mock_detector_success.result_manager.history = MagicMock()
        mock_detector_class.return_value = mock_detector_success

        test_args = ['recalibrate.py', '--config', str(config_path), '--image', sample_image_path]

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                from recalibrate import main
                main()

        # Success returns exit code 0
        assert exc_info.value.code == 0

        # Test failure path
        mock_detector_failure = Mock()
        mock_detector_failure.recalibrate.return_value = False
        mock_detector_class.return_value = mock_detector_failure

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit) as exc_info:
                # Re-import to reset module state
                import importlib
                import recalibrate as recal_module
                importlib.reload(recal_module)
                recal_module.main()

        # Failure returns exit code 1
        assert exc_info.value.code == 1


# Fixtures
@pytest.fixture
def sample_image_path():
    """Provide path to a real sample image for testing"""
    # Use an image from the sample_images directory
    base_path = Path(__file__).parent.parent / "sample_images"

    # Try different locations
    for subdir in ["of_jerusalem", "carmit", "gad"]:
        image_dir = base_path / subdir
        if image_dir.exists():
            images = list(image_dir.glob("*.jpg"))
            if images:
                return str(images[0])

    pytest.skip("No sample images found for testing")
