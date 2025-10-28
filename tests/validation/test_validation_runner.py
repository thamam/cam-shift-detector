"""
Unit tests for ValidationRunner (validation/run_stage3_validation.py)

Tests cover acceptance criteria for Story 3:
- AC1: Validation Runner Orchestration
- AC5: Framework Quality Assurance
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from validation.run_stage3_validation import ValidationRunner, parse_arguments, main
from validation.stage3_test_harness import Metrics
from validation.performance_profiler import PerformanceMetrics


class TestValidationRunnerInitialization:
    """Test ValidationRunner initialization."""

    def test_initialization_with_defaults(self, tmp_path):
        """Test runner initializes with default output directory."""
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json"
        )

        assert runner.detector_config_path == "config/detector_config.json"
        assert runner.baseline_image_path is None
        assert runner.output_dir == Path("validation/results")

    def test_initialization_with_custom_params(self, tmp_path):
        """Test runner initializes with custom parameters."""
        baseline = Path("custom/baseline.jpg")
        output_dir = tmp_path / "custom_output"

        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            baseline_image_path=baseline,
            output_dir=output_dir
        )

        assert runner.baseline_image_path == baseline
        assert runner.output_dir == output_dir

    def test_output_directory_created(self, tmp_path):
        """Test output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_directory"
        assert not output_dir.exists()

        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )

        assert output_dir.exists()


class TestSequentialWorkflowExecution:
    """Test sequential workflow execution (AC1: Sequential execution)."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_sequential_workflow_order(self, mock_harness_class, tmp_path):
        """Test workflow executes in correct sequence: Load → Harness → Reports."""
        # Setup mock harness
        mock_harness = Mock()
        mock_metrics = self._create_mock_metrics()
        mock_perf_metrics = self._create_mock_perf_metrics()
        mock_harness.run_validation.return_value = (mock_metrics, mock_perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Create runner and execute
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=tmp_path
        )

        exit_code, metrics, perf_metrics = runner.run()

        # Verify sequential execution
        assert mock_harness_class.called
        assert mock_harness.run_validation.called
        assert exit_code == 0  # Success

    def _create_mock_metrics(self) -> Metrics:
        """Create mock Metrics object."""
        return Metrics(
            total_images=50,
            true_positives=45,
            true_negatives=3,
            false_positives=1,
            false_negatives=1,
            accuracy=0.96,
            false_positive_rate=0.25,
            false_negative_rate=0.0217,
            confusion_matrix={'TP': 45, 'TN': 3, 'FP': 1, 'FN': 1},
            site_breakdown={},
            total_time_seconds=3000.0,
            errors_count=0
        )

    def _create_mock_perf_metrics(self) -> PerformanceMetrics:
        """Create mock PerformanceMetrics object."""
        return PerformanceMetrics(
            fps=0.02,
            fps_min=0.015,
            fps_max=0.025,
            memory_peak_mb=450.0,
            memory_mean_mb=420.0,
            memory_stddev_mb=15.0,
            cpu_percent_mean=45.0,
            cpu_percent_max=60.0,
            detection_time_mean_ms=500.0,
            detection_time_stddev_ms=50.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )


class TestProgressReporting:
    """Test progress reporting during execution (AC1: Progress reporting)."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    @patch('validation.run_stage3_validation.logger')
    def test_progress_logging(self, mock_logger, mock_harness_class, tmp_path):
        """Test progress is logged throughout execution."""
        # Setup mocks
        mock_harness = Mock()
        mock_metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.0, false_negative_rate=0.0417,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={}, total_time_seconds=100.0, errors_count=0
        )
        mock_perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )
        mock_harness.run_validation.return_value = (mock_metrics, mock_perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=tmp_path
        )
        runner.run()

        # Verify progress logging occurred
        # Check for step logging: [Step 1/3], [Step 2/3], [Step 3/3]
        logged_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("[Step 1/3]" in msg for msg in logged_messages)
        assert any("[Step 2/3]" in msg for msg in logged_messages)
        assert any("[Step 3/3]" in msg for msg in logged_messages)


class TestErrorHandling:
    """Test error handling with graceful degradation (AC1: Graceful error handling)."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_harness_initialization_failure(self, mock_harness_class, tmp_path):
        """Test graceful handling of harness initialization failure."""
        # Simulate initialization failure
        mock_harness_class.side_effect = FileNotFoundError("Config not found")

        runner = ValidationRunner(
            detector_config_path="invalid/path.json",
            output_dir=tmp_path
        )

        exit_code, metrics, perf_metrics = runner.run()

        # Should return system error code
        assert exit_code == 2
        assert metrics is None
        assert perf_metrics is None

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_validation_execution_failure(self, mock_harness_class, tmp_path):
        """Test graceful handling of validation execution failure."""
        # Setup mock that fails during execution
        mock_harness = Mock()
        mock_harness.run_validation.side_effect = RuntimeError("Execution failed")
        mock_harness_class.return_value = mock_harness

        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=tmp_path
        )

        exit_code, metrics, perf_metrics = runner.run()

        # Should return validation error code
        assert exit_code == 1
        assert metrics is None
        assert perf_metrics is None

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_keyboard_interrupt_handling(self, mock_harness_class, tmp_path):
        """Test graceful handling of keyboard interrupt."""
        # Simulate user interruption
        mock_harness = Mock()
        mock_harness.run_validation.side_effect = KeyboardInterrupt()
        mock_harness_class.return_value = mock_harness

        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=tmp_path
        )

        exit_code, metrics, perf_metrics = runner.run()

        # Should return system error code
        assert exit_code == 2

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_generic_exception_handling(self, mock_harness_class, tmp_path):
        """Test graceful handling of unexpected exceptions."""
        # Simulate unexpected error
        mock_harness = Mock()
        mock_harness.run_validation.side_effect = ValueError("Unexpected error")
        mock_harness_class.return_value = mock_harness

        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=tmp_path
        )

        exit_code, metrics, perf_metrics = runner.run()

        # Should return validation error code
        assert exit_code == 1
        assert metrics is None
        assert perf_metrics is None


class TestCLIArgumentParsing:
    """Test command-line argument parsing (AC1: Command-line interface)."""

    def test_default_arguments(self):
        """Test CLI with no arguments uses defaults."""
        with patch('sys.argv', ['run_stage3_validation.py']):
            args = parse_arguments()

            assert args.baseline is None
            assert args.output_dir == Path("validation/results")
            assert args.detector_config == "config/detector_config.json"

    def test_custom_baseline_argument(self):
        """Test CLI with custom baseline path."""
        with patch('sys.argv', ['run_stage3_validation.py', '--baseline', 'custom/baseline.jpg']):
            args = parse_arguments()

            assert args.baseline == Path("custom/baseline.jpg")

    def test_custom_output_dir_argument(self):
        """Test CLI with custom output directory."""
        with patch('sys.argv', ['run_stage3_validation.py', '--output-dir', 'custom/output']):
            args = parse_arguments()

            assert args.output_dir == Path("custom/output")

    def test_all_custom_arguments(self):
        """Test CLI with all custom arguments."""
        with patch('sys.argv', [
            'run_stage3_validation.py',
            '--baseline', 'custom/baseline.jpg',
            '--output-dir', 'custom/output',
            '--detector-config', 'custom/config.json'
        ]):
            args = parse_arguments()

            assert args.baseline == Path("custom/baseline.jpg")
            assert args.output_dir == Path("custom/output")
            assert args.detector_config == "custom/config.json"


class TestExitCodes:
    """Test exit codes (AC1: Exit codes)."""

    @patch('validation.run_stage3_validation.ValidationRunner')
    def test_exit_code_success(self, mock_runner_class):
        """Test exit code 0 for successful validation."""
        # Mock successful execution
        mock_runner = Mock()
        mock_runner.run.return_value = (0, Mock(), Mock())
        mock_runner_class.return_value = mock_runner

        with patch('sys.argv', ['run_stage3_validation.py']):
            exit_code = main()

        assert exit_code == 0

    @patch('validation.run_stage3_validation.ValidationRunner')
    def test_exit_code_validation_error(self, mock_runner_class):
        """Test exit code 1 for validation errors."""
        # Mock validation failure
        mock_runner = Mock()
        mock_runner.run.return_value = (1, None, None)
        mock_runner_class.return_value = mock_runner

        with patch('sys.argv', ['run_stage3_validation.py']):
            exit_code = main()

        assert exit_code == 1

    @patch('validation.run_stage3_validation.ValidationRunner')
    def test_exit_code_system_error(self, mock_runner_class):
        """Test exit code 2 for system errors."""
        # Mock system error
        mock_runner = Mock()
        mock_runner.run.return_value = (2, None, None)
        mock_runner_class.return_value = mock_runner

        with patch('sys.argv', ['run_stage3_validation.py']):
            exit_code = main()

        assert exit_code == 2
