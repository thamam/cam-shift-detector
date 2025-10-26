"""
Integration Tests for Story 3: Complete Validation Workflow

Tests the complete end-to-end validation workflow including:
- ValidationRunner orchestration
- Report generation (JSON + Markdown)
- Go/no-go decision making
- Cross-component integration

Tests cover acceptance criteria for Story 3:
- AC1: Complete workflow execution
- AC2: JSON report generation and validation
- AC3: Markdown report generation
- AC4: Go/no-go decision consistency
- AC5: Integration quality assurance
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from validation.run_stage3_validation import ValidationRunner, main
from validation.stage3_test_harness import Metrics
from validation.performance_profiler import PerformanceMetrics


class TestFullValidationWorkflowEndToEnd:
    """Test complete end-to-end validation workflow execution."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_full_validation_workflow_end_to_end(self, mock_harness_class, tmp_path):
        """Test complete workflow: initialization → execution → reports → exit."""
        # Setup mock harness with realistic data
        mock_harness = Mock()

        # Create realistic validation metrics
        metrics = Metrics(
            total_images=50,
            true_positives=46,
            true_negatives=2,
            false_positives=1,
            false_negatives=1,
            accuracy=0.96,
            false_positive_rate=0.3333,  # 1/3 actual negatives
            false_negative_rate=0.0213,  # 1/47 actual positives
            confusion_matrix={'TP': 46, 'TN': 2, 'FP': 1, 'FN': 1},
            site_breakdown={
                'of_jerusalem': {'accuracy': 0.96, 'correct': 22, 'total': 23},
                'carmit': {'accuracy': 0.94, 'correct': 16, 'total': 17},
                'gad': {'accuracy': 1.0, 'correct': 10, 'total': 10}
            },
            total_time_seconds=2500.0,
            errors_count=0
        )

        # Create realistic performance metrics
        perf_metrics = PerformanceMetrics(
            fps=0.02,
            fps_min=0.016,
            fps_max=0.024,
            memory_peak_mb=475.0,
            memory_mean_mb=440.0,
            memory_stddev_mb=18.0,
            cpu_percent_mean=52.0,
            cpu_percent_max=68.0,
            detection_time_mean_ms=485.0,
            detection_time_stddev_ms=42.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )

        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute complete workflow
        output_dir = tmp_path / "integration_results"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )

        exit_code, result_metrics, result_perf_metrics = runner.run()

        # Verify workflow execution
        assert exit_code == 0, "Workflow should complete successfully"
        assert result_metrics is not None, "Metrics should be returned"
        assert result_perf_metrics is not None, "Performance metrics should be returned"

        # Verify harness was called correctly
        assert mock_harness_class.called, "Test harness should be initialized"
        assert mock_harness.run_validation.called, "Validation should be executed"

        # Verify output directory created
        assert output_dir.exists(), "Output directory should be created"


class TestReportsGeneratedSuccessfully:
    """Test that both JSON and Markdown reports are generated."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_json_report_generated(self, mock_harness_class, tmp_path):
        """Test JSON report is generated with complete structure."""
        # Setup mock harness
        mock_harness = Mock()
        metrics = self._create_passing_metrics()
        perf_metrics = self._create_passing_perf_metrics()
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow
        output_dir = tmp_path / "json_test"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )
        exit_code, _, _ = runner.run()

        # Verify JSON report exists
        json_path = output_dir / "validation_report.json"
        assert json_path.exists(), "JSON report should be generated"

        # Verify JSON is valid and complete
        with open(json_path, 'r') as f:
            report = json.load(f)

        assert "validation_date" in report
        assert "total_images" in report
        assert "metrics" in report
        assert "performance" in report
        assert "site_breakdown" in report
        assert "go_no_go" in report
        assert report["total_images"] == 50

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_markdown_report_generated(self, mock_harness_class, tmp_path):
        """Test Markdown report is generated with all sections."""
        # Setup mock harness
        mock_harness = Mock()
        metrics = self._create_passing_metrics()
        perf_metrics = self._create_passing_perf_metrics()
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow
        output_dir = tmp_path / "markdown_test"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )
        exit_code, _, _ = runner.run()

        # Verify Markdown report exists
        md_path = output_dir / "validation_report.md"
        assert md_path.exists(), "Markdown report should be generated"

        # Verify Markdown has all required sections
        with open(md_path, 'r') as f:
            content = f.read()

        assert "# Stage 3 Validation Report" in content
        assert "## Executive Summary" in content
        assert "## Validation Metrics" in content
        assert "## Performance Benchmarks" in content
        assert "## Per-Site Breakdown" in content
        assert "## Go/No-Go Recommendation" in content

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_both_reports_generated_together(self, mock_harness_class, tmp_path):
        """Test both JSON and Markdown reports are generated in single run."""
        # Setup mock harness
        mock_harness = Mock()
        metrics = self._create_passing_metrics()
        perf_metrics = self._create_passing_perf_metrics()
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow
        output_dir = tmp_path / "both_reports"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )
        exit_code, _, _ = runner.run()

        # Verify both reports exist
        json_path = output_dir / "validation_report.json"
        md_path = output_dir / "validation_report.md"

        assert json_path.exists(), "JSON report should exist"
        assert md_path.exists(), "Markdown report should exist"
        assert exit_code == 0, "Exit code should indicate success"

    def _create_passing_metrics(self) -> Metrics:
        """Create metrics that pass all gate criteria."""
        return Metrics(
            total_images=50,
            true_positives=48,
            true_negatives=0,
            false_positives=0,
            false_negatives=2,
            accuracy=0.96,  # ≥95% ✓
            false_positive_rate=0.0,  # ≤5% ✓
            false_negative_rate=0.0417,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={
                'of_jerusalem': {'accuracy': 0.96, 'correct': 22, 'total': 23},
                'carmit': {'accuracy': 0.94, 'correct': 16, 'total': 17},
                'gad': {'accuracy': 1.0, 'correct': 10, 'total': 10}
            },
            total_time_seconds=2500.0,
            errors_count=0
        )

    def _create_passing_perf_metrics(self) -> PerformanceMetrics:
        """Create performance metrics that pass all gate criteria."""
        return PerformanceMetrics(
            fps=0.02,  # ≥0.0167 ✓
            fps_min=0.016,
            fps_max=0.024,
            memory_peak_mb=475.0,  # ≤500 MB ✓
            memory_mean_mb=440.0,
            memory_stddev_mb=18.0,
            cpu_percent_mean=52.0,
            cpu_percent_max=68.0,
            detection_time_mean_ms=485.0,
            detection_time_stddev_ms=42.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )


class TestReportContentConsistency:
    """Test consistency between JSON and Markdown report content."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_metrics_consistency_across_reports(self, mock_harness_class, tmp_path):
        """Test that metrics are consistent between JSON and Markdown reports."""
        # Setup mock harness
        mock_harness = Mock()
        metrics = Metrics(
            total_images=50,
            true_positives=45,
            true_negatives=3,
            false_positives=1,
            false_negatives=1,
            accuracy=0.96,
            false_positive_rate=0.25,
            false_negative_rate=0.0217,
            confusion_matrix={'TP': 45, 'TN': 3, 'FP': 1, 'FN': 1},
            site_breakdown={
                'of_jerusalem': {'accuracy': 0.96, 'correct': 22, 'total': 23}
            },
            total_time_seconds=2500.0,
            errors_count=0
        )
        perf_metrics = PerformanceMetrics(
            fps=0.02,
            fps_min=0.016,
            fps_max=0.024,
            memory_peak_mb=475.0,
            memory_mean_mb=440.0,
            memory_stddev_mb=18.0,
            cpu_percent_mean=52.0,
            cpu_percent_max=68.0,
            detection_time_mean_ms=485.0,
            detection_time_stddev_ms=42.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow
        output_dir = tmp_path / "consistency_test"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )
        runner.run()

        # Load JSON report
        json_path = output_dir / "validation_report.json"
        with open(json_path, 'r') as f:
            json_report = json.load(f)

        # Load Markdown report
        md_path = output_dir / "validation_report.md"
        with open(md_path, 'r') as f:
            md_content = f.read()

        # Verify key metrics are consistent
        assert json_report["total_images"] == 50
        assert "50 total images" in md_content or "50" in md_content

        assert json_report["metrics"]["accuracy"] == 0.96
        assert "0.96" in md_content or "96" in md_content

        assert json_report["performance"]["peak_memory_mb"] == 475.0
        assert "475" in md_content

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_go_no_go_consistency(self, mock_harness_class, tmp_path):
        """Test go/no-go recommendation is consistent across reports."""
        # Setup mock harness with passing metrics
        mock_harness = Mock()
        metrics = Metrics(
            total_images=50,
            true_positives=48,
            true_negatives=0,
            false_positives=0,
            false_negatives=2,
            accuracy=0.96,  # Pass
            false_positive_rate=0.0,  # Pass
            false_negative_rate=0.0417,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={},
            total_time_seconds=2500.0,
            errors_count=0
        )
        perf_metrics = PerformanceMetrics(
            fps=0.02,  # Pass
            fps_min=0.016,
            fps_max=0.024,
            memory_peak_mb=475.0,  # Pass
            memory_mean_mb=440.0,
            memory_stddev_mb=18.0,
            cpu_percent_mean=52.0,
            cpu_percent_max=68.0,
            detection_time_mean_ms=485.0,
            detection_time_stddev_ms=42.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow
        output_dir = tmp_path / "go_no_go_test"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )
        runner.run()

        # Load JSON report
        json_path = output_dir / "validation_report.json"
        with open(json_path, 'r') as f:
            json_report = json.load(f)

        # Load Markdown report
        md_path = output_dir / "validation_report.md"
        with open(md_path, 'r') as f:
            md_content = f.read()

        # Verify go/no-go consistency
        json_recommendation = json_report["go_no_go"]["recommendation"]

        if json_recommendation == "GO":
            assert "GO" in md_content
            assert "APPROVED FOR PRODUCTION" in md_content or "GO" in md_content
        else:
            assert "NO-GO" in md_content
            assert "NOT READY FOR PRODUCTION" in md_content or "NO-GO" in md_content


class TestPerformanceWithinConstraints:
    """Test that validation execution stays within performance constraints."""

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_validation_execution_time_reasonable(self, mock_harness_class, tmp_path):
        """Test validation completes in reasonable time."""
        import time

        # Setup mock harness with fast execution
        mock_harness = Mock()
        metrics = Metrics(
            total_images=50,
            true_positives=48,
            true_negatives=0,
            false_positives=0,
            false_negatives=2,
            accuracy=0.96,
            false_positive_rate=0.0,
            false_negative_rate=0.0417,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={},
            total_time_seconds=2500.0,
            errors_count=0
        )
        perf_metrics = PerformanceMetrics(
            fps=0.02,
            fps_min=0.016,
            fps_max=0.024,
            memory_peak_mb=475.0,
            memory_mean_mb=440.0,
            memory_stddev_mb=18.0,
            cpu_percent_mean=52.0,
            cpu_percent_max=68.0,
            detection_time_mean_ms=485.0,
            detection_time_stddev_ms=42.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow and measure time
        output_dir = tmp_path / "perf_test"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )

        start_time = time.time()
        exit_code, _, _ = runner.run()
        execution_time = time.time() - start_time

        # Verify execution completes quickly (orchestration overhead should be minimal)
        # Allow up to 5 seconds for test execution (mocked, so should be fast)
        assert execution_time < 5.0, f"Workflow took {execution_time:.2f}s, expected <5s"
        assert exit_code == 0

    @patch('validation.run_stage3_validation.Stage3TestHarness')
    def test_report_generation_not_memory_intensive(self, mock_harness_class, tmp_path):
        """Test report generation doesn't consume excessive memory."""
        # Setup mock harness
        mock_harness = Mock()
        metrics = Metrics(
            total_images=50,
            true_positives=48,
            true_negatives=0,
            false_positives=0,
            false_negatives=2,
            accuracy=0.96,
            false_positive_rate=0.0,
            false_negative_rate=0.0417,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={
                f'site_{i}': {'accuracy': 0.95, 'correct': 45, 'total': 50}
                for i in range(20)  # 20 sites to test memory with larger dataset
            },
            total_time_seconds=2500.0,
            errors_count=0
        )
        perf_metrics = PerformanceMetrics(
            fps=0.02,
            fps_min=0.016,
            fps_max=0.024,
            memory_peak_mb=475.0,
            memory_mean_mb=440.0,
            memory_stddev_mb=18.0,
            cpu_percent_mean=52.0,
            cpu_percent_max=68.0,
            detection_time_mean_ms=485.0,
            detection_time_stddev_ms=42.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )
        mock_harness.run_validation.return_value = (metrics, perf_metrics)
        mock_harness_class.return_value = mock_harness

        # Execute workflow
        output_dir = tmp_path / "memory_test"
        runner = ValidationRunner(
            detector_config_path="config/detector_config.json",
            output_dir=output_dir
        )

        exit_code, _, _ = runner.run()

        # Verify execution completed successfully
        assert exit_code == 0

        # Verify reports were generated
        json_path = output_dir / "validation_report.json"
        md_path = output_dir / "validation_report.md"

        assert json_path.exists()
        assert md_path.exists()

        # Verify file sizes are reasonable (not bloated)
        json_size = json_path.stat().st_size
        md_size = md_path.stat().st_size

        # JSON should be <100KB for 50 images with 20 sites
        assert json_size < 100_000, f"JSON report is {json_size} bytes, expected <100KB"

        # Markdown should be <50KB for 50 images with 20 sites
        assert md_size < 50_000, f"Markdown report is {md_size} bytes, expected <50KB"
