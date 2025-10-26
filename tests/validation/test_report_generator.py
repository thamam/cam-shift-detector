"""
Unit tests for Report Generators and Go/No-Go Decision Logic

Tests cover acceptance criteria for Story 3:
- AC2: JSON Report Generation
- AC3: Markdown Report Generation
- AC4: Go/No-Go Decision Logic
- AC5: Framework Quality Assurance
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock

from validation.report_generator import (
    JSONReportGenerator, MarkdownReportGenerator, GoNoGoDecisionMaker
)
from validation.stage3_test_harness import Metrics
from validation.performance_profiler import PerformanceMetrics


class TestGoNoGoDecisionMaker:
    """Test Go/No-Go decision logic (AC4)."""

    def test_go_recommendation_all_gates_pass(self):
        """Test GO recommendation when all gates pass."""
        decision_maker = GoNoGoDecisionMaker()

        # Create metrics that pass all gates
        metrics = Metrics(
            total_images=50,
            true_positives=48,
            true_negatives=0,
            false_positives=0,
            false_negatives=2,
            accuracy=0.96,  # ≥95%
            false_positive_rate=0.0,  # ≤5%
            false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={},
            total_time_seconds=100.0,
            errors_count=0
        )

        perf_metrics = PerformanceMetrics(
            fps=0.02,  # ≥0.0167
            fps_min=0.015,
            fps_max=0.025,
            memory_peak_mb=450.0,  # ≤500 MB
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

        result = decision_maker.evaluate(metrics, perf_metrics)

        assert result['recommendation'] == "GO"
        assert result['gate_criteria_met']['all_gates'] is True
        assert result['gate_criteria_met']['accuracy'] is True
        assert result['gate_criteria_met']['false_positive_rate'] is True
        assert result['gate_criteria_met']['fps'] is True
        assert result['gate_criteria_met']['memory'] is True

    def test_no_go_accuracy_below_threshold(self):
        """Test NO-GO when accuracy below 95%."""
        decision_maker = GoNoGoDecisionMaker()

        # Create metrics with low accuracy
        metrics = Metrics(
            total_images=50,
            true_positives=45,
            true_negatives=0,
            false_positives=0,
            false_negatives=5,
            accuracy=0.90,  # <95% - FAILS
            false_positive_rate=0.0,
            false_negative_rate=0.10,
            confusion_matrix={'TP': 45, 'TN': 0, 'FP': 0, 'FN': 5},
            site_breakdown={},
            total_time_seconds=100.0,
            errors_count=0
        )

        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )

        result = decision_maker.evaluate(metrics, perf_metrics)

        assert result['recommendation'] == "NO-GO"
        assert result['gate_criteria_met']['all_gates'] is False
        assert result['gate_criteria_met']['accuracy'] is False
        assert "GATE FAILED" in result['rationale']

    def test_no_go_fps_below_target(self):
        """Test NO-GO when FPS below target."""
        decision_maker = GoNoGoDecisionMaker()

        metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.0, false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={}, total_time_seconds=100.0, errors_count=0
        )

        # FPS below target
        perf_metrics = PerformanceMetrics(
            fps=0.01,  # <0.0167 - FAILS
            fps_min=0.009,
            fps_max=0.012,
            memory_peak_mb=450.0,
            memory_mean_mb=420.0,
            memory_stddev_mb=15.0,
            cpu_percent_mean=45.0,
            cpu_percent_max=60.0,
            detection_time_mean_ms=500.0,
            detection_time_stddev_ms=50.0,
            total_images=50,
            meets_fps_target=False,  # FAILS
            meets_memory_target=True
        )

        result = decision_maker.evaluate(metrics, perf_metrics)

        assert result['recommendation'] == "NO-GO"
        assert result['gate_criteria_met']['fps'] is False

    def test_no_go_memory_above_target(self):
        """Test NO-GO when memory above target."""
        decision_maker = GoNoGoDecisionMaker()

        metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.0, false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={}, total_time_seconds=100.0, errors_count=0
        )

        # Memory above target
        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=600.0,  # >500 MB - FAILS
            memory_mean_mb=550.0,
            memory_stddev_mb=30.0,
            cpu_percent_mean=45.0,
            cpu_percent_max=60.0,
            detection_time_mean_ms=500.0,
            detection_time_stddev_ms=50.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=False  # FAILS
        )

        result = decision_maker.evaluate(metrics, perf_metrics)

        assert result['recommendation'] == "NO-GO"
        assert result['gate_criteria_met']['memory'] is False

    def test_rationale_generation(self):
        """Test rationale clearly explains gate status."""
        decision_maker = GoNoGoDecisionMaker()

        metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.0, false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={}, total_time_seconds=100.0, errors_count=0
        )

        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )

        result = decision_maker.evaluate(metrics, perf_metrics)

        # Rationale should include all gate status
        assert "Accuracy" in result['rationale']
        assert "False Positive Rate" in result['rationale']
        assert "FPS" in result['rationale']
        assert "Memory" in result['rationale']

    def test_conservative_decision_logic(self):
        """Test conservative logic: any failure → NO-GO."""
        decision_maker = GoNoGoDecisionMaker()

        # Create scenario with only ONE gate failure
        metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.06,  # Slightly above 5% - SINGLE FAILURE
            false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={}, total_time_seconds=100.0, errors_count=0
        )

        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )

        result = decision_maker.evaluate(metrics, perf_metrics)

        # Should be NO-GO despite only one failure
        assert result['recommendation'] == "NO-GO"


class TestJSONReportGenerator:
    """Test JSON report generation (AC2)."""

    def test_json_report_structure(self, tmp_path):
        """Test JSON report contains all required sections."""
        generator = JSONReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.json"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        # Load and verify structure
        with open(output_path, 'r') as f:
            report = json.load(f)

        # Verify required top-level fields
        assert "validation_date" in report
        assert "total_images" in report
        assert "metrics" in report
        assert "performance" in report
        assert "site_breakdown" in report
        assert "go_no_go" in report

    def test_json_report_content_accuracy(self, tmp_path):
        """Test JSON report content matches input data."""
        generator = JSONReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.json"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        with open(output_path, 'r') as f:
            report = json.load(f)

        # Verify metrics accuracy
        assert report['total_images'] == 50
        assert report['metrics']['accuracy'] == 0.96
        assert report['metrics']['true_positives'] == 48
        assert report['metrics']['false_positives'] == 0

        # Verify performance accuracy
        assert report['performance']['mean_fps'] == 0.02
        assert report['performance']['peak_memory_mb'] == 450.0

        # Verify go/no-go
        assert report['go_no_go']['recommendation'] == "GO"

    def test_json_schema_validation(self, tmp_path):
        """Test JSON is parseable and valid."""
        generator = JSONReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.json"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        # Verify file is valid JSON
        with open(output_path, 'r') as f:
            report = json.load(f)  # Should not raise

        # Verify it's a dictionary
        assert isinstance(report, dict)

    def _create_test_data(self):
        """Create test data for report generation."""
        metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.0, false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={'site1': {'accuracy': 0.95, 'total': 20, 'correct': 19}},
            total_time_seconds=100.0, errors_count=0
        )

        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )

        go_no_go = {
            'recommendation': 'GO',
            'gate_criteria_met': {
                'accuracy': True, 'false_positive_rate': True,
                'fps': True, 'memory': True, 'all_gates': True
            },
            'rationale': 'All gates passed'
        }

        return metrics, perf_metrics, go_no_go


class TestMarkdownReportGenerator:
    """Test Markdown report generation (AC3)."""

    def test_markdown_report_formatting(self, tmp_path):
        """Test Markdown report is properly formatted."""
        generator = MarkdownReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.md"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        # Read and verify formatting
        with open(output_path, 'r') as f:
            content = f.read()

        # Verify Markdown structure
        assert "# Stage 3 Validation Report" in content
        assert "## Executive Summary" in content
        assert "## Validation Metrics" in content
        assert "## Performance Benchmarks" in content
        assert "## Go/No-Go Recommendation" in content

    def test_markdown_executive_summary(self, tmp_path):
        """Test executive summary section."""
        generator = MarkdownReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.md"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        with open(output_path, 'r') as f:
            content = f.read()

        # Verify executive summary content
        assert "Production Readiness: GO" in content
        assert "Detection Accuracy:" in content
        assert "96.00%" in content  # 0.96 * 100

    def test_markdown_confusion_matrix_table(self, tmp_path):
        """Test confusion matrix table formatting."""
        generator = MarkdownReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.md"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        with open(output_path, 'r') as f:
            content = f.read()

        # Verify confusion matrix table
        assert "Confusion Matrix" in content
        assert "(TP)" in content
        assert "(TN)" in content
        assert "(FP)" in content
        assert "(FN)" in content

    def test_markdown_site_breakdown_table(self, tmp_path):
        """Test site breakdown table formatting."""
        generator = MarkdownReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        metrics.site_breakdown = {
            'of_jerusalem': {'accuracy': 0.96, 'total': 23, 'correct': 22},
            'carmit': {'accuracy': 0.94, 'total': 17, 'correct': 16},
            'gad': {'accuracy': 1.00, 'total': 10, 'correct': 10}
        }
        output_path = tmp_path / "test_report.md"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        with open(output_path, 'r') as f:
            content = f.read()

        # Verify site breakdown table
        assert "Per-Site Breakdown" in content
        assert "of_jerusalem" in content
        assert "carmit" in content
        assert "gad" in content

    def test_markdown_failure_analysis_present(self, tmp_path):
        """Test failure analysis section appears when failures exist."""
        generator = MarkdownReportGenerator()

        # Create metrics with failures
        metrics = Metrics(
            total_images=50, true_positives=45, true_negatives=3,
            false_positives=1, false_negatives=1,  # Failures present
            accuracy=0.96, false_positive_rate=0.25, false_negative_rate=0.0217,
            confusion_matrix={'TP': 45, 'TN': 3, 'FP': 1, 'FN': 1},
            site_breakdown={}, total_time_seconds=100.0, errors_count=0
        )

        perf_metrics, go_no_go = self._create_test_perf_data()
        output_path = tmp_path / "test_report.md"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        with open(output_path, 'r') as f:
            content = f.read()

        # Verify failure analysis exists
        assert "Failure Analysis" in content
        assert "False Positives" in content
        assert "False Negatives" in content

    def test_markdown_go_no_go_section(self, tmp_path):
        """Test go/no-go recommendation section."""
        generator = MarkdownReportGenerator()

        metrics, perf_metrics, go_no_go = self._create_test_data()
        output_path = tmp_path / "test_report.md"

        generator.generate(metrics, perf_metrics, go_no_go, output_path)

        with open(output_path, 'r') as f:
            content = f.read()

        # Verify go/no-go section
        assert "Go/No-Go Recommendation" in content
        assert "APPROVED FOR PRODUCTION" in content or "NOT READY FOR PRODUCTION" in content

    def _create_test_data(self):
        """Create test data for report generation."""
        metrics = Metrics(
            total_images=50, true_positives=48, true_negatives=0,
            false_positives=0, false_negatives=2, accuracy=0.96,
            false_positive_rate=0.0, false_negative_rate=0.04,
            confusion_matrix={'TP': 48, 'TN': 0, 'FP': 0, 'FN': 2},
            site_breakdown={'site1': {'accuracy': 0.95, 'total': 20, 'correct': 19}},
            total_time_seconds=100.0, errors_count=0
        )

        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )

        go_no_go = {
            'recommendation': 'GO',
            'gate_criteria_met': {
                'accuracy': True, 'false_positive_rate': True,
                'fps': True, 'memory': True, 'all_gates': True
            },
            'rationale': 'All gates passed'
        }

        return metrics, perf_metrics, go_no_go

    def _create_test_perf_data(self):
        """Create test performance data."""
        perf_metrics = PerformanceMetrics(
            fps=0.02, fps_min=0.015, fps_max=0.025,
            memory_peak_mb=450.0, memory_mean_mb=420.0, memory_stddev_mb=15.0,
            cpu_percent_mean=45.0, cpu_percent_max=60.0,
            detection_time_mean_ms=500.0, detection_time_stddev_ms=50.0,
            total_images=50, meets_fps_target=True, meets_memory_target=True
        )

        go_no_go = {
            'recommendation': 'GO',
            'gate_criteria_met': {
                'accuracy': True, 'false_positive_rate': True,
                'fps': True, 'memory': True, 'all_gates': True
            },
            'rationale': 'All gates passed'
        }

        return perf_metrics, go_no_go
