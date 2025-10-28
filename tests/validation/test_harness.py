"""
Unit tests for Stage3TestHarness (validation/stage3_test_harness.py)

Tests cover all acceptance criteria for Story 2:
- AC1: Test Harness Execution Logic
- AC2: Metrics Calculation
- AC4: Test coverage requirement
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import numpy as np

from validation.stage3_test_harness import (
    Stage3TestHarness,
    DetectionResult,
    Metrics
)
from validation.real_data_loader import ImageMetadata
from datetime import datetime


class TestDetectionResultDataclass:
    """Test DetectionResult dataclass structure."""

    def test_detection_result_creation(self):
        """Test DetectionResult object creation with all fields."""
        result = DetectionResult(
            image_path=Path("/test/image.jpg"),
            site_id="of_jerusalem",
            ground_truth=True,
            predicted=True,
            is_correct=True,
            detection_time_ms=123.45,
            error_message=None
        )

        assert result.image_path == Path("/test/image.jpg")
        assert result.site_id == "of_jerusalem"
        assert result.ground_truth is True
        assert result.predicted is True
        assert result.is_correct is True
        assert result.detection_time_ms == 123.45
        assert result.error_message is None

    def test_detection_result_with_error(self):
        """Test DetectionResult with error message."""
        result = DetectionResult(
            image_path=Path("/test/image.jpg"),
            site_id="carmit",
            ground_truth=False,
            predicted=False,
            is_correct=True,
            detection_time_ms=0.0,
            error_message="Detection failed"
        )

        assert result.error_message == "Detection failed"


class TestMetricsDataclass:
    """Test Metrics dataclass structure."""

    def test_metrics_creation(self):
        """Test Metrics object creation with all fields."""
        metrics = Metrics(
            total_images=50,
            true_positives=25,
            true_negatives=20,
            false_positives=3,
            false_negatives=2,
            accuracy=0.9000,
            false_positive_rate=0.1304,
            false_negative_rate=0.0741,
            confusion_matrix={'TP': 25, 'TN': 20, 'FP': 3, 'FN': 2},
            site_breakdown={'of_jerusalem': {'accuracy': 0.9, 'total': 23, 'correct': 21}},
            total_time_seconds=3000.0,
            errors_count=0
        )

        assert metrics.total_images == 50
        assert metrics.true_positives == 25
        assert metrics.accuracy == 0.9000


class TestStage3TestHarnessInit:
    """Test Stage3TestHarness initialization."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_harness_initialization(self, mock_profiler, mock_loader, mock_detector):
        """Test harness initializes detector, data loader, and profiler."""
        harness = Stage3TestHarness('config.json')

        mock_detector.assert_called_once_with('config.json')
        mock_loader.assert_called_once()
        mock_profiler.assert_called_once()
        assert harness.results == []


class TestCompareWithGroundTruth:
    """Test _compare_with_ground_truth method."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_correct_positive_prediction(self, mock_profiler, mock_loader, mock_detector):
        """Test true positive: predicted=True, ground_truth=True."""
        harness = Stage3TestHarness('config.json')

        is_correct = harness._compare_with_ground_truth(True, True)
        assert is_correct is True

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_correct_negative_prediction(self, mock_profiler, mock_loader, mock_detector):
        """Test true negative: predicted=False, ground_truth=False."""
        harness = Stage3TestHarness('config.json')

        is_correct = harness._compare_with_ground_truth(False, False)
        assert is_correct is True

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_false_positive(self, mock_profiler, mock_loader, mock_detector):
        """Test false positive: predicted=True, ground_truth=False."""
        harness = Stage3TestHarness('config.json')

        is_correct = harness._compare_with_ground_truth(True, False)
        assert is_correct is False

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_false_negative(self, mock_profiler, mock_loader, mock_detector):
        """Test false negative: predicted=False, ground_truth=True."""
        harness = Stage3TestHarness('config.json')

        is_correct = harness._compare_with_ground_truth(False, True)
        assert is_correct is False


class TestMetricsCalculation:
    """Test calculate_metrics method (AC2: Metrics Calculation)."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_metrics_all_correct(self, mock_profiler, mock_loader, mock_detector):
        """Test metrics calculation with 100% accuracy scenario."""
        harness = Stage3TestHarness('config.json')

        # Create results with all correct predictions
        harness.results = [
            DetectionResult(Path(f"/test/{i}.jpg"), "of_jerusalem", True, True, True, 100.0)
            for i in range(25)
        ] + [
            DetectionResult(Path(f"/test/{i}.jpg"), "carmit", False, False, True, 100.0)
            for i in range(25, 50)
        ]

        metrics = harness.calculate_metrics(total_time_seconds=3000.0)

        assert metrics.total_images == 50
        assert metrics.true_positives == 25
        assert metrics.true_negatives == 25
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.accuracy == 1.0000  # 100% accuracy
        assert metrics.false_positive_rate == 0.0
        assert metrics.false_negative_rate == 0.0
        assert metrics.errors_count == 0

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_metrics_all_wrong(self, mock_profiler, mock_loader, mock_detector):
        """Test metrics calculation with 0% accuracy scenario."""
        harness = Stage3TestHarness('config.json')

        # Create results with all wrong predictions
        harness.results = [
            DetectionResult(Path(f"/test/{i}.jpg"), "of_jerusalem", True, False, False, 100.0)
            for i in range(25)
        ] + [
            DetectionResult(Path(f"/test/{i}.jpg"), "carmit", False, True, False, 100.0)
            for i in range(25, 50)
        ]

        metrics = harness.calculate_metrics(total_time_seconds=3000.0)

        assert metrics.total_images == 50
        assert metrics.true_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 25
        assert metrics.false_negatives == 25
        assert metrics.accuracy == 0.0000  # 0% accuracy
        assert metrics.false_positive_rate == 1.0
        assert metrics.false_negative_rate == 1.0

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_metrics_mixed_results(self, mock_profiler, mock_loader, mock_detector):
        """Test metrics calculation with mixed results (realistic accuracy)."""
        harness = Stage3TestHarness('config.json')

        # Create mixed results: TP=25, TN=20, FP=3, FN=2
        harness.results = (
            [DetectionResult(Path(f"/test/{i}.jpg"), "of_jerusalem", True, True, True, 100.0)
             for i in range(25)] +  # TP=25
            [DetectionResult(Path(f"/test/{i}.jpg"), "carmit", False, False, True, 100.0)
             for i in range(25, 45)] +  # TN=20
            [DetectionResult(Path(f"/test/{i}.jpg"), "carmit", False, True, False, 100.0)
             for i in range(45, 48)] +  # FP=3
            [DetectionResult(Path(f"/test/{i}.jpg"), "gad", True, False, False, 100.0)
             for i in range(48, 50)]  # FN=2
        )

        metrics = harness.calculate_metrics(total_time_seconds=3000.0)

        assert metrics.total_images == 50
        assert metrics.true_positives == 25
        assert metrics.true_negatives == 20
        assert metrics.false_positives == 3
        assert metrics.false_negatives == 2

        # Accuracy = (TP + TN) / Total = (25 + 20) / 50 = 0.9
        assert metrics.accuracy == 0.9000

        # FPR = FP / (FP + TN) = 3 / (3 + 20) = 0.1304
        assert abs(metrics.false_positive_rate - 0.1304) < 0.0001

        # FNR = FN / (FN + TP) = 2 / (2 + 25) = 0.0741
        assert abs(metrics.false_negative_rate - 0.0741) < 0.0001


class TestConfusionMatrix:
    """Test confusion matrix generation (AC2)."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_confusion_matrix_structure(self, mock_profiler, mock_loader, mock_detector):
        """Test confusion matrix has correct structure with TP, TN, FP, FN."""
        harness = Stage3TestHarness('config.json')

        # Create specific distribution: TP=25, TN=20, FP=3, FN=2
        harness.results = (
            [DetectionResult(Path(f"/test/{i}.jpg"), "site", True, True, True, 100.0) for i in range(25)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "site", False, False, True, 100.0) for i in range(25, 45)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "site", False, True, False, 100.0) for i in range(45, 48)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "site", True, False, False, 100.0) for i in range(48, 50)]
        )

        metrics = harness.calculate_metrics(total_time_seconds=3000.0)

        assert 'TP' in metrics.confusion_matrix
        assert 'TN' in metrics.confusion_matrix
        assert 'FP' in metrics.confusion_matrix
        assert 'FN' in metrics.confusion_matrix

        assert metrics.confusion_matrix['TP'] == 25
        assert metrics.confusion_matrix['TN'] == 20
        assert metrics.confusion_matrix['FP'] == 3
        assert metrics.confusion_matrix['FN'] == 2


class TestSiteBreakdown:
    """Test calculate_site_breakdown method (AC2)."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_site_breakdown_structure(self, mock_profiler, mock_loader, mock_detector):
        """Test per-site breakdown has correct structure."""
        harness = Stage3TestHarness('config.json')

        # Create results with known distribution
        harness.results = (
            [DetectionResult(Path(f"/test/{i}.jpg"), "of_jerusalem", True, True, True, 100.0)
             for i in range(23)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "carmit", True, True, True, 100.0)
             for i in range(23, 40)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "gad", True, True, True, 100.0)
             for i in range(40, 50)]
        )

        breakdown = harness.calculate_site_breakdown()

        assert 'of_jerusalem' in breakdown
        assert 'carmit' in breakdown
        assert 'gad' in breakdown

        assert breakdown['of_jerusalem']['total'] == 23
        assert breakdown['carmit']['total'] == 17
        assert breakdown['gad']['total'] == 10

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_site_breakdown_accuracy(self, mock_profiler, mock_loader, mock_detector):
        """Test per-site accuracy calculation."""
        harness = Stage3TestHarness('config.json')

        # OF_JERUSALEM: 20/23 correct = 0.8696
        # CARMIT: 17/17 correct = 1.0000
        # GAD: 8/10 correct = 0.8000
        harness.results = (
            [DetectionResult(Path(f"/test/{i}.jpg"), "of_jerusalem", True, True, True, 100.0)
             for i in range(20)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "of_jerusalem", True, False, False, 100.0)
             for i in range(20, 23)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "carmit", True, True, True, 100.0)
             for i in range(23, 40)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "gad", True, True, True, 100.0)
             for i in range(40, 48)] +
            [DetectionResult(Path(f"/test/{i}.jpg"), "gad", True, False, False, 100.0)
             for i in range(48, 50)]
        )

        breakdown = harness.calculate_site_breakdown()

        # OF_JERUSALEM: 20/23 = 0.8696
        assert abs(breakdown['of_jerusalem']['accuracy'] - 0.8696) < 0.0001
        assert breakdown['of_jerusalem']['correct'] == 20

        # CARMIT: 17/17 = 1.0
        assert breakdown['carmit']['accuracy'] == 1.0000
        assert breakdown['carmit']['correct'] == 17

        # GAD: 8/10 = 0.8
        assert breakdown['gad']['accuracy'] == 0.8000
        assert breakdown['gad']['correct'] == 8


class TestGracefulErrorHandling:
    """Test graceful error handling (AC1)."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    def test_detection_error_handling(self, mock_profiler, mock_loader, mock_detector):
        """Test harness handles detection errors gracefully."""
        harness = Stage3TestHarness('config.json')

        # Mock detector to raise exception
        mock_profiler_instance = mock_profiler.return_value
        mock_profiler_instance.profile_detection.side_effect = Exception("Detection failed")

        # Mock data loader
        mock_loader_instance = mock_loader.return_value
        mock_loader_instance.load_image.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        image_metadata = ImageMetadata(
            image_path=Path("/test/image.jpg"),
            site_id="of_jerusalem",
            timestamp=datetime.now(),
            has_shift=True
        )

        result = harness._execute_single_detection(image_metadata)

        # Should return result with error message, not crash
        assert result.error_message is not None
        assert "Detection failed" in result.error_message
        assert result.predicted is False  # Conservative: assume stable on error


class TestIntegrationWithMockDetector:
    """Integration test: Full harness execution with mock detector (AC4)."""

    @patch('validation.stage3_test_harness.CameraMovementDetector')
    @patch('validation.stage3_test_harness.RealDataLoader')
    @patch('validation.stage3_test_harness.PerformanceProfiler')
    @patch('validation.stage3_test_harness.cv2.cvtColor')
    def test_full_harness_execution(self, mock_cvt, mock_profiler, mock_loader, mock_detector):
        """Test complete harness execution from start to finish."""
        # Setup mock detector
        mock_detector_instance = mock_detector.return_value
        mock_detector_instance.process_frame.return_value = {
            'status': 'VALID',
            'translation_displacement': 0.5,
            'confidence': 0.95,
            'frame_id': 'test',
            'timestamp': '2025-10-25T00:00:00Z'
        }

        # Setup mock data loader
        mock_loader_instance = mock_loader.return_value
        mock_images = [
            ImageMetadata(
                image_path=Path(f"/test/{i}.jpg"),
                site_id="of_jerusalem" if i < 5 else "carmit",
                timestamp=datetime.now(),
                has_shift=False  # All stable
            )
            for i in range(10)
        ]
        mock_loader_instance.load_dataset.return_value = mock_images
        mock_loader_instance.load_image.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Setup mock profiler
        mock_profiler_instance = mock_profiler.return_value
        mock_profiler_instance.profile_detection.return_value = (
            mock_detector_instance.process_frame.return_value,
            0.5  # 0.5 seconds detection time
        )
        mock_profiler_instance.get_metrics.return_value = Mock(
            fps=2.0, memory_peak_mb=300.0, meets_fps_target=True, meets_memory_target=True
        )

        # Setup color conversion
        mock_cvt.return_value = np.zeros((480, 640, 3), dtype=np.uint8)

        # Execute validation
        harness = Stage3TestHarness('config.json')
        metrics, performance_metrics = harness.run_validation()

        # Verify execution completed
        assert metrics.total_images == 10
        assert metrics.errors_count == 0

        # Verify detector was set with baseline
        mock_detector_instance.set_baseline.assert_called_once()

        # Verify all images were processed
        assert len(harness.results) == 10
