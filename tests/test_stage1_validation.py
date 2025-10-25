"""
Unit tests for Stage 1 Test Harness

Tests synthetic transformation functions, accuracy calculation,
and validation workflow.
"""

import pytest
import numpy as np
import cv2
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from validation.stage1_test_harness import (
    Stage1TestHarness,
    TransformationSpec,
    GroundTruthLabel,
    DetectionResult,
    AccuracyMetrics,
    run_stage1_validation
)


@pytest.fixture
def test_harness(tmp_path):
    """Create test harness with temporary directory"""
    return Stage1TestHarness(validation_dir=str(tmp_path / "validation"))


@pytest.fixture
def sample_image():
    """Create a simple test image"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some features for transformation visibility
    cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(image, (400, 300), 50, (128, 128, 128), -1)
    return image


class TestTransformations:
    """Tests for image transformation functions"""

    def test_apply_translation_right(self, test_harness, sample_image):
        """Test right translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'right')

        assert transformed.shape == sample_image.shape
        # Pixel at (110, 100) in original should be at (120, 100) in transformed
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_left(self, test_harness, sample_image):
        """Test left translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'left')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_up(self, test_harness, sample_image):
        """Test up translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'up')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_down(self, test_harness, sample_image):
        """Test down translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'down')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_diagonal_ur(self, test_harness, sample_image):
        """Test diagonal up-right translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'diagonal_ur')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_diagonal_ul(self, test_harness, sample_image):
        """Test diagonal up-left translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'diagonal_ul')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_diagonal_dr(self, test_harness, sample_image):
        """Test diagonal down-right translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'diagonal_dr')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_diagonal_dl(self, test_harness, sample_image):
        """Test diagonal down-left translation"""
        transformed = test_harness.apply_translation(sample_image, 10, 'diagonal_dl')

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_translation_invalid_direction(self, test_harness, sample_image):
        """Test that invalid direction raises ValueError"""
        with pytest.raises(ValueError, match="Unknown direction"):
            test_harness.apply_translation(sample_image, 10, 'invalid_direction')

    def test_apply_rotation(self, test_harness, sample_image):
        """Test rotation transformation"""
        transformed = test_harness.apply_rotation(sample_image, 45.0)

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_apply_rotation_with_scale(self, test_harness, sample_image):
        """Test rotation with scaling"""
        transformed = test_harness.apply_rotation(sample_image, 30.0, scale=1.2)

        assert transformed.shape == sample_image.shape
        assert not np.array_equal(transformed, sample_image)

    def test_zero_shift_produces_similar_image(self, test_harness, sample_image):
        """Test that zero shift produces nearly identical image"""
        transformed = test_harness.apply_translation(sample_image, 0, 'right')

        # Should be very similar (allowing for minor interpolation differences)
        diff = np.abs(transformed.astype(float) - sample_image.astype(float)).mean()
        assert diff < 1.0  # Less than 1 pixel difference on average


class TestAccuracyMetrics:
    """Tests for accuracy metrics calculation"""

    def test_calculate_accuracy_perfect(self, test_harness):
        """Test accuracy calculation with perfect predictions"""
        results = [
            DetectionResult("img1", "INVALID", 5.0, 0.9, "INVALID"),
            DetectionResult("img2", "INVALID", 6.0, 0.85, "INVALID"),
            DetectionResult("img3", "VALID", 0.3, 0.95, "VALID"),
            DetectionResult("img4", "VALID", 0.2, 0.92, "VALID"),
        ]

        metrics = test_harness.calculate_accuracy_metrics(results)

        assert metrics.true_positives == 2
        assert metrics.true_negatives == 2
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 0
        assert metrics.accuracy == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.total_samples == 4

    def test_calculate_accuracy_with_errors(self, test_harness):
        """Test accuracy calculation with some errors"""
        results = [
            DetectionResult("img1", "INVALID", 5.0, 0.9, "INVALID"),  # TP
            DetectionResult("img2", "VALID", 0.3, 0.95, "INVALID"),   # FN
            DetectionResult("img3", "INVALID", 2.0, 0.8, "VALID"),    # FP
            DetectionResult("img4", "VALID", 0.2, 0.92, "VALID"),     # TN
        ]

        metrics = test_harness.calculate_accuracy_metrics(results)

        assert metrics.true_positives == 1
        assert metrics.true_negatives == 1
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1
        assert metrics.accuracy == 0.5  # 2 correct out of 4
        assert metrics.precision == 0.5  # 1 / (1 + 1)
        assert metrics.recall == 0.5  # 1 / (1 + 1)
        assert metrics.f1_score == 0.5
        assert metrics.total_samples == 4

    def test_calculate_accuracy_all_false_negatives(self, test_harness):
        """Test when all predictions are false negatives"""
        results = [
            DetectionResult("img1", "VALID", 0.3, 0.9, "INVALID"),
            DetectionResult("img2", "VALID", 0.2, 0.95, "INVALID"),
        ]

        metrics = test_harness.calculate_accuracy_metrics(results)

        assert metrics.true_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 0
        assert metrics.false_negatives == 2
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_calculate_accuracy_all_false_positives(self, test_harness):
        """Test when all predictions are false positives"""
        results = [
            DetectionResult("img1", "INVALID", 2.0, 0.8, "VALID"),
            DetectionResult("img2", "INVALID", 3.0, 0.85, "VALID"),
        ]

        metrics = test_harness.calculate_accuracy_metrics(results)

        assert metrics.true_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_positives == 2
        assert metrics.false_negatives == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0  # No actual positives to recall

    def test_calculate_accuracy_empty_results(self, test_harness):
        """Test accuracy calculation with empty results"""
        results = []

        metrics = test_harness.calculate_accuracy_metrics(results)

        assert metrics.total_samples == 0
        assert metrics.accuracy == 0.0


class TestDatasetGeneration:
    """Tests for test dataset generation"""

    def test_generate_test_dataset_creates_directories(self, test_harness, sample_image, tmp_path):
        """Test that dataset generation creates expected directories"""
        # Save sample image
        baseline_path = tmp_path / "baseline.jpg"
        cv2.imwrite(str(baseline_path), sample_image)

        # Generate dataset
        images, labels = test_harness.generate_test_dataset(
            [str(baseline_path)],
            shifts=[2],
            directions=['right', 'left'],
            include_baseline=True
        )

        # Check directories exist
        assert test_harness.baseline_dir.exists()
        assert test_harness.shift_dirs["2px"].exists()

        # Check images generated: 1 baseline + 2 shifts = 3 images
        assert len(images) == 3
        assert len(labels) == 3

    def test_generate_test_dataset_ground_truth_labels(self, test_harness, sample_image, tmp_path):
        """Test that ground truth labels are correct"""
        baseline_path = tmp_path / "test_baseline.jpg"
        cv2.imwrite(str(baseline_path), sample_image)

        images, labels = test_harness.generate_test_dataset(
            [str(baseline_path)],
            shifts=[5],
            directions=['right'],
            include_baseline=True
        )

        # Check baseline label
        baseline_label = next(l for l in labels if l.expected_status == "VALID")
        assert baseline_label.transformation["type"] == "none"
        assert baseline_label.expected_displacement_range == (0.0, 0.5)

        # Check shifted label
        shifted_label = next(l for l in labels if l.expected_status == "INVALID")
        assert shifted_label.transformation["type"] == "translate"
        assert shifted_label.transformation["magnitude_px"] == 5.0
        assert shifted_label.transformation["direction"] == "right"
        assert shifted_label.expected_displacement_range == (4.0, 7.5)  # 5px * 0.8 to 5px * 1.5

    def test_generate_test_dataset_saves_ground_truth_json(self, test_harness, sample_image, tmp_path):
        """Test that ground truth is saved to JSON file"""
        baseline_path = tmp_path / "test_baseline.jpg"
        cv2.imwrite(str(baseline_path), sample_image)

        test_harness.generate_test_dataset(
            [str(baseline_path)],
            shifts=[2],
            directions=['right'],
            include_baseline=False
        )

        # Check JSON file exists
        assert test_harness.ground_truth_file.exists()

        # Load and verify JSON structure
        with open(test_harness.ground_truth_file, 'r') as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert "image_id" in data[0]
        assert "transformation" in data[0]
        assert "expected_status" in data[0]

    def test_load_ground_truth(self, test_harness, tmp_path):
        """Test loading ground truth from JSON"""
        # Create ground truth JSON manually
        test_harness.ground_truth_file.parent.mkdir(parents=True, exist_ok=True)

        test_data = [
            {
                "image_id": "test_001",
                "baseline_image": "baseline.jpg",
                "transformation": {"type": "translate", "magnitude_px": 5.0, "direction": "right"},
                "expected_status": "INVALID",
                "expected_displacement_range": [4.0, 7.5]
            }
        ]

        with open(test_harness.ground_truth_file, 'w') as f:
            json.dump(test_data, f)

        # Load ground truth
        labels = test_harness.load_ground_truth()

        assert len(labels) == 1
        assert labels[0].image_id == "test_001"
        assert labels[0].transformation["magnitude_px"] == 5.0


class TestReportGeneration:
    """Tests for report generation"""

    def test_generate_metrics_report_passed(self, test_harness):
        """Test report generation when validation passed"""
        overall_metrics = AccuracyMetrics(
            true_positives=95,
            true_negatives=95,
            false_positives=5,
            false_negatives=5,
            accuracy=0.95,
            precision=0.95,
            recall=0.95,
            f1_score=0.95,
            total_samples=200
        )

        performance_results = {
            "overall_metrics": overall_metrics,
            "metrics_by_shift": {
                "2px": overall_metrics,
            },
            "detection_results": [],
            "passed": True,
            "timestamp": "2025-10-24T10:00:00"
        }

        report = test_harness.generate_metrics_report(performance_results)

        assert "PASSED" in report
        assert "95.00%" in report
        assert "Stage 1 Validation Results" in report
        assert "Confusion Matrix" in report

    def test_generate_metrics_report_failed(self, test_harness):
        """Test report generation when validation failed"""
        overall_metrics = AccuracyMetrics(
            true_positives=85,
            true_negatives=85,
            false_positives=15,
            false_negatives=15,
            accuracy=0.85,
            precision=0.85,
            recall=0.85,
            f1_score=0.85,
            total_samples=200
        )

        performance_results = {
            "overall_metrics": overall_metrics,
            "metrics_by_shift": {},
            "detection_results": [],
            "passed": False,
            "timestamp": "2025-10-24T10:00:00"
        }

        report = test_harness.generate_metrics_report(performance_results)

        assert "FAILED" in report
        assert "NO-GO" in report
        assert "85.00%" in report


class TestIntegration:
    """Integration tests for full validation workflow"""

    @patch('validation.stage1_test_harness.cv2.imread')
    @patch('validation.stage1_test_harness.cv2.imwrite')
    def test_run_stage1_validation_integration(self, mock_imwrite, mock_imread, tmp_path, sample_image):
        """Test complete Stage 1 validation workflow"""
        # Mock file I/O
        mock_imread.return_value = sample_image
        mock_imwrite.return_value = True

        # Create mock detector
        mock_detector = Mock()
        mock_detector.process_frame = Mock(return_value={
            "status": "INVALID",
            "displacement": 5.2,
            "confidence": 0.88
        })

        # Create baseline image path
        baseline_path = str(tmp_path / "baseline.jpg")

        # Run validation
        results = run_stage1_validation(
            detector=mock_detector,
            baseline_images=[baseline_path],
            validation_dir=str(tmp_path / "validation")
        )

        # Verify results structure
        assert "overall_metrics" in results
        assert "metrics_by_shift" in results
        assert "detection_results" in results
        assert "passed" in results
        assert "timestamp" in results

        # Verify detector was called
        assert mock_detector.process_frame.called


def test_transformation_spec_dataclass():
    """Test TransformationSpec dataclass"""
    spec = TransformationSpec(
        type="translate",
        magnitude_px=5.0,
        direction="right"
    )

    assert spec.type == "translate"
    assert spec.magnitude_px == 5.0
    assert spec.direction == "right"
    assert spec.angle_deg == 0.0  # default
    assert spec.scale_factor == 1.0  # default


def test_ground_truth_label_dataclass():
    """Test GroundTruthLabel dataclass"""
    label = GroundTruthLabel(
        image_id="test_001",
        baseline_image="baseline.jpg",
        transformation={"type": "translate", "magnitude_px": 5.0},
        expected_status="INVALID",
        expected_displacement_range=(4.0, 7.5)
    )

    assert label.image_id == "test_001"
    assert label.expected_status == "INVALID"


def test_detection_result_dataclass():
    """Test DetectionResult dataclass"""
    result = DetectionResult(
        image_id="test_001",
        predicted_status="INVALID",
        displacement=5.2,
        confidence=0.88,
        expected_status="INVALID"
    )

    assert result.predicted_status == "INVALID"
    assert result.expected_status == "INVALID"


def test_accuracy_metrics_dataclass():
    """Test AccuracyMetrics dataclass"""
    metrics = AccuracyMetrics(
        true_positives=95,
        true_negatives=95,
        false_positives=5,
        false_negatives=5,
        accuracy=0.95,
        precision=0.95,
        recall=0.95,
        f1_score=0.95,
        total_samples=200
    )

    assert metrics.accuracy == 0.95
    assert metrics.total_samples == 200
