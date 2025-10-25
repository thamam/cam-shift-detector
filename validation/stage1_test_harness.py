"""
Stage 1 Test Harness - Synthetic Transformation Validation

Automated testing framework for validating camera movement detection
using synthetic image transformations with known ground truth.

Acceptance Criteria:
- AC-1.9.1: Achieve >95% detection accuracy
- AC-1.9.4: Automated test harness with transformation and accuracy measurement
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import cv2


@dataclass
class TransformationSpec:
    """Specification for a synthetic transformation"""
    type: str  # 'translate', 'rotate', 'scale'
    magnitude_px: float  # For translation
    angle_deg: float = 0.0  # For rotation
    scale_factor: float = 1.0  # For scaling
    direction: str = ""  # For translation: 'right', 'left', 'up', 'down', 'diagonal_ur', 'diagonal_ul', 'diagonal_dr', 'diagonal_dl'


@dataclass
class GroundTruthLabel:
    """Ground truth label for a transformed image"""
    image_id: str
    baseline_image: str
    transformation: Dict[str, Any]
    expected_status: str  # "VALID" or "INVALID"
    expected_displacement_range: Tuple[float, float]


@dataclass
class DetectionResult:
    """Detection result from camera movement detector"""
    image_id: str
    predicted_status: str
    displacement: float
    confidence: float
    expected_status: str


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics"""
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_samples: int


class Stage1TestHarness:
    """
    Test harness for Stage 1 validation using synthetic transformations

    This harness:
    1. Applies known transformations to baseline images
    2. Runs the detector on transformed images
    3. Compares results against ground truth
    4. Calculates comprehensive accuracy metrics
    5. Generates validation reports
    """

    def __init__(self, validation_dir: str = "validation"):
        self.validation_dir = Path(validation_dir)
        self.stage1_data_dir = self.validation_dir / "stage1_data"
        self.baseline_dir = self.stage1_data_dir / "baseline"
        self.ground_truth_file = self.stage1_data_dir / "ground_truth.json"

        # Transformation directories
        self.shift_dirs = {
            "2px": self.stage1_data_dir / "shifted_2px",
            "5px": self.stage1_data_dir / "shifted_5px",
            "10px": self.stage1_data_dir / "shifted_10px",
        }

    def apply_translation(
        self,
        image: np.ndarray,
        shift_px: int,
        direction: str
    ) -> np.ndarray:
        """
        Apply horizontal, vertical, or diagonal translation to an image

        Args:
            image: Input image array
            shift_px: Number of pixels to shift
            direction: Direction of shift - one of:
                      'right', 'left', 'up', 'down',
                      'diagonal_ur' (up-right), 'diagonal_ul' (up-left),
                      'diagonal_dr' (down-right), 'diagonal_dl' (down-left)

        Returns:
            Transformed image array
        """
        height, width = image.shape[:2]

        # Define translation vectors for each direction
        direction_vectors = {
            'right': (shift_px, 0),
            'left': (-shift_px, 0),
            'down': (0, shift_px),
            'up': (0, -shift_px),
            'diagonal_ur': (shift_px, -shift_px),  # right + up
            'diagonal_ul': (-shift_px, -shift_px),  # left + up
            'diagonal_dr': (shift_px, shift_px),  # right + down
            'diagonal_dl': (-shift_px, shift_px),  # left + down
        }

        if direction not in direction_vectors:
            raise ValueError(f"Unknown direction: {direction}. Must be one of {list(direction_vectors.keys())}")

        dx, dy = direction_vectors[direction]

        # Create translation matrix
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply translation
        translated = cv2.warpAffine(
            image,
            translation_matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return translated

    def apply_rotation(
        self,
        image: np.ndarray,
        angle_deg: float,
        scale: float = 1.0
    ) -> np.ndarray:
        """
        Apply rotation around center of image

        Args:
            image: Input image array
            angle_deg: Rotation angle in degrees (positive = counter-clockwise)
            scale: Optional scaling factor (default 1.0)

        Returns:
            Rotated image array
        """
        height, width = image.shape[:2]
        center = (width / 2, height / 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, scale)

        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return rotated

    def generate_test_dataset(
        self,
        baseline_images: List[str],
        shifts: List[int] = [2, 5, 10],
        directions: List[str] = None,
        include_baseline: bool = True
    ) -> Tuple[List[str], List[GroundTruthLabel]]:
        """
        Generate complete synthetic test dataset with ground truth labels

        Args:
            baseline_images: List of paths to baseline images
            shifts: List of shift magnitudes in pixels (default: [2, 5, 10])
            directions: List of shift directions (default: all 8 directions)
            include_baseline: Whether to include unmodified baseline images

        Returns:
            Tuple of (list of generated image paths, list of ground truth labels)
        """
        if directions is None:
            directions = ['right', 'left', 'up', 'down',
                         'diagonal_ur', 'diagonal_ul', 'diagonal_dr', 'diagonal_dl']

        generated_images = []
        ground_truth_labels = []

        for baseline_path in baseline_images:
            baseline_name = Path(baseline_path).stem
            image = cv2.imread(baseline_path)

            if image is None:
                print(f"Warning: Could not load image {baseline_path}")
                continue

            # Include unmodified baseline as VALID example
            if include_baseline:
                baseline_output_path = self.baseline_dir / Path(baseline_path).name
                baseline_output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(baseline_output_path), image)

                label = GroundTruthLabel(
                    image_id=f"{baseline_name}_baseline",
                    baseline_image=Path(baseline_path).name,
                    transformation={"type": "none"},
                    expected_status="VALID",
                    expected_displacement_range=(0.0, 0.5)
                )
                generated_images.append(str(baseline_output_path))
                ground_truth_labels.append(label)

            # Generate shifted versions
            for shift_px in shifts:
                shift_dir = self.shift_dirs[f"{shift_px}px"]
                shift_dir.mkdir(parents=True, exist_ok=True)

                for direction in directions:
                    # Apply transformation
                    transformed = self.apply_translation(image, shift_px, direction)

                    # Generate output filename
                    image_id = f"{baseline_name}_shift_{shift_px}px_{direction}"
                    output_filename = f"{image_id}.jpg"
                    output_path = shift_dir / output_filename

                    # Save transformed image
                    cv2.imwrite(str(output_path), transformed)

                    # Create ground truth label
                    label = GroundTruthLabel(
                        image_id=image_id,
                        baseline_image=Path(baseline_path).name,
                        transformation={
                            "type": "translate",
                            "magnitude_px": float(shift_px),
                            "direction": direction
                        },
                        expected_status="INVALID",
                        expected_displacement_range=(float(shift_px) * 0.8, float(shift_px) * 1.5)
                    )

                    generated_images.append(str(output_path))
                    ground_truth_labels.append(label)

        # Save ground truth to JSON
        self.save_ground_truth(ground_truth_labels)

        return generated_images, ground_truth_labels

    def save_ground_truth(self, labels: List[GroundTruthLabel]):
        """Save ground truth labels to JSON file"""
        self.ground_truth_file.parent.mkdir(parents=True, exist_ok=True)

        labels_dict = [asdict(label) for label in labels]

        with open(self.ground_truth_file, 'w') as f:
            json.dump(labels_dict, f, indent=2)

        print(f"Ground truth saved to {self.ground_truth_file}")

    def load_ground_truth(self) -> List[GroundTruthLabel]:
        """Load ground truth labels from JSON file"""
        with open(self.ground_truth_file, 'r') as f:
            labels_dict = json.load(f)

        labels = [
            GroundTruthLabel(**label_dict)
            for label_dict in labels_dict
        ]

        return labels

    def calculate_accuracy_metrics(
        self,
        detection_results: List[DetectionResult]
    ) -> AccuracyMetrics:
        """
        Calculate comprehensive accuracy metrics (TP, TN, FP, FN, precision, recall, F1)

        Args:
            detection_results: List of detection results with predictions and ground truth

        Returns:
            AccuracyMetrics object with all calculated metrics
        """
        tp = sum(1 for r in detection_results
                 if r.predicted_status == "INVALID" and r.expected_status == "INVALID")
        tn = sum(1 for r in detection_results
                 if r.predicted_status == "VALID" and r.expected_status == "VALID")
        fp = sum(1 for r in detection_results
                 if r.predicted_status == "INVALID" and r.expected_status == "VALID")
        fn = sum(1 for r in detection_results
                 if r.predicted_status == "VALID" and r.expected_status == "INVALID")

        total = len(detection_results)

        # Calculate metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return AccuracyMetrics(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_samples=total
        )

    def evaluate_stage1_performance(
        self,
        detector,
        test_images: List[str],
        ground_truth_labels: List[GroundTruthLabel]
    ) -> Dict[str, Any]:
        """
        Run detector on synthetic dataset and measure accuracy

        Args:
            detector: CameraMovementDetector instance
            test_images: List of test image paths
            ground_truth_labels: List of ground truth labels

        Returns:
            Dictionary containing:
            - overall_metrics: Overall accuracy metrics
            - metrics_by_shift: Metrics broken down by shift magnitude
            - detection_results: Individual detection results
            - passed: Boolean indicating if >95% accuracy threshold met
        """
        # Create lookup for ground truth
        gt_lookup = {label.image_id: label for label in ground_truth_labels}

        detection_results = []
        results_by_shift = {"baseline": [], "2px": [], "5px": [], "10px": []}

        for image_path in test_images:
            # Extract image_id from path
            image_stem = Path(image_path).stem

            # Find corresponding ground truth
            if image_stem not in gt_lookup:
                print(f"Warning: No ground truth for {image_stem}")
                continue

            gt_label = gt_lookup[image_stem]

            # Load image and run detection
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load {image_path}")
                continue

            # Run detector
            result = detector.process_frame(image, frame_id=image_stem)

            # Create detection result
            det_result = DetectionResult(
                image_id=image_stem,
                predicted_status=result["status"],
                displacement=result.get("displacement", 0.0),
                confidence=result.get("confidence", 0.0),
                expected_status=gt_label.expected_status
            )

            detection_results.append(det_result)

            # Categorize by shift magnitude
            if "baseline" in image_stem:
                results_by_shift["baseline"].append(det_result)
            elif "2px" in image_path:
                results_by_shift["2px"].append(det_result)
            elif "5px" in image_path:
                results_by_shift["5px"].append(det_result)
            elif "10px" in image_path:
                results_by_shift["10px"].append(det_result)

        # Calculate overall metrics
        overall_metrics = self.calculate_accuracy_metrics(detection_results)

        # Calculate metrics by shift magnitude
        metrics_by_shift = {}
        for shift, results in results_by_shift.items():
            if results:
                metrics_by_shift[shift] = self.calculate_accuracy_metrics(results)

        # Check if accuracy threshold met (>95%)
        passed = overall_metrics.accuracy > 0.95

        return {
            "overall_metrics": overall_metrics,
            "metrics_by_shift": metrics_by_shift,
            "detection_results": detection_results,
            "passed": passed,
            "timestamp": datetime.now().isoformat()
        }

    def generate_metrics_report(
        self,
        performance_results: Dict[str, Any],
        output_file: str = None
    ) -> str:
        """
        Generate comprehensive accuracy metrics report

        Args:
            performance_results: Results from evaluate_stage1_performance()
            output_file: Optional path to save report (default: validation/stage1_results.md)

        Returns:
            Report content as string
        """
        if output_file is None:
            output_file = self.validation_dir / "stage1_results.md"

        overall = performance_results["overall_metrics"]
        by_shift = performance_results["metrics_by_shift"]
        passed = performance_results["passed"]

        # Build report
        lines = [
            "# Stage 1 Validation Results - Synthetic Transformations",
            "",
            f"**Test Date**: {performance_results['timestamp']}",
            f"**Total Samples**: {overall.total_samples}",
            f"**Accuracy Threshold**: >95%",
            f"**Result**: {'✅ PASSED' if passed else '❌ FAILED'}",
            "",
            "## Overall Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Accuracy | {overall.accuracy:.2%} |",
            f"| Precision | {overall.precision:.2%} |",
            f"| Recall | {overall.recall:.2%} |",
            f"| F1-Score | {overall.f1_score:.2%} |",
            "",
            "## Confusion Matrix",
            "",
            f"| | Predicted VALID | Predicted INVALID |",
            f"|---|---|---|",
            f"| **Actual VALID** | {overall.true_negatives} (TN) | {overall.false_positives} (FP) |",
            f"| **Actual INVALID** | {overall.false_negatives} (FN) | {overall.true_positives} (TP) |",
            "",
            "## Performance by Shift Magnitude",
            "",
        ]

        for shift, metrics in sorted(by_shift.items()):
            lines.extend([
                f"### {shift.upper()}",
                "",
                f"- **Samples**: {metrics.total_samples}",
                f"- **Accuracy**: {metrics.accuracy:.2%}",
                f"- **Precision**: {metrics.precision:.2%}",
                f"- **Recall**: {metrics.recall:.2%}",
                f"- **F1-Score**: {metrics.f1_score:.2%}",
                f"- **Confusion**: TP={metrics.true_positives}, TN={metrics.true_negatives}, FP={metrics.false_positives}, FN={metrics.false_negatives}",
                "",
            ])

        # Add failure analysis if any failures
        if overall.false_negatives > 0 or overall.false_positives > 0:
            lines.extend([
                "## Failure Analysis",
                "",
            ])

            if overall.false_negatives > 0:
                lines.append(f"- **False Negatives**: {overall.false_negatives} cases where camera movement was NOT detected")

            if overall.false_positives > 0:
                lines.append(f"- **False Positives**: {overall.false_positives} cases where stable images were incorrectly flagged as moved")

            lines.append("")

        # Add recommendations
        lines.extend([
            "## Recommendations",
            "",
        ])

        if passed:
            lines.append("✅ **GO**: Detection accuracy exceeds 95% threshold. System is ready for Stage 2 validation.")
        else:
            lines.extend([
                "❌ **NO-GO**: Detection accuracy below 95% threshold. Recommendations:",
                "",
                "1. Review false negative cases to identify systematic detection failures",
                "2. Analyze false positive cases to reduce over-sensitivity",
                "3. Consider adjusting detection thresholds or feature matching parameters",
                "4. Re-run validation after improvements",
            ])

        report_content = "\n".join(lines)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_content)

        print(f"\n{'='*60}")
        print(f"Stage 1 Results Report saved to: {output_path}")
        print(f"{'='*60}\n")

        return report_content


def run_stage1_validation(
    detector,
    baseline_images: List[str],
    validation_dir: str = "validation"
) -> Dict[str, Any]:
    """
    Convenience function to run complete Stage 1 validation

    Args:
        detector: Initialized CameraMovementDetector with baseline set
        baseline_images: List of paths to baseline images
        validation_dir: Path to validation directory

    Returns:
        Performance results dictionary
    """
    harness = Stage1TestHarness(validation_dir)

    # Generate test dataset
    print("Generating synthetic test dataset...")
    test_images, ground_truth = harness.generate_test_dataset(baseline_images)
    print(f"Generated {len(test_images)} test images")

    # Run validation
    print("\nRunning Stage 1 validation...")
    results = harness.evaluate_stage1_performance(detector, test_images, ground_truth)

    # Generate report
    print("\nGenerating metrics report...")
    harness.generate_metrics_report(results)

    return results
