"""
Stage 3 Test Harness - Automated Validation Framework

Executes camera shift detector against real DAF imagery, compares results
with ground truth annotations, and calculates systematic performance metrics.
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import cv2
import numpy as np

from validation.utilities.real_data_loader import RealDataLoader, ImageMetadata
from validation.utilities.performance_profiler import PerformanceProfiler, PerformanceMetrics
from src.camera_movement_detector import CameraMovementDetector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result for single image detection with ground truth comparison.

    Attributes:
        image_path: Path to validated image
        site_id: DAF site identifier
        ground_truth: Expected result (True if shift detected)
        predicted: Detector prediction (True if shift detected)
        is_correct: Whether prediction matches ground truth
        detection_time_ms: Time taken for detection (milliseconds)
        error_message: Error details if detection failed (None if successful)
    """
    image_path: Path
    site_id: str
    ground_truth: bool
    predicted: bool
    is_correct: bool
    detection_time_ms: float
    error_message: Optional[str] = None


@dataclass
class Metrics:
    """Comprehensive validation metrics for test harness execution.

    Attributes:
        total_images: Total number of images processed
        true_positives: Correctly detected shifts (TP)
        true_negatives: Correctly detected stable cameras (TN)
        false_positives: Incorrectly detected shifts (FP)
        false_negatives: Missed camera shifts (FN)
        accuracy: Overall detection accuracy (TP + TN) / Total
        false_positive_rate: FP / (FP + TN)
        false_negative_rate: FN / (FN + TP)
        confusion_matrix: Dict with TP, TN, FP, FN counts
        site_breakdown: Per-site metrics (accuracy, image counts)
        total_time_seconds: Total validation execution time
        errors_count: Number of detection failures
    """
    total_images: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    confusion_matrix: Dict[str, int]
    site_breakdown: Dict[str, Dict]
    total_time_seconds: float
    errors_count: int


class Stage3TestHarness:
    """Automated test harness for Stage 3 real-world validation.

    Executes camera shift detector against 50 real DAF images, compares
    predictions with ground truth annotations, and generates comprehensive
    performance metrics with per-site breakdown.
    """

    def __init__(self, detector_config_path: str,
                 sample_images_root: Optional[Path] = None,
                 ground_truth_path: Optional[Path] = None):
        """Initialize test harness with detector and data loader.

        Args:
            detector_config_path: Path to detector configuration JSON
            sample_images_root: Optional custom sample images directory
            ground_truth_path: Optional custom ground truth file path
        """
        self.detector = CameraMovementDetector(detector_config_path)
        self.data_loader = RealDataLoader(sample_images_root, ground_truth_path)
        self.profiler = PerformanceProfiler()
        self.results: List[DetectionResult] = []

        logger.info(f"Initialized Stage3TestHarness with config: {detector_config_path}")
        logger.info("Performance profiler enabled")

    def run_validation(self, baseline_image_path: Optional[Path] = None) -> tuple:
        """Execute full validation on all 50 images with progress reporting.

        Main execution loop that:
        1. Loads all validation images with ground truth
        2. Sets baseline for detector
        3. Executes detection on each image with performance profiling
        4. Compares predictions with ground truth
        5. Calculates comprehensive metrics

        Args:
            baseline_image_path: Optional explicit baseline image path.
                                If None, uses first image as baseline.

        Returns:
            Tuple of (Metrics, PerformanceMetrics) with comprehensive validation results.
        """
        logger.info("=" * 60)
        logger.info("Starting Stage 3 Validation")
        logger.info("=" * 60)

        start_time = time.time()

        # Load validation dataset
        logger.info("Loading validation dataset...")
        images = self.data_loader.load_dataset()
        logger.info(f"Loaded {len(images)} images from 3 DAF sites")

        # Set baseline (use first image if not specified)
        if baseline_image_path is None:
            baseline_image_path = images[0].image_path
            logger.info(f"Using first image as baseline: {baseline_image_path.name}")

        baseline_image = self._load_and_convert_image(baseline_image_path)
        self.detector.set_baseline(baseline_image)
        logger.info("Baseline set successfully")

        # Execute detection on all images
        logger.info(f"\nProcessing {len(images)} images...")
        for i, image_metadata in enumerate(images, 1):
            logger.info(f"Processing image {i}/{len(images)}: {image_metadata.image_path.name}")
            result = self._execute_single_detection(image_metadata)
            self.results.append(result)

            # Progress indicator
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(images)} images processed ({i*100//len(images)}%)")

        # Calculate metrics
        total_time = time.time() - start_time
        logger.info(f"\nDetection complete! Total time: {total_time:.2f}s")

        metrics = self.calculate_metrics(total_time)
        self._log_metrics_summary(metrics)

        # Get performance metrics
        performance_metrics = self.profiler.get_metrics()
        self.profiler.log_summary()

        return metrics, performance_metrics

    def _execute_single_detection(self, image_metadata: ImageMetadata) -> DetectionResult:
        """Execute detector on single image with error handling and profiling.

        Loads image, converts RGB→BGR, runs detector with performance profiling,
        compares with ground truth. Handles detection errors gracefully by logging
        and continuing validation.

        Args:
            image_metadata: Metadata for image to process

        Returns:
            DetectionResult with prediction and timing information.
        """
        error_message = None
        predicted = False  # Default to no shift if detection fails
        detection_time_ms = 0.0

        try:
            # Load image (returns RGB)
            image_rgb = self.data_loader.load_image(image_metadata.image_path)

            # Convert RGB → BGR for detector
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Run detection with performance profiling
            detection_result, detection_time_seconds = self.profiler.profile_detection(
                self.detector.process_frame,
                image_bgr,
                frame_id=image_metadata.image_path.stem
            )

            detection_time_ms = detection_time_seconds * 1000

            # Interpret result: INVALID status indicates camera shift
            predicted = (detection_result['status'] == 'INVALID')

        except Exception as e:
            error_message = str(e)
            logger.error(f"Detection failed for {image_metadata.image_path.name}: {error_message}")
            predicted = False  # Conservative: assume stable on error

        # Compare with ground truth
        ground_truth = image_metadata.has_shift or False  # Handle None case
        is_correct = self._compare_with_ground_truth(predicted, ground_truth)

        return DetectionResult(
            image_path=image_metadata.image_path,
            site_id=image_metadata.site_id,
            ground_truth=ground_truth,
            predicted=predicted,
            is_correct=is_correct,
            detection_time_ms=detection_time_ms,
            error_message=error_message
        )

    def _compare_with_ground_truth(self, predicted: bool, ground_truth: bool) -> bool:
        """Compare detector prediction with ground truth annotation.

        Args:
            predicted: Detector prediction (True if shift detected)
            ground_truth: Expected result from annotations

        Returns:
            True if prediction matches ground truth, False otherwise.
        """
        return predicted == ground_truth

    def _load_and_convert_image(self, image_path: Path) -> np.ndarray:
        """Load image and convert RGB→BGR for detector.

        Args:
            image_path: Path to image file

        Returns:
            Image in BGR format (H × W × 3, uint8)
        """
        image_rgb = self.data_loader.load_image(image_path)
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def calculate_metrics(self, total_time_seconds: float) -> Metrics:
        """Calculate comprehensive metrics from detection results.

        Computes:
        - Confusion matrix (TP, TN, FP, FN)
        - Overall accuracy
        - False positive rate (FPR)
        - False negative rate (FNR)
        - Per-site breakdown

        Args:
            total_time_seconds: Total validation execution time

        Returns:
            Metrics object with all calculated statistics.
        """
        # Calculate confusion matrix
        tp = sum(1 for r in self.results if r.ground_truth and r.predicted)
        tn = sum(1 for r in self.results if not r.ground_truth and not r.predicted)
        fp = sum(1 for r in self.results if not r.ground_truth and r.predicted)
        fn = sum(1 for r in self.results if r.ground_truth and not r.predicted)

        total = len(self.results)
        errors_count = sum(1 for r in self.results if r.error_message is not None)

        # Calculate rates (with safe division)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # Per-site breakdown
        site_breakdown = self.calculate_site_breakdown()

        return Metrics(
            total_images=total,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy=round(accuracy, 4),  # 4 decimal places precision
            false_positive_rate=round(fpr, 4),
            false_negative_rate=round(fnr, 4),
            confusion_matrix={
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            },
            site_breakdown=site_breakdown,
            total_time_seconds=round(total_time_seconds, 2),
            errors_count=errors_count
        )

    def calculate_site_breakdown(self) -> Dict[str, Dict]:
        """Calculate per-site accuracy breakdown.

        Returns:
            Dictionary mapping site_id to metrics:
            {
                'of_jerusalem': {'accuracy': float, 'total': int, 'correct': int},
                'carmit': {...},
                'gad': {...}
            }
        """
        # Group results by site
        site_results = {}
        for result in self.results:
            site_id = result.site_id
            if site_id not in site_results:
                site_results[site_id] = []
            site_results[site_id].append(result)

        # Calculate per-site accuracy
        site_breakdown = {}
        for site_id, results in site_results.items():
            total = len(results)
            correct = sum(1 for r in results if r.is_correct)
            accuracy = correct / total if total > 0 else 0.0

            site_breakdown[site_id] = {
                'accuracy': round(accuracy, 4),
                'total': total,
                'correct': correct
            }

        return site_breakdown

    def _log_metrics_summary(self, metrics: Metrics):
        """Log comprehensive metrics summary to console.

        Args:
            metrics: Calculated metrics to display
        """
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nOverall Performance:")
        logger.info(f"  Total Images: {metrics.total_images}")
        logger.info(f"  Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
        logger.info(f"  Detection Errors: {metrics.errors_count}")
        logger.info(f"  Execution Time: {metrics.total_time_seconds:.2f}s")

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Positives (TP):  {metrics.true_positives}")
        logger.info(f"  True Negatives (TN):  {metrics.true_negatives}")
        logger.info(f"  False Positives (FP): {metrics.false_positives}")
        logger.info(f"  False Negatives (FN): {metrics.false_negatives}")

        logger.info(f"\nError Rates:")
        logger.info(f"  False Positive Rate: {metrics.false_positive_rate:.4f}")
        logger.info(f"  False Negative Rate: {metrics.false_negative_rate:.4f}")

        logger.info(f"\nPer-Site Breakdown:")
        for site_id, site_metrics in metrics.site_breakdown.items():
            logger.info(f"  {site_id:15s} - Accuracy: {site_metrics['accuracy']:.4f} "
                       f"({site_metrics['correct']}/{site_metrics['total']} correct)")

        logger.info("=" * 60)
