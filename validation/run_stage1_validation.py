#!/usr/bin/env python3
"""
Stage 1 Validation Execution Script

Runs the camera movement detector on the Stage 1 synthetic test dataset
and measures accuracy against ground truth labels. Validates that the
detector meets the >95% accuracy threshold (AC-1.9.1).

Usage:
    python validation/run_stage1_validation.py

Inputs:
    - validation/stage1_data/ground_truth.json - Ground truth labels
    - validation/stage1_data/baseline/*.jpg - Baseline images
    - validation/stage1_data/shifted_*/*.jpg - Transformed test images
    - config.json - Detector configuration

Outputs:
    - validation/stage1_results.json - Complete validation results
    - validation/stage1_results_report.txt - Human-readable report
    - Console output with real-time progress and final metrics

Validation Logic:
    1. Load ground truth labels (1250 images)
    2. Group images by baseline (50 unique baselines)
    3. For each baseline:
       - Initialize detector with config.json
       - Set baseline image
       - Process all transformed versions (baseline + 24 shifts = 25 images per baseline)
       - Collect detection results
    4. Calculate confusion matrix metrics (TP, TN, FP, FN)
    5. Calculate derived metrics (accuracy, precision, recall, F1-score)
    6. Analyze per-shift-magnitude performance (2px, 5px, 10px)
    7. Generate comprehensive results report
    8. Verify >95% accuracy threshold (GO/NO-GO decision)
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_movement_detector import CameraMovementDetector
from validation.stage1_test_harness import (
    Stage1TestHarness, DetectionResult, GroundTruthLabel
)


def load_ground_truth(gt_file: Path) -> list:
    """Load ground truth labels from JSON file"""
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    # Convert to GroundTruthLabel objects
    labels = []
    for entry in gt_data:
        label = GroundTruthLabel(
            image_id=entry['image_id'],
            baseline_image=entry['baseline_image'],
            transformation=entry['transformation'],
            expected_status=entry['expected_status'],
            expected_displacement_range=tuple(entry['expected_displacement_range'])
        )
        labels.append(label)

    return labels


def group_by_baseline(labels: list) -> dict:
    """Group ground truth labels by baseline image"""
    grouped = defaultdict(list)
    for label in labels:
        grouped[label.baseline_image].append(label)
    return grouped


def run_validation(config_path: str = 'config.json') -> dict:
    """Execute Stage 1 validation"""

    print("=" * 80)
    print("Stage 1 Validation Execution")
    print("=" * 80)
    print()

    # Load ground truth
    print("📋 Loading ground truth labels...")
    gt_file = Path("validation/stage1_data/ground_truth.json")
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")

    labels = load_ground_truth(gt_file)
    print(f"   Loaded {len(labels)} ground truth labels")

    # Group by baseline
    print("\n🔍 Grouping images by baseline...")
    baseline_groups = group_by_baseline(labels)
    print(f"   Found {len(baseline_groups)} unique baselines")
    print(f"   Images per baseline: {len(labels) // len(baseline_groups)}")

    # Initialize test harness for metrics calculation
    print("\n🔧 Initializing Stage1TestHarness...")
    harness = Stage1TestHarness(validation_dir="validation")

    # Detection results storage
    all_detection_results = []

    # Process each baseline group
    print(f"\n🚀 Processing {len(baseline_groups)} baseline groups...")
    print(f"   Total images to process: {len(labels)}")
    print()

    start_time = datetime.now()
    processed_count = 0

    for baseline_idx, (baseline_name, baseline_labels) in enumerate(baseline_groups.items(), 1):
        print(f"[{baseline_idx}/{len(baseline_groups)}] Processing baseline: {baseline_name}")

        # Find the baseline image label
        baseline_label = next((l for l in baseline_labels if l.expected_status == "VALID"), None)
        if not baseline_label:
            print(f"   ⚠️  Warning: No baseline label found for {baseline_name}, skipping...")
            continue

        # Load baseline image
        baseline_image_path = Path("validation/stage1_data/baseline") / baseline_name
        if not baseline_image_path.exists():
            print(f"   ⚠️  Warning: Baseline image not found: {baseline_image_path}, skipping...")
            continue

        baseline_image = cv2.imread(str(baseline_image_path))
        if baseline_image is None:
            print(f"   ⚠️  Warning: Failed to load baseline image: {baseline_image_path}, skipping...")
            continue

        # Initialize detector for this baseline
        try:
            detector = CameraMovementDetector(config_path=config_path)
            detector.set_baseline(baseline_image)
        except Exception as e:
            print(f"   ❌ Error initializing detector: {e}, skipping...")
            continue

        # Process all images for this baseline
        baseline_results = []

        for label in baseline_labels:
            # Determine image path based on transformation
            if label.transformation['type'] == 'none':
                # Baseline image - strip "_baseline" suffix from image_id
                base_id = label.image_id.replace("_baseline", "")
                image_path = Path("validation/stage1_data/baseline") / f"{base_id}.jpg"
            else:
                # Transformed image
                shift_px = int(label.transformation['magnitude_px'])
                image_path = Path(f"validation/stage1_data/shifted_{shift_px}px") / f"{label.image_id}.jpg"

            # Load test image
            if not image_path.exists():
                print(f"      ⚠️  Warning: Image not found: {image_path}, skipping...")
                continue

            test_image = cv2.imread(str(image_path))
            if test_image is None:
                print(f"      ⚠️  Warning: Failed to load image: {image_path}, skipping...")
                continue

            # Run detection
            try:
                result = detector.process_frame(test_image, frame_id=label.image_id)

                # Create detection result
                detection_result = DetectionResult(
                    image_id=label.image_id,
                    predicted_status=result['status'],
                    displacement=result['translation_displacement'],
                    confidence=result['confidence'],
                    expected_status=label.expected_status
                )

                baseline_results.append(detection_result)
                processed_count += 1

            except Exception as e:
                print(f"      ❌ Error processing {label.image_id}: {e}, skipping...")
                continue

        # Store results for this baseline
        all_detection_results.extend(baseline_results)

        # Show progress
        correct = sum(1 for r in baseline_results if r.predicted_status == r.expected_status)
        accuracy = (correct / len(baseline_results) * 100) if baseline_results else 0
        print(f"   Processed {len(baseline_results)} images, accuracy: {accuracy:.1f}%")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n✅ Detection complete!")
    print(f"   Processed: {processed_count} images")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Processing rate: {processed_count / duration:.1f} images/second")

    # Calculate accuracy metrics
    print(f"\n📊 Calculating accuracy metrics...")
    overall_metrics = harness.calculate_accuracy_metrics(all_detection_results)

    print(f"\n📈 Overall Performance:")
    print(f"   Accuracy: {overall_metrics.accuracy * 100:.2f}%")
    print(f"   Precision: {overall_metrics.precision * 100:.2f}%")
    print(f"   Recall: {overall_metrics.recall * 100:.2f}%")
    print(f"   F1-Score: {overall_metrics.f1_score * 100:.2f}%")
    print(f"\n   Confusion Matrix:")
    print(f"      True Positives: {overall_metrics.true_positives}")
    print(f"      True Negatives: {overall_metrics.true_negatives}")
    print(f"      False Positives: {overall_metrics.false_positives}")
    print(f"      False Negatives: {overall_metrics.false_negatives}")

    # Calculate per-shift-magnitude metrics
    print(f"\n📊 Per-Shift-Magnitude Performance:")
    metrics_by_shift = {}

    for shift_magnitude in [2, 5, 10]:
        # Filter results for this shift magnitude
        shift_results = []
        for result in all_detection_results:
            # Find corresponding label
            label = next((l for l in labels if l.image_id == result.image_id), None)
            if label and label.transformation.get('magnitude_px') == float(shift_magnitude):
                shift_results.append(result)

        if shift_results:
            shift_metrics = harness.calculate_accuracy_metrics(shift_results)
            metrics_by_shift[f"{shift_magnitude}px"] = shift_metrics

            print(f"\n   {shift_magnitude}px shifts:")
            print(f"      Accuracy: {shift_metrics.accuracy * 100:.2f}%")
            print(f"      Samples: {shift_metrics.total_samples}")
        else:
            print(f"\n   {shift_magnitude}px shifts: No results")

    # Check if >95% accuracy threshold achieved
    accuracy_threshold = 0.95
    passed = overall_metrics.accuracy >= accuracy_threshold

    print(f"\n{'='*80}")
    if passed:
        print(f"✅ PASS: Stage 1 Validation SUCCESSFUL")
        print(f"   Accuracy {overall_metrics.accuracy * 100:.2f}% >= {accuracy_threshold * 100:.0f}% threshold")
        print(f"   GO for Stage 2 validation")
    else:
        print(f"❌ FAIL: Stage 1 Validation FAILED")
        print(f"   Accuracy {overall_metrics.accuracy * 100:.2f}% < {accuracy_threshold * 100:.0f}% threshold")
        print(f"   NO-GO: Detector requires improvement before Stage 2")
    print(f"{'='*80}")

    # Build performance results dictionary
    performance_results = {
        "overall_metrics": overall_metrics,
        "metrics_by_shift": metrics_by_shift,
        "detection_results": all_detection_results,
        "passed": passed,
        "timestamp": datetime.now().isoformat(),
        "config_used": config_path,
        "total_images_processed": processed_count,
        "duration_seconds": duration,
        "processing_rate": processed_count / duration if duration > 0 else 0
    }

    return performance_results


def save_results(results: dict, output_dir: Path):
    """Save validation results to files"""

    print(f"\n💾 Saving results...")

    # Save JSON results
    json_output = output_dir / "stage1_results.json"

    # Convert results to JSON-serializable format
    json_results = {
        "overall_metrics": {
            "true_positives": results["overall_metrics"].true_positives,
            "true_negatives": results["overall_metrics"].true_negatives,
            "false_positives": results["overall_metrics"].false_positives,
            "false_negatives": results["overall_metrics"].false_negatives,
            "accuracy": results["overall_metrics"].accuracy,
            "precision": results["overall_metrics"].precision,
            "recall": results["overall_metrics"].recall,
            "f1_score": results["overall_metrics"].f1_score,
            "total_samples": results["overall_metrics"].total_samples
        },
        "metrics_by_shift": {
            shift: {
                "true_positives": metrics.true_positives,
                "true_negatives": metrics.true_negatives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "total_samples": metrics.total_samples
            }
            for shift, metrics in results["metrics_by_shift"].items()
        },
        "detection_results": [
            {
                "image_id": r.image_id,
                "predicted_status": r.predicted_status,
                "displacement": r.displacement,
                "confidence": r.confidence,
                "expected_status": r.expected_status
            }
            for r in results["detection_results"]
        ],
        "passed": results["passed"],
        "timestamp": results["timestamp"],
        "config_used": results["config_used"],
        "total_images_processed": results["total_images_processed"],
        "duration_seconds": results["duration_seconds"],
        "processing_rate": results["processing_rate"]
    }

    with open(json_output, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   JSON results: {json_output}")

    # Generate and save text report
    harness = Stage1TestHarness(validation_dir=str(output_dir))
    report = harness.generate_metrics_report(results)

    report_output = output_dir / "stage1_results_report.txt"
    with open(report_output, 'w') as f:
        f.write(report)
    print(f"   Text report: {report_output}")

    print(f"\n✅ Results saved successfully!")


def main():
    """Main execution function"""

    try:
        # Run validation
        results = run_validation(config_path='config.json')

        # Save results
        output_dir = Path("validation")
        save_results(results, output_dir)

        # Return exit code based on pass/fail
        return 0 if results["passed"] else 1

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
