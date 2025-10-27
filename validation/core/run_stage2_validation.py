#!/usr/bin/env python3
"""
Stage 2 Validation Execution Script

Runs the camera movement detector on Stage 2 temporal sequences and measures
detection rate against AC-1.9.2: 100% detection rate (0% false negatives).

Usage:
    python validation/run_stage2_validation.py

Inputs:
    - validation/stage2_data/ground_truth_sequences.json - Sequence metadata
    - validation/stage2_data/pattern_*/sequence_*/frame_*.jpg - Frame images
    - validation/stage2_data/pattern_*/sequence_*/frame_annotations.json - Ground truth
    - config.json - Detector configuration

Outputs:
    - validation/stage2_results.json - Complete validation results
    - validation/stage2_results_report.txt - Human-readable report
    - Console output with real-time progress and final metrics

Validation Logic:
    1. Load all 60 sequences with ground truth annotations
    2. For each sequence:
       - Initialize detector with baseline frame (frame 0)
       - Process all subsequent frames maintaining detector state
       - Compare predictions against ground truth annotations
       - Track false negatives, false positives, latency
    3. Calculate per-pattern and overall detection metrics
    4. Verify 100% detection rate (AC-1.9.2)
    5. Generate comprehensive results report
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_movement_detector import CameraMovementDetector
from validation.harnesses.stage2_test_harness import (
    Stage2TestHarness, FrameAnnotation, SequenceMetadata, SequenceDetectionMetrics
)


def load_ground_truth(stage2_data_dir: Path) -> Dict:
    """Load global ground truth sequences metadata"""
    gt_file = stage2_data_dir / "ground_truth_sequences.json"

    with open(gt_file, 'r') as f:
        return json.load(f)


def load_sequence_annotations(sequence_dir: Path) -> List[FrameAnnotation]:
    """Load per-frame annotations for a sequence"""
    annotations_file = sequence_dir / "frame_annotations.json"

    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Convert to FrameAnnotation objects
    annotations = []
    for frame_data in data['frames']:
        annotation = FrameAnnotation(
            frame_number=frame_data['frame_number'],
            frame_id=frame_data['frame_id'],
            timestamp=frame_data['timestamp'],
            cumulative_shift_px=frame_data['cumulative_shift_px'],
            shift_from_previous_px=frame_data['shift_from_previous_px'],
            expected_status=frame_data['expected_status'],
            expected_displacement_range=tuple(frame_data['expected_displacement_range']),
            movement_vector=frame_data['movement_vector'],
            is_critical_transition=frame_data['is_critical_transition'],
            transition_notes=frame_data.get('transition_notes', "")
        )
        annotations.append(annotation)

    return annotations


def load_sequence_metadata(sequence_dir: Path) -> SequenceMetadata:
    """Load sequence metadata"""
    metadata_file = sequence_dir / "sequence_metadata.json"

    with open(metadata_file, 'r') as f:
        data = json.load(f)

    metadata = SequenceMetadata(
        sequence_id=data['sequence_id'],
        pattern_type=data['pattern_type'],
        baseline_image=data['baseline_image'],
        total_frames=data['total_frames'],
        frame_rate=data['frame_rate'],
        duration_seconds=data['duration_seconds'],
        movement_description=data['movement_description'],
        expected_detections=data['expected_detections'],
        expected_first_invalid_frame=data.get('expected_first_invalid_frame'),
        critical_frames=data['critical_frames'],
        direction=data.get('direction', ""),
        max_displacement_px=data.get('max_displacement_px', 0.0)
    )

    return metadata


def process_sequence(
    sequence_dir: Path,
    detector: CameraMovementDetector,
    metadata: SequenceMetadata,
    annotations: List[FrameAnnotation]
) -> List[Dict]:
    """
    Process a temporal sequence with the detector.

    Args:
        sequence_dir: Path to sequence directory
        detector: Initialized detector instance
        metadata: Sequence metadata
        annotations: Per-frame ground truth annotations

    Returns:
        List of detection results
    """
    results = []

    # Process frames in sequence
    frame_files = sorted(sequence_dir.glob("frame_*.jpg"), key=lambda x: int(x.stem.split('_')[1]))

    # Set baseline (frame 0)
    baseline_path = frame_files[0]
    baseline_image = cv2.imread(str(baseline_path))
    if baseline_image is None:
        raise ValueError(f"Failed to load baseline: {baseline_path}")

    detector.set_baseline(baseline_image)

    # Process all frames (including baseline as frame 0)
    for frame_file in frame_files:
        frame_image = cv2.imread(str(frame_file))
        if frame_image is None:
            print(f"  ⚠️  Warning: Failed to load frame: {frame_file}")
            continue

        frame_id = frame_file.stem
        result = detector.process_frame(frame_image, frame_id=frame_id)
        results.append(result)

    return results


def run_validation(config_path: str = 'config.json') -> Dict:
    """Execute Stage 2 validation on temporal sequences"""

    print("=" * 80)
    print("Stage 2 Validation Execution - Temporal Sequences")
    print("=" * 80)
    print()

    # Load ground truth
    print("📋 Loading ground truth sequences...")
    stage2_data_dir = Path("validation/stage2_data")
    if not stage2_data_dir.exists():
        raise FileNotFoundError(f"Stage 2 data directory not found: {stage2_data_dir}")

    ground_truth = load_ground_truth(stage2_data_dir)
    total_sequences = ground_truth['total_sequences']
    total_frames = ground_truth['total_frames']

    print(f"   Total sequences: {total_sequences}")
    print(f"   Total frames: {total_frames}")
    print(f"   Patterns: {ground_truth['patterns']}")

    # Initialize test harness
    print("\n🔧 Initializing Stage2TestHarness...")
    harness = Stage2TestHarness(threshold_px=1.5)

    # Process all sequences
    print(f"\n🚀 Processing {total_sequences} temporal sequences...")
    print()

    start_time = datetime.now()
    all_sequence_metrics = []
    processed_frames_count = 0

    for seq_idx, seq_info in enumerate(ground_truth['sequences'], 1):
        sequence_id = seq_info['sequence_id']
        pattern_type = seq_info['pattern_type']

        print(f"[{seq_idx}/{total_sequences}] {sequence_id}")

        # Locate sequence directory - map pattern type to directory name
        pattern_dir_map = {
            'gradual_onset': 'pattern_1_gradual_onset',
            'sudden_onset': 'pattern_2_sudden_onset',
            'progressive': 'pattern_3_progressive',
            'oscillation': 'pattern_4_oscillation',
            'recovery': 'pattern_5_recovery',
            'multi_axis': 'pattern_6_multi_axis'
        }

        pattern_dir_name = pattern_dir_map.get(pattern_type)
        if not pattern_dir_name:
            print(f"  ⚠️  Warning: Unknown pattern type: {pattern_type}, skipping...")
            continue

        pattern_dir = stage2_data_dir / pattern_dir_name
        sequence_dir = pattern_dir / sequence_id

        if not sequence_dir.exists():
            print(f"  ⚠️  Warning: Sequence directory not found: {sequence_dir}, skipping...")
            continue

        # Load sequence data
        try:
            metadata = load_sequence_metadata(sequence_dir)
            annotations = load_sequence_annotations(sequence_dir)
        except Exception as e:
            print(f"  ❌ Error loading sequence data: {e}, skipping...")
            continue

        # Initialize detector
        try:
            detector = CameraMovementDetector(config_path=config_path)
        except Exception as e:
            print(f"  ❌ Error initializing detector: {e}, skipping...")
            continue

        # Process sequence
        try:
            detection_results = process_sequence(
                sequence_dir=sequence_dir,
                detector=detector,
                metadata=metadata,
                annotations=annotations
            )
            processed_frames_count += len(detection_results)
        except Exception as e:
            print(f"  ❌ Error processing sequence: {e}, skipping...")
            continue

        # Validate results
        try:
            metrics = harness.validate_sequence_detection(
                detection_results=detection_results,
                annotations=annotations,
                metadata=metadata
            )
            all_sequence_metrics.append(metrics)
        except Exception as e:
            print(f"  ❌ Error validating sequence: {e}, skipping...")
            continue

        # Show progress
        detection_rate = metrics.detection_rate * 100
        symbol = "✅" if metrics.false_negatives == 0 else "❌"
        print(f"   {symbol} Detection rate: {detection_rate:.1f}% | FN: {metrics.false_negatives} | FP: {metrics.false_positives}")

        # Show latency for sudden onset patterns
        if pattern_type == "sudden_onset" and metrics.detection_latency_frames <= 2:
            print(f"   ⚡ Latency: {metrics.detection_latency_frames} frames (within threshold)")
        elif pattern_type == "sudden_onset":
            print(f"   ⚠️  Latency: {metrics.detection_latency_frames} frames (EXCEEDS 2-frame threshold)")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n✅ Detection complete!")
    print(f"   Processed: {processed_frames_count} frames across {len(all_sequence_metrics)} sequences")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Processing rate: {processed_frames_count / duration:.1f} frames/second")

    # Calculate overall metrics
    print(f"\n📊 Calculating overall metrics...")

    # Aggregate metrics across all sequences
    total_expected_invalid = sum(m.expected_invalid_frames for m in all_sequence_metrics)
    total_detected_invalid = sum(m.detected_invalid_frames for m in all_sequence_metrics)
    total_false_negatives = sum(m.false_negatives for m in all_sequence_metrics)
    total_false_positives = sum(m.false_positives for m in all_sequence_metrics)
    total_frames_processed = sum(m.total_frames for m in all_sequence_metrics)

    overall_detection_rate = (
        total_detected_invalid / total_expected_invalid
        if total_expected_invalid > 0
        else 1.0
    )

    # Critical frame analysis
    total_critical_frames = sum(m.critical_frames_tested for m in all_sequence_metrics)
    total_critical_correct = sum(m.critical_frames_correct for m in all_sequence_metrics)
    critical_frame_accuracy = (
        total_critical_correct / total_critical_frames
        if total_critical_frames > 0
        else 1.0
    )

    print(f"\n📈 Overall Performance:")
    print(f"   Detection Rate: {overall_detection_rate * 100:.2f}%")
    print(f"   False Negatives: {total_false_negatives} / {total_expected_invalid} expected detections")
    print(f"   False Positives: {total_false_positives}")
    print(f"   Critical Frame Accuracy: {critical_frame_accuracy * 100:.2f}% ({total_critical_correct}/{total_critical_frames})")

    # Per-pattern analysis
    print(f"\n📊 Per-Pattern Performance:")
    patterns = ['gradual_onset', 'sudden_onset', 'progressive', 'oscillation', 'recovery', 'multi_axis']

    pattern_metrics = {}
    for pattern in patterns:
        pattern_sequences = [m for m in all_sequence_metrics if m.pattern_type == pattern]

        if not pattern_sequences:
            continue

        pattern_expected_invalid = sum(m.expected_invalid_frames for m in pattern_sequences)
        pattern_detected_invalid = sum(m.detected_invalid_frames for m in pattern_sequences)
        pattern_false_negatives = sum(m.false_negatives for m in pattern_sequences)
        pattern_false_positives = sum(m.false_positives for m in pattern_sequences)

        pattern_detection_rate = (
            pattern_detected_invalid / pattern_expected_invalid
            if pattern_expected_invalid > 0
            else 1.0
        )

        pattern_metrics[pattern] = {
            'sequences': len(pattern_sequences),
            'detection_rate': pattern_detection_rate,
            'false_negatives': pattern_false_negatives,
            'false_positives': pattern_false_positives,
            'expected_invalid': pattern_expected_invalid,
            'detected_invalid': pattern_detected_invalid
        }

        symbol = "✅" if pattern_false_negatives == 0 else "❌"
        print(f"   {pattern:20s}: {symbol} {pattern_detection_rate * 100:.1f}% | FN: {pattern_false_negatives}")

    # Latency analysis for sudden onset
    sudden_onset_sequences = [m for m in all_sequence_metrics if m.pattern_type == "sudden_onset"]
    if sudden_onset_sequences:
        avg_latency = sum(m.detection_latency_frames for m in sudden_onset_sequences) / len(sudden_onset_sequences)
        max_latency = max(m.detection_latency_frames for m in sudden_onset_sequences)
        within_threshold = sum(1 for m in sudden_onset_sequences if m.detection_latency_frames <= 2)

        print(f"\n⚡ Sudden Onset Latency Analysis:")
        print(f"   Average latency: {avg_latency:.1f} frames")
        print(f"   Max latency: {max_latency} frames")
        print(f"   Within 2-frame threshold: {within_threshold}/{len(sudden_onset_sequences)} sequences")

    # Check AC-1.9.2: 100% detection rate (0% false negatives)
    ac_192_passed = (total_false_negatives == 0)

    print(f"\n{'='*80}")
    if ac_192_passed:
        print(f"✅ PASS: Stage 2 Validation SUCCESSFUL")
        print(f"   Detection Rate: {overall_detection_rate * 100:.2f}% = 100% (AC-1.9.2)")
        print(f"   False Negatives: {total_false_negatives} (0% - PERFECT)")
        print(f"   GO for Stage 3 validation")
    else:
        print(f"❌ FAIL: Stage 2 Validation FAILED")
        print(f"   Detection Rate: {overall_detection_rate * 100:.2f}% < 100% (AC-1.9.2)")
        print(f"   False Negatives: {total_false_negatives} (MUST BE 0)")
        print(f"   NO-GO: Detector requires improvement before Stage 3")
    print(f"{'='*80}")

    # Build results dictionary
    validation_results = {
        "overall_metrics": {
            "detection_rate": overall_detection_rate,
            "false_negatives": total_false_negatives,
            "false_positives": total_false_positives,
            "expected_invalid_frames": total_expected_invalid,
            "detected_invalid_frames": total_detected_invalid,
            "critical_frame_accuracy": critical_frame_accuracy,
            "critical_frames_tested": total_critical_frames,
            "critical_frames_correct": total_critical_correct,
            "total_frames_processed": total_frames_processed,
            "total_sequences": len(all_sequence_metrics)
        },
        "pattern_metrics": pattern_metrics,
        "sequence_metrics": [
            {
                "sequence_id": m.sequence_id,
                "pattern_type": m.pattern_type,
                "detection_rate": m.detection_rate,
                "false_negatives": m.false_negatives,
                "false_positives": m.false_positives,
                "detection_latency_frames": m.detection_latency_frames,
                "critical_frame_accuracy": m.critical_frame_accuracy
            }
            for m in all_sequence_metrics
        ],
        "ac_192_passed": ac_192_passed,
        "timestamp": datetime.now().isoformat(),
        "config_used": config_path,
        "duration_seconds": duration,
        "processing_rate_fps": processed_frames_count / duration if duration > 0 else 0
    }

    return validation_results


def save_results(results: Dict, output_dir: Path):
    """Save Stage 2 validation results to files"""

    print(f"\n💾 Saving results...")

    # Save JSON results
    json_output = output_dir / "stage2_results.json"
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"   JSON results: {json_output}")

    # Generate text report
    report_output = output_dir / "stage2_results_report.txt"
    with open(report_output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage 2 Validation Results - Temporal Sequence Detection\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Configuration: {results['config_used']}\n")
        f.write(f"Duration: {results['duration_seconds']:.2f} seconds\n")
        f.write(f"Processing Rate: {results['processing_rate_fps']:.1f} frames/second\n\n")

        f.write("Overall Performance:\n")
        f.write(f"  Detection Rate: {results['overall_metrics']['detection_rate'] * 100:.2f}%\n")
        f.write(f"  False Negatives: {results['overall_metrics']['false_negatives']}\n")
        f.write(f"  False Positives: {results['overall_metrics']['false_positives']}\n")
        f.write(f"  Critical Frame Accuracy: {results['overall_metrics']['critical_frame_accuracy'] * 100:.2f}%\n\n")

        f.write("Per-Pattern Performance:\n")
        for pattern, metrics in results['pattern_metrics'].items():
            f.write(f"  {pattern:20s}: {metrics['detection_rate'] * 100:.1f}% | FN: {metrics['false_negatives']}\n")

        f.write("\nAcceptance Criteria:\n")
        status = "✅ PASS" if results['ac_192_passed'] else "❌ FAIL"
        f.write(f"  AC-1.9.2 (100% Detection Rate): {status}\n\n")

        if results['ac_192_passed']:
            f.write("DECISION: GO for Stage 3 validation\n")
        else:
            f.write("DECISION: NO-GO - Detector requires improvement\n")

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
        return 0 if results["ac_192_passed"] else 1

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
