#!/usr/bin/env python3
"""
Sequence 20 Frame-by-Frame Investigation

Analyzes the sudden onset sequence with baseline 00018752 that showed 0% detection rate.
Previous investigation confirmed the baseline image itself is functional (detects 5px shift
correctly in isolation). This script analyzes the sequence processing to identify why
temporal sequence processing fails for this baseline.

Key Finding from baseline investigation:
- Baseline 00018752: 500 ORB features, initialization successful
- Simple 5px shift test: PASSED (4.31px detected, 0.88 confidence, INVALID status)
- Conclusion: Baseline functional, issue is sequence-specific

Investigation Focus:
1. Process sequence frame-by-frame with detector
2. Compare actual detector results with expected ground truth
3. Identify at which frame detection fails
4. Analyze feature matching quality across frames
5. Check if detector state is properly maintained

Usage:
    python validation/investigate_sequence_20.py
"""

import sys
import json
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.camera_movement_detector import CameraMovementDetector


def load_sequence_metadata(sequence_dir: Path) -> dict:
    """Load sequence metadata and annotations"""
    metadata_file = sequence_dir / "sequence_metadata.json"
    annotations_file = sequence_dir / "frame_annotations.json"

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    return metadata, annotations


def process_sequence_detailed(sequence_dir: Path, config_path: Path) -> list:
    """Process sequence frame-by-frame with detailed logging"""

    # Load sequence data
    metadata, annotations = load_sequence_metadata(sequence_dir)

    # Initialize detector
    detector = CameraMovementDetector(str(config_path))

    # Get frame files in order
    frame_files = sorted(sequence_dir.glob("frame_*.jpg"))

    print(f"📁 Sequence: {metadata['sequence_id']}")
    print(f"   Pattern: {metadata['pattern_type']}")
    print(f"   Total Frames: {len(frame_files)}")
    print(f"   Expected Detections: {metadata['expected_detections']}")
    print()

    # Set baseline (frame 0)
    baseline_file = frame_files[0]
    baseline_image = cv2.imread(str(baseline_file))

    print("🔧 Setting Baseline (frame 0)")
    detector.set_baseline(baseline_image)
    print(f"   Baseline Set: {'✅' if detector.baseline_set else '❌'}")
    print()

    # Process all frames
    results = []
    false_negatives = []

    print("🎬 Processing Frames:")
    print()

    for i, frame_file in enumerate(frame_files):
        frame_image = cv2.imread(str(frame_file))
        result = detector.process_frame(frame_image, frame_id=f"frame_{i:03d}")

        # Get expected ground truth for this frame
        expected_annotation = annotations['frames'][i]
        expected_status = expected_annotation['expected_status']
        expected_displacement = expected_annotation['cumulative_shift_px']

        # Check if detection matches expectation
        is_correct = (result['status'] == expected_status)
        is_false_negative = (expected_status == "INVALID" and result['status'] == "VALID")

        # Store result
        frame_result = {
            "frame_number": i,
            "frame_id": frame_file.name,
            "expected_status": expected_status,
            "actual_status": result['status'],
            "expected_displacement": expected_displacement,
            "actual_displacement": result['translation_displacement'],
            "confidence": result['confidence'],
            "is_correct": is_correct,
            "is_false_negative": is_false_negative
        }
        results.append(frame_result)

        if is_false_negative:
            false_negatives.append(frame_result)

        # Print frame result
        status_icon = "✅" if is_correct else "❌"
        fn_marker = " [FN]" if is_false_negative else ""

        print(f"   Frame {i:2d}: {status_icon}{fn_marker}")
        print(f"      Expected: {expected_status:7s} @ {expected_displacement:5.2f}px")
        print(f"      Actual:   {result['status']:7s} @ {result['translation_displacement']:5.2f}px (conf: {result['confidence']:.4f})")

        # For frames 1-5 (should all be INVALID with 5px shift), show detailed analysis
        if i >= 1 and i <= 5:
            if is_false_negative:
                print(f"      🔴 FALSE NEGATIVE: Expected INVALID (5px shift), got VALID")
                print(f"         Displacement measured: {result['translation_displacement']:.2f}px (threshold: 1.5px)")
                print(f"         Confidence: {result['confidence']:.4f}")

        print()

    return results, false_negatives, metadata


def main():
    """Main investigation execution"""

    print("=" * 80)
    print("Sequence 20 Frame-by-Frame Investigation")
    print("Sudden Onset Pattern with Baseline 00018752")
    print("=" * 80)
    print()

    # Paths
    config_path = Path("config.json")
    sequence_dir = Path("validation/stage2_data/pattern_2_sudden_onset/pattern_2_sudden_onset_00018752-d1e8-44fb-9cba-5107c18eb386_up")

    if not sequence_dir.exists():
        print(f"❌ ERROR: Sequence directory not found: {sequence_dir}")
        return 1

    # Process sequence
    results, false_negatives, metadata = process_sequence_detailed(sequence_dir, config_path)

    # Summary
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    total_frames = len(results)
    expected_invalid = metadata['expected_detections']
    false_negative_count = len(false_negatives)
    detection_rate = (expected_invalid - false_negative_count) / expected_invalid * 100

    print(f"📊 Detection Performance:")
    print(f"   Total Frames: {total_frames}")
    print(f"   Expected INVALID: {expected_invalid}")
    print(f"   False Negatives: {false_negative_count}")
    print(f"   Detection Rate: {detection_rate:.2f}%")
    print()

    if false_negative_count == expected_invalid:
        print("🔴 COMPLETE FAILURE: 0% detection rate")
        print()

        # Analyze first few frames (should be easiest to detect)
        print("🔍 Analysis of First 5 Frames After Onset:")
        print()

        for i in range(1, min(6, len(results))):
            r = results[i]
            print(f"   Frame {i}:")
            print(f"      Expected: INVALID @ 5.0px shift")
            print(f"      Measured: {r['actual_displacement']:.2f}px (threshold: 1.5px)")
            print(f"      Status: {r['actual_status']} (confidence: {r['confidence']:.4f})")
            print(f"      Problem: Displacement below threshold" if r['actual_displacement'] < 1.5 else "      Problem: Unknown")
            print()

        print("💡 Possible Root Causes:")
        print("   1. Feature matching failing across entire sequence")
        print("   2. Transformation matrix calculation incorrect")
        print("   3. Static region mask eliminating all features")
        print("   4. Baseline features not properly stored/retrieved")
        print()

        print("📋 RECOMMENDED ACTIONS:")
        print("   1. Debug feature matching for frame 1 vs baseline")
        print("   2. Inspect static region mask for this baseline")
        print("   3. Verify feature extractor returns baseline features")
        print("   4. Test homography estimation with known transformation")

    elif false_negative_count > 0:
        print(f"🟡 PARTIAL FAILURE: {detection_rate:.2f}% detection rate")
        print()
        print(f"   Frames with false negatives:")
        for fn in false_negatives:
            print(f"      Frame {fn['frame_number']}: {fn['actual_displacement']:.2f}px measured (expected {fn['expected_displacement']:.2f}px)")

    else:
        print("🟢 SEQUENCE PASSES: 100% detection rate")
        print("   This contradicts Stage 2 results - investigate validation script")

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
