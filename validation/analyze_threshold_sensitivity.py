#!/usr/bin/env python3
"""
Analyze detection rate vs displacement threshold

Tests how FN rate drops as we increase the detection threshold from 1.5px
to higher values. Helps determine minimum acceptable threshold for production.
"""

import sys
from pathlib import Path
import json
import math
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.camera_movement_detector import CameraMovementDetector

def collect_all_measurements(stage2_data_dir: Path, config_path: Path) -> list:
    """Collect measured displacements for all INVALID frames across all sequences"""

    detector = CameraMovementDetector(str(config_path))

    all_frames = []

    # Pattern directory mapping
    pattern_dirs = {
        'gradual_onset': 'pattern_1_gradual_onset',
        'sudden_onset': 'pattern_2_sudden_onset',
        'progressive': 'pattern_3_progressive',
        'oscillation': 'pattern_4_oscillation',
        'recovery': 'pattern_5_recovery',
        'multi_axis': 'pattern_6_multi_axis'
    }

    for pattern_type, pattern_dir in pattern_dirs.items():
        pattern_path = stage2_data_dir / pattern_dir

        if not pattern_path.exists():
            continue

        for sequence_dir in sorted(pattern_path.iterdir()):
            if not sequence_dir.is_dir():
                continue

            # Load annotations
            annotations_file = sequence_dir / 'frame_annotations.json'
            if not annotations_file.exists():
                continue

            with open(annotations_file) as f:
                annotations = json.load(f)

            # Set baseline (frame 0)
            baseline_img = cv2.imread(str(sequence_dir / 'frame_000.jpg'))
            detector.set_baseline(baseline_img)

            # Process all frames
            for frame_data in annotations['frames']:
                frame_num = frame_data['frame_number']
                expected_status = frame_data['expected_status']

                if expected_status == 'INVALID':
                    frame_file = sequence_dir / f"frame_{frame_num:03d}.jpg"
                    frame_img = cv2.imread(str(frame_file))

                    result = detector.process_frame(frame_img, f"frame_{frame_num:03d}")

                    all_frames.append({
                        'sequence': sequence_dir.name,
                        'pattern': pattern_type,
                        'frame': frame_num,
                        'expected_displacement': frame_data['cumulative_shift_px'],
                        'measured_displacement': result['translation_displacement'],
                        'confidence': result['confidence']
                    })

    return all_frames

def analyze_fn_by_displacement(all_frames: list, thresholds: list) -> dict:
    """Analyze FN distribution by displacement magnitude at various thresholds"""

    print("=" * 80)
    print("Detection Rate vs Displacement Threshold Analysis")
    print("=" * 80)
    print()

    total_invalid_frames = len(all_frames)
    print(f"Total frames expected INVALID: {total_invalid_frames}")
    print()

    # Analyze at each threshold
    threshold_results = {}

    for threshold in thresholds:
        fn_frames = []

        for frame in all_frames:
            measured = frame['measured_displacement']
            if measured < threshold:
                fn_frames.append(frame)

        fn_count = len(fn_frames)
        detection_rate = ((total_invalid_frames - fn_count) / total_invalid_frames) * 100

        threshold_results[threshold] = {
            'fn_count': fn_count,
            'detection_rate': detection_rate,
            'fn_frames': fn_frames
        }

        print(f"Threshold: {threshold:.1f}px")
        print(f"   Detection Rate: {detection_rate:.2f}%")
        print(f"   False Negatives: {fn_count} / {total_invalid_frames}")
        print(f"   Improvement from 1.5px: {threshold_results[1.5]['fn_count'] - fn_count} FN eliminated")

        if fn_count == 0:
            print(f"   ✅ 100% DETECTION ACHIEVED!")
        elif fn_count <= 5:
            print(f"   ⚠️  Near-perfect detection ({fn_count} FN remaining)")

        print()

    # Find threshold for 100% detection
    print("=" * 80)
    print("Threshold Analysis Summary")
    print("=" * 80)
    print()

    for threshold in sorted(threshold_results.keys()):
        result = threshold_results[threshold]
        fn = result['fn_count']
        rate = result['detection_rate']

        if fn == 0:
            print(f"✅ {threshold:.1f}px: 100% detection ({rate:.2f}%)")
            break
        elif fn <= 5:
            print(f"⚠️  {threshold:.1f}px: {rate:.2f}% detection ({fn} FN)")
        else:
            print(f"❌ {threshold:.1f}px: {rate:.2f}% detection ({fn} FN)")

    print()

    # Analyze FN by pattern at original threshold
    print("=" * 80)
    print("False Negative Distribution by Pattern (1.5px threshold)")
    print("=" * 80)
    print()

    fn_by_pattern = {}
    for frame in threshold_results[1.5]['fn_frames']:
        pattern = frame['pattern']
        if pattern not in fn_by_pattern:
            fn_by_pattern[pattern] = []
        fn_by_pattern[pattern].append(frame)

    for pattern, fn_frames in sorted(fn_by_pattern.items()):
        print(f"{pattern:20s}: {len(fn_frames):3d} FN")

        # Show displacement distribution
        displacements = [f['measured_displacement'] for f in fn_frames]
        if displacements:
            avg_disp = sum(displacements) / len(displacements)
            min_disp = min(displacements)
            max_disp = max(displacements)
            print(f"{'':20s}   Displacement: {min_disp:.2f} - {max_disp:.2f}px (avg: {avg_disp:.2f}px)")

    print()

    # Find minimum threshold for 0 FN
    print("=" * 80)
    print("Recommended Operational Threshold")
    print("=" * 80)
    print()

    for threshold in sorted(threshold_results.keys()):
        result = threshold_results[threshold]
        if result['fn_count'] == 0:
            increase = ((threshold - 1.5) / 1.5) * 100
            print(f"✅ Minimum threshold for 100% detection: {threshold:.1f}px")
            print(f"   Increase from specification: +{threshold - 1.5:.1f}px ({increase:.1f}%)")
            print(f"   All {total_invalid_frames} INVALID frames would be detected")
            break
        elif result['fn_count'] <= 3:
            print(f"⚠️  Near-perfect at {threshold:.1f}px:")
            print(f"   Detection Rate: {result['detection_rate']:.2f}%")
            print(f"   Remaining FN: {result['fn_count']}")
            print(f"   These {result['fn_count']} FN may be edge cases")

    return threshold_results


def main():
    stage2_data_dir = Path("validation/stage2_data")
    config_path = Path("config.json")

    if not stage2_data_dir.exists():
        print(f"❌ ERROR: Stage 2 data not found: {stage2_data_dir}")
        print("   Run validation/generate_stage2_data.py first")
        return 1

    print("📊 Collecting measured displacements from all sequences...")
    print("   (This may take a few minutes)")
    print()

    all_frames = collect_all_measurements(stage2_data_dir, config_path)

    # Test thresholds from 1.5px to 5.0px
    thresholds = [1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]

    results = analyze_fn_by_displacement(all_frames, thresholds)

    return 0


if __name__ == "__main__":
    sys.exit(main())
