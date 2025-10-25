#!/usr/bin/env python3
"""
Quick diagnostic for recovery pattern failures
"""

import sys
from pathlib import Path
import cv2
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_movement_detector import CameraMovementDetector

def main():
    sequence_dir = Path("validation/stage2_data/pattern_5_recovery/pattern_5_recovery_00018752-d1e8-44fb-9cba-5107c18eb386_up")
    config_path = Path("config.json")

    # Load annotations
    with open(sequence_dir / "frame_annotations.json") as f:
        annotations = json.load(f)

    # Initialize detector
    detector = CameraMovementDetector(str(config_path))

    # Set baseline (frame 0)
    baseline_img = cv2.imread(str(sequence_dir / "frame_000.jpg"))
    detector.set_baseline(baseline_img)

    print("Recovery Pattern Diagnostic")
    print("=" * 80)
    print(f"Threshold: 1.5px")
    print()

    # Process frames around the critical transition (frames 18-25)
    for i in range(18, 26):
        frame_file = sequence_dir / f"frame_{i:03d}.jpg"
        frame_img = cv2.imread(str(frame_file))

        result = detector.process_frame(frame_img, f"frame_{i:03d}")

        expected_annotation = annotations['frames'][i]
        expected_status = expected_annotation['expected_status']
        expected_displacement = expected_annotation['cumulative_shift_px']

        is_match = result['status'] == expected_status
        symbol = "✅" if is_match else "❌"

        print(f"Frame {i:2d}: {symbol}")
        print(f"   Expected: {expected_status:7s} @ {expected_displacement:5.2f}px")
        print(f"   Measured: {result['status']:7s} @ {result['translation_displacement']:5.2f}px")
        print(f"   Confidence: {result['confidence']:.4f}")

        if not is_match:
            if expected_status == "INVALID" and result['status'] == "VALID":
                print(f"   🔴 FALSE NEGATIVE")
            else:
                print(f"   🔴 FALSE POSITIVE")
        print()

if __name__ == "__main__":
    main()
