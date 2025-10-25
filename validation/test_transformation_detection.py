#!/usr/bin/env python3
"""
Direct transformation detection test

Tests if the detector can accurately measure affine transformations applied
with cv2.warpAffine vs what it measures with homography estimation.

This isolates whether the issue is:
1. The transformation generation (warpAffine)
2. The homography estimation (detector measurement)
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.camera_movement_detector import CameraMovementDetector


def test_direction(baseline_img, direction_name, dx, dy, config_path):
    """Test a specific translation direction"""

    print(f"\n🧪 Testing {direction_name}: dx={dx:+.1f}, dy={dy:+.1f}")
    print(f"   Expected displacement: {np.sqrt(dx**2 + dy**2):.2f}px")

    # Apply transformation
    h, w = baseline_img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    transformed = cv2.warpAffine(baseline_img, M, (w, h))

    # Initialize detector and test
    detector = CameraMovementDetector(str(config_path))
    detector.set_baseline(baseline_img)
    result = detector.process_frame(transformed, "test_frame")

    # Results
    measured = result['translation_displacement']
    expected = np.sqrt(dx**2 + dy**2)
    error = abs(measured - expected)
    error_pct = (error / expected * 100) if expected > 0 else 0

    print(f"   Measured displacement: {measured:.2f}px")
    print(f"   Error: {error:.2f}px ({error_pct:.1f}%)")
    print(f"   Confidence: {result['confidence']:.4f}")
    print(f"   Status: {result['status']} (threshold: 1.5px)")

    if error > 1.0:
        print(f"   ⚠️  HIGH ERROR: {error:.2f}px off")
    elif error > 0.5:
        print(f"   ⚠️  MODERATE ERROR: {error:.2f}px off")
    else:
        print(f"   ✅ GOOD ACCURACY")

    return measured, expected, error, result['confidence']


def main():
    """Main test execution"""

    print("=" * 80)
    print("Transformation Detection Accuracy Test")
    print("=" * 80)

    # Load baseline image 00018752
    baseline_path = Path("sample_images/gad/00018752-d1e8-44fb-9cba-5107c18eb386.jpg")
    if not baseline_path.exists():
        print(f"❌ ERROR: Baseline not found: {baseline_path}")
        return 1

    baseline_img = cv2.imread(str(baseline_path))
    config_path = Path("config.json")

    print(f"\n📷 Baseline Image: {baseline_path.name}")
    print(f"   Shape: {baseline_img.shape}")
    print()

    # Test various transformations
    tests = [
        ("Right 5px", 5.0, 0.0),
        ("Up 5px", 0.0, -5.0),
        ("Down 5px", 0.0, 5.0),
        ("Left 5px", -5.0, 0.0),
        ("Diagonal UR 5px", 5.0, -5.0),
        ("Right 3px", 3.0, 0.0),
        ("Up 3px", 0.0, -3.0),
        ("Right 7px", 7.0, 0.0),
        ("Up 7px", 0.0, -7.0),
    ]

    results = []
    for name, dx, dy in tests:
        measured, expected, error, conf = test_direction(
            baseline_img, name, dx, dy, config_path
        )
        results.append((name, expected, measured, error, conf))

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()

    print(f"{'Direction':<20} {'Expected':>10} {'Measured':>10} {'Error':>10} {'Confidence':>12}")
    print("-" * 80)

    for name, expected, measured, error, conf in results:
        print(f"{name:<20} {expected:>10.2f} {measured:>10.2f} {error:>10.2f} {conf:>12.4f}")

    # Analyze patterns
    print()
    print("🔍 Error Analysis:")

    avg_error = np.mean([r[3] for r in results])
    max_error = max([r[3] for r in results])
    max_error_test = [r for r in results if r[3] == max_error][0]

    print(f"   Average error: {avg_error:.2f}px")
    print(f"   Max error: {max_error:.2f}px ({max_error_test[0]})")

    # Check if vertical movements have higher error
    horizontal = [r for r in results if r[0].startswith(('Right', 'Left'))]
    vertical = [r for r in results if r[0].startswith(('Up', 'Down'))]

    if horizontal and vertical:
        h_avg_error = np.mean([r[3] for r in horizontal])
        v_avg_error = np.mean([r[3] for r in vertical])

        print(f"   Horizontal movements avg error: {h_avg_error:.2f}px")
        print(f"   Vertical movements avg error: {v_avg_error:.2f}px")

        if v_avg_error > h_avg_error * 1.5:
            print(f"   ⚠️  VERTICAL BIAS: Vertical errors {v_avg_error/h_avg_error:.1f}x worse than horizontal")
        elif h_avg_error > v_avg_error * 1.5:
            print(f"   ⚠️  HORIZONTAL BIAS: Horizontal errors {h_avg_error/v_avg_error:.1f}x worse than vertical")

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
