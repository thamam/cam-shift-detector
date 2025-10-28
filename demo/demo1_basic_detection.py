#!/usr/bin/env python3
"""
Demo 1: Basic Detection API

Demonstrates:
- Initializing the detector with configuration
- Setting baseline reference
- Processing a test frame
- Displaying detection results
"""

import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_movement_detector import CameraMovementDetector
import cv2


def main():
    print("=" * 70)
    print("DEMO 1: Basic Camera Movement Detection API")
    print("=" * 70)
    print()

    # Configuration
    config_path = 'config/config_session_001.json'
    image_dir = 'sample_images/of_jerusalem'

    # Get first two images from directory
    import glob
    images = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    if len(images) < 2:
        print(f"   ✗ Need at least 2 images in {image_dir}")
        return 1

    baseline_image_path = images[0]
    test_image_path = images[1]

    print(f"1. Initializing detector with config: {config_path}")
    detector = CameraMovementDetector(config_path)
    print("   ✓ Detector initialized")
    print()

    print(f"2. Loading baseline image: {baseline_image_path}")
    baseline = cv2.imread(baseline_image_path)
    if baseline is None:
        print(f"   ✗ Error: Could not load baseline image")
        return 1
    print(f"   ✓ Baseline loaded: {baseline.shape[1]}×{baseline.shape[0]} pixels")
    print()

    print("3. Setting baseline reference...")
    try:
        detector.set_baseline(baseline)
        print("   ✓ Baseline set successfully")
    except ValueError as e:
        print(f"   ✗ Error setting baseline: {e}")
        return 1
    print()

    print(f"4. Loading test frame: {test_image_path}")
    test_frame = cv2.imread(test_image_path)
    if test_frame is None:
        print(f"   ✗ Error: Could not load test image")
        return 1
    print(f"   ✓ Test frame loaded: {test_frame.shape[1]}×{test_frame.shape[0]} pixels")
    print()

    print("5. Processing frame and detecting movement...")
    result = detector.process_frame(test_frame, frame_id="demo_frame_100")
    print()

    # Display results
    print("=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(f"Status:       {result['status']}")
    print(f"Displacement: {result['translation_displacement']:.2f} pixels")
    print(f"Confidence:   {result['confidence']:.3f}")
    print(f"Frame ID:     {result['frame_id']}")
    print(f"Timestamp:    {result['timestamp']}")
    print()

    if result['status'] == 'INVALID':
        print("⚠️  CAMERA MOVEMENT DETECTED!")
        print(f"    Camera moved {result['translation_displacement']:.2f} pixels")
        print("    → Action: Halt measurements, alert operator")
    else:
        print("✓  Camera position is STABLE")
        print("   → Action: Proceed with water quality analysis")
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
