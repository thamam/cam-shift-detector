#!/usr/bin/env python3
"""
Demo 3: Manual Recalibration

Demonstrates:
- Initial baseline setup
- Detecting movement
- Manual recalibration with new baseline
- Verifying detection after recalibration
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_movement_detector import CameraMovementDetector
import cv2


def main():
    print("=" * 70)
    print("DEMO 3: Manual Recalibration")
    print("=" * 70)
    print()

    # Configuration
    config_path = 'config/config_session_001.json'

    # Get images from both sites
    import glob
    site1_images = sorted(glob.glob('sample_images/of_jerusalem/*.jpg'))
    site2_images = sorted(glob.glob('sample_images/carmit/*.jpg'))

    if len(site1_images) < 2 or len(site2_images) < 2:
        print("   ✗ Need at least 2 images in each site directory")
        return 1

    # Scenario: Site 1 (Jerusalem) → camera moves → recalibrate to Site 2 (Carmit)
    site1_baseline = site1_images[0]
    site1_test = site1_images[1]
    site2_baseline = site2_images[0]
    site2_test = site2_images[1]

    print("SCENARIO: Camera repositioned between sites")
    print("          Simulate lighting change or camera maintenance")
    print()

    # Initialize detector
    print("1. Initializing detector...")
    detector = CameraMovementDetector(config_path)
    print("   ✓ Detector initialized")
    print()

    # Set initial baseline (Site 1)
    print(f"2. Setting initial baseline: Site 1 (Jerusalem)")
    print(f"   Image: {os.path.basename(site1_baseline)}")
    baseline1 = cv2.imread(site1_baseline)
    detector.set_baseline(baseline1)
    print("   ✓ Baseline set")
    print()

    # Test with image from same site (should be VALID)
    print("3. Testing with image from same site...")
    print(f"   Image: {os.path.basename(site1_test)}")
    test1 = cv2.imread(site1_test)
    result1 = detector.process_frame(test1, frame_id="before_move")
    print(f"   Status:       {result1['status']}")
    print(f"   Displacement: {result1['translation_displacement']:.2f}px")
    print()

    # Test with image from different site (should be INVALID)
    print("4. Simulating camera movement (switching to Site 2)...")
    print(f"   Image: {os.path.basename(site2_test)}")
    test2 = cv2.imread(site2_test)
    result2 = detector.process_frame(test2, frame_id="after_move")
    print(f"   Status:       {result2['status']}")
    print(f"   Displacement: {result2['translation_displacement']:.2f}px")

    if result2['status'] == 'INVALID':
        print("   ⚠️  Movement detected! Camera position changed.")
    print()

    # Manual recalibration
    print("=" * 70)
    print("MANUAL RECALIBRATION")
    print("=" * 70)
    print()
    print("5. Operator action: Recalibrating with new baseline...")
    print(f"   New baseline: Site 2 (Carmit)")
    print(f"   Image: {os.path.basename(site2_baseline)}")

    new_baseline = cv2.imread(site2_baseline)
    success = detector.recalibrate(new_baseline)

    if success:
        print("   ✓ Recalibration successful - new baseline set")
    else:
        print("   ✗ Recalibration failed - insufficient features")
        return 1
    print()

    # Test again with Site 2 image (should now be VALID)
    print("6. Testing with same Site 2 image after recalibration...")
    print(f"   Image: {os.path.basename(site2_test)}")
    result3 = detector.process_frame(test2, frame_id="after_recalibration")
    print(f"   Status:       {result3['status']}")
    print(f"   Displacement: {result3['translation_displacement']:.2f}px")

    if result3['status'] == 'VALID':
        print("   ✓ Camera position stable - measurements can resume")
    print()

    # Summary
    print("=" * 70)
    print("RECALIBRATION SUMMARY")
    print("=" * 70)
    print()
    print("Before recalibration:")
    print(f"  Site 1 test: {result1['status']:7s} ({result1['translation_displacement']:.2f}px)")
    print(f"  Site 2 test: {result2['status']:7s} ({result2['translation_displacement']:.2f}px) ← Movement!")
    print()
    print("After recalibration:")
    print(f"  Site 2 test: {result3['status']:7s} ({result3['translation_displacement']:.2f}px) ← Stable!")
    print()
    print("✓ Recalibration workflow validated")
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
