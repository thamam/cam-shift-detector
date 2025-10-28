#!/usr/bin/env python3
"""
Demo 2: History Buffer Query

Demonstrates:
- Processing multiple frames
- Querying history buffer
- Retrieving specific frame results
- Displaying detection history
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_movement_detector import CameraMovementDetector
import cv2
import glob


def main():
    print("=" * 70)
    print("DEMO 2: History Buffer Query")
    print("=" * 70)
    print()

    # Configuration
    config_path = 'config/config_session_001.json'
    image_dir = 'sample_images/of_jerusalem'

    print("1. Initializing detector...")
    detector = CameraMovementDetector(config_path)
    print("   ✓ Detector initialized")
    print()

    # Load all images from directory
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:10]

    if not image_paths:
        print(f"   ✗ No images found in {image_dir}")
        return 1

    print(f"2. Found {len(image_paths)} images in {image_dir}")
    print()

    # Set baseline (first image)
    print("3. Setting baseline from first image...")
    baseline = cv2.imread(image_paths[0])
    detector.set_baseline(baseline)
    print(f"   ✓ Baseline set: {os.path.basename(image_paths[0])}")
    print()

    # Process remaining images
    print("4. Processing frames and building history...")
    for i, img_path in enumerate(image_paths[1:], start=1):
        frame = cv2.imread(img_path)
        frame_id = f"demo_frame_{i:03d}"
        result = detector.process_frame(frame, frame_id=frame_id)
        print(f"   Frame {i:2d}: {result['status']:7s} - "
              f"{result['translation_displacement']:5.2f}px - {os.path.basename(img_path)}")
    print()

    # Query history buffer
    print("=" * 70)
    print("HISTORY BUFFER QUERIES")
    print("=" * 70)
    print()

    # Query last 5 detections
    print("5. Query: Last 5 detections")
    print("-" * 70)
    recent = detector.get_history(limit=5)
    for r in recent:
        print(f"   {r['timestamp']}: {r['status']:7s} - {r['translation_displacement']:5.2f}px - {r['frame_id']}")
    print()

    # Query specific frame
    print("6. Query: Specific frame (demo_frame_003)")
    print("-" * 70)
    frame_result = detector.get_history(frame_id="demo_frame_003")
    if frame_result:
        r = frame_result[0]
        print(f"   Frame ID:     {r['frame_id']}")
        print(f"   Status:       {r['status']}")
        print(f"   Displacement: {r['translation_displacement']:.2f}px")
        print(f"   Confidence:   {r['confidence']:.3f}")
        print(f"   Timestamp:    {r['timestamp']}")
    else:
        print("   Frame not found in history")
    print()

    # Query all history
    print("7. Query: Complete history buffer")
    print("-" * 70)
    all_history = detector.get_history()
    print(f"   Total entries in buffer: {len(all_history)}")

    valid_count = sum(1 for r in all_history if r['status'] == 'VALID')
    invalid_count = sum(1 for r in all_history if r['status'] == 'INVALID')

    print(f"   VALID entries:   {valid_count}")
    print(f"   INVALID entries: {invalid_count}")

    if all_history:
        displacements = [r['translation_displacement'] for r in all_history]
        avg_displacement = sum(displacements) / len(displacements)
        max_displacement = max(displacements)
        print(f"   Average displacement: {avg_displacement:.2f}px")
        print(f"   Maximum displacement: {max_displacement:.2f}px")
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
