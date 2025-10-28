#!/usr/bin/env python3
"""
Demo 4: Multi-Frame Sequence Processing

Demonstrates:
- Processing a sequence of frames
- Tracking displacement over time
- Identifying movement events
- Generating summary statistics
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_movement_detector import CameraMovementDetector
import cv2
import glob


def main():
    print("=" * 70)
    print("DEMO 4: Multi-Frame Sequence Processing")
    print("=" * 70)
    print()

    # Configuration
    config_path = 'config/config_session_001.json'
    image_dir = 'sample_images/of_jerusalem'

    print("1. Initializing detector...")
    detector = CameraMovementDetector(config_path)
    print("   ✓ Detector initialized")
    print()

    # Load image sequence
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:20]

    if len(image_paths) < 2:
        print(f"   ✗ Need at least 2 images in {image_dir}")
        return 1

    print(f"2. Found {len(image_paths)} images in sequence")
    print()

    # Set baseline
    print("3. Setting baseline from first frame...")
    baseline = cv2.imread(image_paths[0])
    detector.set_baseline(baseline)
    print(f"   ✓ Baseline: {os.path.basename(image_paths[0])}")
    print()

    # Process sequence
    print("4. Processing image sequence...")
    print("=" * 70)
    print(f"{'Frame':>6} {'Status':>8} {'Displacement':>12} {'Image'}")
    print("-" * 70)

    results = []
    for i, img_path in enumerate(image_paths[1:], start=1):
        frame = cv2.imread(img_path)
        frame_id = f"seq_frame_{i:03d}"
        result = detector.process_frame(frame, frame_id=frame_id)
        results.append(result)

        status_icon = "⚠️ " if result['status'] == 'INVALID' else "✓ "
        print(f"{i:>6} {status_icon}{result['status']:>7} "
              f"{result['translation_displacement']:>10.2f}px  {os.path.basename(img_path)}")

    print("=" * 70)
    print()

    # Generate statistics
    print("5. Sequence Statistics")
    print("-" * 70)

    total_frames = len(results)
    valid_frames = sum(1 for r in results if r['status'] == 'VALID')
    invalid_frames = sum(1 for r in results if r['status'] == 'INVALID')

    displacements = [r['translation_displacement'] for r in results]
    avg_disp = sum(displacements) / len(displacements)
    min_disp = min(displacements)
    max_disp = max(displacements)

    print(f"   Total frames processed:  {total_frames}")
    print(f"   VALID frames:            {valid_frames} ({valid_frames/total_frames*100:.1f}%)")
    print(f"   INVALID frames:          {invalid_frames} ({invalid_frames/total_frames*100:.1f}%)")
    print()
    print(f"   Average displacement:    {avg_disp:.2f}px")
    print(f"   Minimum displacement:    {min_disp:.2f}px")
    print(f"   Maximum displacement:    {max_disp:.2f}px")
    print()

    # Identify movement events
    print("6. Movement Events")
    print("-" * 70)

    movement_events = [r for r in results if r['status'] == 'INVALID']

    if movement_events:
        print(f"   {len(movement_events)} movement event(s) detected:")
        print()
        for event in movement_events:
            print(f"   Frame: {event['frame_id']}")
            print(f"   Displacement: {event['translation_displacement']:.2f}px")
            print(f"   Timestamp: {event['timestamp']}")
            print()
    else:
        print("   ✓ No movement events detected - camera stable throughout sequence")
    print()

    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
