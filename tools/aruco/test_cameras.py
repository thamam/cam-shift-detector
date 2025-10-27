#!/usr/bin/env python3
"""
Quick script to test all available cameras and show their feeds.
Press 'q' to close a camera window and try the next one.
"""
import cv2 as cv
import sys

def test_camera(index):
    """Test if camera at given index works and show preview."""
    print(f"\n[Testing] Camera {index}...")
    cap = cv.VideoCapture(index)

    if not cap.isOpened():
        print(f"  [FAILED] Camera {index} cannot be opened")
        return False

    # Get camera properties
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    print(f"  [SUCCESS] Camera {index} opened: {width}x{height} @ {fps:.1f}fps")
    print(f"  Press 'q' to close this camera and try the next one...")

    window_name = f"Camera {index} - {width}x{height}"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARNING] Failed to read frame from camera {index}")
            break

        # Add text overlay
        cv.putText(frame, f"Camera {index}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f"{width}x{height}", (20, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.putText(frame, "Press 'q' to close", (20, 120),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv.imshow(window_name, frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

    cap.release()
    cv.destroyWindow(window_name)
    return True

def main():
    print("=" * 60)
    print("Camera Tester - Finding available cameras")
    print("=" * 60)

    max_cameras = 10  # Test up to 10 camera indices
    working_cameras = []

    for i in range(max_cameras):
        if test_camera(i):
            working_cameras.append(i)

    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    if working_cameras:
        print(f"Working cameras found: {working_cameras}")
        print(f"\nTo use a specific camera, run:")
        print(f"  python camshift_annotator.py --source N --mode charuco --draw")
        print(f"  (where N is one of: {', '.join(map(str, working_cameras))})")
    else:
        print("No working cameras found!")
        print("Check that:")
        print("  1. A camera is connected")
        print("  2. Camera drivers are installed")
        print("  3. You have permission to access /dev/video*")
        sys.exit(1)

if __name__ == "__main__":
    main()
