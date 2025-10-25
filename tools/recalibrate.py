#!/usr/bin/env python3
"""Recalibration Script for Camera Movement Detection

Manual baseline reset helper for DAF water quality monitoring systems.

Usage:
    python tools/recalibrate.py --config config.json --image current_frame.jpg
    python tools/recalibrate.py --config config.json --image current_frame.jpg --clear-history
"""

import argparse
import cv2
import numpy as np
import sys
from datetime import datetime, UTC
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_movement_detector import CameraMovementDetector


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Manual recalibration tool for camera movement detection",
        epilog="Example: python tools/recalibrate.py --config config.json --image frame.jpg"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json file"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to new reference image for baseline reset"
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear detection history buffer after successful recalibration"
    )
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """Load and validate reference image.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as numpy array

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image file is corrupted or unreadable
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. File may be corrupted.")

    return image


def main():
    """Main execution flow - THIN WRAPPER around detector.recalibrate()."""
    args = parse_arguments()

    # Simple validation: config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load reference image (simple I/O)
    try:
        image = load_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize detector (delegates to CameraMovementDetector)
    try:
        detector = CameraMovementDetector(args.config)
    except Exception as e:
        print(f"Error: Failed to initialize detector: {e}")
        sys.exit(1)

    # Call intrinsic method - ALL validation happens here
    print(f"Recalibrating baseline from: {args.image}")
    print(f"Using config: {args.config}")

    success = detector.recalibrate(image)  # ← ALL LOGIC IN THIS METHOD

    # Report result (simple output formatting)
    timestamp = datetime.now(UTC).isoformat()

    if success:
        print(f"✓ Recalibration successful at {timestamp}")

        # Optional: clear history (simple flag check)
        if args.clear_history:
            detector.result_manager.history.clear()
            print("✓ Detection history buffer cleared")

        sys.exit(0)
    else:
        print(f"✗ Recalibration failed at {timestamp}")
        print("Reason: Insufficient features detected in ROI (<50 required)")
        print("Action: Try a different image with more texture/features in the ROI")
        sys.exit(1)


if __name__ == "__main__":
    main()
