#!/usr/bin/env python3
"""ROI Selection Tool for Camera Movement Detection Setup

Interactive OpenCV GUI for defining static regions and generating config.json.

Usage:
    python tools/select_roi.py --source image --path sample_images/of_jerusalem/001.jpg
    python tools/select_roi.py --source camera  # Future: live camera capture
"""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_extractor import FeatureExtractor


MIN_FEATURES_REQUIRED = 50
DEFAULT_THRESHOLD_PIXELS = 2.0
DEFAULT_HISTORY_BUFFER_SIZE = 100


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive ROI selection tool for camera movement detection setup"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["image", "camera"],
        help="Input source type (image file or camera)"
    )
    parser.add_argument(
        "--path",
        help="Path to image file (required if --source=image)"
    )
    return parser.parse_args()


def load_image(image_path: str) -> np.ndarray:
    """Load and validate image from file path.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as NumPy array (H×W×3, uint8, BGR)

    Raises:
        FileNotFoundError: If image file not found
        ValueError: If image cannot be loaded or is corrupted
    """
    # Check file exists
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load image
    image = cv2.imread(image_path)

    # Validate loaded successfully
    if image is None:
        raise ValueError(
            f"Failed to load image from {image_path}. File may be corrupted."
        )

    return image


def create_mask_from_roi(roi_coords: tuple, image_shape: tuple) -> np.ndarray:
    """Generate binary mask for ROI coordinates.

    Args:
        roi_coords: Tuple of (x, y, width, height)
        image_shape: Tuple of (height, width)

    Returns:
        Binary mask (H×W, uint8): 255 inside ROI, 0 outside
    """
    img_height, img_width = image_shape
    x, y, width, height = roi_coords

    # Create zero-filled mask (all dynamic by default)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Set ROI region to 255 (static)
    mask[y:y+height, x:x+width] = 255

    return mask


def validate_roi_features(image: np.ndarray, roi_coords: tuple) -> tuple:
    """Validate sufficient features detected in ROI.

    Args:
        image: Input image (H×W×3, uint8, BGR)
        roi_coords: Tuple of (x, y, width, height)

    Returns:
        Tuple of (feature_count, success_flag)
    """
    # Create mask from ROI coordinates
    mask = create_mask_from_roi(roi_coords, image.shape[:2])

    # Initialize FeatureExtractor
    extractor = FeatureExtractor(min_features_required=MIN_FEATURES_REQUIRED)

    # Extract features
    keypoints, descriptors = extractor.extract_features(image, mask)

    # Count features
    feature_count = len(keypoints) if keypoints else 0
    success = feature_count >= MIN_FEATURES_REQUIRED

    return (feature_count, success)


def save_config(roi_coords: tuple, output_path: str = 'config.json'):
    """Save validated ROI to config.json.

    Args:
        roi_coords: Tuple of (x, y, width, height)
        output_path: Path for output config file
    """
    x, y, width, height = roi_coords
    config = {
        "roi": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height)
        },
        "threshold_pixels": DEFAULT_THRESHOLD_PIXELS,
        "history_buffer_size": DEFAULT_HISTORY_BUFFER_SIZE,
        "min_features_required": MIN_FEATURES_REQUIRED
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\u2713 Config saved to {output_path}")


def select_roi_interactive(image: np.ndarray) -> tuple:
    """Interactive ROI selection using OpenCV GUI.

    Args:
        image: Input image to display

    Returns:
        Tuple of (x, y, width, height) or None if user cancelled
    """
    # Display instructions
    window_name = "ROI Selection - Press SPACE to confirm, ESC to cancel"

    # Call cv2.selectROI()
    roi = cv2.selectROI(window_name, image, showCrosshair=True, fromCenter=False)

    # Close window
    cv2.destroyAllWindows()

    # Extract coordinates
    x, y, width, height = roi

    # Handle user cancellation or empty selection
    if width == 0 or height == 0:
        return None

    return (x, y, width, height)


def main():
    """Main execution flow."""
    args = parse_arguments()

    # Validate arguments
    if args.source == "image":
        if not args.path:
            print("Error: --path required when --source=image")
            sys.exit(1)
    else:
        # Camera source not yet implemented
        print("Error: Camera source not yet implemented")
        sys.exit(1)

    # Load image
    try:
        image = load_image(args.path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Instructions: Click and drag to select static region.")
    print("              Press SPACE to confirm, ESC to cancel.")
    print()

    # Interactive ROI selection loop
    attempt_count = 0
    while True:
        attempt_count += 1

        # Select ROI
        roi_coords = select_roi_interactive(image)

        if roi_coords is None:  # User cancelled
            print("Operation cancelled by user")
            sys.exit(0)

        # Validate features
        try:
            feature_count, valid = validate_roi_features(image, roi_coords)
            print(f"Features detected: {feature_count}")

            if valid:
                print(f"\u2713 Validation passed (\u2265{MIN_FEATURES_REQUIRED} features)")
                save_config(roi_coords)
                break
            else:
                print(f"\u2717 Validation failed: {feature_count} < {MIN_FEATURES_REQUIRED} required features")
                print("Please select a different region with more texture/features")
                print()

                # Warn if too many attempts
                if attempt_count > 5:
                    print(f"Warning: {attempt_count} attempts made. Consider selecting a different image area.")
                    print()

        except Exception as e:
            print(f"Error during validation: {e}")
            print("Please try selecting a different region")
            print()


if __name__ == "__main__":
    main()
