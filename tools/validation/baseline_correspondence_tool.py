#!/usr/bin/env python3
"""
Mode B - Baseline Correspondence Tool

Interactive tool for visualizing feature correspondence against a pinned baseline frame.
Shows motion vectors, match quality metrics, and optional difference heatmap.

Usage:
    python tools/validation/baseline_correspondence_tool.py \
        --input-dir sample_images/of_jerusalem \
        --config config.json \
        --output-dir baseline_correspondence_results
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import logging

# Add project root to Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import cv2 as cv
import numpy as np

from src.camera_movement_detector import CameraMovementDetector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Mode B: Baseline Correspondence Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir sample_images/of_jerusalem \\
      --config config.json --output-dir baseline_correspondence_results
"""
    )

    # Required arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing image sequence"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to CSD detector configuration JSON (default: config.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="baseline_correspondence_results",
        help="Output directory for results (default: baseline_correspondence_results)"
    )

    return parser.parse_args()


def load_and_cache_images(input_dir: Path) -> List[Tuple[np.ndarray, str]]:
    """Load all images from directory and cache in memory.

    Args:
        input_dir: Path to directory containing images

    Returns:
        List of (image, filename) tuples sorted by filename
    """
    logger.info("📦 Loading images into memory...")

    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        sys.exit(1)

    logger.info(f"📁 Found {len(image_files)} images")

    images = []
    for img_path in image_files:
        img = cv.imread(str(img_path))
        if img is not None:
            images.append((img, img_path.name))
        else:
            logger.warning(f"Skipping unreadable image: {img_path}")

    if not images:
        logger.error("No valid images loaded")
        sys.exit(1)

    logger.info(f"✅ Loaded {len(images)} images")
    return images


def draw_roi_outline(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Draw ROI outline on image.

    Args:
        image: Image to draw on
        roi_mask: Binary ROI mask

    Returns:
        Image with ROI outline drawn
    """
    display = image.copy()

    # Find contours of ROI mask
    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours in yellow
    cv.drawContours(display, contours, -1, (0, 255, 255), 2)

    return display


def draw_roi_outline_thick(image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Draw thick ROI outline for diagnostic mode.

    Args:
        image: Image to draw on
        roi_mask: Binary ROI mask

    Returns:
        Image with thick ROI outline drawn
    """
    display = image.copy()

    # Find contours of ROI mask
    contours, _ = cv.findContours(roi_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw thick contours in cyan for better visibility
    cv.drawContours(display, contours, -1, (255, 255, 0), 4)

    return display


def create_side_by_side_display(
    baseline: np.ndarray,
    current: np.ndarray,
    matches: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    mask: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    diagnostic_mode: bool = False,
    baseline_keypoints: Optional[List] = None,
    current_keypoints: Optional[List] = None
) -> np.ndarray:
    """Create side-by-side display with baseline and current frame with motion vectors.

    Args:
        baseline: Baseline frame
        current: Current frame
        matches: List of ((x0, y0), (x1, y1)) baseline-to-current point pairs
        mask: Inlier mask (1=inlier, 0=outlier)
        roi_mask: Optional ROI mask to draw on both images
        diagnostic_mode: If True, show feature points and enhanced ROI visualization
        baseline_keypoints: List of cv2.KeyPoint for baseline (for diagnostic mode)
        current_keypoints: List of cv2.KeyPoint for current (for diagnostic mode)

    Returns:
        Side-by-side composite image with motion vectors crossing between frames
    """
    h, w = current.shape[:2]

    # Draw ROI on both images if provided
    baseline_display = baseline.copy()
    current_display = current.copy()

    if roi_mask is not None:
        if diagnostic_mode:
            # Enhanced ROI visualization in diagnostic mode
            baseline_display = draw_roi_outline_thick(baseline_display, roi_mask)
            current_display = draw_roi_outline_thick(current_display, roi_mask)
        else:
            baseline_display = draw_roi_outline(baseline_display, roi_mask)
            current_display = draw_roi_outline(current_display, roi_mask)

    # Diagnostic mode: Draw all feature points
    if diagnostic_mode:
        if baseline_keypoints:
            for kp in baseline_keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv.circle(baseline_display, (x, y), 3, (255, 0, 0), -1)  # Blue for baseline

        if current_keypoints:
            for kp in current_keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv.circle(current_display, (x, y), 3, (0, 0, 255), -1)  # Red for current

    # Create side-by-side canvas
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = baseline_display
    canvas[:, w:] = current_display

    # Draw motion vectors from baseline (left) to current (right)
    for idx, ((x0, y0), (x1, y1)) in enumerate(matches):
        is_inlier = mask[idx] == 1
        color = (0, 255, 0) if is_inlier else (0, 0, 255)

        # Baseline point on left side
        pt1 = (int(x0), int(y0))
        # Current point on right side (shift x by width)
        pt2 = (int(x1 + w), int(y1))

        cv.arrowedLine(canvas, pt1, pt2, color, 1, tipLength=0.2)

    # Diagnostic info overlay
    if diagnostic_mode:
        baseline_count = len(baseline_keypoints) if baseline_keypoints else 0
        current_count = len(current_keypoints) if current_keypoints else 0
        match_count = len(matches)

        # Draw diagnostic info on top
        info_y = 30
        cv.putText(canvas, f"DIAGNOSTIC MODE", (10, info_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        info_y += 30
        cv.putText(canvas, f"Baseline features: {baseline_count}", (10, info_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        info_y += 25
        cv.putText(canvas, f"Current features: {current_count}", (10, info_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        info_y += 25
        cv.putText(canvas, f"Matches: {match_count}", (10, info_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return canvas


def draw_motion_vectors(
    image: np.ndarray,
    matches: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    mask: np.ndarray
) -> np.ndarray:
    """Draw motion vector arrows from baseline to current frame features.

    Args:
        image: Current frame image to draw on
        matches: List of ((x0, y0), (x1, y1)) baseline-to-current point pairs
        mask: Inlier mask (1=inlier, 0=outlier)

    Returns:
        Annotated image with motion vector arrows
    """
    display = image.copy()

    for idx, ((x0, y0), (x1, y1)) in enumerate(matches):
        is_inlier = mask[idx] == 1

        # Color: GREEN for inliers, RED for outliers
        color = (0, 255, 0) if is_inlier else (0, 0, 255)

        # Draw arrow from baseline point to current point
        pt1 = (int(x0), int(y0))
        pt2 = (int(x1), int(y1))
        cv.arrowedLine(display, pt1, pt2, color, 2, tipLength=0.3)

    return display


def draw_match_quality_metrics(
    image: np.ndarray,
    inliers: int,
    total: int,
    rmse: float,
    confidence: float
) -> np.ndarray:
    """Draw match quality metrics panel on image.

    Args:
        image: Image to draw on
        inliers: Number of inlier matches
        total: Total number of matches
        rmse: Root mean squared error (pixels)
        confidence: Confidence score [0.0-1.0]

    Returns:
        Image with metrics panel overlay
    """
    display = image.copy()
    h, w = display.shape[:2]

    # Calculate inlier ratio
    ratio = (inliers / total * 100) if total > 0 else 0.0

    # Color-code based on ratio thresholds
    if ratio > 80.0:
        metrics_color = (0, 255, 0)  # GREEN
    elif ratio >= 50.0:
        metrics_color = (0, 165, 255)  # ORANGE
    else:
        metrics_color = (0, 0, 255)  # RED

    # Draw semi-transparent background
    overlay = display.copy()
    panel_x = w - 280
    panel_y = 10
    panel_w = 270
    panel_h = 120
    cv.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                 (40, 40, 40), -1)
    cv.addWeighted(overlay, 0.7, display, 0.3, 0, display)

    # Draw metrics text
    y_offset = panel_y + 25
    line_spacing = 25

    cv.putText(display, f"Inliers: {inliers}/{total}",
               (panel_x + 10, y_offset), cv.FONT_HERSHEY_SIMPLEX,
               0.5, metrics_color, 1, cv.LINE_AA)
    y_offset += line_spacing

    cv.putText(display, f"Ratio: {ratio:.1f}%",
               (panel_x + 10, y_offset), cv.FONT_HERSHEY_SIMPLEX,
               0.5, metrics_color, 1, cv.LINE_AA)
    y_offset += line_spacing

    cv.putText(display, f"RMSE: {rmse:.2f} px",
               (panel_x + 10, y_offset), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (255, 255, 255), 1, cv.LINE_AA)
    y_offset += line_spacing

    cv.putText(display, f"Confidence: {confidence:.2f}",
               (panel_x + 10, y_offset), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (255, 255, 255), 1, cv.LINE_AA)

    return display


def compute_diff_heatmap(
    baseline: np.ndarray,
    current: np.ndarray,
    H: np.ndarray
) -> Optional[np.ndarray]:
    """Compute difference heatmap between warped baseline and current frame.

    Args:
        baseline: Baseline frame
        current: Current frame
        H: Homography matrix from baseline to current

    Returns:
        Heatmap image with color-coded differences, or None if warping fails
    """
    h, w = current.shape[:2]

    try:
        # Warp baseline to current frame perspective
        warped = cv.warpPerspective(baseline, H, (w, h))

        # Convert to grayscale
        warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        current_gray = cv.cvtColor(current, cv.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv.absdiff(warped_gray, current_gray)

        # Apply color map (blue=low, red=high)
        heatmap = cv.applyColorMap(diff, cv.COLORMAP_JET)

        return heatmap
    except cv.error as e:
        logger.warning(f"Heatmap generation failed: {e}")
        return None


def draw_baseline_thumbnail(
    image: np.ndarray,
    baseline: np.ndarray,
    position: str = "top-left"
) -> np.ndarray:
    """Draw baseline frame thumbnail in corner of image.

    Args:
        image: Main display image
        baseline: Baseline frame to show as thumbnail
        position: Corner position ("top-left", "top-right", etc.)

    Returns:
        Image with baseline thumbnail overlay
    """
    display = image.copy()
    h, w = display.shape[:2]

    # Resize baseline to thumbnail (160x120)
    thumb_w, thumb_h = 160, 120
    thumbnail = cv.resize(baseline, (thumb_w, thumb_h))

    # Position (top-left with margin)
    if position == "top-left":
        x_offset = 10
        y_offset = 10
    else:
        x_offset = 10
        y_offset = 10

    # Draw thumbnail with border
    cv.rectangle(display, (x_offset - 2, y_offset - 2),
                 (x_offset + thumb_w + 2, y_offset + thumb_h + 2),
                 (255, 255, 255), 2)
    display[y_offset:y_offset+thumb_h, x_offset:x_offset+thumb_w] = thumbnail

    # Add label
    cv.putText(display, "BASELINE", (x_offset, y_offset - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return display


def draw_status_bar(
    image: np.ndarray,
    frame_idx: int,
    total_frames: int,
    baseline_set: bool,
    show_vectors: bool,
    show_heatmap: bool,
    show_diagnostic: bool = False
) -> np.ndarray:
    """Draw status bar at bottom of image.

    Args:
        image: Image to draw on
        frame_idx: Current frame index
        total_frames: Total number of frames
        baseline_set: Whether baseline has been set
        show_vectors: Whether motion vectors are visible
        show_heatmap: Whether heatmap is visible
        show_diagnostic: Whether diagnostic mode is active

    Returns:
        Image with status bar
    """
    display = image.copy()
    h, w = display.shape[:2]

    # Draw status bar background
    bar_h = 50
    overlay = display.copy()
    cv.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    cv.addWeighted(overlay, 0.8, display, 0.2, 0, display)

    # Frame counter
    frame_text = f"Frame {frame_idx + 1}/{total_frames}"
    cv.putText(display, frame_text, (10, h - 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Baseline status
    baseline_text = "Baseline: SET" if baseline_set else "Baseline: NOT SET (press 'b')"
    baseline_color = (0, 255, 0) if baseline_set else (0, 165, 255)
    cv.putText(display, baseline_text, (10, h - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, baseline_color, 1, cv.LINE_AA)

    # Toggle states
    vectors_text = f"Vectors [v]: {'ON' if show_vectors else 'OFF'}"
    heatmap_text = f"Heatmap [h]: {'ON' if show_heatmap else 'OFF'}"
    diagnostic_text = f"Diagnostic [d]: {'ON' if show_diagnostic else 'OFF'}"

    cv.putText(display, vectors_text, (300, h - 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
    cv.putText(display, heatmap_text, (500, h - 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)
    cv.putText(display, diagnostic_text, (300, h - 8),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if show_diagnostic else (200, 200, 200), 1, cv.LINE_AA)

    # Controls hint
    controls_text = "[b] Pin | [\u2190\u2192] Navigate | [d] Debug | [q] Quit"
    cv.putText(display, controls_text, (w - 400, h - 18),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv.LINE_AA)

    return display


def run_mode_b(args: argparse.Namespace) -> None:
    """Run Mode B: Baseline Correspondence Tool.

    Args:
        args: Parsed command-line arguments
    """
    logger.info("🎬 Starting Mode B: Baseline Correspondence Tool")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CSD detector
    logger.info("🔧 Initializing CSD detector...")
    detector = CameraMovementDetector(config_path=args.config)

    # Load and cache all images
    input_path = Path(args.input_dir)
    images = load_and_cache_images(input_path)

    logger.info("📝 Controls:")
    logger.info("   [b] - Pin/change baseline")
    logger.info("   [\u2192] - Next frame")
    logger.info("   [\u2190] - Previous frame")
    logger.info("   [v] - Toggle motion vectors")
    logger.info("   [h] - Toggle difference heatmap")
    logger.info("   [d] - Toggle diagnostic mode (show feature points)")
    logger.info("   [q] - Quit")
    logger.info("")
    logger.info("💡 TIP: Set baseline first by pressing 'b' key")

    # Create resizable window
    cv.namedWindow("Mode B: Baseline Correspondence", cv.WINDOW_NORMAL)

    # Set reasonable window size (fits most screens)
    window_width = 1280
    window_height = 720
    cv.resizeWindow("Mode B: Baseline Correspondence", window_width, window_height)

    # State variables
    frame_idx = 0
    baseline_frame = None
    baseline_idx = None
    baseline_set = False
    baseline_keypoints = None
    show_vectors = True
    show_heatmap = False
    show_diagnostic = False

    # Main loop
    while True:
        # Get current frame
        current_image, current_name = images[frame_idx]
        display = current_image.copy()

        # Draw ROI even when baseline not set
        if not baseline_set:
            roi_mask = detector.region_manager.get_static_mask(current_image.shape[:2])
            display = draw_roi_outline(display, roi_mask)

        # Process frame if baseline is set
        if baseline_set:
            # Process current frame against baseline
            result = detector.process_frame(current_image, frame_id=current_name)

            # Get match correspondences and keypoints for diagnostic mode
            matches, mask = detector.movement_detector.get_last_matches()
            current_keypoints = detector.movement_detector.last_current_keypoints

            if matches and len(matches) >= 50:
                # Get ROI mask from detector
                roi_mask = detector.region_manager.get_static_mask(current_image.shape[:2])

                # Create side-by-side display showing baseline and current with vectors
                if show_vectors:
                    display = create_side_by_side_display(
                        baseline_frame, current_image, matches, mask, roi_mask,
                        show_diagnostic, baseline_keypoints, current_keypoints
                    )
                else:
                    # Side-by-side without vectors but with ROI
                    display = create_side_by_side_display(
                        baseline_frame, current_image, [], np.array([]), roi_mask,
                        show_diagnostic, baseline_keypoints, current_keypoints
                    )

                # Calculate metrics
                inliers = int(np.sum(mask))
                total = len(matches)
                # Use translation_displacement as RMSE proxy (both measure error in pixels)
                rmse = result.get("translation_displacement", 0.0)
                confidence = result.get("confidence", 0.0)

                # Draw metrics panel
                display = draw_match_quality_metrics(
                    display, inliers, total, rmse, confidence
                )

                # Draw heatmap if enabled (on right side only)
                if show_heatmap:
                    H = detector.movement_detector.get_last_homography()
                    if H is not None:
                        heatmap = compute_diff_heatmap(baseline_frame, current_image, H)
                        if heatmap is not None:
                            # Apply heatmap to right side (current frame)
                            h, w = current_image.shape[:2]
                            right_side = display[:, w:].copy()
                            blended = cv.addWeighted(right_side, 0.5, heatmap, 0.5, 0)
                            display[:, w:] = blended

                # Add labels
                cv.putText(display, "BASELINE", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
                h, w = current_image.shape[:2]
                cv.putText(display, "CURRENT", (w + 10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
            else:
                # Not enough matches - show warning on side-by-side view with ROI
                roi_mask = detector.region_manager.get_static_mask(current_image.shape[:2])
                display = create_side_by_side_display(
                    baseline_frame, current_image, [], np.array([]), roi_mask,
                    show_diagnostic, baseline_keypoints, current_keypoints
                )

                h, w = current_image.shape[:2]
                match_count = len(matches) if matches else 0
                cv.putText(display, f"WARNING: Only {match_count} matches (need 50+)",
                           (w + 10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Add labels
                cv.putText(display, "BASELINE", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
                cv.putText(display, "CURRENT", (w + 10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

        # Draw status bar
        display = draw_status_bar(
            display, frame_idx, len(images),
            baseline_set, show_vectors, show_heatmap, show_diagnostic
        )

        # Display
        cv.imshow("Mode B: Baseline Correspondence", display)

        # Wait for key press (0 = wait indefinitely)
        key = cv.waitKey(0) & 0xFF

        # Handle keyboard controls
        if key == ord('b'):
            # Pin baseline
            baseline_frame = current_image.copy()
            baseline_idx = frame_idx
            baseline_set = True
            detector.set_baseline(baseline_frame)
            # Store baseline keypoints for diagnostic mode
            baseline_keypoints, _ = detector.feature_extractor.get_baseline()
            logger.info(f"✅ Baseline set: Frame {frame_idx + 1} ({current_name})")

        elif key == 83:  # Right arrow
            frame_idx = min(frame_idx + 1, len(images) - 1)

        elif key == 81:  # Left arrow
            frame_idx = max(frame_idx - 1, 0)

        elif key == ord('v'):
            show_vectors = not show_vectors
            logger.info(f"Motion vectors: {'ON' if show_vectors else 'OFF'}")

        elif key == ord('h'):
            show_heatmap = not show_heatmap
            logger.info(f"Difference heatmap: {'ON' if show_heatmap else 'OFF'}")

        elif key == ord('d'):
            show_diagnostic = not show_diagnostic
            logger.info(f"Diagnostic mode: {'ON' if show_diagnostic else 'OFF'}")

        elif key == ord('q'):
            logger.info("⏹️  User requested quit")
            break

    # Close display windows
    cv.destroyAllWindows()
    logger.info("🎉 Mode B session complete!")


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    try:
        run_mode_b(args)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
