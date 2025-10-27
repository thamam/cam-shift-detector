#!/usr/bin/env python3
"""
ChArUco vs Cam-Shift Comparison Tool

Standalone tool for visually comparing ChArUco 6-DOF pose estimation with
cam-shift feature-based detection through dual OpenCV display windows.

Supports two operation modes:
- Offline: Batch processing of image directories for systematic validation
- Online: Real-time camera capture for interactive debugging

Usage:
    # Offline mode (batch processing)
    python tools/validation/comparison_tool.py \
        --mode offline \
        --input-dir session_001/frames \
        --camera-yaml camera.yaml \
        --charuco-config comparison_config.json \
        --camshift-config config.json \
        --output-dir comparison_results

    # Online mode (live camera)
    python tools/validation/comparison_tool.py \
        --mode online \
        --camera-id 0 \
        --camera-yaml camera.yaml \
        --charuco-config comparison_config.json \
        --camshift-config config.json \
        --output-dir comparison_results
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Add project root to Python path for imports
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

import cv2 as cv
import numpy as np

from validation.utilities.dual_detector_runner import DualDetectorRunner, DualDetectionResult
from validation.utilities.comparison_logger import ComparisonLogger


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
        description="ChArUco vs Cam-Shift Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Offline mode:
    %(prog)s --mode offline --input-dir session_001/frames \\
        --camera-yaml camera.yaml --charuco-config comparison_config.json \\
        --camshift-config config.json --output-dir comparison_results

  Online mode:
    %(prog)s --mode online --camera-id 0 \\
        --camera-yaml camera.yaml --charuco-config comparison_config.json \\
        --camshift-config config.json --output-dir comparison_results
"""
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "online"],
        required=True,
        help="Operation mode: 'offline' for batch processing, 'online' for live camera"
    )
    parser.add_argument(
        "--camera-yaml",
        type=str,
        required=True,
        help="Path to camera calibration YAML file"
    )
    parser.add_argument(
        "--charuco-config",
        type=str,
        required=True,
        help="Path to ChArUco configuration JSON file"
    )
    parser.add_argument(
        "--camshift-config",
        type=str,
        required=True,
        help="Path to cam-shift detector configuration JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output files (logs, graphs, reports)"
    )

    # Offline mode arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory with images (required for offline mode)"
    )

    # Online mode arguments
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0, used for online mode)"
    )

    # Optional display arguments
    parser.add_argument(
        "--show-features",
        action="store_true",
        help="Display ORB features on cam-shift window"
    )

    args = parser.parse_args()

    # Validate mode-specific arguments
    if args.mode == "offline" and not args.input_dir:
        parser.error("--input-dir is required for offline mode")

    return args


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate arguments and check file existence.

    Args:
        args: Parsed arguments

    Raises:
        SystemExit: If validation fails
    """
    # Check camera YAML exists
    if not Path(args.camera_yaml).exists():
        logger.error(f"Camera YAML not found: {args.camera_yaml}")
        sys.exit(1)

    # Check ChArUco config exists
    if not Path(args.charuco_config).exists():
        logger.error(f"ChArUco config not found: {args.charuco_config}")
        sys.exit(1)

    # Check cam-shift config exists
    if not Path(args.camshift_config).exists():
        logger.error(f"Cam-shift config not found: {args.camshift_config}")
        sys.exit(1)

    # Check input directory exists (for offline mode)
    if args.mode == "offline":
        input_path = Path(args.input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)
        if not input_path.is_dir():
            logger.error(f"Input path is not a directory: {args.input_dir}")
            sys.exit(1)

    logger.info("✅ All required files validated")


def load_charuco_config(config_path: str) -> Dict[str, Any]:
    """Load ChArUco configuration from JSON file.

    Args:
        config_path: Path to configuration JSON

    Returns:
        Configuration dictionary

    Raises:
        SystemExit: If configuration cannot be loaded
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ["charuco_board", "comparison_settings", "display_settings"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in ChArUco config: {field}")
                sys.exit(1)

        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in ChArUco config: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load ChArUco config: {e}")
        sys.exit(1)


def create_display_windows(
    charuco_frame: np.ndarray,
    camshift_frame: np.ndarray,
    result: DualDetectionResult,
    config: Dict[str, Any],
    fps: float
) -> np.ndarray:
    """Create annotated dual display windows with status bar.

    Args:
        charuco_frame: ChArUco detector frame (BGR)
        camshift_frame: Cam-shift detector frame (BGR)
        result: DualDetectionResult from process_frame()
        config: Display configuration
        fps: Current frames per second

    Returns:
        Combined display image (ChArUco left, Cam-shift right, status bar below)
    """
    # Get display settings
    display_settings = config.get("display_settings", {})
    window_width = display_settings.get("window_width", 640)
    window_height = display_settings.get("window_height", 480)

    # Resize frames to display size
    charuco_display = cv.resize(charuco_frame, (window_width, window_height))
    camshift_display = cv.resize(camshift_frame, (window_width, window_height))

    # Annotate ChArUco frame
    charuco_display = annotate_charuco_frame(charuco_display, result)

    # Annotate Cam-shift frame
    camshift_display = annotate_camshift_frame(camshift_display, result)

    # Create status bar
    status_bar = draw_comparison_status_bar(window_width * 2, result, fps)

    # Combine frames horizontally
    combined_top = np.hstack([charuco_display, camshift_display])

    # Add status bar below
    combined = np.vstack([combined_top, status_bar])

    return combined


def annotate_charuco_frame(frame: np.ndarray, result: DualDetectionResult) -> np.ndarray:
    """Annotate ChArUco frame with detection info.

    Args:
        frame: ChArUco frame (BGR)
        result: DualDetectionResult

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Overlay displacement
    disp_text = f"Disp: {result.charuco_displacement_px:.2f}px" if not np.isnan(result.charuco_displacement_px) else "Disp: N/A"
    cv.putText(annotated, disp_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay status
    status_text = "Status: DETECTED" if result.charuco_detected else "Status: NOT DETECTED"
    status_color = (0, 255, 0) if result.charuco_detected else (0, 0, 255)
    cv.putText(annotated, status_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Overlay confidence
    conf_text = f"Confidence: {result.charuco_confidence:.0f}" if not np.isnan(result.charuco_confidence) else "Confidence: N/A"
    cv.putText(annotated, conf_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated


def annotate_camshift_frame(frame: np.ndarray, result: DualDetectionResult) -> np.ndarray:
    """Annotate Cam-shift frame with detection info.

    Args:
        frame: Cam-shift frame (BGR)
        result: DualDetectionResult

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    # Overlay displacement
    disp_text = f"Disp: {result.camshift_displacement_px:.2f}px"
    cv.putText(annotated, disp_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay status
    status_text = f"Status: {result.camshift_status}"
    status_color = (0, 255, 0) if result.camshift_status == "VALID" else (0, 165, 255)
    cv.putText(annotated, status_text, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Overlay confidence
    conf_text = f"Confidence: {result.camshift_confidence:.2f}"
    cv.putText(annotated, conf_text, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated


def draw_comparison_status_bar(width: int, result: DualDetectionResult, fps: float) -> np.ndarray:
    """Draw comparison status bar showing agreement metrics.

    Args:
        width: Status bar width in pixels
        result: DualDetectionResult
        fps: Current frames per second

    Returns:
        Status bar image (BGR)
    """
    # Create status bar (height: 80px)
    status_bar = np.zeros((80, width, 3), dtype=np.uint8)

    # Determine bar color based on agreement
    if result.agreement_status == "GREEN":
        bar_color = (0, 255, 0)  # Green
    elif result.agreement_status == "RED":
        bar_color = (0, 0, 255)  # Red
    else:
        bar_color = (128, 128, 128)  # Gray (no comparison available)

    # Fill background with bar color
    status_bar[:] = bar_color

    # Add comparison metric text
    if not np.isnan(result.displacement_diff):
        comparison_text = f"||d1-d2||_2 = {result.displacement_diff:.2f}px [{result.agreement_status}]"
    else:
        comparison_text = "||d1-d2||_2 = N/A [ChArUco not detected]"

    cv.putText(status_bar, comparison_text, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add threshold text
    threshold_text = f"Threshold: {result.threshold_px:.2f}px"
    cv.putText(status_bar, threshold_text, (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Add frame info and FPS
    frame_text = f"Frame: {result.frame_idx} | FPS: {fps:.1f}"
    cv.putText(status_bar, frame_text, (width - 350, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return status_bar


def run_offline_comparison(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run offline comparison mode (batch processing).

    Args:
        args: Parsed command-line arguments
        config: ChArUco configuration
    """
    logger.info("🎬 Starting offline comparison mode")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DualDetectorRunner
    charuco_board = config["charuco_board"]
    comparison_settings = config["comparison_settings"]

    runner = DualDetectorRunner(
        camera_yaml_path=args.camera_yaml,
        camshift_config_path=args.camshift_config,
        charuco_squares_x=charuco_board["squares_x"],
        charuco_squares_y=charuco_board["squares_y"],
        charuco_square_len_m=charuco_board["square_len_m"],
        charuco_marker_len_m=charuco_board["marker_len_m"],
        charuco_dict_name=charuco_board["dict_name"],
        z_distance_m=comparison_settings["default_z_distance_m"]
    )

    # Initialize ComparisonLogger
    session_name = Path(args.input_dir).name
    logger_obj = ComparisonLogger(output_dir=str(output_dir), session_name=session_name)

    # Load all images (sorted)
    input_path = Path(args.input_dir)
    image_files = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))

    if not image_files:
        logger.error(f"No images found in {args.input_dir}")
        sys.exit(1)

    logger.info(f"📁 Found {len(image_files)} images")

    # Load first image and set baseline
    logger.info("🎯 Setting baseline from first image...")
    first_image = cv.imread(str(image_files[0]))
    if first_image is None:
        logger.error(f"Failed to load first image: {image_files[0]}")
        sys.exit(1)

    if not runner.set_baseline(first_image):
        logger.error("❌ Failed to set baseline (ChArUco board not detected)")
        sys.exit(1)

    logger.info("✅ Baseline set successfully")

    # Process all frames
    logger.info("🔄 Processing frames...")
    start_time = time.time()
    frame_times = []

    for i, image_file in enumerate(image_files):
        frame_start = time.time()

        # Load image
        image = cv.imread(str(image_file))
        if image is None:
            logger.warning(f"Skipping unreadable image: {image_file}")
            continue

        # Process frame
        result = runner.process_frame(image, frame_id=image_file.name)

        # Log result
        logger_obj.log_frame(result)

        # Calculate FPS
        frame_times.append(time.time() - frame_start)
        recent_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0.0

        # Create and display frames
        combined_display = create_display_windows(image, image, result, config, recent_fps)
        cv.imshow("ChArUco vs Cam-Shift Comparison", combined_display)

        # Wait 1ms for key press
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("⏹️  User requested quit")
            break

        # Progress update
        if (i + 1) % 10 == 0:
            logger.info(f"   Processed {i + 1}/{len(image_files)} frames ({recent_fps:.1f} FPS)")

    # Calculate elapsed time and FPS
    elapsed_time = time.time() - start_time
    avg_fps = len(logger_obj.results) / elapsed_time if elapsed_time > 0 else 0.0

    logger.info(f"✅ Processed {len(logger_obj.results)} frames in {elapsed_time:.2f}s ({avg_fps:.2f} FPS)")

    # Close display windows
    cv.destroyAllWindows()

    # Save log
    logger.info("💾 Saving results...")
    log_path = logger_obj.save_log()
    logger.info(f"   Log saved: {log_path}")

    # Generate MSE graph
    try:
        graph_path = logger_obj.generate_mse_graph()
        logger.info(f"   MSE graph saved: {graph_path}")
    except ValueError as e:
        logger.warning(f"   MSE graph generation failed: {e}")

    # Get worst matches
    worst_matches = logger_obj.get_worst_matches(n=10)
    logger.info(f"   Worst matches retrieved: {len(worst_matches)}")

    # Save worst matches report
    worst_matches_path = output_dir / f"{session_name}_worst_matches.txt"
    with open(worst_matches_path, 'w') as f:
        f.write("Top 10 Worst Matches (largest displacement differences)\n")
        f.write("=" * 70 + "\n\n")
        for i, match in enumerate(worst_matches, 1):
            f.write(f"{i}. Frame {match.frame_idx}:\n")
            f.write(f"   ChArUco displacement: {match.charuco_displacement_px:.2f}px\n")
            f.write(f"   Cam-shift displacement: {match.camshift_displacement_px:.2f}px\n")
            f.write(f"   Difference: {match.displacement_diff:.2f}px\n")
            f.write(f"   Agreement: {match.agreement_status}\n\n")

    logger.info(f"   Worst matches report saved: {worst_matches_path}")
    logger.info("🎉 Offline comparison complete!")


def run_online_comparison(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    """Run online comparison mode (live camera).

    Args:
        args: Parsed command-line arguments
        config: ChArUco configuration
    """
    logger.info("🎬 Starting online comparison mode")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize DualDetectorRunner
    charuco_board = config["charuco_board"]
    comparison_settings = config["comparison_settings"]

    runner = DualDetectorRunner(
        camera_yaml_path=args.camera_yaml,
        camshift_config_path=args.camshift_config,
        charuco_squares_x=charuco_board["squares_x"],
        charuco_squares_y=charuco_board["squares_y"],
        charuco_square_len_m=charuco_board["square_len_m"],
        charuco_marker_len_m=charuco_board["marker_len_m"],
        charuco_dict_name=charuco_board["dict_name"],
        z_distance_m=comparison_settings["default_z_distance_m"]
    )

    # Initialize ComparisonLogger
    session_name = f"online_session_{int(time.time())}"
    logger_obj = ComparisonLogger(output_dir=str(output_dir), session_name=session_name)

    # Open camera
    logger.info(f"📹 Opening camera {args.camera_id}...")
    cap = cv.VideoCapture(args.camera_id)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open camera {args.camera_id}")
        sys.exit(1)

    logger.info("✅ Camera opened successfully")
    logger.info("📝 Instructions: Press 's' to set baseline, 'q' to quit")

    # Wait for baseline
    baseline_set = False
    while not baseline_set:
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ Failed to read from camera")
            cap.release()
            sys.exit(1)

        # Display instructions
        display_frame = frame.copy()
        cv.putText(display_frame, "Press 's' to set baseline, 'q' to quit",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("ChArUco vs Cam-Shift Comparison", display_frame)

        key = cv.waitKey(30) & 0xFF
        if key == ord('s'):
            # Set baseline
            logger.info("🎯 Setting baseline...")
            if runner.set_baseline(frame):
                logger.info("✅ Baseline set successfully")
                baseline_set = True
            else:
                logger.warning("❌ Baseline failed (ChArUco not detected), try again")
        elif key == ord('q'):
            logger.info("⏹️  User quit before setting baseline")
            cap.release()
            cv.destroyAllWindows()
            return

    # Continuous processing loop
    logger.info("🔄 Starting continuous processing...")
    logger.info("📝 Press 'q' to quit and generate reports")

    frame_times = []
    frame_count = 0

    while True:
        frame_start = time.time()

        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.warning("⚠️  Failed to read frame, ending session")
            break

        # Process frame
        result = runner.process_frame(frame, frame_id=f"frame_{frame_count}")
        frame_count += 1

        # Log result
        logger_obj.log_frame(result)

        # Calculate FPS
        frame_times.append(time.time() - frame_start)
        recent_fps = 1.0 / np.mean(frame_times[-30:]) if frame_times else 0.0

        # Create and display frames
        combined_display = create_display_windows(frame, frame, result, config, recent_fps)
        cv.imshow("ChArUco vs Cam-Shift Comparison", combined_display)

        # Check for quit
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("⏹️  User requested quit")
            break

    # Release camera and close windows
    cap.release()
    cv.destroyAllWindows()

    logger.info(f"✅ Processed {len(logger_obj.results)} frames")

    # Save log
    logger.info("💾 Saving results...")
    log_path = logger_obj.save_log()
    logger.info(f"   Log saved: {log_path}")

    # Generate MSE graph
    try:
        graph_path = logger_obj.generate_mse_graph()
        logger.info(f"   MSE graph saved: {graph_path}")
    except ValueError as e:
        logger.warning(f"   MSE graph generation failed: {e}")

    # Get worst matches
    worst_matches = logger_obj.get_worst_matches(n=10)
    logger.info(f"   Worst matches retrieved: {len(worst_matches)}")

    # Save worst matches report
    worst_matches_path = output_dir / f"{session_name}_worst_matches.txt"
    with open(worst_matches_path, 'w') as f:
        f.write("Top 10 Worst Matches (largest displacement differences)\n")
        f.write("=" * 70 + "\n\n")
        for i, match in enumerate(worst_matches, 1):
            f.write(f"{i}. Frame {match.frame_idx}:\n")
            f.write(f"   ChArUco displacement: {match.charuco_displacement_px:.2f}px\n")
            f.write(f"   Cam-shift displacement: {match.camshift_displacement_px:.2f}px\n")
            f.write(f"   Difference: {match.displacement_diff:.2f}px\n")
            f.write(f"   Agreement: {match.agreement_status}\n\n")

    logger.info(f"   Worst matches report saved: {worst_matches_path}")
    logger.info("🎉 Online comparison complete!")


def main():
    """Main entry point."""
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(args)

    # Load ChArUco configuration
    config = load_charuco_config(args.charuco_config)

    # Route to appropriate mode
    if args.mode == "offline":
        run_offline_comparison(args, config)
    elif args.mode == "online":
        run_online_comparison(args, config)
    else:
        logger.error(f"Invalid mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
