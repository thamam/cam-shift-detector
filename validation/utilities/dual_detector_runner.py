"""
Dual Detector Runner - ChArUco and Cam-Shift Orchestration

Orchestrates parallel execution of ChArUco 6-DOF pose estimation and cam-shift
feature-based detector, enabling systematic comparison of their agreement.
"""

import time
import logging
import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

from src.camera_movement_detector import CameraMovementDetector
from tools.aruco.camshift_annotator import (
    make_charuco_board,
    read_yaml_camera,
    estimate_pose_charuco
)
from validation.utilities.comparison_metrics import (
    calculate_displacement_difference,
    calculate_threshold,
    classify_agreement,
    calculate_charuco_displacement_2d,
    calculate_charuco_displacement_2d_components
)


logger = logging.getLogger(__name__)


@dataclass
class DualDetectionResult:
    """Result from dual detector processing.

    Attributes:
        frame_idx: Frame index in sequence
        timestamp_ns: Nanosecond timestamp (time.time_ns())
        charuco_detected: Whether ChArUco board was detected
        charuco_displacement_px: ChArUco 2D displacement in pixels (NaN if not detected)
        charuco_dx: ChArUco displacement X component in pixels (NaN if not detected)
        charuco_dy: ChArUco displacement Y component in pixels (NaN if not detected)
        charuco_confidence: ChArUco confidence (corner count, NaN if not detected)
        camshift_status: Cam-shift detector status ("VALID" or "INVALID")
        camshift_displacement_px: Cam-shift 2D displacement in pixels
        camshift_dx: Cam-shift displacement X component in pixels
        camshift_dy: Cam-shift displacement Y component in pixels
        camshift_confidence: Cam-shift confidence score
        displacement_diff: ||d1-d2||_2 comparison metric (NaN if ChArUco not detected)
        agreement_status: "GREEN" if diff <= threshold, "RED" otherwise (None if ChArUco not detected)
        threshold_px: Comparison threshold in pixels
    """
    frame_idx: int
    timestamp_ns: int
    charuco_detected: bool
    charuco_displacement_px: float
    charuco_dx: float
    charuco_dy: float
    charuco_confidence: float
    camshift_status: str
    camshift_displacement_px: float
    camshift_dx: float
    camshift_dy: float
    camshift_confidence: float
    displacement_diff: float
    agreement_status: Optional[str]
    threshold_px: float


class DualDetectorRunner:
    """Orchestrates ChArUco and cam-shift detectors for comparison.

    Initializes both detectors, manages baseline configuration, and processes
    frames through both detection pipelines for systematic comparison.
    """

    def __init__(
        self,
        camera_yaml_path: str,
        camshift_config_path: str,
        charuco_squares_x: int = 7,
        charuco_squares_y: int = 5,
        charuco_square_len_m: float = 0.035,
        charuco_marker_len_m: float = 0.026,
        charuco_dict_name: str = "DICT_4X4_50",
        z_distance_m: float = 1.15
    ):
        """Initialize dual detector runner.

        Args:
            camera_yaml_path: Path to camera calibration YAML file
            camshift_config_path: Path to cam-shift detector config JSON
            charuco_squares_x: ChArUco board squares in X direction (default: 7)
            charuco_squares_y: ChArUco board squares in Y direction (default: 5)
            charuco_square_len_m: ChArUco square size in meters (default: 0.035)
            charuco_marker_len_m: ChArUco marker size in meters (default: 0.026)
            charuco_dict_name: ArUco dictionary name (default: "DICT_4X4_50")
            z_distance_m: Camera-to-board distance for 3D-to-2D projection (default: 1.15m)

        Raises:
            FileNotFoundError: If camera_yaml_path or camshift_config_path not found
            ValueError: If camera calibration cannot be loaded
        """
        # Validate paths
        if not Path(camera_yaml_path).exists():
            raise FileNotFoundError(f"Camera YAML not found: {camera_yaml_path}")
        if not Path(camshift_config_path).exists():
            raise FileNotFoundError(f"Camshift config not found: {camshift_config_path}")

        # Load camera calibration
        result = read_yaml_camera(camera_yaml_path)
        if result is None:
            raise ValueError(f"Failed to load camera calibration from {camera_yaml_path}")

        self.K, self.dist, (self.image_width, self.image_height) = result

        # Initialize ChArUco detector
        self.charuco_dict, self.charuco_board, self.charuco_detector = make_charuco_board(
            charuco_squares_x,
            charuco_squares_y,
            charuco_square_len_m,
            charuco_marker_len_m,
            charuco_dict_name
        )

        # Initialize cam-shift detector
        self.camshift_detector = CameraMovementDetector(config_path=camshift_config_path)

        # Store configuration
        self.z_distance_m = z_distance_m
        self.threshold_px = calculate_threshold(self.image_width, self.image_height, percent=0.03)

        # Baseline storage
        self.tvec_baseline: Optional[np.ndarray] = None
        self.baseline_set = False
        self.frame_counter = 0

        logger.info(f"Initialized DualDetectorRunner:")
        logger.info(f"  Image size: {self.image_width}x{self.image_height}")
        logger.info(f"  Threshold: {self.threshold_px:.2f}px (3% of {min(self.image_width, self.image_height)}px)")
        logger.info(f"  Z distance: {self.z_distance_m}m")
        logger.info(f"  ChArUco board: {charuco_squares_x}x{charuco_squares_y}, {charuco_square_len_m}m squares")

    def set_baseline(self, image: np.ndarray) -> bool:
        """Set baseline for both detectors from given image.

        Args:
            image: Baseline image (BGR or grayscale)

        Returns:
            True if baselines set successfully, False otherwise

        Note:
            ChArUco baseline requires successful board detection.
            Cam-shift baseline always succeeds.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Set cam-shift baseline (always succeeds)
        self.camshift_detector.set_baseline(image)

        # Attempt ChArUco baseline detection
        pose_result = estimate_pose_charuco(
            image_gray,
            self.charuco_detector,
            self.charuco_board,
            self.K,
            self.dist
        )

        if pose_result is None:
            logger.warning("ChArUco board not detected in baseline image")
            self.baseline_set = False
            self.tvec_baseline = None
            return False

        rvec, tvec, n_corners = pose_result
        self.tvec_baseline = tvec
        self.baseline_set = True
        self.frame_counter = 0

        logger.info(f"Baseline set successfully (ChArUco: {n_corners} corners detected)")
        return True

    def process_frame(self, image: np.ndarray, frame_id: Optional[str] = None) -> DualDetectionResult:
        """Process frame through both detectors and calculate comparison metrics.

        Args:
            image: Input image (BGR or grayscale)
            frame_id: Optional frame identifier for logging

        Returns:
            DualDetectionResult with both detector outputs and comparison metrics

        Raises:
            RuntimeError: If baseline not set before processing

        Note:
            ChArUco detection failures are handled gracefully (displacement=NaN, no crash)
        """
        if not self.baseline_set:
            raise RuntimeError("Baseline must be set before processing frames")

        timestamp_ns = time.time_ns()
        frame_idx = self.frame_counter
        self.frame_counter += 1

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # --- ChArUco Detection ---
        pose_result = estimate_pose_charuco(
            image_gray,
            self.charuco_detector,
            self.charuco_board,
            self.K,
            self.dist
        )

        if pose_result is None:
            # ChArUco not detected - graceful handling
            charuco_detected = False
            charuco_displacement_px = np.nan
            charuco_dx = np.nan
            charuco_dy = np.nan
            charuco_confidence = np.nan
        else:
            # ChArUco detected - calculate displacement
            rvec, tvec, n_corners = pose_result
            charuco_detected = True
            charuco_displacement_px = calculate_charuco_displacement_2d(
                tvec, self.tvec_baseline, self.K, self.z_distance_m
            )
            charuco_dx, charuco_dy = calculate_charuco_displacement_2d_components(
                tvec, self.tvec_baseline, self.K, self.z_distance_m
            )
            charuco_confidence = float(n_corners)

        # --- Cam-shift Detection ---
        camshift_result = self.camshift_detector.process_frame(image, frame_id=frame_id)
        camshift_status = camshift_result["status"]
        camshift_displacement_px = camshift_result["translation_displacement"]
        camshift_confidence = camshift_result["confidence"]

        # Extract component-wise displacements from movement detector
        movement_detector = self.camshift_detector.movement_detector
        camshift_dx = movement_detector.last_tx
        camshift_dy = movement_detector.last_ty

        # --- Comparison Metrics ---
        if charuco_detected:
            displacement_diff = calculate_displacement_difference(
                charuco_displacement_px,
                camshift_displacement_px
            )
            agreement_status = classify_agreement(displacement_diff, self.threshold_px)
        else:
            # Cannot compare if ChArUco not detected
            displacement_diff = np.nan
            agreement_status = None

        # Build result
        result = DualDetectionResult(
            frame_idx=frame_idx,
            timestamp_ns=timestamp_ns,
            charuco_detected=charuco_detected,
            charuco_displacement_px=charuco_displacement_px,
            charuco_dx=charuco_dx,
            charuco_dy=charuco_dy,
            charuco_confidence=charuco_confidence,
            camshift_status=camshift_status,
            camshift_displacement_px=camshift_displacement_px,
            camshift_dx=camshift_dx,
            camshift_dy=camshift_dy,
            camshift_confidence=camshift_confidence,
            displacement_diff=displacement_diff,
            agreement_status=agreement_status,
            threshold_px=self.threshold_px
        )

        return result
