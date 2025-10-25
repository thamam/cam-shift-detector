"""Camera Movement Detector - Main Black-Box API.

This module provides the CameraMovementDetector class, the primary interface for
integrating camera movement detection into DAF water quality monitoring systems.
It orchestrates all core components (StaticRegionManager, FeatureExtractor,
MovementDetector, ResultManager) through a simple black-box API.
"""

import json
import numpy as np
from typing import Dict, List, Optional

from src.static_region_manager import StaticRegionManager
from src.feature_extractor import FeatureExtractor
from src.movement_detector import MovementDetector
from src.result_manager import ResultManager


class CameraMovementDetector:
    """Black-box interface for DAF system integration.

    This class provides a simple API for camera movement detection that hides
    all implementation complexity. It loads configuration, manages component
    lifecycle, orchestrates the detection pipeline, and returns standardized
    results.

    Attributes:
        config_path (str): Path to JSON configuration file
        config (dict): Loaded configuration dictionary
        region_manager (StaticRegionManager): ROI mask generation component
        feature_extractor (FeatureExtractor): ORB feature management component
        movement_detector (MovementDetector): Homography-based movement detection
        result_manager (ResultManager): Result building and history management
        baseline_set (bool): Flag indicating if baseline has been captured
    """

    def __init__(self, config_path: str = 'config.json') -> None:
        """Initialize detector with configuration.

        Loads configuration from JSON file, validates schema, and initializes
        all internal components. The detector is ready for baseline capture
        after successful initialization.

        Args:
            config_path: Path to JSON config file with ROI and parameters.
                        Defaults to 'config.json' in current directory.

        Raises:
            FileNotFoundError: If config file not found at specified path
            ValueError: If config validation fails (missing fields, invalid types,
                       invalid value ranges)

        Example:
            >>> detector = CameraMovementDetector('config.json')
            >>> detector.set_baseline(reference_image)
            >>> result = detector.process_frame(current_image)
        """
        self.config_path = config_path

        # Load config file
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        # Validate config schema
        self._validate_config(self.config)

        # Initialize all component instances
        self.region_manager = StaticRegionManager(config_path)
        self.feature_extractor = FeatureExtractor(
            min_features_required=self.config['min_features_required'],
            orb_features=self.config.get('orb_features', 1000)  # Default 1000 if not in config
        )
        self.movement_detector = MovementDetector(
            threshold_pixels=self.config['threshold_pixels'],
            use_affine_model=self.config.get('use_affine_model', False)
        )
        self.result_manager = ResultManager(
            threshold_pixels=self.config['threshold_pixels'],
            history_buffer_size=self.config['history_buffer_size']
        )

        # Baseline capture flag
        self.baseline_set = False

    def _validate_config(self, config: dict) -> None:
        """Validate configuration schema and value ranges.

        Checks that all required fields are present, have correct types,
        and contain reasonable values. Raises descriptive errors for any
        validation failures.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If any validation check fails with descriptive message
        """
        # Check required top-level fields
        required_fields = ['roi', 'threshold_pixels', 'history_buffer_size',
                          'min_features_required']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Invalid config: missing required field '{field}'")

        # Validate ROI structure
        roi = config['roi']
        if not isinstance(roi, dict):
            raise ValueError(f"Invalid config: roi must be dict, got {type(roi).__name__}")

        roi_fields = ['x', 'y', 'width', 'height']
        for field in roi_fields:
            if field not in roi:
                raise ValueError(f"Invalid config: missing required ROI field '{field}'")
            if not isinstance(roi[field], (int, float)):
                raise ValueError(
                    f"Invalid config: roi.{field} must be number, got {type(roi[field]).__name__}"
                )
            if roi[field] < 0:
                raise ValueError(
                    f"Invalid config: roi.{field} must be non-negative, got {roi[field]}"
                )

        # Validate threshold_pixels
        threshold = config['threshold_pixels']
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"Invalid config: threshold_pixels must be number, got {type(threshold).__name__}"
            )
        if threshold <= 0:
            raise ValueError(
                f"Invalid config: threshold_pixels must be positive, got {threshold}"
            )

        # Validate history_buffer_size
        buffer_size = config['history_buffer_size']
        if not isinstance(buffer_size, int):
            raise ValueError(
                f"Invalid config: history_buffer_size must be int, got {type(buffer_size).__name__}"
            )
        if buffer_size <= 0:
            raise ValueError(
                f"Invalid config: history_buffer_size must be positive, got {buffer_size}"
            )

        # Validate min_features_required
        min_features = config['min_features_required']
        if not isinstance(min_features, int):
            raise ValueError(
                f"Invalid config: min_features_required must be int, got {type(min_features).__name__}"
            )
        if min_features <= 0:
            raise ValueError(
                f"Invalid config: min_features_required must be positive, got {min_features}"
            )

    def set_baseline(self, image_array: np.ndarray) -> None:
        """Capture initial baseline features (setup phase).

        Validates image format, generates static region mask, extracts ORB
        features, and validates that sufficient features were detected.
        Must be called before process_frame().

        Args:
            image_array: Reference image for baseline (H×W×3, uint8, BGR format)

        Raises:
            ValueError: If insufficient features detected (< min_features_required)
            ValueError: If image_array has invalid format

        Example:
            >>> detector = CameraMovementDetector('config.json')
            >>> baseline_image = cv2.imread('reference.jpg')
            >>> detector.set_baseline(baseline_image)  # Captures baseline features
        """
        # Validate image format
        self._validate_image_format(image_array)

        # Generate static mask
        mask = self.region_manager.get_static_mask(image_array.shape[:2])

        # Set baseline (FeatureExtractor validates min features)
        self.feature_extractor.set_baseline(image_array, mask)

        # Mark baseline as set
        self.baseline_set = True

    def process_frame(self, image_array: np.ndarray, frame_id: Optional[str] = None) -> Dict:
        """Detect camera movement in single frame.

        Orchestrates full detection pipeline: mask generation, feature extraction,
        movement detection, result building, and history storage. Returns
        standardized result dictionary with status, displacement, confidence,
        frame_id, and timestamp.

        Args:
            image_array: NumPy array (H × W × 3, uint8, BGR format)
            frame_id: Optional identifier for tracking (auto-generated if None)

        Returns:
            Detection result dictionary:
            {
                "status": "VALID" | "INVALID",
                "translation_displacement": float,  # pixels (translation only)
                "confidence": float,    # [0.0, 1.0] inlier ratio
                "frame_id": str,
                "timestamp": str  # ISO 8601 UTC
            }

        Raises:
            RuntimeError: If baseline not set (call set_baseline() first)
            ValueError: If image_array has invalid format

        Example:
            >>> detector = CameraMovementDetector('config.json')
            >>> detector.set_baseline(baseline_image)
            >>> result = detector.process_frame(current_image, frame_id="frame_001")
            >>> if result['status'] == 'INVALID':
            ...     print(f"Camera moved {result['translation_displacement']:.2f}px")
        """
        # Check baseline set
        if not self.baseline_set:
            raise RuntimeError("Baseline not set. Call set_baseline() before process_frame()")

        # Validate image format
        self._validate_image_format(image_array)

        # Generate static mask
        mask = self.region_manager.get_static_mask(image_array.shape[:2])

        # Get baseline features
        baseline_features = self.feature_extractor.get_baseline()

        # Extract current features
        current_features = self.feature_extractor.extract_features(image_array, mask)

        # Detect movement (handle edge cases per AC-1.5.4)
        try:
            moved, displacement, confidence = self.movement_detector.detect_movement(
                baseline_features, current_features
            )
        except (ValueError, RuntimeError):
            # Insufficient matches or homography failure → return INVALID with inf displacement
            displacement = float('inf')
            confidence = 0.0

        # Build result dict
        result = self.result_manager.create_result(displacement, confidence, frame_id)

        # Store in history
        self.result_manager.add_to_history(result)

        # Return result
        return result

    def get_history(self, frame_id: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict]:
        """Query detection history buffer.

        Retrieves results from the circular history buffer with optional
        filtering by frame_id or limiting to most recent N results.

        Args:
            frame_id: Return results for specific frame_id (optional)
            limit: Return last N results (optional)

        Returns:
            List of detection result dicts (empty list if no matches)

        Example:
            >>> # Get all history
            >>> all_results = detector.get_history()

            >>> # Get specific frame
            >>> frame_result = detector.get_history(frame_id="frame_001")

            >>> # Get last 10 results
            >>> recent = detector.get_history(limit=10)
        """
        # Delegate to ResultManager with appropriate method
        if frame_id is not None:
            # Get by frame_id, return as list
            result = self.result_manager.get_by_frame_id(frame_id)
            return [result] if result is not None else []
        elif limit is not None:
            # Get last N results
            return self.result_manager.get_last_n(limit)
        else:
            # Get all results
            return self.result_manager.get_history()

    def recalibrate(self, image_array: np.ndarray) -> bool:
        """Manually reset baseline features.

        Attempts to set a new baseline using the provided image. Returns
        True if successful, False if insufficient features detected.
        Useful for handling lighting changes or maintenance scenarios.

        Args:
            image_array: New reference image (H×W×3, uint8, BGR format)

        Returns:
            True if recalibration successful, False if insufficient features

        Example:
            >>> # Attempt recalibration
            >>> success = detector.recalibrate(new_reference_image)
            >>> if success:
            ...     print("Recalibration successful")
            ... else:
            ...     print("Insufficient features for recalibration")
        """
        try:
            self.set_baseline(image_array)
            return True
        except ValueError:
            # Insufficient features or invalid format
            return False

    def _validate_image_format(self, image_array: np.ndarray) -> None:
        """Validate image array format and raise descriptive error if invalid.

        Args:
            image_array: Image to validate

        Raises:
            ValueError: If image format is invalid
        """
        if not isinstance(image_array, np.ndarray):
            raise ValueError(
                f"image_array must be NumPy array, got {type(image_array).__name__}"
            )

        if image_array.ndim != 3:
            raise ValueError(
                f"image_array must have shape (H, W, 3), got shape {image_array.shape}"
            )

        if image_array.shape[2] != 3:
            raise ValueError(
                f"image_array must have 3 channels, got {image_array.shape[2]}"
            )

        if image_array.dtype != np.uint8:
            raise ValueError(
                f"image_array must have dtype uint8, got {image_array.dtype}"
            )
