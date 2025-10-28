"""
Stub Implementation of CameraMovementDetector

This stub provides a mock implementation of the CameraMovementDetector API
for integration testing and parallel development. It returns realistic mock
data without performing actual computer vision operations.

Usage:
    from src.camera_movement_detector_stub import CameraMovementDetector

    # Same API as real implementation
    detector = CameraMovementDetector('config.json')
    detector.set_baseline(initial_frame)
    result = detector.process_frame(current_frame)

Swapping to Real Implementation:
    Simply change the import:
    from src.camera_movement_detector import CameraMovementDetector

Author: BMAD Dev Team
Version: 0.1.0
Date: 2025-10-26
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class CameraMovementDetector:
    """
    Stub implementation of CameraMovementDetector for integration testing.

    This class provides the same public API as the real implementation but
    returns mock data without performing actual computer vision operations.
    Useful for:
    - Integration testing without production dependencies
    - Parallel development (integrate now, swap later)
    - System testing with predictable responses
    """

    def __init__(self, config_path: str = 'config.json') -> None:
        """
        Initialize detector with configuration (stub).

        Args:
            config_path: Path to JSON config file with ROI and parameters

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config validation fails
        """
        self.config_path = config_path
        self.baseline_set = False
        self.config = {}
        self.frame_counter = 0

        # Load and validate config
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration file (stub validation)"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)

            # Basic validation
            if 'roi' not in self.config:
                raise ValueError("Missing 'roi' in config")
            if 'threshold_pixels' not in self.config:
                raise ValueError("Missing 'threshold_pixels' in config")

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

    def set_baseline(self, image_array: np.ndarray) -> None:
        """
        Capture initial baseline features (stub).

        Args:
            image_array: Reference image for baseline

        Raises:
            ValueError: If image_array is invalid or insufficient features
        """
        # Validate image format
        if not isinstance(image_array, np.ndarray):
            raise ValueError("image_array must be numpy.ndarray")

        if image_array.ndim != 3:
            raise ValueError(f"image_array must be 3D (H×W×3), got shape {image_array.shape}")

        if image_array.shape[2] != 3:
            raise ValueError(f"image_array must have 3 channels (BGR), got {image_array.shape[2]}")

        if image_array.dtype != np.uint8:
            raise ValueError(f"image_array must be uint8, got {image_array.dtype}")

        # Simulate feature validation
        # In real implementation, would check for ≥50 features in ROI
        min_features = self.config.get('min_features_required', 50)
        simulated_feature_count = np.random.randint(min_features, min_features + 200)

        if simulated_feature_count < min_features:
            raise ValueError(
                f"Insufficient features detected: {simulated_feature_count} "
                f"(minimum required: {min_features})"
            )

        self.baseline_set = True

    def process_frame(
        self,
        image_array: np.ndarray,
        frame_id: Optional[str] = None
    ) -> Dict:
        """
        Detect camera movement in single frame (stub).

        Args:
            image_array: NumPy array (H × W × 3, uint8, BGR format)
            frame_id: Optional identifier for tracking (auto-generated if None)

        Returns:
            {
                "status": "VALID" | "INVALID",
                "displacement": float,  # pixels
                "confidence": float,    # [0.0, 1.0] inlier ratio
                "frame_id": str,
                "timestamp": str  # ISO 8601 UTC
            }

        Raises:
            RuntimeError: If baseline not set (call set_baseline() first)
            ValueError: If image_array invalid format
        """
        if not self.baseline_set:
            raise RuntimeError(
                "Baseline not set. Call set_baseline() before process_frame()"
            )

        # Validate image format
        if not isinstance(image_array, np.ndarray):
            raise ValueError("image_array must be numpy.ndarray")

        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError(
                f"image_array must be H×W×3, got shape {image_array.shape}"
            )

        if image_array.dtype != np.uint8:
            raise ValueError(f"image_array must be uint8, got {image_array.dtype}")

        # Generate frame ID if not provided
        if frame_id is None:
            self.frame_counter += 1
            frame_id = f"stub_frame_{self.frame_counter:06d}"

        # Generate mock result
        # Simulate realistic behavior:
        # - 95% VALID, 5% INVALID
        # - Displacement between 0.0 and 5.0 pixels
        # - Confidence between 0.7 and 1.0

        threshold = self.config.get('threshold_pixels', 2.0)
        is_valid = np.random.random() > 0.05  # 95% valid

        if is_valid:
            # VALID: displacement below threshold
            displacement = np.random.uniform(0.0, threshold - 0.1)
            confidence = np.random.uniform(0.85, 1.0)
            status = "VALID"
        else:
            # INVALID: displacement above threshold
            displacement = np.random.uniform(threshold, threshold * 2.0)
            confidence = np.random.uniform(0.7, 0.95)
            status = "INVALID"

        # Build result dict
        result = {
            "status": status,
            "displacement": round(displacement, 2),
            "confidence": round(confidence, 2),
            "frame_id": frame_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        return result

    def recalibrate(self, image_array: np.ndarray) -> bool:
        """
        Manually reset baseline features (stub).

        Args:
            image_array: New reference image (required)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.set_baseline(image_array)
            return True
        except (ValueError, Exception):
            return False

    def get_history(
        self,
        frame_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Query detection history buffer (stub).

        Args:
            frame_id: Return results for specific frame_id (optional)
            limit: Return last N results (optional)

        Returns:
            List of detection result dicts (empty list in stub)

        Note:
            Stub implementation returns empty list.
            Real implementation maintains FIFO buffer of last 100 results.
        """
        # Stub: return empty history
        # Real implementation would return actual detection results
        return []


# Mock responses for specific testing scenarios
class MockCameraMovementDetector(CameraMovementDetector):
    """
    Extended stub with configurable mock responses for testing.

    Example:
        detector = MockCameraMovementDetector('config.json')
        detector.set_mock_response(status="INVALID", displacement=3.5)
        result = detector.process_frame(frame)  # Returns mocked response
    """

    def __init__(self, config_path: str = 'config.json') -> None:
        super().__init__(config_path)
        self.mock_response = None

    def set_mock_response(
        self,
        status: str = "VALID",
        displacement: float = 0.5,
        confidence: float = 0.95
    ) -> None:
        """
        Set a specific mock response for testing.

        Args:
            status: "VALID" or "INVALID"
            displacement: Displacement in pixels
            confidence: Confidence score [0.0, 1.0]
        """
        self.mock_response = {
            "status": status,
            "displacement": round(displacement, 2),
            "confidence": round(confidence, 2)
        }

    def process_frame(
        self,
        image_array: np.ndarray,
        frame_id: Optional[str] = None
    ) -> Dict:
        """Process frame with mocked response if set"""
        if self.mock_response is None:
            return super().process_frame(image_array, frame_id)

        # Validate baseline
        if not self.baseline_set:
            raise RuntimeError(
                "Baseline not set. Call set_baseline() before process_frame()"
            )

        # Validate image format (basic)
        if not isinstance(image_array, np.ndarray):
            raise ValueError("image_array must be numpy.ndarray")

        # Generate frame ID
        if frame_id is None:
            self.frame_counter += 1
            frame_id = f"mock_frame_{self.frame_counter:06d}"

        # Return mocked response
        result = {
            **self.mock_response,
            "frame_id": frame_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        return result
