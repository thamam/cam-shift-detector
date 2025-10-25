"""Result Manager for Camera Movement Detection

This module provides functionality to manage detection results and maintain
a FIFO history buffer. It builds standardized result dictionaries with status,
translation displacement, confidence, frame ID, and timestamps.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional
import uuid


class ResultManager:
    """Manages detection results and maintains FIFO history buffer.

    This class builds standardized result dictionaries from movement detection
    data and maintains a circular buffer of recent results. It determines status
    based on translation displacement thresholds and provides query methods for
    retrieving history.

    ARCHITECTURAL LIMITATION (MVP Constraint):
        The translation_displacement field only measures translation (tx, ty)
        from the homography matrix. It does NOT include rotation, scale, or
        shear/perspective distortion. This is an architectural constraint from
        the MovementDetector implementation (Story 1.3).

    Attributes:
        threshold_pixels (float): Translation displacement threshold in pixels (default 2.0)
        _buffer (deque): Circular buffer storing recent detection results
    """

    def __init__(self, threshold_pixels: float = 2.0, history_buffer_size: int = 100) -> None:
        """Initialize ResultManager with threshold and buffer size.

        Args:
            threshold_pixels: Translation displacement threshold in pixels.
                            Displacements >= this value result in INVALID status.
                            Default is 2.0 pixels per technical specification.
            history_buffer_size: Maximum number of results to store in history buffer.
                               Uses FIFO eviction when full. Default is 100.

        Raises:
            ValueError: If threshold_pixels is not positive
            ValueError: If history_buffer_size is not a positive integer
        """
        # Validate threshold_pixels (Subtask 4.3)
        if not isinstance(threshold_pixels, (int, float)) or threshold_pixels <= 0:
            raise ValueError(
                f"threshold_pixels must be positive, got {threshold_pixels}"
            )

        # Validate history_buffer_size (Subtask 4.4, 2.3)
        if not isinstance(history_buffer_size, int) or history_buffer_size <= 0:
            raise ValueError(
                f"history_buffer_size must be positive integer, got {history_buffer_size}"
            )

        self.threshold_pixels = float(threshold_pixels)
        # Initialize FIFO buffer with maxlen for automatic eviction (Subtask 2.1, 2.4)
        self._buffer = deque(maxlen=history_buffer_size)

    def create_result(
        self,
        translation_displacement: float,
        confidence: float,
        frame_id: Optional[str] = None
    ) -> Dict[str, any]:
        """Create standardized result dictionary from detection data.

        Builds a result dictionary with status determination, timestamp generation,
        and automatic frame ID generation if not provided.

        Args:
            translation_displacement: Translation magnitude in pixels from MovementDetector.
                                    Must be non-negative. Rounded to 2 decimals internally.
            confidence: Detection confidence [0.0, 1.0] from MovementDetector.
                       Based on inlier ratio from homography estimation.
            frame_id: Optional frame identifier. Auto-generated (UUID) if None.

        Returns:
            Result dictionary with fields in exact order:
            {
                "status": "VALID" | "INVALID",
                "translation_displacement": float (2 decimals),
                "confidence": float (2 decimals),
                "frame_id": str,
                "timestamp": str (ISO 8601 UTC with milliseconds and 'Z')
            }

        Raises:
            ValueError: If translation_displacement is negative
            ValueError: If confidence is not in range [0.0, 1.0]

        Note:
            translation_displacement only measures translation (tx, ty) from homography.
            Rotation, scale, and shear are not detected (architectural limitation).
        """
        # Validate translation_displacement (Subtask 4.1)
        if not isinstance(translation_displacement, (int, float)) or translation_displacement < 0:
            raise ValueError(
                f"translation_displacement must be non-negative float, got {translation_displacement}"
            )

        # Validate confidence (Subtask 4.2)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            raise ValueError(
                f"confidence must be in range [0.0, 1.0], got {confidence}"
            )

        # Round to 2 decimals (constraint from context)
        translation_displacement = round(float(translation_displacement), 2)
        confidence = round(float(confidence), 2)

        # Determine status from threshold comparison (Subtask 1.3, AC-1.4.2)
        # displacement < threshold → "VALID" (no translation movement detected)
        # displacement >= threshold → "INVALID" (translation detected, measurements corrupted)
        status = "VALID" if translation_displacement < self.threshold_pixels else "INVALID"

        # Generate ISO 8601 UTC timestamp with milliseconds (Subtask 1.4, AC-1.4.3)
        timestamp = datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

        # Auto-generate frame_id if not provided (Subtask 1.5)
        if frame_id is None:
            frame_id = str(uuid.uuid4())

        # Build result dictionary with exact field order (Subtask 1.6, AC-1.4.1)
        result = {
            "status": status,
            "translation_displacement": translation_displacement,
            "confidence": confidence,
            "frame_id": frame_id,
            "timestamp": timestamp
        }

        return result

    def add_to_history(self, result: Dict[str, any]) -> None:
        """Add result to FIFO history buffer.

        Appends result to buffer with automatic eviction of oldest entry when full.
        Validates that required fields are present in result dictionary.

        Args:
            result: Result dictionary to add. Must contain required fields:
                   status, translation_displacement, confidence, frame_id, timestamp

        Raises:
            ValueError: If result is missing required fields
        """
        # Validate result dict has required fields (Subtask 2.2)
        required_fields = ["status", "translation_displacement", "confidence", "frame_id", "timestamp"]
        for field in required_fields:
            if field not in result:
                raise ValueError(
                    f"result dict missing required field: {field}"
                )

        # Append to deque (automatic eviction if full) (Subtask 2.2, 2.4)
        self._buffer.append(result)

    def get_history(self) -> List[Dict[str, any]]:
        """Return all results in history buffer.

        Args:
            None

        Returns:
            List of result dictionaries in chronological order (oldest to newest).
            Returns empty list if buffer is empty.
        """
        # Return list (not deque object) for consistent API (Subtask 3.1, constraint)
        return list(self._buffer)

    def get_last_n(self, n: int) -> List[Dict[str, any]]:
        """Return most recent n results from history buffer.

        Args:
            n: Number of recent results to retrieve. Must be positive integer.

        Returns:
            List of up to n most recent result dictionaries.
            If n > buffer size, returns all available results.
            Returns empty list if buffer is empty.

        Raises:
            ValueError: If n is not a positive integer
        """
        # Validate n is positive integer (Subtask 3.2, 3.4)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(
                f"n must be positive integer, got {n}"
            )

        # Handle edge case: n > buffer size (Subtask 3.4)
        if n >= len(self._buffer):
            return list(self._buffer)

        # Return last n items (Subtask 3.2)
        return list(self._buffer)[-n:]

    def get_by_frame_id(self, frame_id: str) -> Optional[Dict[str, any]]:
        """Search buffer for result with matching frame_id.

        Args:
            frame_id: Frame identifier to search for

        Returns:
            First matching result dictionary, or None if not found.

        Raises:
            ValueError: If frame_id is not a string
        """
        # Validate frame_id type (Subtask 3.3)
        if not isinstance(frame_id, str):
            raise ValueError(
                f"frame_id must be string, got {type(frame_id).__name__}"
            )

        # Iterate buffer, return first match or None (Subtask 3.3, 3.4)
        for result in self._buffer:
            if result.get("frame_id") == frame_id:
                return result

        return None
