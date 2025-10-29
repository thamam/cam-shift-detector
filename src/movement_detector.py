"""Movement Detector for Camera Movement Detection

This module provides functionality to detect camera movement by comparing
baseline features to current frame features using homography estimation.
It calculates displacement magnitude and confidence scores to determine
if the camera has moved beyond a configured threshold.
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np


class MovementDetector:
    """Detects camera movement through feature matching and homography estimation.

    This class compares baseline ORB features to current frame features using
    Brute Force matching with Hamming distance. It estimates homography transformation,
    calculates displacement from the translation vector, and returns movement status
    with confidence scores based on inlier ratios.

    ARCHITECTURAL LIMITATION (MVP Constraint):
        This implementation only measures TRANSLATION displacement (tx, ty) from the
        homography matrix. It does NOT detect rotation, scale, or shear/perspective
        distortion. A camera could rotate significantly (corrupting measurements) while
        showing <2.0px translation, resulting in false "no movement" detection.

        Full homography decomposition with separate rotation/scale thresholds is
        planned for Story 1.5 enhancement.

    Attributes:
        threshold_pixels (float): Translation displacement threshold in pixels (default 2.0)
        matcher (cv2.BFMatcher): Brute Force matcher configured for ORB descriptors
    """

    def __init__(self, threshold_pixels: float = 2.0, use_affine_model: bool = False) -> None:
        """Initialize movement detector with displacement threshold.

        Args:
            threshold_pixels: Minimum displacement in pixels to consider movement.
                            Default is 2.0 pixels per technical specification.
            use_affine_model: If True, use 6-DOF affine transformation instead of 8-DOF homography.
                            Affine model is more stable for pure translations and vertical movements.
                            Default is False for backward compatibility.

        Raises:
            ValueError: If threshold_pixels is not a positive number
        """
        if not isinstance(threshold_pixels, (int, float)) or threshold_pixels <= 0:
            raise ValueError(
                f"threshold_pixels must be a positive number, got {threshold_pixels}"
            )

        self.threshold_pixels = float(threshold_pixels)
        self.use_affine_model = use_affine_model
        # Use Brute Force Matcher with Hamming distance for ORB descriptors
        # crossCheck=True ensures bidirectional matching for higher quality matches
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Store last translation components for external access (Mode A enhanced metrics)
        self.last_tx: float = 0.0
        self.last_ty: float = 0.0

        # Store last matches, mask, and homography for Mode B visualization
        self.last_matches: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.last_mask: Optional[np.ndarray] = None
        self.last_homography: Optional[np.ndarray] = None
        self.last_baseline_keypoints: List[cv2.KeyPoint] = []
        self.last_current_keypoints: List[cv2.KeyPoint] = []

    def detect_movement(
        self,
        baseline_features: Tuple[List[cv2.KeyPoint], np.ndarray],
        current_features: Tuple[List[cv2.KeyPoint], np.ndarray],
    ) -> Tuple[bool, float, float]:
        """Detect camera movement between baseline and current features.

        Matches features using BFMatcher, estimates homography transformation,
        calculates TRANSLATION displacement from translation vector (tx, ty), and
        determines if movement exceeds threshold. Returns confidence based on inlier ratio.

        Note:
            This method only measures translation displacement. Rotation, scale, and
            shear are not detected. See class docstring for architectural limitation details.

        Args:
            baseline_features: Tuple of (keypoints, descriptors) from baseline image
            current_features: Tuple of (keypoints, descriptors) from current frame

        Returns:
            Tuple of (moved, displacement, confidence):
                - moved (bool): True if translation displacement >= threshold, False otherwise
                - displacement (float): Translation magnitude in pixels (rounded to 2 decimals)
                - confidence (float): Inlier ratio [0.0, 1.0] from homography

        Raises:
            ValueError: If features have invalid format or insufficient matches (<10)
            RuntimeError: If homography estimation fails
        """
        # Validate feature tuple formats (AC-1.3.6)
        self._validate_features(baseline_features, "baseline_features")
        self._validate_features(current_features, "current_features")

        baseline_keypoints, baseline_descriptors = baseline_features
        current_keypoints, current_descriptors = current_features

        # Match features using BFMatcher (AC-1.3.1)
        matches = self.matcher.match(baseline_descriptors, current_descriptors)

        # Validate minimum match count (AC-1.3.2)
        if len(matches) < 10:
            raise ValueError(
                f"Insufficient feature matches: found {len(matches)} < 10 required"
            )

        # Extract matched keypoint coordinates (Task 2.1)
        src_pts = np.float32(
            [baseline_keypoints[m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [current_keypoints[m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Estimate transformation matrix (Task 2.2, AC-1.3.2)
        if self.use_affine_model:
            # Use 6-DOF affine transformation (more stable for pure translations)
            # estimateAffinePartial2D returns 2x3 matrix [R|t] and inliers mask
            # This model is appropriate for translation + rotation + scale (no perspective)
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

            # Handle affine estimation failure (Task 2.3, AC-1.3.6)
            if M is None:
                raise RuntimeError(
                    "Affine estimation failed: insufficient or degenerate matches"
                )

            # Extract translation vector from affine matrix (Task 2.4, AC-1.3.3)
            # M = [r11  r12  tx]
            #     [r21  r22  ty]
            # NOTE: This only extracts translation. Rotation/scale are ignored (MVP constraint).
            tx = M[0, 2]
            ty = M[1, 2]
        else:
            # Use 8-DOF homography (original implementation)
            # Using simple method (method=0) initially as per constraints
            H, mask = cv2.findHomography(src_pts, dst_pts, method=0)

            # Handle homography estimation failure (Task 2.3, AC-1.3.6)
            if H is None:
                raise RuntimeError(
                    "Homography estimation failed: singular matrix or degenerate configuration"
                )

            # Extract translation vector from homography matrix (Task 2.4, AC-1.3.3)
            # H = [h11  h12  tx]
            #     [h21  h22  ty]
            #     [h31  h32  1 ]
            # NOTE: This only extracts translation. Rotation/scale/shear are ignored (MVP constraint).
            tx = H[0, 2]
            ty = H[1, 2]

        # Store translation components (for Mode A enhanced metrics)
        self.last_tx = float(tx)
        self.last_ty = float(ty)

        # Store matches and transformation matrix for Mode B visualization
        self.last_baseline_keypoints = baseline_keypoints
        self.last_current_keypoints = current_keypoints
        self.last_matches = [
            (
                (baseline_keypoints[m.queryIdx].pt[0], baseline_keypoints[m.queryIdx].pt[1]),
                (current_keypoints[m.trainIdx].pt[0], current_keypoints[m.trainIdx].pt[1])
            )
            for m in matches
        ]
        self.last_mask = mask
        # Store homography matrix (H for homography, M for affine)
        self.last_homography = H if not self.use_affine_model else M

        # Calculate translation displacement magnitude (Task 2.5, AC-1.3.3)
        displacement = np.sqrt(tx**2 + ty**2)

        # Round to 2 decimal places (Task 2.6, AC-1.3.3)
        displacement = round(displacement, 2)

        # Calculate confidence score from inlier ratio (Task 3.3, AC-1.3.5)
        # mask contains 1 for inliers, 0 for outliers
        inliers = np.sum(mask)
        confidence = inliers / len(matches)

        # Ensure confidence is in range [0.0, 1.0] (Task 3.4, AC-1.3.5)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        confidence = round(confidence, 2)  # Round for consistent precision

        # Compare displacement to threshold (Task 3.1, AC-1.3.4)
        # Set moved flag (Task 3.2, AC-1.3.4)
        moved = bool(displacement >= self.threshold_pixels)

        # Return tuple: (moved, displacement, confidence) (Task 3.5, AC-1.3.4, AC-1.3.5)
        return (moved, displacement, confidence)

    def _validate_features(
        self, features: Tuple[List, np.ndarray], param_name: str
    ) -> None:
        """Validate feature tuple format.

        Args:
            features: Feature tuple to validate
            param_name: Parameter name for error messages

        Raises:
            ValueError: If features have invalid format
        """
        if not isinstance(features, tuple) or len(features) != 2:
            raise ValueError(
                f"Invalid {param_name} format: expected tuple of (keypoints, descriptors), "
                f"got {type(features)}"
            )

        keypoints, descriptors = features

        if not isinstance(keypoints, list):
            raise ValueError(
                f"Invalid {param_name} format: keypoints must be list, "
                f"got {type(keypoints)}"
            )

        if not isinstance(descriptors, np.ndarray):
            raise ValueError(
                f"Invalid {param_name} format: descriptors must be numpy.ndarray, "
                f"got {type(descriptors)}"
            )

        if len(keypoints) == 0:
            raise ValueError(
                f"Invalid {param_name} format: keypoints list is empty"
            )

        if descriptors.size == 0:
            raise ValueError(
                f"Invalid {param_name} format: descriptors array is empty"
            )

    def get_last_matches(self) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], np.ndarray]:
        """Get match correspondences from last detect_movement() call.

        Returns:
            Tuple of (matches, mask) where:
                - matches: List of ((x0, y0), (x1, y1)) baseline-to-current point pairs
                - mask: Inlier mask array (1=inlier, 0=outlier)

        Note:
            This method is designed for Mode B visualization. Call detect_movement()
            before calling this method to ensure fresh data.
        """
        return (self.last_matches, self.last_mask if self.last_mask is not None else np.array([]))

    def get_last_homography(self) -> Optional[np.ndarray]:
        """Get transformation matrix from last detect_movement() call.

        Returns:
            Homography matrix (3x3) or Affine matrix (2x3), or None if not computed

        Note:
            Returns homography (8-DOF) when use_affine_model=False, or
            affine matrix (6-DOF) when use_affine_model=True.
        """
        return self.last_homography
