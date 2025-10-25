"""Feature Extractor for Camera Movement Detection

This module provides functionality to extract ORB (Oriented FAST and Rotated BRIEF)
features from camera images using binary masks, focusing detection on static regions
while ignoring dynamic elements like water and bubbles.

The module stores baseline features for comparison with current frames to detect
camera movement through feature matching and homography estimation.
"""

from typing import List, Tuple

import cv2
import numpy as np


class FeatureExtractor:
    """Extracts and manages ORB features for camera movement detection.

    This class uses OpenCV's ORB feature detector with binary mask support to
    extract keypoints and descriptors from static regions only. It maintains
    baseline features for movement detection and validates minimum feature counts
    to ensure reliable detection.

    Attributes:
        min_features_required (int): Minimum features needed for reliable detection
        orb (cv2.ORB): OpenCV ORB feature detector instance
        baseline_features (tuple | None): Stored baseline (keypoints, descriptors) or None
    """

    def __init__(self, min_features_required: int = 50, orb_features: int = 1000) -> None:
        """Initialize ORB feature detector.

        Args:
            min_features_required: Minimum features required during baseline capture.
                                  Default is 50 features for reliable detection.
            orb_features: Maximum number of ORB features to extract per image.
                         Default is 1000. Increased from 500 to improve vertical
                         movement detection accuracy.

        Raises:
            ValueError: If min_features_required is not a positive integer
            ValueError: If orb_features is not a positive integer
        """
        if not isinstance(min_features_required, int) or min_features_required <= 0:
            raise ValueError(
                f"min_features_required must be a positive integer, got {min_features_required}"
            )

        if not isinstance(orb_features, int) or orb_features <= 0:
            raise ValueError(
                f"orb_features must be a positive integer, got {orb_features}"
            )

        self.min_features_required = min_features_required
        self.orb_features = orb_features
        self.orb = cv2.ORB_create(nfeatures=orb_features)
        self.baseline_features: Tuple[List[cv2.KeyPoint], np.ndarray] | None = None

    def extract_features(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Extract ORB features from image using binary mask.

        Features are detected only in regions where mask value is 255 (static regions).
        Areas where mask is 0 (dynamic regions) are ignored.

        Args:
            image: Input image as NumPy array (H×W×3, uint8, BGR format)
            mask: Binary mask as NumPy array (H×W, uint8) where 255=detect, 0=ignore

        Returns:
            Tuple of (keypoints, descriptors):
                - keypoints: List of cv2.KeyPoint objects
                - descriptors: NumPy array (n_features×32, uint8) or None if no features

        Raises:
            ValueError: If image or mask format is invalid or dimensions don't match
        """
        # Validate image format
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Invalid image format: expected numpy.ndarray, got {type(image).__name__}"
            )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Invalid image format: expected shape (H, W, 3), got {image.shape}"
            )

        if image.dtype != np.uint8:
            raise ValueError(
                f"Invalid image format: expected dtype uint8, got {image.dtype}"
            )

        # Validate mask format
        if not isinstance(mask, np.ndarray):
            raise ValueError(
                f"Invalid mask format: expected numpy.ndarray, got {type(mask).__name__}"
            )

        if mask.ndim != 2:
            raise ValueError(
                f"Invalid mask format: expected shape (H, W), got {mask.shape}"
            )

        if mask.dtype != np.uint8:
            raise ValueError(
                f"Invalid mask format: expected dtype uint8, got {mask.dtype}"
            )

        # Validate dimensions match
        img_height, img_width = image.shape[:2]
        mask_height, mask_width = mask.shape

        if mask_height != img_height or mask_width != img_width:
            raise ValueError(
                f"Mask dimensions ({mask_height}, {mask_width}) must match "
                f"image dimensions ({img_height}, {img_width})"
            )

        # Extract features using ORB with mask
        keypoints, descriptors = self.orb.detectAndCompute(image, mask=mask)

        # Convert keypoints tuple to list for consistency
        keypoints = list(keypoints) if keypoints else []

        return keypoints, descriptors

    def set_baseline(self, image: np.ndarray, mask: np.ndarray) -> None:
        """Capture and store baseline features for movement detection.

        Extracts features from the baseline image and validates that sufficient
        features were detected for reliable movement detection.

        Args:
            image: Baseline reference image (H×W×3, uint8, BGR format)
            mask: Binary mask (H×W, uint8) where 255=detect, 0=ignore

        Raises:
            ValueError: If image/mask format invalid, dimensions mismatch, or
                       insufficient features detected (< min_features_required)
        """
        # Extract features (this validates inputs)
        keypoints, descriptors = self.extract_features(image, mask)

        # Validate feature count
        feature_count = len(keypoints) if keypoints else 0

        if feature_count < self.min_features_required:
            raise ValueError(
                f"Insufficient features detected: {feature_count} < {self.min_features_required}. "
                "Try different ROI or larger region."
            )

        # Store baseline
        self.baseline_features = (keypoints, descriptors)

    def get_baseline(self) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Get stored baseline features.

        Returns:
            Tuple of (keypoints, descriptors) stored during set_baseline()

        Raises:
            RuntimeError: If baseline features not set (call set_baseline() first)
        """
        if self.baseline_features is None:
            raise RuntimeError(
                "Baseline features not set. Call set_baseline() first."
            )

        return self.baseline_features
