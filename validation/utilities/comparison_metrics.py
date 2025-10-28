"""
Comparison Metrics - Displacement Comparison Calculations

Provides pure calculation functions for comparing ChArUco 6-DOF pose estimation
against cam-shift detector 2D displacement measurements.
"""

import numpy as np
from typing import List


def calculate_displacement_difference(d1: float, d2: float) -> float:
    """Calculate L2 norm (Euclidean distance) between two displacements.

    Args:
        d1: First displacement value in pixels
        d2: Second displacement value in pixels

    Returns:
        Absolute difference (L2 norm): |d1 - d2|

    Example:
        >>> calculate_displacement_difference(18.5, 15.2)
        3.3
        >>> calculate_displacement_difference(10.0, 10.0)
        0.0
    """
    return abs(d1 - d2)


def calculate_threshold(width: int, height: int, percent: float = 0.03) -> float:
    """Calculate comparison threshold as percentage of minimum image dimension.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        percent: Threshold percentage (default: 0.03 = 3%)

    Returns:
        Threshold value in pixels

    Example:
        >>> calculate_threshold(640, 480, 0.03)
        14.4  # 3% of min(640, 480) = 3% of 480 = 14.4
    """
    min_dimension = min(width, height)
    return min_dimension * percent


def classify_agreement(diff: float, threshold: float) -> str:
    """Classify detector agreement based on displacement difference.

    Args:
        diff: Displacement difference in pixels (from calculate_displacement_difference)
        threshold: Threshold value in pixels (from calculate_threshold)

    Returns:
        "GREEN" if diff <= threshold (detectors agree)
        "RED" if diff > threshold (detectors disagree)

    Example:
        >>> classify_agreement(10.0, 14.4)
        'GREEN'
        >>> classify_agreement(20.0, 14.4)
        'RED'
        >>> classify_agreement(14.4, 14.4)  # Boundary case
        'GREEN'
    """
    return "GREEN" if diff <= threshold else "RED"


def calculate_charuco_displacement_2d(
    tvec_current: np.ndarray,
    tvec_baseline: np.ndarray,
    K: np.ndarray,
    z_distance_m: float = 1.15
) -> float:
    """Convert 3D ChArUco translation to 2D pixel displacement.

    Projects 3D translation vectors onto 2D image plane using camera intrinsics.

    Approach:
    - Calculate 3D delta: delta = tvec_current - tvec_baseline
    - Extract focal lengths: fx = K[0,0], fy = K[1,1]
    - Project to 2D: dx_px = (delta_x * fx) / z_distance_m
    - Project to 2D: dy_px = (delta_y * fy) / z_distance_m
    - Calculate magnitude: displacement = sqrt(dx_px^2 + dy_px^2)

    Args:
        tvec_current: Current translation vector [x, y, z] in meters (3D)
        tvec_baseline: Baseline translation vector [x, y, z] in meters (3D)
        K: Camera matrix (3x3) with focal lengths at K[0,0] and K[1,1]
        z_distance_m: Camera-to-board distance in meters (default: 1.15m)

    Returns:
        2D pixel displacement magnitude

    Example:
        >>> K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        >>> tvec_baseline = np.array([[0.0], [0.0], [1.15]])
        >>> tvec_current = np.array([[0.01], [0.02], [1.15]])  # 1cm X, 2cm Y displacement
        >>> calculate_charuco_displacement_2d(tvec_current, tvec_baseline, K, 1.15)
        15.52...  # sqrt((0.01*800/1.15)^2 + (0.02*800/1.15)^2)
    """
    # Extract focal lengths from camera matrix
    fx = K[0, 0]  # Focal length X (pixels)
    fy = K[1, 1]  # Focal length Y (pixels)

    # Calculate 3D displacement (delta in meters)
    delta_x_m = tvec_current[0, 0] - tvec_baseline[0, 0]
    delta_y_m = tvec_current[1, 0] - tvec_baseline[1, 0]

    # Project 3D displacement to 2D image plane
    # Formula: dx_px = (delta_x * fx) / z_distance
    dx_px = (delta_x_m * fx) / z_distance_m
    dy_px = (delta_y_m * fy) / z_distance_m

    # Calculate 2D displacement magnitude
    displacement_2d_px = np.sqrt(dx_px**2 + dy_px**2)

    return float(displacement_2d_px)


def calculate_mse(charuco_list: List[float], camshift_list: List[float]) -> float:
    """Calculate Mean Squared Error between ChArUco and cam-shift displacement sequences.

    Args:
        charuco_list: List of ChArUco displacement values in pixels
        camshift_list: List of cam-shift displacement values in pixels

    Returns:
        Mean Squared Error (MSE) across the sequence

    Raises:
        ValueError: If lists have different lengths or are empty

    Example:
        >>> calculate_mse([10.0, 12.0], [11.0, 13.0])
        1.0  # MSE = ((10-11)^2 + (12-13)^2) / 2 = (1 + 1) / 2 = 1.0
        >>> calculate_mse([5.0, 10.0, 15.0], [5.0, 10.0, 15.0])
        0.0  # Perfect agreement
    """
    if len(charuco_list) != len(camshift_list):
        raise ValueError(
            f"Lists must have same length: charuco={len(charuco_list)}, "
            f"camshift={len(camshift_list)}"
        )

    if len(charuco_list) == 0:
        raise ValueError("Cannot calculate MSE for empty lists")

    # Calculate squared differences
    squared_errors = [(c - cs)**2 for c, cs in zip(charuco_list, camshift_list)]

    # Return mean
    return float(np.mean(squared_errors))
