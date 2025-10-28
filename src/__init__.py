"""
Camera Shift Detector

Computer vision system for detecting camera movement in time-series imagery
using feature matching and transformation analysis.

Main Classes:
    CameraMovementDetector: Primary API for camera shift detection

Public API:
    - CameraMovementDetector: Main detector class
    - __version__: Package version string
"""

from .camera_movement_detector import CameraMovementDetector

__version__ = "0.1.0"
__all__ = ["CameraMovementDetector", "__version__"]
