"""Static Region Manager for Camera Movement Detection

This module provides functionality to load ROI (Region of Interest) coordinates
from configuration files and generate binary masks for static regions, avoiding
dynamic elements like water and bubbles.

The mask-based approach allows for flexible region definitions (rectangles,
polygons, arbitrary masks) and integrates cleanly with OpenCV feature detection.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np


class StaticRegionManager:
    """Manages static region definition and binary mask generation for camera movement detection.

    This class loads ROI coordinates from a JSON configuration file and provides
    methods to generate binary masks for the defined static region. The mask-based
    approach allows OpenCV feature detectors to focus on static areas while ignoring
    dynamic elements. It validates both configuration schema and image dimensions
    to ensure reliable operation.

    Attributes:
        config_path (str): Path to the configuration JSON file
        roi (Dict[str, int]): ROI coordinates with keys: x, y, width, height
    """

    def __init__(self, config_path: str) -> None:
        """Load and validate ROI configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file containing ROI coordinates

        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config schema is invalid or ROI values are not positive integers
        """
        self.config_path = config_path
        self.roi = self._load_and_validate_config()

    def _load_and_validate_config(self) -> Dict[str, int]:
        """Load configuration file and validate schema.

        Returns:
            Dictionary with validated ROI coordinates (x, y, width, height)

        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If JSON is invalid or schema validation fails
        """
        config_file = Path(self.config_path)

        # Check if file exists
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load JSON
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        # Validate schema - must have 'roi' key
        if 'roi' not in config:
            raise ValueError("Invalid config schema: missing required field 'roi'")

        roi = config['roi']

        # Validate required fields
        required_fields = ['x', 'y', 'width', 'height']
        for field in required_fields:
            if field not in roi:
                raise ValueError(f"Invalid config schema: missing required field 'roi.{field}'")

        # Validate all values are integers
        for field in required_fields:
            if not isinstance(roi[field], int):
                raise ValueError(f"Invalid config schema: 'roi.{field}' must be an integer")

        # Validate x and y are non-negative (can be 0 for top-left corner)
        if roi['x'] < 0:
            raise ValueError(f"Invalid config schema: 'roi.x' must be non-negative (got {roi['x']})")
        if roi['y'] < 0:
            raise ValueError(f"Invalid config schema: 'roi.y' must be non-negative (got {roi['y']})")

        # Validate width and height are positive (must be > 0)
        if roi['width'] <= 0:
            raise ValueError(f"Invalid config schema: 'roi.width' must be positive (got {roi['width']})")
        if roi['height'] <= 0:
            raise ValueError(f"Invalid config schema: 'roi.height' must be positive (got {roi['height']})")

        return {
            'x': roi['x'],
            'y': roi['y'],
            'width': roi['width'],
            'height': roi['height']
        }

    def get_static_mask(self, image_shape: tuple) -> np.ndarray:
        """Get binary mask for static region.

        Generates a binary mask where pixels inside the static ROI are 255
        and pixels outside (dynamic areas) are 0. This mask can be used directly
        with OpenCV feature detection methods like ORB.detectAndCompute().

        Args:
            image_shape: Tuple of (height, width) representing image dimensions

        Returns:
            Binary mask as NumPy array (H×W, uint8):
                - 255 = static region (features will be detected here)
                - 0 = dynamic region (features will be ignored)

        Raises:
            ValueError: If image_shape is invalid or ROI exceeds image bounds

        Example:
            >>> manager = StaticRegionManager("config.json")
            >>> mask = manager.get_static_mask((480, 640))  # 480 height, 640 width
            >>> # Use with ORB: orb.detectAndCompute(image, mask=mask)
        """
        # Validate image_shape is a tuple
        if not isinstance(image_shape, tuple):
            raise ValueError(f"Invalid image_shape: expected tuple, got {type(image_shape).__name__}")

        # Validate image_shape has exactly 2 elements
        if len(image_shape) != 2:
            raise ValueError(f"Invalid image_shape: expected (height, width), got {len(image_shape)} elements")

        # Validate both dimensions are positive integers
        img_height, img_width = image_shape
        if not isinstance(img_height, int) or not isinstance(img_width, int):
            raise ValueError(f"Invalid image_shape: dimensions must be integers, got ({type(img_height).__name__}, {type(img_width).__name__})")

        if img_height <= 0 or img_width <= 0:
            raise ValueError(f"Invalid image_shape: dimensions must be positive, got ({img_height}, {img_width})")

        # Extract ROI coordinates
        x = self.roi['x']
        y = self.roi['y']
        width = self.roi['width']
        height = self.roi['height']

        # Validate ROI is within image bounds
        if x + width > img_width or y + height > img_height:
            raise ValueError(
                f"ROI exceeds image bounds: ROI({x}, {y}, {width}, {height}) "
                f"vs Image({img_height}, {img_width})"
            )

        # Generate binary mask using the rectangular ROI
        mask = self._generate_rectangular_mask(image_shape)
        return mask

    def _generate_rectangular_mask(self, image_shape: tuple) -> np.ndarray:
        """Generate binary mask from rectangular ROI coordinates.

        Creates a mask where pixels inside the ROI rectangle are 255 (static)
        and pixels outside are 0 (dynamic). This is the core mask generation
        logic that converts ROI coordinates into a binary mask.

        Args:
            image_shape: Tuple of (height, width)

        Returns:
            Binary mask (H×W, uint8): 255 inside ROI, 0 outside

        Note:
            This method assumes ROI coordinates have already been validated
            by get_static_mask(). It does not perform bounds checking.
        """
        img_height, img_width = image_shape

        # Create zero-filled mask (all dynamic by default)
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        # Extract ROI coordinates
        x = self.roi['x']
        y = self.roi['y']
        width = self.roi['width']
        height = self.roi['height']

        # Set ROI region to 255 (static)
        mask[y:y+height, x:x+width] = 255

        return mask
