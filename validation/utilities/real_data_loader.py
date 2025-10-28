"""
Real Data Loader for Stage 3 Validation

Loads real DAF imagery from sample_images/ directory with metadata extraction
and image validation for systematic testing against ground truth annotations.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np


@dataclass
class ImageMetadata:
    """Metadata for a single validation image.
    
    Attributes:
        image_path: Absolute path to image file
        site_id: DAF site identifier ('of_jerusalem', 'carmit', 'gad')
        timestamp: Image capture/modification timestamp
        has_shift: Ground truth annotation (True if camera shift detected)
    """
    image_path: Path
    site_id: str
    timestamp: datetime
    has_shift: Optional[bool] = None  # Populated from ground truth


class RealDataLoader:
    """Load real DAF imagery with metadata for validation testing.
    
    Scans sample_images/ directory, extracts metadata from directory structure,
    and provides validated image loading with ground truth integration.
    """
    
    def __init__(self, sample_images_root: Optional[Path] = None, 
                 ground_truth_path: Optional[Path] = None):
        """Initialize data loader.
        
        Args:
            sample_images_root: Root directory containing site subdirectories.
                              Defaults to {project_root}/sample_images/
            ground_truth_path: Path to ground_truth.json file.
                             Defaults to {project_root}/validation/ground_truth/ground_truth.json
        """
        if sample_images_root is None:
            # Default: assume validation/ is sibling to sample_images/ (go up to project root)
            self.sample_images_root = Path(__file__).parent.parent.parent / "sample_images"
        else:
            self.sample_images_root = Path(sample_images_root)
            
        if ground_truth_path is None:
            # Use the ground_truth symlink at validation/ground_truth/
            self.ground_truth_path = Path(__file__).parent.parent / "ground_truth" / "ground_truth.json"
        else:
            self.ground_truth_path = Path(ground_truth_path)
        
        self._ground_truth_data: Optional[dict] = None
        
    def load_dataset(self) -> List[ImageMetadata]:
        """Scan sample_images/ directory and return metadata for all images.
        
        Walks through site subdirectories (of_jerusalem, carmit, gad) and collects
        all .jpg images with their metadata.
        
        Returns:
            List of ImageMetadata objects for all discovered images, sorted by path.
            
        Raises:
            FileNotFoundError: If sample_images/ directory doesn't exist.
        """
        if not self.sample_images_root.exists():
            raise FileNotFoundError(
                f"Sample images directory not found: {self.sample_images_root}"
            )
        
        images: List[ImageMetadata] = []
        
        # Expected site directories
        site_dirs = ["of_jerusalem", "carmit", "gad"]
        
        for site_id in site_dirs:
            site_path = self.sample_images_root / site_id
            if not site_path.exists():
                continue
                
            # Find all .jpg files in site directory
            for image_path in sorted(site_path.glob("*.jpg")):
                # Extract timestamp from file modification time
                # (UUID filenames don't contain timestamps, so we use mtime)
                timestamp = datetime.fromtimestamp(image_path.stat().st_mtime)
                
                # Create metadata object
                metadata = ImageMetadata(
                    image_path=image_path,
                    site_id=site_id,
                    timestamp=timestamp,
                    has_shift=None  # Will be populated from ground truth if available
                )
                
                images.append(metadata)
        
        # Load ground truth annotations if available
        if self.ground_truth_path.exists():
            self._load_ground_truth_annotations(images)
        
        return images
    
    def load_image(self, path: Path) -> np.ndarray:
        """Load and validate single image using OpenCV.
        
        Args:
            path: Absolute or relative path to image file
            
        Returns:
            Image as numpy array in RGB color space (OpenCV loads as BGR, converted to RGB)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image file is invalid or corrupted
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        # Load image with OpenCV
        image = cv2.imread(str(path))
        
        if image is None:
            raise ValueError(f"Failed to load image (corrupted or invalid format): {path}")
        
        # Convert BGR to RGB for consistency with detection system
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Validate dimensions (images should be non-zero)
        if image_rgb.shape[0] == 0 or image_rgb.shape[1] == 0:
            raise ValueError(f"Image has invalid dimensions: {image_rgb.shape}")
        
        return image_rgb
    
    def _load_ground_truth_annotations(self, images: List[ImageMetadata]) -> None:
        """Populate has_shift field from ground truth annotations.
        
        Args:
            images: List of ImageMetadata objects to annotate
        """
        try:
            with open(self.ground_truth_path, 'r') as f:
                self._ground_truth_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Ground truth not yet available or invalid - skip annotation
            return
        
        # Create lookup dictionary: relative_path -> has_camera_shift
        ground_truth_lookup = {}
        for annotation in self._ground_truth_data.get("images", []):
            image_path_str = annotation.get("image_path", "")
            has_shift = annotation.get("has_camera_shift", None)
            ground_truth_lookup[image_path_str] = has_shift
        
        # Annotate images with ground truth
        for metadata in images:
            # Convert absolute path to relative path from project root
            try:
                relative_path = str(metadata.image_path.relative_to(
                    self.sample_images_root.parent
                ))
            except ValueError:
                # Path is not relative to expected root
                relative_path = str(metadata.image_path)
            
            # Look up ground truth annotation
            if relative_path in ground_truth_lookup:
                metadata.has_shift = ground_truth_lookup[relative_path]
    
    def get_site_distribution(self, images: List[ImageMetadata]) -> dict:
        """Calculate distribution of images per site.
        
        Args:
            images: List of ImageMetadata objects
            
        Returns:
            Dictionary mapping site_id -> image count
        """
        distribution = {}
        for metadata in images:
            site_id = metadata.site_id
            distribution[site_id] = distribution.get(site_id, 0) + 1
        return distribution
