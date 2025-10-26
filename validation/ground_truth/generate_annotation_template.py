#!/usr/bin/env python3
"""
Generate ground_truth.json template with all 50 images pre-populated.

This script creates an initial ground_truth.json file with all image paths
filled in, ready for manual annotation. The annotator only needs to set
has_camera_shift, confidence, and optional notes for each image.
"""

import json
from datetime import date
from pathlib import Path


def generate_template():
    """Generate ground_truth.json template with all images."""
    
    # Path configuration
    project_root = Path(__file__).parent.parent.parent
    sample_images_root = project_root / "sample_images"
    output_path = Path(__file__).parent / "ground_truth.json"
    
    # Collect all images
    images = []
    site_dirs = ["of_jerusalem", "carmit", "gad"]
    
    for site_id in site_dirs:
        site_path = sample_images_root / site_id
        if not site_path.exists():
            print(f"Warning: {site_id} directory not found")
            continue
        
        for image_path in sorted(site_path.glob("*.jpg")):
            # Convert to relative path from project root
            relative_path = str(image_path.relative_to(project_root))
            
            # Create annotation entry (to be filled manually)
            entry = {
                "image_path": relative_path,
                "site_id": site_id,
                "has_camera_shift": None,  # TO BE ANNOTATED
                "confidence": "high",      # TO BE ANNOTATED
                "notes": ""                # OPTIONAL
            }
            images.append(entry)
    
    # Create ground truth structure
    ground_truth = {
        "version": "1.0",
        "annotator": "Tomer",
        "annotation_date": str(date.today()),
        "images": images
    }
    
    # Write JSON with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"✓ Generated ground_truth.json template")
    print(f"✓ Location: {output_path}")
    print(f"✓ Total images: {len(images)}")
    print(f"✓ Distribution:")
    
    # Show distribution
    distribution = {}
    for entry in images:
        site = entry["site_id"]
        distribution[site] = distribution.get(site, 0) + 1
    
    for site_id, count in sorted(distribution.items()):
        print(f"  - {site_id}: {count} images")
    
    print(f"\n✓ Next step: Manually annotate each image")
    print(f"  - Set has_camera_shift: true or false")
    print(f"  - Set confidence: high, medium, or low")
    print(f"  - Optionally add notes")
    print(f"\nSee ANNOTATION_GUIDELINES.md for criteria")


if __name__ == "__main__":
    generate_template()
