#!/usr/bin/env python3
"""
Stage 1 Test Data Generation Script

Generates synthetic test dataset for Stage 1 validation by applying
known transformations (2px, 5px, 10px shifts in 8 directions) to
baseline images from sample_images/ directory.

Usage:
    python validation/generate_stage1_data.py

Outputs:
    - validation/stage1_data/baseline/ - Original baseline images
    - validation/stage1_data/shifted_2px/ - 2px transformed images
    - validation/stage1_data/shifted_5px/ - 5px transformed images
    - validation/stage1_data/shifted_10px/ - 10px transformed images
    - validation/stage1_data/ground_truth.json - Complete ground truth labels

Test Coverage:
    - 50 baseline images (of_jerusalem: 23, carmit: 17, gad: 10)
    - 3 shift magnitudes (2px, 5px, 10px)
    - 8 directions per shift (right, left, up, down, diagonal_ur, diagonal_ul, diagonal_dr, diagonal_dl)
    - Total: 50 baselines + 1200 transformed images (50 * 3 * 8) = 1250 images
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import glob

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.harnesses.stage1_test_harness import Stage1TestHarness


def main():
    """Generate Stage 1 test dataset"""

    print("=" * 80)
    print("Stage 1 Test Data Generation")
    print("=" * 80)
    print()

    # Configuration
    validation_dir = "validation"

    # Collect all baseline images from sample_images/
    baseline_images = []
    sample_images_dir = Path("sample_images")

    print("📸 Collecting baseline images...")
    for dataset in ["of_jerusalem", "carmit", "gad"]:
        dataset_path = sample_images_dir / dataset
        if dataset_path.exists():
            images = sorted(dataset_path.glob("*.jpg"))
            baseline_images.extend([str(img) for img in images])
            print(f"   {dataset}: {len(images)} images")

    total_baselines = len(baseline_images)
    print(f"\n✅ Total baseline images: {total_baselines}")

    if total_baselines == 0:
        print("❌ ERROR: No baseline images found in sample_images/")
        return 1

    # Initialize test harness
    print(f"\n🔧 Initializing Stage1TestHarness...")
    harness = Stage1TestHarness(validation_dir=validation_dir)

    # Define transformations
    shifts = [2, 5, 10]  # pixels
    directions = [
        'right', 'left', 'up', 'down',
        'diagonal_ur', 'diagonal_ul', 'diagonal_dr', 'diagonal_dl'
    ]

    # Calculate expected output
    num_transformed_per_baseline = len(shifts) * len(directions)
    total_transformed = total_baselines * num_transformed_per_baseline
    total_images = total_baselines + total_transformed

    print(f"\n📊 Test Dataset Specification:")
    print(f"   Baseline images: {total_baselines}")
    print(f"   Shift magnitudes: {shifts}")
    print(f"   Directions per shift: {len(directions)}")
    print(f"   Transformed images: {total_transformed} ({total_baselines} × {num_transformed_per_baseline})")
    print(f"   Total dataset size: {total_images} images")
    print()

    # Generate test dataset
    print("🔄 Generating synthetic test dataset...")
    print(f"   This will take approximately {total_images // 20} seconds...")
    print()

    start_time = datetime.now()

    try:
        generated_images, ground_truth_labels = harness.generate_test_dataset(
            baseline_images=baseline_images,
            shifts=shifts,
            directions=directions,
            include_baseline=True
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print(f"\n✅ Dataset generation complete!")
        print(f"   Generated images: {len(generated_images)}")
        print(f"   Ground truth labels: {len(ground_truth_labels)}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Processing rate: {len(generated_images) / duration:.1f} images/second")

        # Verify directory structure
        print(f"\n📁 Verifying directory structure...")
        stage1_data_dir = Path(validation_dir) / "stage1_data"

        subdirs = {
            "baseline": stage1_data_dir / "baseline",
            "shifted_2px": stage1_data_dir / "shifted_2px",
            "shifted_5px": stage1_data_dir / "shifted_5px",
            "shifted_10px": stage1_data_dir / "shifted_10px"
        }

        for name, path in subdirs.items():
            if path.exists():
                count = len(list(path.glob("*.jpg")))
                print(f"   {name}/: {count} images")
            else:
                print(f"   ❌ {name}/: directory not found!")

        # Verify ground truth JSON
        gt_file = stage1_data_dir / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
            print(f"   ground_truth.json: {len(gt_data)} labels")

            # Show sample label
            if gt_data:
                print(f"\n📝 Sample ground truth label:")
                sample = gt_data[0]
                print(f"   Image ID: {sample['image_id']}")
                print(f"   Baseline: {sample['baseline_image']}")
                print(f"   Transformation: {sample['transformation']['type']} "
                      f"{sample['transformation'].get('magnitude_px', 0)}px "
                      f"{sample['transformation'].get('direction', 'N/A')}")
                print(f"   Expected status: {sample['expected_status']}")
                print(f"   Expected displacement: {sample['expected_displacement_range']}")
        else:
            print(f"   ❌ ground_truth.json: file not found!")

        # Summary statistics
        print(f"\n📊 Dataset Statistics:")

        # Count by status
        valid_count = sum(1 for label in ground_truth_labels if label.expected_status == "VALID")
        invalid_count = sum(1 for label in ground_truth_labels if label.expected_status == "INVALID")

        print(f"   VALID (baseline): {valid_count}")
        print(f"   INVALID (shifted): {invalid_count}")
        print(f"   Total: {len(ground_truth_labels)}")

        # Count by shift magnitude
        print(f"\n   By shift magnitude:")
        for shift in shifts:
            shift_count = sum(1 for label in ground_truth_labels
                            if label.transformation.get('magnitude_px') == float(shift))
            print(f"      {shift}px: {shift_count} images")

        # Count by direction
        print(f"\n   By direction:")
        for direction in directions:
            dir_count = sum(1 for label in ground_truth_labels
                          if label.transformation.get('direction') == direction)
            print(f"      {direction}: {dir_count} images")

        print(f"\n✅ Stage 1 test data generation complete!")
        print(f"\n📂 Output directory: {stage1_data_dir}")
        print(f"   Ready for Stage 1 validation execution (Task 3)")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR during dataset generation:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
