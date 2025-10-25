#!/usr/bin/env python3
"""
Baseline Image 00018752 Investigation Script

Investigates the complete detection failure (0% detection rate) for baseline image
00018752-d1e8-44fb-9cba-5107c18eb386.jpg which caused 29 false negatives in
sudden onset pattern testing.

Root Cause Hypotheses:
1. Insufficient ORB features extracted from baseline image
2. Poor feature quality preventing reliable matching
3. Image characteristics incompatible with detector initialization

Investigation Steps:
1. Load baseline image and extract features
2. Compare feature count against min_features_required threshold
3. Analyze feature distribution and quality
4. Test detector initialization and basic transformation detection
5. Compare with successful baseline images

Usage:
    python validation/investigate_baseline_00018752.py
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.camera_movement_detector import CameraMovementDetector


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def analyze_orb_features(image: np.ndarray, orb_detector) -> dict:
    """Analyze ORB features extracted from an image"""
    keypoints, descriptors = orb_detector.detectAndCompute(image, None)

    analysis = {
        "feature_count": len(keypoints) if keypoints else 0,
        "has_descriptors": descriptors is not None,
        "descriptor_shape": descriptors.shape if descriptors is not None else None
    }

    if keypoints:
        # Analyze feature distribution
        x_coords = [kp.pt[0] for kp in keypoints]
        y_coords = [kp.pt[1] for kp in keypoints]
        responses = [kp.response for kp in keypoints]

        analysis.update({
            "spatial_distribution": {
                "x_range": (min(x_coords), max(x_coords)),
                "y_range": (min(y_coords), max(y_coords)),
                "x_std": np.std(x_coords),
                "y_std": np.std(y_coords)
            },
            "feature_quality": {
                "mean_response": np.mean(responses),
                "min_response": min(responses),
                "max_response": max(responses),
                "std_response": np.std(responses)
            }
        })

    return analysis


def test_detector_initialization(image_path: Path, config_path: Path) -> dict:
    """Test detector initialization with problematic baseline"""
    detector = CameraMovementDetector(str(config_path))

    # Load baseline image
    baseline_image = cv2.imread(str(image_path))
    if baseline_image is None:
        return {"error": "Failed to load baseline image"}

    # Set baseline and capture state
    try:
        detector.set_baseline(baseline_image)
        init_success = True
        error_msg = None
    except Exception as e:
        init_success = False
        error_msg = str(e)

    init_result = {
        "baseline_set": detector.baseline_set,
        "init_success": init_success,
        "error": error_msg if not init_success else None
    }

    return init_result


def test_simple_transformation(image_path: Path, config_path: Path) -> dict:
    """Test detection with a simple 5px shift (should be detected)"""
    detector = CameraMovementDetector(str(config_path))

    # Load baseline
    baseline_image = cv2.imread(str(image_path))
    if baseline_image is None:
        return {"error": "Failed to load baseline image"}

    detector.set_baseline(baseline_image)

    # Create 5px right shift (well above 1.5px threshold)
    h, w = baseline_image.shape[:2]
    M = np.float32([[1, 0, 5.0], [0, 1, 0]])
    shifted_image = cv2.warpAffine(baseline_image, M, (w, h))

    # Process shifted frame
    result = detector.process_frame(shifted_image, "test_frame")

    return {
        "expected_status": "INVALID",
        "actual_status": result["status"],
        "displacement": result["translation_displacement"],
        "confidence": result["confidence"],
        "detection_correct": result["status"] == "INVALID"
    }


def compare_with_successful_baseline(problematic_path: Path, successful_path: Path, config: dict) -> dict:
    """Compare problematic baseline with a successful one"""

    # Analyze both images
    problematic_img = cv2.imread(str(problematic_path))
    successful_img = cv2.imread(str(successful_path))

    if problematic_img is None or successful_img is None:
        return {"error": "Failed to load comparison images"}

    # Create ORB detector with config settings
    orb_features = config.get("orb_features", 500)
    orb = cv2.ORB_create(nfeatures=orb_features)

    problematic_analysis = analyze_orb_features(problematic_img, orb)
    successful_analysis = analyze_orb_features(successful_img, orb)

    return {
        "problematic": {
            "path": str(problematic_path.name),
            "shape": problematic_img.shape,
            "features": problematic_analysis
        },
        "successful": {
            "path": str(successful_path.name),
            "shape": successful_img.shape,
            "features": successful_analysis
        },
        "comparison": {
            "feature_count_ratio": (
                problematic_analysis["feature_count"] / successful_analysis["feature_count"]
                if successful_analysis["feature_count"] > 0 else 0
            ),
            "meets_minimum": problematic_analysis["feature_count"] >= config.get("min_features_required", 50)
        }
    }


def main():
    """Main investigation execution"""

    print("=" * 80)
    print("Baseline Image 00018752 Investigation")
    print("=" * 80)
    print()

    # Paths
    config_path = Path("config.json")
    problematic_image = Path("sample_images/gad/00018752-d1e8-44fb-9cba-5107c18eb386.jpg")
    successful_image = Path("sample_images/of_jerusalem/00000001-b1e8-44fb-9cba-5107c18eb386.jpg")

    # Load config
    config = load_config(config_path)
    print(f"📋 Configuration:")
    print(f"   Threshold: {config.get('threshold_pixels', 'N/A')}px")
    print(f"   ORB Features: {config.get('orb_features', 'N/A')}")
    print(f"   Min Features Required: {config.get('min_features_required', 'N/A')}")
    print()

    # 1. Feature Extraction Analysis
    print("🔍 Step 1: Feature Extraction Analysis")
    print()

    if not problematic_image.exists():
        print(f"❌ ERROR: Problematic image not found: {problematic_image}")
        return 1

    img = cv2.imread(str(problematic_image))
    if img is None:
        print(f"❌ ERROR: Failed to load image: {problematic_image}")
        return 1

    print(f"   Image: {problematic_image.name}")
    print(f"   Shape: {img.shape}")

    orb = cv2.ORB_create(nfeatures=config.get("orb_features", 500))
    feature_analysis = analyze_orb_features(img, orb)

    print(f"   Feature Count: {feature_analysis['feature_count']}")
    print(f"   Min Required: {config.get('min_features_required', 50)}")

    if feature_analysis['feature_count'] < config.get('min_features_required', 50):
        print(f"   ⚠️  INSUFFICIENT FEATURES: {feature_analysis['feature_count']} < 50")
    else:
        print(f"   ✅ Feature count meets minimum requirement")

    if feature_analysis.get('spatial_distribution'):
        dist = feature_analysis['spatial_distribution']
        print(f"   Spatial Distribution:")
        print(f"      X range: {dist['x_range']}")
        print(f"      Y range: {dist['y_range']}")
        print(f"      X std: {dist['x_std']:.2f}")
        print(f"      Y std: {dist['y_std']:.2f}")

    if feature_analysis.get('feature_quality'):
        qual = feature_analysis['feature_quality']
        print(f"   Feature Quality:")
        print(f"      Mean response: {qual['mean_response']:.4f}")
        print(f"      Response range: [{qual['min_response']:.4f}, {qual['max_response']:.4f}]")

    print()

    # 2. Detector Initialization Test
    print("🔧 Step 2: Detector Initialization Test")
    print()

    init_result = test_detector_initialization(problematic_image, config_path)

    if init_result.get('error'):
        print(f"   ❌ ERROR: {init_result['error']}")
    else:
        print(f"   Baseline Set: {'✅' if init_result['baseline_set'] else '❌'}")
        print(f"   Initialization Success: {'✅' if init_result['init_success'] else '❌'}")

    print()

    # 3. Simple Transformation Test
    print("🎯 Step 3: Simple 5px Shift Detection Test")
    print()

    transform_result = test_simple_transformation(problematic_image, config_path)

    if "error" in transform_result:
        print(f"   ❌ ERROR: {transform_result['error']}")
    else:
        print(f"   Expected Status: {transform_result['expected_status']}")
        print(f"   Actual Status: {transform_result['actual_status']}")
        print(f"   Displacement: {transform_result['displacement']:.2f}px")
        print(f"   Confidence: {transform_result['confidence']:.4f}")

        if transform_result['detection_correct']:
            print(f"   ✅ Detection CORRECT (5px shift detected)")
        else:
            print(f"   ❌ Detection FAILED (5px shift NOT detected)")

    print()

    # 4. Comparison with Successful Baseline
    print("📊 Step 4: Comparison with Successful Baseline")
    print()

    if successful_image.exists():
        comparison = compare_with_successful_baseline(
            problematic_image, successful_image, config
        )

        if "error" in comparison:
            print(f"   ❌ ERROR: {comparison['error']}")
        else:
            prob = comparison['problematic']
            succ = comparison['successful']
            comp = comparison['comparison']

            print(f"   Problematic: {prob['path']}")
            print(f"      Features: {prob['features']['feature_count']}")
            print(f"      Shape: {prob['shape']}")

            print(f"   Successful: {succ['path']}")
            print(f"      Features: {succ['features']['feature_count']}")
            print(f"      Shape: {succ['shape']}")

            print(f"   Comparison:")
            print(f"      Feature Ratio: {comp['feature_count_ratio']:.2%}")
            print(f"      Meets Minimum: {'✅' if comp['meets_minimum'] else '❌'}")
    else:
        print(f"   ⚠️  Successful baseline not found for comparison")

    print()

    # Summary and Recommendations
    print("=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Determine primary issue
    if feature_analysis['feature_count'] < config.get('min_features_required', 50):
        print("🔴 PRIMARY ISSUE: Insufficient ORB Features")
        print(f"   Feature count ({feature_analysis['feature_count']}) below minimum threshold (50)")
        print()
        print("📋 RECOMMENDED ACTIONS:")
        print("   1. Implement baseline validation in detector initialization")
        print("   2. Reject baselines with feature_count < min_features_required")
        print("   3. Add pre-validation step in dataset generation")
        print("   4. Consider excluding low-feature images from test dataset")
        print()
        print("   Code change required in CameraMovementDetector.set_baseline():")
        print("   ```python")
        print("   if len(self.baseline_keypoints) < self.config['min_features_required']:")
        print("       raise ValueError(f'Insufficient features: {len(self.baseline_keypoints)}')")
        print("   ```")
    elif not transform_result.get('detection_correct', False):
        print("🔴 PRIMARY ISSUE: Detection Logic Failure")
        print("   Image has sufficient features but 5px shift not detected")
        print()
        print("📋 RECOMMENDED ACTIONS:")
        print("   1. Debug feature matching logic in process_frame()")
        print("   2. Verify RANSAC homography estimation parameters")
        print("   3. Check displacement calculation from transformation matrix")
        print("   4. Investigate if image characteristics affect matching")
    else:
        print("🟢 BASELINE APPEARS FUNCTIONAL")
        print("   Features extracted successfully")
        print("   Simple transformation detected correctly")
        print()
        print("📋 NEXT INVESTIGATION:")
        print("   Issue may be specific to temporal sequence processing")
        print("   Analyze sequence-level state management")
        print("   Review sudden onset pattern frame-by-frame results")

    print()
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
