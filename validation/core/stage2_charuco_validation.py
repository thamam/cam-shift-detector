#!/usr/bin/env python3
"""
Stage 2 ChArUco Ground Truth Validation

Validates the camera movement detector against high-precision ChArUco ground truth
measurements. This provides validation with real camera movements (not synthetic
transformations) using 6-DOF pose estimation as ground truth.

Usage:
    python validation/stage2_charuco_validation.py

Inputs:
    - session_001/poses.csv - ChArUco ground truth (203 frames, 158 detected)
    - session_001/frames/*.jpg - Annotated video frames
    - config_session_001.json - ROI configuration for session
    - camera.yaml - Camera intrinsics for 3D-to-2D projection

Outputs:
    - validation/stage2_results.json - Complete validation results with metrics
    - validation/stage2_results_report.txt - Human-readable report
    - Console output with analysis and recommendations

Validation Approach:
    1. Load ChArUco ground truth poses (3D displacement in meters)
    2. Convert 3D displacement to 2D pixel displacement using camera projection
    3. Run detector on session frames using static ROI (walls/furniture)
    4. Compare detector status vs ground truth displacement
    5. Calculate TPR, FPR, accuracy metrics
    6. Test with multiple thresholds (1.5px from Stage 1, 16.8px from handoff)

Success Criteria (from handoff):
    - Detect movements ≥3% of average frame dimension
    - For 640×480: avg = 560px → 3% = 16.8 pixels
    - Goal: 100% detection rate (zero false negatives)

Key Insight:
    - ChArUco provides ground truth measurement (3D pose)
    - Detector uses static objects in ROI for detection (walls/furniture)
    - ChArUco board itself is NOT used for detection, only for ground truth
"""

import sys
import json
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_movement_detector import CameraMovementDetector


@dataclass
class CharucoGroundTruth:
    """ChArUco ground truth for a single frame"""
    frame_idx: int
    timestamp_ns: int
    detected: bool  # ChArUco detection flag

    # 3D displacement from first detection (meters)
    base_tx_m: float
    base_ty_m: float
    base_tz_m: float

    # Orientation change from first detection (degrees)
    base_roll_deg: float
    base_pitch_deg: float
    base_yaw_deg: float

    inliers_count: int  # Quality metric

    # Calculated fields
    displacement_3d_mm: float = 0.0  # 3D Euclidean distance in mm
    displacement_2d_px: float = 0.0  # Projected 2D displacement in pixels


@dataclass
class Stage2ValidationResult:
    """Validation result for a single frame"""
    frame_idx: int
    frame_path: str

    # Ground truth
    gt_detected: bool
    gt_displacement_3d_mm: float
    gt_displacement_2d_px: float

    # Detector output
    detector_status: str  # "VALID" or "INVALID"
    detector_displacement_px: float
    detector_confidence: float

    # Classification
    classification: str  # "TP", "TN", "FP", "FN"
    ground_truth_label: str  # "MOVEMENT" or "NO_MOVEMENT"

    # Notes
    notes: str = ""


@dataclass
class Stage2Metrics:
    """Comprehensive Stage 2 validation metrics"""
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Derived metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    tpr: float  # True Positive Rate
    fpr: float  # False Positive Rate

    # Counts
    total_frames: int
    frames_with_ground_truth: int
    frames_analyzed: int

    # Threshold used
    threshold_px: float
    threshold_description: str


class CharucoValidator:
    """Stage 2 validator using ChArUco ground truth"""

    def __init__(
        self,
        session_dir: str,
        config_path: str,
        camera_yaml_path: str = "camera.yaml"
    ):
        """
        Initialize validator

        Args:
            session_dir: Path to session directory (e.g., "session_001")
            config_path: Path to ROI config (e.g., "config_session_001.json")
            camera_yaml_path: Path to camera calibration file
        """
        self.session_dir = Path(session_dir)
        self.poses_csv = self.session_dir / "poses.csv"
        self.frames_dir = self.session_dir / "frames"
        self.config_path = config_path

        # Load camera intrinsics
        self.camera_matrix, self.image_size = self._load_camera_intrinsics(camera_yaml_path)

        # Will be set during validation
        self.detector = None
        self.ground_truth: List[CharucoGroundTruth] = []

    def _load_camera_intrinsics(self, yaml_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load camera intrinsics from OpenCV YAML calibration file"""
        # OpenCV YAML format requires special handling
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)

        # Extract camera matrix
        camera_matrix = fs.getNode('camera_matrix').mat()

        # Extract image size
        width = int(fs.getNode('image_width').real())
        height = int(fs.getNode('image_height').real())

        fs.release()

        return camera_matrix, (width, height)

    def load_ground_truth(self) -> None:
        """Load ChArUco ground truth poses from CSV"""
        print("📋 Loading ChArUco ground truth...")

        if not self.poses_csv.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {self.poses_csv}")

        # Load CSV
        df = pd.read_csv(self.poses_csv)

        print(f"   Total frames in CSV: {len(df)}")
        print(f"   Frames with ChArUco detection: {df['detected'].sum()}")

        # Convert to ground truth objects
        self.ground_truth = []
        for _, row in df.iterrows():
            # Calculate 3D displacement magnitude (mm)
            dx = row['base_tx_m'] * 1000  # meters to mm
            dy = row['base_ty_m'] * 1000
            dz = row['base_tz_m'] * 1000
            displacement_3d_mm = np.sqrt(dx**2 + dy**2 + dz**2)

            gt = CharucoGroundTruth(
                frame_idx=int(row['frame_idx']),
                timestamp_ns=int(row['timestamp_ns']),
                detected=bool(row['detected']),
                base_tx_m=row['base_tx_m'],
                base_ty_m=row['base_ty_m'],
                base_tz_m=row['base_tz_m'],
                base_roll_deg=row['base_roll_deg'],
                base_pitch_deg=row['base_pitch_deg'],
                base_yaw_deg=row['base_yaw_deg'],
                inliers_count=int(row['inliers_count']),
                displacement_3d_mm=displacement_3d_mm,
            )

            self.ground_truth.append(gt)

        print(f"   ✓ Loaded {len(self.ground_truth)} ground truth entries")

    def convert_3d_to_2d_displacement(self) -> None:
        """
        Convert 3D displacement (meters) to approximate 2D displacement (pixels)

        Approach:
        - Use camera matrix focal lengths to estimate pixel displacement
        - Assume displacement primarily in X-Y plane (base_tx_m, base_ty_m)
        - Approximate: dx_px ≈ dx_m * fx / z_m, dy_px ≈ dy_m * fy / z_m
        - Use average Z distance from camera for conversion

        Note: This is an approximation since:
        - Detector measures 2D image displacement
        - ChArUco measures 3D world displacement
        - Actual projection depends on depth (Z coordinate)
        """
        print("\n🔄 Converting 3D displacement to 2D pixel displacement...")

        # Calculate average Z distance (camera to board) for detected frames
        z_distances = [
            gt.base_tz_m for gt in self.ground_truth
            if gt.detected and not np.isnan(gt.base_tz_m)
        ]

        if not z_distances:
            raise ValueError("No valid Z distances in ground truth")

        # Use baseline Z (first detected frame typically has base_tz_m = 0)
        # Get typical Z distance from first few detected frames
        detected_frames = [gt for gt in self.ground_truth if gt.detected]
        if len(detected_frames) < 2:
            raise ValueError("Need at least 2 detected frames")

        # Estimate typical camera-to-board distance (use frame with small displacement)
        # First detected frame has displacement=0, subsequent frames show actual distance
        typical_z_m = 1.15  # From handoff: typical ~1.15m distance

        # Camera intrinsics
        fx = self.camera_matrix[0, 0]  # Focal length X (pixels)
        fy = self.camera_matrix[1, 1]  # Focal length Y (pixels)

        print(f"   Camera focal length: fx={fx:.1f}px, fy={fy:.1f}px")
        print(f"   Typical Z distance: {typical_z_m:.2f}m")

        # Convert each ground truth entry
        for gt in self.ground_truth:
            if gt.detected:
                # Convert meters to pixels using projection
                # dx_px = (dx_m * fx) / z_m
                dx_px = (gt.base_tx_m * fx) / typical_z_m
                dy_px = (gt.base_ty_m * fy) / typical_z_m

                # 2D displacement magnitude
                gt.displacement_2d_px = np.sqrt(dx_px**2 + dy_px**2)
            else:
                # No ChArUco detection → cannot calculate displacement
                gt.displacement_2d_px = np.nan

        # Report statistics
        valid_2d = [gt.displacement_2d_px for gt in self.ground_truth if not np.isnan(gt.displacement_2d_px)]
        if valid_2d:
            print(f"   2D displacement range: {min(valid_2d):.2f} - {max(valid_2d):.2f} px")
            print(f"   2D displacement mean: {np.mean(valid_2d):.2f} px")
            print(f"   2D displacement median: {np.median(valid_2d):.2f} px")

    def run_detector_validation(self, threshold_px: float) -> List[Stage2ValidationResult]:
        """
        Run detector on all session frames and compare against ground truth

        Args:
            threshold_px: Displacement threshold in pixels for classification

        Returns:
            List of validation results for each frame
        """
        print(f"\n🔍 Running detector validation (threshold={threshold_px}px)...")

        # Initialize detector
        print(f"   Loading detector config: {self.config_path}")
        self.detector = CameraMovementDetector(self.config_path)

        # Find baseline frame (first detected ChArUco frame)
        baseline_gt = None
        for gt in self.ground_truth:
            if gt.detected:
                baseline_gt = gt
                break

        if baseline_gt is None:
            raise ValueError("No baseline frame found (need at least one detected ChArUco frame)")

        # Load baseline image
        baseline_path = self.frames_dir / f"frame_{baseline_gt.frame_idx:06d}.jpg"
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline frame not found: {baseline_path}")

        baseline_image = cv2.imread(str(baseline_path))
        print(f"   Setting baseline: frame {baseline_gt.frame_idx}")
        self.detector.set_baseline(baseline_image)

        # Process all frames
        results = []
        total_frames = len(self.ground_truth)

        for i, gt in enumerate(self.ground_truth):
            # Progress indicator
            if (i + 1) % 25 == 0 or (i + 1) == total_frames:
                print(f"   Processing frame {i + 1}/{total_frames}...", end='\r')

            # Load frame
            frame_path = self.frames_dir / f"frame_{gt.frame_idx:06d}.jpg"
            if not frame_path.exists():
                # Frame missing → skip
                continue

            frame_image = cv2.imread(str(frame_path))

            # Run detector
            try:
                detection = self.detector.process_frame(frame_image, frame_id=f"frame_{gt.frame_idx}")
                detector_status = detection['status']
                detector_displacement = detection['translation_displacement']
                detector_confidence = detection['confidence']
            except Exception as e:
                # Detector failed → treat as INVALID with inf displacement
                detector_status = "INVALID"
                detector_displacement = float('inf')
                detector_confidence = 0.0

            # Ground truth classification
            if not gt.detected or np.isnan(gt.displacement_2d_px):
                # No ChArUco detection → skip (cannot establish ground truth)
                continue

            # Classify ground truth: MOVEMENT if displacement >= threshold
            gt_has_movement = gt.displacement_2d_px >= threshold_px
            gt_label = "MOVEMENT" if gt_has_movement else "NO_MOVEMENT"

            # Detector classification
            detector_detected_movement = (detector_status == "INVALID")

            # Confusion matrix classification
            if gt_has_movement and detector_detected_movement:
                classification = "TP"  # True Positive
            elif not gt_has_movement and not detector_detected_movement:
                classification = "TN"  # True Negative
            elif not gt_has_movement and detector_detected_movement:
                classification = "FP"  # False Positive
            else:  # gt_has_movement and not detector_detected_movement
                classification = "FN"  # False Negative

            # Create result
            result = Stage2ValidationResult(
                frame_idx=gt.frame_idx,
                frame_path=str(frame_path),
                gt_detected=gt.detected,
                gt_displacement_3d_mm=gt.displacement_3d_mm,
                gt_displacement_2d_px=gt.displacement_2d_px,
                detector_status=detector_status,
                detector_displacement_px=detector_displacement,
                detector_confidence=detector_confidence,
                classification=classification,
                ground_truth_label=gt_label,
                notes=""
            )

            results.append(result)

        print(f"\n   ✓ Processed {len(results)} frames with ground truth")

        return results

    def calculate_metrics(
        self,
        results: List[Stage2ValidationResult],
        threshold_px: float,
        threshold_description: str
    ) -> Stage2Metrics:
        """Calculate comprehensive validation metrics"""

        # Count classifications
        tp = sum(1 for r in results if r.classification == "TP")
        tn = sum(1 for r in results if r.classification == "TN")
        fp = sum(1 for r in results if r.classification == "FP")
        fn = sum(1 for r in results if r.classification == "FN")

        total = len(results)

        # Calculate derived metrics
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # TPR = TP / (TP + FN) = recall
        tpr = recall

        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return Stage2Metrics(
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            tpr=tpr,
            fpr=fpr,
            total_frames=len(self.ground_truth),
            frames_with_ground_truth=len([gt for gt in self.ground_truth if gt.detected]),
            frames_analyzed=total,
            threshold_px=threshold_px,
            threshold_description=threshold_description
        )

    def generate_report(
        self,
        metrics_list: List[Tuple[Stage2Metrics, str]],
        results_by_threshold: Dict[str, List[Stage2ValidationResult]]
    ) -> Dict:
        """Generate comprehensive validation report"""

        report = {
            "session": str(self.session_dir),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_info": {
                "total_frames": len(self.ground_truth),
                "frames_with_charuco_detection": len([gt for gt in self.ground_truth if gt.detected]),
                "charuco_detection_rate": sum(gt.detected for gt in self.ground_truth) / len(self.ground_truth),
            },
            "thresholds_tested": [],
            "recommendations": {},
        }

        # Add metrics for each threshold
        for metrics, description in metrics_list:
            threshold_data = {
                "threshold_px": metrics.threshold_px,
                "description": metrics.threshold_description,
                "metrics": asdict(metrics),
            }
            report["thresholds_tested"].append(threshold_data)

        # Add detailed results for primary threshold (16.8px)
        primary_threshold = 16.8
        if f"{primary_threshold:.1f}px" in results_by_threshold:
            primary_results = results_by_threshold[f"{primary_threshold:.1f}px"]
            report["detailed_results_primary"] = [asdict(r) for r in primary_results]

        return report


def main():
    """Execute Stage 2 validation"""

    print("=" * 80)
    print("Stage 2 ChArUco Ground Truth Validation")
    print("=" * 80)
    print()

    # Configuration
    session_dir = "session_001"
    config_path = "config_session_001.json"
    camera_yaml = "camera.yaml"

    # Initialize validator
    validator = CharucoValidator(session_dir, config_path, camera_yaml)

    # Load ground truth
    validator.load_ground_truth()

    # Convert 3D to 2D displacement
    validator.convert_3d_to_2d_displacement()

    # Test multiple thresholds
    thresholds = [
        (1.5, "Stage 1 corrected threshold"),
        (16.8, "Handoff success criterion (3% of 560px avg dimension)"),
        (10.0, "Moderate threshold for comparison"),
    ]

    all_metrics = []
    results_by_threshold = {}

    for threshold_px, description in thresholds:
        print(f"\n{'=' * 80}")
        print(f"Testing threshold: {threshold_px}px - {description}")
        print('=' * 80)

        # Run validation
        results = validator.run_detector_validation(threshold_px)

        # Calculate metrics
        metrics = validator.calculate_metrics(results, threshold_px, description)

        # Store
        all_metrics.append((metrics, description))
        results_by_threshold[f"{threshold_px:.1f}px"] = results

        # Print summary
        print("\n📊 Metrics Summary:")
        print(f"   Accuracy:  {metrics.accuracy:.2%}")
        print(f"   Precision: {metrics.precision:.2%}")
        print(f"   Recall:    {metrics.recall:.2%} (TPR)")
        print(f"   F1-Score:  {metrics.f1_score:.2%}")
        print(f"   FPR:       {metrics.fpr:.2%}")
        print(f"\n   Confusion Matrix:")
        print(f"   TP: {metrics.true_positives:3d}  |  FN: {metrics.false_negatives:3d}")
        print(f"   FP: {metrics.false_positives:3d}  |  TN: {metrics.true_negatives:3d}")

    # Generate comprehensive report
    print("\n📝 Generating validation report...")
    report = validator.generate_report(all_metrics, results_by_threshold)

    # Save JSON report
    output_path = Path("validation/stage2_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"   ✓ Saved JSON report: {output_path}")

    # Generate text report
    text_report_path = Path("validation/stage2_results_report.txt")
    with open(text_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Stage 2 ChArUco Ground Truth Validation - Results Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Session: {session_dir}\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")

        f.write("Session Statistics:\n")
        f.write(f"  Total frames: {report['session_info']['total_frames']}\n")
        f.write(f"  ChArUco detected: {report['session_info']['frames_with_charuco_detection']}\n")
        f.write(f"  Detection rate: {report['session_info']['charuco_detection_rate']:.1%}\n\n")

        for threshold_data in report['thresholds_tested']:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"Threshold: {threshold_data['threshold_px']}px\n")
            f.write(f"Description: {threshold_data['description']}\n")
            f.write("-" * 80 + "\n\n")

            m = threshold_data['metrics']
            f.write(f"Accuracy:   {m['accuracy']:.2%}\n")
            f.write(f"Precision:  {m['precision']:.2%}\n")
            f.write(f"Recall:     {m['recall']:.2%} (True Positive Rate)\n")
            f.write(f"F1-Score:   {m['f1_score']:.2%}\n")
            f.write(f"FPR:        {m['fpr']:.2%} (False Positive Rate)\n\n")

            f.write("Confusion Matrix:\n")
            f.write(f"  TP: {m['true_positives']:3d}  |  FN: {m['false_negatives']:3d}\n")
            f.write(f"  FP: {m['false_positives']:3d}  |  TN: {m['true_negatives']:3d}\n\n")

            f.write(f"Frames analyzed: {m['frames_analyzed']} / {m['frames_with_ground_truth']} with ground truth\n")

    print(f"   ✓ Saved text report: {text_report_path}")

    # Print recommendations
    print("\n" + "=" * 80)
    print("🎯 Recommendations & Analysis")
    print("=" * 80)

    # Find best performing threshold
    best_metrics = max(all_metrics, key=lambda x: x[0].accuracy)
    print(f"\n✓ Best performing threshold: {best_metrics[0].threshold_px}px ({best_metrics[1]})")
    print(f"  Accuracy: {best_metrics[0].accuracy:.2%}")
    print(f"  Recall (TPR): {best_metrics[0].recall:.2%}")
    print(f"  FPR: {best_metrics[0].fpr:.2%}")

    # Check if 100% detection rate achieved
    print("\n📋 Stage 2 Success Criteria Check:")
    print("   Goal: 100% detection rate (zero false negatives) for intentional movements")

    for metrics, description in all_metrics:
        fn_count = metrics.false_negatives
        recall_pct = metrics.recall * 100

        status = "✓ PASS" if fn_count == 0 else "✗ FAIL"
        print(f"   {status} - {metrics.threshold_px}px: {fn_count} false negatives, {recall_pct:.1f}% recall")

    print("\n🔍 Key Insights:")
    print("   1. ChArUco provides 3D ground truth, detector measures 2D displacement")
    print("   2. Detector uses static ROI (walls/furniture), not ChArUco board")
    print("   3. 3D-to-2D conversion is approximate (depends on depth)")
    print("   4. Different thresholds reveal detector sensitivity trade-offs")

    print("\n✓ Stage 2 validation complete!")
    print(f"   Results saved to: {output_path}")


if __name__ == "__main__":
    main()
