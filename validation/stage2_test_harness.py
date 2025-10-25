#!/usr/bin/env python3
"""
Stage 2 Test Harness - Temporal Sequence Generation and Validation

Generates realistic temporal sequences of camera movements to validate
detection capability across time. Tests 6 movement patterns:
1. Gradual Onset (slow drift)
2. Sudden Onset (impact/bump)
3. Progressive Displacement (incremental creep)
4. Oscillation (vibration)
5. Recovery Sequence (return to baseline)
6. Multi-Axis Movement (diagonal drift)

Validates AC-1.9.2: 0% false negatives / 100% detection rate
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import math


@dataclass
class FrameAnnotation:
    """Ground truth annotation for a single frame in a sequence"""
    frame_number: int
    frame_id: str
    timestamp: float
    cumulative_shift_px: float
    shift_from_previous_px: float
    expected_status: str  # "VALID" or "INVALID"
    expected_displacement_range: Tuple[float, float]
    movement_vector: Dict[str, float]  # {"x": dx, "y": dy}
    is_critical_transition: bool
    transition_notes: str = ""


@dataclass
class SequenceMetadata:
    """Metadata for a temporal movement sequence"""
    sequence_id: str
    pattern_type: str
    baseline_image: str
    total_frames: int
    frame_rate: float
    duration_seconds: float
    movement_description: str
    expected_detections: int
    expected_first_invalid_frame: Optional[int]
    critical_frames: List[int]
    direction: str = ""
    max_displacement_px: float = 0.0


@dataclass
class SequenceDetectionMetrics:
    """Detection metrics for a temporal sequence"""
    sequence_id: str
    pattern_type: str
    total_frames: int
    expected_invalid_frames: int
    detected_invalid_frames: int
    false_negatives: int
    false_positives: int
    detection_rate: float
    first_invalid_detected: Optional[int]
    expected_first_invalid: Optional[int]
    detection_latency_frames: int
    critical_frame_accuracy: float
    critical_frames_tested: int
    critical_frames_correct: int


class Stage2TestHarness:
    """
    Test harness for Stage 2 validation - temporal sequence generation and validation.

    Generates synthetic temporal sequences simulating realistic camera movement
    patterns over time. Each sequence contains multiple frames showing progressive
    or sudden movement dynamics.
    """

    def __init__(self, threshold_px: float = 1.5):
        """
        Initialize Stage 2 test harness.

        Args:
            threshold_px: Detection threshold in pixels (from config.json)
        """
        self.threshold_px = threshold_px

    def generate_gradual_onset_trajectory(
        self,
        total_frames: int,
        max_displacement: float,
        direction: str
    ) -> List[Tuple[float, float]]:
        """
        Generate linear drift trajectory (0 → max_displacement over time).

        Simulates: Camera mounting slowly loosening, thermal expansion.

        Args:
            total_frames: Number of frames in sequence
            max_displacement: Final displacement in pixels
            direction: Movement direction (right, up, diagonal_ur, diagonal_dr)

        Returns:
            List of (dx, dy) tuples for each frame
        """
        # Direction vectors
        direction_map = {
            'right': (1.0, 0.0),
            'up': (0.0, -1.0),
            'diagonal_ur': (1.0, -1.0),  # Up-right (normalize to √2)
            'diagonal_dr': (1.0, 1.0),   # Down-right (normalize to √2)
        }

        if direction not in direction_map:
            raise ValueError(f"Invalid direction: {direction}")

        unit_x, unit_y = direction_map[direction]

        # Normalize diagonal directions
        if 'diagonal' in direction:
            magnitude = math.sqrt(unit_x**2 + unit_y**2)
            unit_x /= magnitude
            unit_y /= magnitude

        # Linear trajectory: displacement increases linearly
        trajectory = []
        for frame in range(total_frames):
            progress = frame / (total_frames - 1)  # 0.0 → 1.0
            displacement = max_displacement * progress

            dx = displacement * unit_x
            dy = displacement * unit_y
            trajectory.append((dx, dy))

        return trajectory

    def generate_sudden_onset_trajectory(
        self,
        total_frames: int,
        displacement: float,
        direction: str,
        onset_frame: int = 1
    ) -> List[Tuple[float, float]]:
        """
        Generate instant shift trajectory (0 → displacement at onset_frame, sustained).

        Simulates: Camera mount impact, sudden external force.

        Args:
            total_frames: Number of frames in sequence
            displacement: Sustained displacement in pixels
            direction: Movement direction
            onset_frame: Frame number where movement occurs

        Returns:
            List of (dx, dy) tuples for each frame
        """
        direction_map = {
            'right': (1.0, 0.0),
            'up': (0.0, -1.0),
            'diagonal_ur': (1.0, -1.0),
            'diagonal_dr': (1.0, 1.0),
        }

        unit_x, unit_y = direction_map[direction]

        if 'diagonal' in direction:
            magnitude = math.sqrt(unit_x**2 + unit_y**2)
            unit_x /= magnitude
            unit_y /= magnitude

        dx = displacement * unit_x
        dy = displacement * unit_y

        # Frame 0: baseline (0, 0)
        # Frames >= onset_frame: sustained displacement
        trajectory = []
        for frame in range(total_frames):
            if frame < onset_frame:
                trajectory.append((0.0, 0.0))
            else:
                trajectory.append((dx, dy))

        return trajectory

    def generate_progressive_trajectory(
        self,
        total_frames: int,
        step_size: float,
        direction: str
    ) -> List[Tuple[float, float]]:
        """
        Generate incremental step trajectory (step_size per N frames).

        Simulates: Incremental mounting degradation, step-wise drift.

        Args:
            total_frames: Number of frames in sequence
            step_size: Displacement increment per step (pixels)
            direction: Movement direction

        Returns:
            List of (dx, dy) tuples for each frame
        """
        direction_map = {
            'right': (1.0, 0.0),
            'up': (0.0, -1.0),
            'diagonal_ur': (1.0, -1.0),
            'diagonal_dr': (1.0, 1.0),
        }

        unit_x, unit_y = direction_map[direction]

        if 'diagonal' in direction:
            magnitude = math.sqrt(unit_x**2 + unit_y**2)
            unit_x /= magnitude
            unit_y /= magnitude

        # Incremental steps every 4 frames
        frames_per_step = 4
        trajectory = []
        for frame in range(total_frames):
            step_count = frame // frames_per_step
            displacement = step_count * step_size

            dx = displacement * unit_x
            dy = displacement * unit_y
            trajectory.append((dx, dy))

        return trajectory

    def generate_oscillation_trajectory(
        self,
        total_frames: int,
        amplitude: float,
        axis: str = 'horizontal'
    ) -> List[Tuple[float, float]]:
        """
        Generate sinusoidal oscillation trajectory (vibration).

        Simulates: Wind-induced vibration, mechanical resonance.

        Args:
            total_frames: Number of frames in sequence
            amplitude: Oscillation amplitude in pixels
            axis: 'horizontal' or 'vertical'

        Returns:
            List of (dx, dy) tuples for each frame
        """
        trajectory = []
        period = total_frames  # One complete cycle over sequence

        for frame in range(total_frames):
            # Sinusoidal displacement: amplitude * sin(2π * frame / period)
            angle = 2 * math.pi * frame / period
            displacement = amplitude * math.sin(angle)

            if axis == 'horizontal':
                trajectory.append((displacement, 0.0))
            else:  # vertical
                trajectory.append((0.0, displacement))

        return trajectory

    def generate_recovery_trajectory(
        self,
        total_frames: int,
        initial_displacement: float,
        direction: str
    ) -> List[Tuple[float, float]]:
        """
        Generate recovery trajectory (displacement → 0 linearly).

        Simulates: Automatic correction mechanism, manual adjustment.

        Args:
            total_frames: Number of frames in sequence
            initial_displacement: Starting displacement in pixels
            direction: Movement direction

        Returns:
            List of (dx, dy) tuples for each frame
        """
        direction_map = {
            'right': (1.0, 0.0),
            'up': (0.0, -1.0),
            'diagonal_ur': (1.0, -1.0),
            'diagonal_dr': (1.0, 1.0),
        }

        unit_x, unit_y = direction_map[direction]

        if 'diagonal' in direction:
            magnitude = math.sqrt(unit_x**2 + unit_y**2)
            unit_x /= magnitude
            unit_y /= magnitude

        # Linear recovery: displacement decreases linearly to 0
        trajectory = []
        for frame in range(total_frames):
            progress = 1.0 - (frame / (total_frames - 1))  # 1.0 → 0.0
            displacement = initial_displacement * progress

            dx = displacement * unit_x
            dy = displacement * unit_y
            trajectory.append((dx, dy))

        return trajectory

    def generate_multi_axis_trajectory(
        self,
        total_frames: int,
        x_rate: float,
        y_rate: float
    ) -> List[Tuple[float, float]]:
        """
        Generate independent X and Y drift trajectory.

        Simulates: Non-uniform mounting loosening, asymmetric forces.

        Args:
            total_frames: Number of frames in sequence
            x_rate: X-axis drift rate (pixels per frame)
            y_rate: Y-axis drift rate (pixels per frame)

        Returns:
            List of (dx, dy) tuples for each frame
        """
        trajectory = []
        for frame in range(total_frames):
            dx = frame * x_rate
            dy = frame * y_rate
            trajectory.append((dx, dy))

        return trajectory

    def apply_temporal_transformation(
        self,
        baseline_image: np.ndarray,
        trajectory: List[Tuple[float, float]],
        output_dir: Path,
        sequence_id: str
    ) -> List[str]:
        """
        Generate frame sequence by applying trajectory transformations.

        Args:
            baseline_image: Original baseline image
            trajectory: List of (dx, dy) displacements per frame
            output_dir: Directory to save frame images
            sequence_id: Unique sequence identifier

        Returns:
            List of frame file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []

        h, w = baseline_image.shape[:2]

        for frame_num, (dx, dy) in enumerate(trajectory):
            # Create transformation matrix
            M = np.float32([[1, 0, dx], [0, 1, dy]])

            # Apply transformation
            transformed = cv2.warpAffine(
                baseline_image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Save frame
            frame_filename = f"frame_{frame_num:03d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), transformed)
            frame_paths.append(str(frame_path))

        return frame_paths

    def create_frame_annotations(
        self,
        sequence_id: str,
        trajectory: List[Tuple[float, float]],
        pattern_type: str,
        frame_rate: float = 1.0
    ) -> List[FrameAnnotation]:
        """
        Create per-frame ground truth annotations with critical transition markers.

        Args:
            sequence_id: Unique sequence identifier
            trajectory: List of (dx, dy) displacements per frame
            pattern_type: Movement pattern type
            frame_rate: Frames per second

        Returns:
            List of FrameAnnotation objects
        """
        annotations = []

        for frame_num, (dx, dy) in enumerate(trajectory):
            # Calculate displacement magnitude
            # For recovery and progressive patterns, measure from baseline (Frame 0)
            # For other patterns, measure absolute applied transformation
            if pattern_type in ['recovery', 'progressive']:
                if frame_num == 0:
                    # Baseline frame - no displacement from itself
                    cumulative_shift = 0.0
                else:
                    # Calculate displacement from baseline (Frame 0)
                    baseline_dx, baseline_dy = trajectory[0]
                    cumulative_shift = math.sqrt((baseline_dx - dx)**2 + (baseline_dy - dy)**2)
            else:
                # For other patterns (gradual_onset, sudden_onset, oscillation, multi_axis)
                # Use the applied transformation as the displacement
                cumulative_shift = math.sqrt(dx**2 + dy**2)

            # Calculate shift from previous frame
            if frame_num == 0:
                shift_from_previous = 0.0
            else:
                prev_dx, prev_dy = trajectory[frame_num - 1]

                # Calculate previous shift using same pattern-aware logic
                if pattern_type in ['recovery', 'progressive']:
                    if frame_num == 1:
                        prev_shift = 0.0  # Frame 0 is baseline
                    else:
                        baseline_dx, baseline_dy = trajectory[0]
                        prev_shift = math.sqrt((baseline_dx - prev_dx)**2 + (baseline_dy - prev_dy)**2)
                else:
                    prev_shift = math.sqrt(prev_dx**2 + prev_dy**2)

                shift_from_previous = cumulative_shift - prev_shift

            # Determine expected status
            expected_status = "INVALID" if cumulative_shift >= self.threshold_px else "VALID"

            # Expected displacement range (80%-150% tolerance)
            min_disp = cumulative_shift * 0.8
            max_disp = cumulative_shift * 1.5

            # Identify critical transitions
            is_critical = False
            transition_notes = ""

            if frame_num > 0:
                prev_dx, prev_dy = trajectory[frame_num - 1]

                # Calculate previous shift using same pattern-aware logic
                if pattern_type in ['recovery', 'progressive']:
                    if frame_num == 1:
                        prev_shift = 0.0  # Frame 0 is baseline
                    else:
                        baseline_dx, baseline_dy = trajectory[0]
                        prev_shift = math.sqrt((baseline_dx - prev_dx)**2 + (baseline_dy - prev_dy)**2)
                else:
                    prev_shift = math.sqrt(prev_dx**2 + prev_dy**2)

                prev_status = "INVALID" if prev_shift >= self.threshold_px else "VALID"

                # Check for status transition
                if prev_status != expected_status:
                    is_critical = True
                    if expected_status == "INVALID":
                        transition_notes = f"Threshold crossing: VALID→INVALID (displacement {cumulative_shift:.2f}px ≥ {self.threshold_px}px)"
                    else:
                        transition_notes = f"Recovery: INVALID→VALID (displacement {cumulative_shift:.2f}px < {self.threshold_px}px)"

                # Mark first INVALID frame for sudden onset
                if pattern_type == "sudden_onset" and frame_num == 1:
                    is_critical = True
                    transition_notes = "Sudden onset - first INVALID frame (latency test)"

            frame_id = f"{sequence_id}_frame_{frame_num:03d}"

            annotation = FrameAnnotation(
                frame_number=frame_num,
                frame_id=frame_id,
                timestamp=frame_num / frame_rate,
                cumulative_shift_px=cumulative_shift,
                shift_from_previous_px=shift_from_previous,
                expected_status=expected_status,
                expected_displacement_range=(min_disp, max_disp),
                movement_vector={"x": dx, "y": dy},
                is_critical_transition=is_critical,
                transition_notes=transition_notes
            )

            annotations.append(annotation)

        return annotations

    def create_sequence_metadata(
        self,
        sequence_id: str,
        pattern_type: str,
        baseline_image: str,
        annotations: List[FrameAnnotation],
        direction: str = "",
        frame_rate: float = 1.0
    ) -> SequenceMetadata:
        """
        Create metadata for a temporal sequence.

        Args:
            sequence_id: Unique sequence identifier
            pattern_type: Movement pattern type
            baseline_image: Baseline image filename
            annotations: Frame annotations
            direction: Movement direction
            frame_rate: Frames per second

        Returns:
            SequenceMetadata object
        """
        total_frames = len(annotations)
        duration = total_frames / frame_rate

        # Count expected INVALID frames
        expected_detections = sum(1 for a in annotations if a.expected_status == "INVALID")

        # Find first INVALID frame
        first_invalid = next(
            (a.frame_number for a in annotations if a.expected_status == "INVALID"),
            None
        )

        # Extract critical frame numbers
        critical_frames = [a.frame_number for a in annotations if a.is_critical_transition]

        # Calculate max displacement
        max_displacement = max(a.cumulative_shift_px for a in annotations)

        # Generate movement description
        if pattern_type == "gradual_onset":
            movement_desc = f"Linear drift 0→{max_displacement:.1f}px over {duration:.0f}s"
        elif pattern_type == "sudden_onset":
            movement_desc = f"Instant {max_displacement:.1f}px shift at frame 1, sustained"
        elif pattern_type == "progressive":
            movement_desc = f"Incremental steps to {max_displacement:.1f}px over {duration:.0f}s"
        elif pattern_type == "oscillation":
            movement_desc = f"Sinusoidal ±{max_displacement:.1f}px oscillation"
        elif pattern_type == "recovery":
            movement_desc = f"Recovery {max_displacement:.1f}px→0 over {duration:.0f}s"
        elif pattern_type == "multi_axis":
            movement_desc = f"Multi-axis drift to {max_displacement:.1f}px"
        else:
            movement_desc = f"Unknown pattern: {pattern_type}"

        metadata = SequenceMetadata(
            sequence_id=sequence_id,
            pattern_type=pattern_type,
            baseline_image=baseline_image,
            total_frames=total_frames,
            frame_rate=frame_rate,
            duration_seconds=duration,
            movement_description=movement_desc,
            expected_detections=expected_detections,
            expected_first_invalid_frame=first_invalid,
            critical_frames=critical_frames,
            direction=direction,
            max_displacement_px=max_displacement
        )

        return metadata

    def validate_sequence_detection(
        self,
        detection_results: List[Dict],
        annotations: List[FrameAnnotation],
        metadata: SequenceMetadata
    ) -> SequenceDetectionMetrics:
        """
        Validate detection results against ground truth annotations.

        Args:
            detection_results: List of detection results from detector
            annotations: Ground truth frame annotations
            metadata: Sequence metadata

        Returns:
            SequenceDetectionMetrics with validation results
        """
        total_frames = len(annotations)
        expected_invalid = sum(1 for a in annotations if a.expected_status == "INVALID")

        # Count detections
        detected_invalid = 0
        false_negatives = 0
        false_positives = 0

        for i, (result, annotation) in enumerate(zip(detection_results, annotations)):
            predicted_status = result['status']
            expected_status = annotation.expected_status

            if expected_status == "INVALID":
                if predicted_status == "INVALID":
                    detected_invalid += 1
                else:
                    false_negatives += 1
            else:  # expected_status == "VALID"
                if predicted_status == "INVALID":
                    false_positives += 1

        # Calculate detection rate
        detection_rate = detected_invalid / expected_invalid if expected_invalid > 0 else 1.0

        # Find first INVALID detection
        first_detected_invalid = next(
            (i for i, r in enumerate(detection_results) if r['status'] == "INVALID"),
            None
        )

        # Calculate detection latency (for sudden onset patterns)
        detection_latency = 0
        if metadata.pattern_type == "sudden_onset" and metadata.expected_first_invalid_frame is not None:
            if first_detected_invalid is not None:
                detection_latency = first_detected_invalid - metadata.expected_first_invalid_frame
            else:
                detection_latency = 999  # Not detected

        # Validate critical frames
        critical_annotations = [a for a in annotations if a.is_critical_transition]
        critical_correct = 0

        for annotation in critical_annotations:
            frame_num = annotation.frame_number
            if frame_num < len(detection_results):
                result = detection_results[frame_num]
                if result['status'] == annotation.expected_status:
                    critical_correct += 1

        critical_frame_accuracy = (
            critical_correct / len(critical_annotations)
            if len(critical_annotations) > 0
            else 1.0
        )

        metrics = SequenceDetectionMetrics(
            sequence_id=metadata.sequence_id,
            pattern_type=metadata.pattern_type,
            total_frames=total_frames,
            expected_invalid_frames=expected_invalid,
            detected_invalid_frames=detected_invalid,
            false_negatives=false_negatives,
            false_positives=false_positives,
            detection_rate=detection_rate,
            first_invalid_detected=first_detected_invalid,
            expected_first_invalid=metadata.expected_first_invalid_frame,
            detection_latency_frames=detection_latency,
            critical_frame_accuracy=critical_frame_accuracy,
            critical_frames_tested=len(critical_annotations),
            critical_frames_correct=critical_correct
        )

        return metrics

    def save_sequence_data(
        self,
        output_dir: Path,
        metadata: SequenceMetadata,
        annotations: List[FrameAnnotation]
    ):
        """
        Save sequence metadata and frame annotations to JSON files.

        Args:
            output_dir: Sequence output directory
            metadata: Sequence metadata
            annotations: Frame annotations
        """
        # Save metadata
        metadata_path = output_dir / "sequence_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Save frame annotations
        annotations_path = output_dir / "frame_annotations.json"
        annotations_dict = {
            "frames": [asdict(a) for a in annotations]
        }
        with open(annotations_path, 'w') as f:
            json.dump(annotations_dict, f, indent=2)


def main():
    """Example usage and testing"""
    harness = Stage2TestHarness(threshold_px=1.5)

    # Example: Generate gradual onset trajectory
    print("Generating gradual onset trajectory...")
    trajectory = harness.generate_gradual_onset_trajectory(
        total_frames=60,
        max_displacement=5.0,
        direction='right'
    )

    print(f"Generated {len(trajectory)} frames")
    print(f"Frame 0: {trajectory[0]}")
    print(f"Frame 30: {trajectory[30]}")
    print(f"Frame 59: {trajectory[59]}")

    # Create frame annotations
    annotations = harness.create_frame_annotations(
        sequence_id="test_gradual_onset_001",
        trajectory=trajectory,
        pattern_type="gradual_onset",
        frame_rate=1.0
    )

    # Find critical frames
    critical_frames = [a for a in annotations if a.is_critical_transition]
    print(f"\nCritical frames: {len(critical_frames)}")
    for cf in critical_frames:
        print(f"  Frame {cf.frame_number}: {cf.transition_notes}")


if __name__ == "__main__":
    main()
