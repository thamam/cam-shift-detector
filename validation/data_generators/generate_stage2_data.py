#!/usr/bin/env python3
"""
Stage 2 Dataset Generator

Generates temporal sequences with realistic camera movement patterns for Stage 2 validation.
Creates 6 movement patterns × 10 baseline images = 60 sequences with ~1,900 total frames.

Patterns:
1. Gradual Onset (60 frames) - Linear drift 0→5px
2. Sudden Onset (30 frames) - Instant 5px shift sustained
3. Progressive (20 frames) - 0.5px incremental steps
4. Oscillation (20 frames) - Sinusoidal ±3px
5. Recovery (30 frames) - 5px→0 gradual return
6. Multi-Axis (30 frames) - Independent X/Y drift

Usage:
    python validation/generate_stage2_data.py
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.harnesses.stage2_test_harness import Stage2TestHarness, SequenceMetadata


def select_baseline_images(sample_images_dir: Path, count: int = 10) -> List[str]:
    """
    Select baseline images from sample_images directory.

    Args:
        sample_images_dir: Path to sample_images directory
        count: Number of images to select

    Returns:
        List of baseline image paths
    """
    # Collect all images from subdirectories
    all_images = []

    for subdir in ['of_jerusalem', 'carmit', 'gad']:
        subdir_path = sample_images_dir / subdir
        if subdir_path.exists():
            images = list(subdir_path.glob('*.jpg'))
            all_images.extend(images)

    # Select evenly across available images
    if len(all_images) < count:
        print(f"⚠️  Warning: Found only {len(all_images)} images, requested {count}")
        return [str(img) for img in all_images]

    # Systematic sampling
    step = len(all_images) / count
    selected = [all_images[int(i * step)] for i in range(count)]

    return [str(img) for img in selected]


def generate_pattern_1_gradual_onset(
    harness: Stage2TestHarness,
    baseline_images: List[str],
    output_dir: Path
) -> List[SequenceMetadata]:
    """Generate Pattern 1: Gradual Onset sequences"""
    print("\n🔄 Generating Pattern 1: Gradual Onset (Linear Drift 0→5px)")

    pattern_dir = output_dir / "pattern_1_gradual_onset"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    sequences_metadata = []
    directions = ['right', 'up', 'diagonal_ur', 'diagonal_dr']

    total_frames = 60
    max_displacement = 5.0

    for baseline_idx, baseline_path in enumerate(baseline_images[:10], 1):
        baseline_image = cv2.imread(baseline_path)
        if baseline_image is None:
            print(f"  ⚠️  Failed to load: {baseline_path}")
            continue

        baseline_name = Path(baseline_path).stem
        direction = directions[(baseline_idx - 1) % len(directions)]

        sequence_id = f"pattern_1_gradual_onset_{baseline_name}_{direction}"
        print(f"  [{baseline_idx}/10] {sequence_id}")

        # Generate trajectory
        trajectory = harness.generate_gradual_onset_trajectory(
            total_frames=total_frames,
            max_displacement=max_displacement,
            direction=direction
        )

        # Generate frame sequence
        sequence_dir = pattern_dir / sequence_id
        frame_paths = harness.apply_temporal_transformation(
            baseline_image=baseline_image,
            trajectory=trajectory,
            output_dir=sequence_dir,
            sequence_id=sequence_id
        )

        # Create annotations
        annotations = harness.create_frame_annotations(
            sequence_id=sequence_id,
            trajectory=trajectory,
            pattern_type="gradual_onset",
            frame_rate=1.0
        )

        # Create metadata
        metadata = harness.create_sequence_metadata(
            sequence_id=sequence_id,
            pattern_type="gradual_onset",
            baseline_image=baseline_name,
            annotations=annotations,
            direction=direction,
            frame_rate=1.0
        )

        # Save sequence data
        harness.save_sequence_data(sequence_dir, metadata, annotations)
        sequences_metadata.append(metadata)

    print(f"  ✅ Generated {len(sequences_metadata)} sequences ({len(sequences_metadata) * total_frames} frames)")
    return sequences_metadata


def generate_pattern_2_sudden_onset(
    harness: Stage2TestHarness,
    baseline_images: List[str],
    output_dir: Path
) -> List[SequenceMetadata]:
    """Generate Pattern 2: Sudden Onset sequences"""
    print("\n⚡ Generating Pattern 2: Sudden Onset (Instant 5px Shift)")

    pattern_dir = output_dir / "pattern_2_sudden_onset"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    sequences_metadata = []
    directions = ['right', 'up', 'diagonal_ur', 'diagonal_dr']

    total_frames = 30
    displacement = 5.0

    for baseline_idx, baseline_path in enumerate(baseline_images[:10], 1):
        baseline_image = cv2.imread(baseline_path)
        if baseline_image is None:
            continue

        baseline_name = Path(baseline_path).stem
        direction = directions[(baseline_idx - 1) % len(directions)]

        sequence_id = f"pattern_2_sudden_onset_{baseline_name}_{direction}"
        print(f"  [{baseline_idx}/10] {sequence_id}")

        # Generate trajectory
        trajectory = harness.generate_sudden_onset_trajectory(
            total_frames=total_frames,
            displacement=displacement,
            direction=direction,
            onset_frame=1
        )

        # Generate frame sequence
        sequence_dir = pattern_dir / sequence_id
        frame_paths = harness.apply_temporal_transformation(
            baseline_image=baseline_image,
            trajectory=trajectory,
            output_dir=sequence_dir,
            sequence_id=sequence_id
        )

        # Create annotations
        annotations = harness.create_frame_annotations(
            sequence_id=sequence_id,
            trajectory=trajectory,
            pattern_type="sudden_onset",
            frame_rate=1.0
        )

        # Create metadata
        metadata = harness.create_sequence_metadata(
            sequence_id=sequence_id,
            pattern_type="sudden_onset",
            baseline_image=baseline_name,
            annotations=annotations,
            direction=direction,
            frame_rate=1.0
        )

        # Save sequence data
        harness.save_sequence_data(sequence_dir, metadata, annotations)
        sequences_metadata.append(metadata)

    print(f"  ✅ Generated {len(sequences_metadata)} sequences ({len(sequences_metadata) * total_frames} frames)")
    return sequences_metadata


def generate_pattern_3_progressive(
    harness: Stage2TestHarness,
    baseline_images: List[str],
    output_dir: Path
) -> List[SequenceMetadata]:
    """Generate Pattern 3: Progressive Displacement sequences"""
    print("\n📈 Generating Pattern 3: Progressive Displacement (0.5px Steps)")

    pattern_dir = output_dir / "pattern_3_progressive"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    sequences_metadata = []
    directions = ['right', 'up', 'diagonal_ur', 'diagonal_dr']

    total_frames = 20
    step_size = 0.5

    for baseline_idx, baseline_path in enumerate(baseline_images[:10], 1):
        baseline_image = cv2.imread(baseline_path)
        if baseline_image is None:
            continue

        baseline_name = Path(baseline_path).stem
        direction = directions[(baseline_idx - 1) % len(directions)]

        sequence_id = f"pattern_3_progressive_{baseline_name}_{direction}"
        print(f"  [{baseline_idx}/10] {sequence_id}")

        # Generate trajectory
        trajectory = harness.generate_progressive_trajectory(
            total_frames=total_frames,
            step_size=step_size,
            direction=direction
        )

        # Generate frame sequence
        sequence_dir = pattern_dir / sequence_id
        frame_paths = harness.apply_temporal_transformation(
            baseline_image=baseline_image,
            trajectory=trajectory,
            output_dir=sequence_dir,
            sequence_id=sequence_id
        )

        # Create annotations
        annotations = harness.create_frame_annotations(
            sequence_id=sequence_id,
            trajectory=trajectory,
            pattern_type="progressive",
            frame_rate=1.0
        )

        # Create metadata
        metadata = harness.create_sequence_metadata(
            sequence_id=sequence_id,
            pattern_type="progressive",
            baseline_image=baseline_name,
            annotations=annotations,
            direction=direction,
            frame_rate=1.0
        )

        # Save sequence data
        harness.save_sequence_data(sequence_dir, metadata, annotations)
        sequences_metadata.append(metadata)

    print(f"  ✅ Generated {len(sequences_metadata)} sequences ({len(sequences_metadata) * total_frames} frames)")
    return sequences_metadata


def generate_pattern_4_oscillation(
    harness: Stage2TestHarness,
    baseline_images: List[str],
    output_dir: Path
) -> List[SequenceMetadata]:
    """Generate Pattern 4: Oscillation sequences"""
    print("\n🌊 Generating Pattern 4: Oscillation (Sinusoidal ±3px)")

    pattern_dir = output_dir / "pattern_4_oscillation"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    sequences_metadata = []
    axes = ['horizontal', 'vertical']

    total_frames = 20
    amplitude = 3.0

    for baseline_idx, baseline_path in enumerate(baseline_images[:10], 1):
        baseline_image = cv2.imread(baseline_path)
        if baseline_image is None:
            continue

        baseline_name = Path(baseline_path).stem
        axis = axes[(baseline_idx - 1) % len(axes)]

        sequence_id = f"pattern_4_oscillation_{baseline_name}_{axis}"
        print(f"  [{baseline_idx}/10] {sequence_id}")

        # Generate trajectory
        trajectory = harness.generate_oscillation_trajectory(
            total_frames=total_frames,
            amplitude=amplitude,
            axis=axis
        )

        # Generate frame sequence
        sequence_dir = pattern_dir / sequence_id
        frame_paths = harness.apply_temporal_transformation(
            baseline_image=baseline_image,
            trajectory=trajectory,
            output_dir=sequence_dir,
            sequence_id=sequence_id
        )

        # Create annotations
        annotations = harness.create_frame_annotations(
            sequence_id=sequence_id,
            trajectory=trajectory,
            pattern_type="oscillation",
            frame_rate=1.0
        )

        # Create metadata
        metadata = harness.create_sequence_metadata(
            sequence_id=sequence_id,
            pattern_type="oscillation",
            baseline_image=baseline_name,
            annotations=annotations,
            direction=axis,
            frame_rate=1.0
        )

        # Save sequence data
        harness.save_sequence_data(sequence_dir, metadata, annotations)
        sequences_metadata.append(metadata)

    print(f"  ✅ Generated {len(sequences_metadata)} sequences ({len(sequences_metadata) * total_frames} frames)")
    return sequences_metadata


def generate_pattern_5_recovery(
    harness: Stage2TestHarness,
    baseline_images: List[str],
    output_dir: Path
) -> List[SequenceMetadata]:
    """Generate Pattern 5: Recovery sequences"""
    print("\n🔙 Generating Pattern 5: Recovery (5px→0 Gradual Return)")

    pattern_dir = output_dir / "pattern_5_recovery"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    sequences_metadata = []
    directions = ['right', 'up', 'diagonal_ur', 'diagonal_dr']

    total_frames = 30
    initial_displacement = 5.0

    for baseline_idx, baseline_path in enumerate(baseline_images[:10], 1):
        baseline_image = cv2.imread(baseline_path)
        if baseline_image is None:
            continue

        baseline_name = Path(baseline_path).stem
        direction = directions[(baseline_idx - 1) % len(directions)]

        sequence_id = f"pattern_5_recovery_{baseline_name}_{direction}"
        print(f"  [{baseline_idx}/10] {sequence_id}")

        # Generate trajectory
        trajectory = harness.generate_recovery_trajectory(
            total_frames=total_frames,
            initial_displacement=initial_displacement,
            direction=direction
        )

        # Generate frame sequence
        sequence_dir = pattern_dir / sequence_id
        frame_paths = harness.apply_temporal_transformation(
            baseline_image=baseline_image,
            trajectory=trajectory,
            output_dir=sequence_dir,
            sequence_id=sequence_id
        )

        # Create annotations
        annotations = harness.create_frame_annotations(
            sequence_id=sequence_id,
            trajectory=trajectory,
            pattern_type="recovery",
            frame_rate=1.0
        )

        # Create metadata
        metadata = harness.create_sequence_metadata(
            sequence_id=sequence_id,
            pattern_type="recovery",
            baseline_image=baseline_name,
            annotations=annotations,
            direction=direction,
            frame_rate=1.0
        )

        # Save sequence data
        harness.save_sequence_data(sequence_dir, metadata, annotations)
        sequences_metadata.append(metadata)

    print(f"  ✅ Generated {len(sequences_metadata)} sequences ({len(sequences_metadata) * total_frames} frames)")
    return sequences_metadata


def generate_pattern_6_multi_axis(
    harness: Stage2TestHarness,
    baseline_images: List[str],
    output_dir: Path
) -> List[SequenceMetadata]:
    """Generate Pattern 6: Multi-Axis Movement sequences"""
    print("\n📐 Generating Pattern 6: Multi-Axis Movement (Independent X/Y)")

    pattern_dir = output_dir / "pattern_6_multi_axis"
    pattern_dir.mkdir(parents=True, exist_ok=True)

    sequences_metadata = []
    rate_combinations = [
        (0.15, 0.20),  # X faster
        (0.20, 0.15),  # Y faster
        (0.18, 0.18),  # Equal rates
        (0.10, 0.25),  # Y much faster
    ]

    total_frames = 30

    for baseline_idx, baseline_path in enumerate(baseline_images[:10], 1):
        baseline_image = cv2.imread(baseline_path)
        if baseline_image is None:
            continue

        baseline_name = Path(baseline_path).stem
        x_rate, y_rate = rate_combinations[(baseline_idx - 1) % len(rate_combinations)]

        sequence_id = f"pattern_6_multi_axis_{baseline_name}_x{x_rate:.2f}_y{y_rate:.2f}"
        print(f"  [{baseline_idx}/10] {sequence_id}")

        # Generate trajectory
        trajectory = harness.generate_multi_axis_trajectory(
            total_frames=total_frames,
            x_rate=x_rate,
            y_rate=y_rate
        )

        # Generate frame sequence
        sequence_dir = pattern_dir / sequence_id
        frame_paths = harness.apply_temporal_transformation(
            baseline_image=baseline_image,
            trajectory=trajectory,
            output_dir=sequence_dir,
            sequence_id=sequence_id
        )

        # Create annotations
        annotations = harness.create_frame_annotations(
            sequence_id=sequence_id,
            trajectory=trajectory,
            pattern_type="multi_axis",
            frame_rate=1.0
        )

        # Create metadata
        metadata = harness.create_sequence_metadata(
            sequence_id=sequence_id,
            pattern_type="multi_axis",
            baseline_image=baseline_name,
            annotations=annotations,
            direction=f"x_rate={x_rate:.2f},y_rate={y_rate:.2f}",
            frame_rate=1.0
        )

        # Save sequence data
        harness.save_sequence_data(sequence_dir, metadata, annotations)
        sequences_metadata.append(metadata)

    print(f"  ✅ Generated {len(sequences_metadata)} sequences ({len(sequences_metadata) * total_frames} frames)")
    return sequences_metadata


def create_global_ground_truth(
    all_sequences: List[SequenceMetadata],
    output_dir: Path
):
    """Create global ground truth JSON with all sequence metadata"""
    print("\n💾 Creating global ground truth file...")

    ground_truth = {
        "generation_date": datetime.now().isoformat(),
        "total_sequences": len(all_sequences),
        "total_frames": sum(s.total_frames for s in all_sequences),
        "patterns": {
            "gradual_onset": len([s for s in all_sequences if s.pattern_type == "gradual_onset"]),
            "sudden_onset": len([s for s in all_sequences if s.pattern_type == "sudden_onset"]),
            "progressive": len([s for s in all_sequences if s.pattern_type == "progressive"]),
            "oscillation": len([s for s in all_sequences if s.pattern_type == "oscillation"]),
            "recovery": len([s for s in all_sequences if s.pattern_type == "recovery"]),
            "multi_axis": len([s for s in all_sequences if s.pattern_type == "multi_axis"])
        },
        "sequences": [
            {
                "sequence_id": s.sequence_id,
                "pattern_type": s.pattern_type,
                "baseline_image": s.baseline_image,
                "total_frames": s.total_frames,
                "expected_detections": s.expected_detections,
                "critical_frames": s.critical_frames,
                "direction": s.direction,
                "max_displacement_px": s.max_displacement_px
            }
            for s in all_sequences
        ]
    }

    output_path = output_dir / "ground_truth_sequences.json"
    with open(output_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print(f"  ✅ Saved: {output_path}")


def main():
    """Main dataset generation execution"""
    print("=" * 80)
    print("Stage 2 Dataset Generation")
    print("=" * 80)

    # Setup paths
    sample_images_dir = Path("sample_images")
    output_dir = Path("validation/stage2_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize test harness
    print("\n🔧 Initializing Stage 2 test harness...")
    harness = Stage2TestHarness(threshold_px=1.5)

    # Select baseline images
    print(f"\n📋 Selecting baseline images from {sample_images_dir}...")
    baseline_images = select_baseline_images(sample_images_dir, count=10)
    print(f"  Selected {len(baseline_images)} baseline images")

    start_time = datetime.now()

    # Generate all pattern types
    all_sequences = []

    all_sequences.extend(generate_pattern_1_gradual_onset(harness, baseline_images, output_dir))
    all_sequences.extend(generate_pattern_2_sudden_onset(harness, baseline_images, output_dir))
    all_sequences.extend(generate_pattern_3_progressive(harness, baseline_images, output_dir))
    all_sequences.extend(generate_pattern_4_oscillation(harness, baseline_images, output_dir))
    all_sequences.extend(generate_pattern_5_recovery(harness, baseline_images, output_dir))
    all_sequences.extend(generate_pattern_6_multi_axis(harness, baseline_images, output_dir))

    # Create global ground truth
    create_global_ground_truth(all_sequences, output_dir)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Summary statistics
    total_frames = sum(s.total_frames for s in all_sequences)
    total_expected_detections = sum(s.expected_detections for s in all_sequences)

    print("\n" + "=" * 80)
    print("✅ Stage 2 Dataset Generation Complete!")
    print("=" * 80)
    print(f"\n📊 Summary Statistics:")
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Expected INVALID detections: {total_expected_detections}")
    print(f"  Processing rate: {total_frames / duration:.1f} frames/second")
    print(f"  Duration: {duration:.2f} seconds")

    print(f"\n📂 Output directory: {output_dir.absolute()}")
    print(f"\n✅ Dataset ready for Stage 2 validation!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
