"""
Stage 3 Validation Framework

This package provides real-world validation infrastructure for the camera shift detector.

Modules:
- real_data_loader: Load real DAF imagery with metadata
- stage3_test_harness: Execute detector against ground truth
- performance_profiler: Measure FPS, memory, CPU usage
- run_stage3_validation: Orchestrate complete validation workflow
"""

__version__ = "1.0.0"
__author__ = "Tomer"

from pathlib import Path

# Package root directory
VALIDATION_ROOT = Path(__file__).parent

# Key directories
GROUND_TRUTH_DIR = VALIDATION_ROOT / "ground_truth"
RESULTS_DIR = VALIDATION_ROOT / "results"
SAMPLE_IMAGES_DIR = VALIDATION_ROOT.parent / "sample_images"

__all__ = [
    "VALIDATION_ROOT",
    "GROUND_TRUTH_DIR",
    "RESULTS_DIR",
    "SAMPLE_IMAGES_DIR",
]
