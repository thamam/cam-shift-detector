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
import sys

# Package root directory
VALIDATION_ROOT = Path(__file__).parent

# Key directories
GROUND_TRUTH_DIR = VALIDATION_ROOT / "ground_truth"
RESULTS_DIR = VALIDATION_ROOT / "results"
SAMPLE_IMAGES_DIR = VALIDATION_ROOT.parent / "sample_images"

# Create module aliases for backward compatibility with old import paths
# Import modules one by one and register aliases to avoid circular dependencies

# Import and register harnesses first
from validation.harnesses import stage1_test_harness, stage2_test_harness, stage3_test_harness
sys.modules['validation.stage1_test_harness'] = stage1_test_harness
sys.modules['validation.stage2_test_harness'] = stage2_test_harness
sys.modules['validation.stage3_test_harness'] = stage3_test_harness

# Import and register utilities one by one (order matters due to dependencies)
from validation.utilities import real_data_loader
sys.modules['validation.real_data_loader'] = real_data_loader

from validation.utilities import performance_profiler
sys.modules['validation.performance_profiler'] = performance_profiler

from validation.utilities import report_generator
sys.modules['validation.report_generator'] = report_generator

# Import and register core modules
from validation.core import run_stage3_validation
sys.modules['validation.run_stage3_validation'] = run_stage3_validation

__all__ = [
    "VALIDATION_ROOT",
    "GROUND_TRUTH_DIR",
    "RESULTS_DIR",
    "SAMPLE_IMAGES_DIR",
    # Module aliases for backward compatibility
    "stage1_test_harness",
    "stage2_test_harness",
    "stage3_test_harness",
    "real_data_loader",
    "performance_profiler",
    "report_generator",
    "run_stage3_validation",
]
