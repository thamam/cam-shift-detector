# Camera Movement Detector - Validation System

Comprehensive validation framework for testing camera movement detection across synthetic, temporal, and real-world scenarios.

## Directory Structure

```
validation/
├── core/                   # Production validation scripts
├── harnesses/              # Test harness modules
├── utilities/              # Support modules (loaders, profilers, reporters)
├── data_generators/        # Synthetic data generation
├── data/                   # Test datasets (stage 1-3)
├── results/                # Validation results by stage
└── archive/                # Historical investigation scripts
```

## Quick Start

### Stage 1: Synthetic Validation (AC-1.9.1 - 95% accuracy)
```bash
python validation/core/run_stage1_validation.py
```
- **Purpose**: Validate detector on synthetic 2px, 5px, 10px shifts
- **Dataset**: 1,250 transformed images (50 baselines × 25 variants)
- **Results**: `validation/results/stage1/`

### Stage 2: Temporal Validation (AC-1.9.2 - 100% detection rate)
```bash
# Using ChArUco ground truth
python validation/core/stage2_charuco_validation.py

# Using synthetic temporal sequences
python validation/core/run_stage2_validation.py
```
- **Purpose**: Validate movement detection over time
- **ChArUco**: 157 frames with precise 6-DOF pose measurements
- **Synthetic**: 1,900 frames across 6 temporal patterns
- **Results**: `validation/results/stage2/`

### Stage 3: Real-World Validation
```bash
python validation/core/run_stage3_validation.py
```
- **Purpose**: Validate on real DAF agricultural images
- **Dataset**: 50 images from 3 pilot sites
- **Results**: `validation/results/stage3/`

## Validation Stages

### Stage 1: Synthetic Single-Frame Shifts
**Goal**: Verify detector correctly identifies static vs shifted images

**Data**: `validation/data/stage1/`
- 50 baseline images from DAF sites
- 24 synthetic shifts per baseline (8 directions × 3 magnitudes)
- Ground truth: `data/stage1/ground_truth.json`

**Acceptance**: Accuracy ≥95% (AC-1.9.1)

### Stage 2: Temporal Sequence Detection
**Goal**: Verify detector maintains correct status across frame sequences

**ChArUco Option** (COMPLETED):
- 2 sessions: 203 + 56 frames with 6-DOF ground truth
- Data: ChArUco sessions in project root
- Results: 100% detection rate achieved ✅

**Synthetic Option**:
- Data: `validation/data/stage2/`
- 6 patterns: gradual onset, sudden onset, progressive, oscillation, recovery, multi-axis
- Ground truth: `data/stage2/ground_truth_sequences.json`

**Acceptance**: 100% detection rate / 0% false negatives (AC-1.9.2)

### Stage 3: Real-World Deployment
**Goal**: Validate on actual pilot site imagery

**Data**: `validation/data/stage3/`
- 50 real DAF images (OF_JERUSALEM, CARMIT, GAD sites)
- Ground truth: Manual annotations

**Status**: Investigation pending (98% FPR issue identified)

## Test Harnesses

Located in `validation/harnesses/`:
- `stage1_test_harness.py` - Single-frame validation framework
- `stage2_test_harness.py` - Temporal sequence validation
- `stage3_test_harness.py` - Real-world validation with profiling

## Utilities

Located in `validation/utilities/`:
- `real_data_loader.py` - Load and process DAF images
- `performance_profiler.py` - FPS, memory, CPU profiling
- `report_generator.py` - JSON/Markdown report generation

## Data Generators

Located in `validation/data_generators/`:
- `generate_stage1_data.py` - Create synthetic shifted images
- `generate_stage2_data.py` - Generate temporal sequences

## Results Organization

All validation results organized by stage:
```
results/
├── stage1/
│   ├── stage1_results.json
│   ├── stage1_results.md
│   └── stage1_results_report.txt
├── stage2/
│   ├── charuco/          # ChArUco validation results
│   └── STAGE2_DESIGN.md
└── stage3/
    └── validation_report.json
```

## Historical Archive

Investigation scripts from Stage 2 debugging preserved in `validation/archive/`:
- `stage2_investigations/` - Analysis scripts (threshold, failures, patterns)
- `one_time_scripts/` - Diagnostic and regeneration utilities

These are **historical artifacts** - not needed for production validation.

## Common Tasks

### Generate New Test Data
```bash
# Regenerate Stage 1 synthetic data
python validation/data_generators/generate_stage1_data.py

# Regenerate Stage 2 temporal sequences
python validation/data_generators/generate_stage2_data.py
```

### View Results
```bash
# Stage 1 summary
cat validation/results/stage1/stage1_results_report.txt

# Stage 2 ChArUco results
cat validation/results/stage2/charuco/stage2_results_report.txt

# Stage 3 detailed results
python -m json.tool validation/results/stage3/validation_report.json | less
```

## Validation Acceptance Criteria

| Stage | Metric | Target | Status |
|-------|--------|--------|--------|
| Stage 1 | Accuracy | ≥95% | ✅ PASSED (95.59%) |
| Stage 2 (ChArUco) | Detection Rate | 100% | ✅ PASSED (100%) |
| Stage 2 (Synthetic) | False Negatives | 0% | ⚠️ PENDING |
| Stage 3 | Production Ready | Go/No-Go | 🔍 INVESTIGATING |

## Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- Camera movement detector: `src.camera_movement_detector`

## Documentation

- Stage 1 data: `validation/data/stage1/README.md`
- Stage 2 data: `validation/data/stage2/README.md`
- ChArUco handoff: `claudedocs/charuco_validation_handoff.md`
- Stage 2 completion: `validation/results/stage2/charuco/STAGE2_COMPLETION_SUMMARY.md`
