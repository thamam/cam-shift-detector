# Camera Shift Detector

Computer vision system for detecting camera movement in time-series imagery using feature matching and transformation analysis.

## Project Status

Development in progress - Stage 3 validation framework complete.

**Recent Updates:**
- ✅ Stage 1: Validation dataset and ground truth generation
- ✅ Stage 2: Test harness with performance profiling
- ✅ Stage 3: Automated validation runner with comprehensive reporting
- ✅ Integration documentation and stub implementation

## Integration

### Quick Integration

```bash
# Install package
pip install -e .

# Basic usage
from src.camera_movement_detector import CameraMovementDetector

detector = CameraMovementDetector('config.json')
detector.set_baseline(initial_frame)
result = detector.process_frame(current_frame)
```

### Documentation

- **Integration Guide:** `docs/integration-guide.md` - Comprehensive integration documentation
- **Installation Guide:** `docs/installation.md` - Step-by-step installation instructions
- **Integration Cheat Sheet:** `docs/integration-cheat-sheet.md` - Quick reference for meetings

### Stub Implementation (Parallel Development)

For integration development before production module is ready:

```python
# Use stub for integration testing
from src.camera_movement_detector_stub import CameraMovementDetector

# Same API as real implementation
detector = CameraMovementDetector('config.json')
detector.set_baseline(initial_frame)
result = detector.process_frame(current_frame)

# Later, swap to real implementation (no code changes needed):
# from src.camera_movement_detector import CameraMovementDetector
```

**Stub Features:**
- Same API as real implementation
- Returns realistic mock data
- Validates input formats
- Enables parallel integration development
- Zero code changes when swapping to real module

## Stage 3 Validation

The Stage 3 validation framework provides automated end-to-end validation with comprehensive reporting and production readiness assessment.

### Quick Start

Run the complete validation workflow:

```bash
python validation/run_stage3_validation.py
```

This will:
1. Load and validate the test dataset (50 real DAF site images)
2. Execute camera shift detection with performance profiling
3. Generate comprehensive validation reports (JSON + Markdown)
4. Provide GO/NO-GO production deployment recommendation

### Command-Line Options

```bash
# Use custom baseline image
python validation/run_stage3_validation.py --baseline sample_images/of_jerusalem/img_001.jpg

# Specify custom output directory
python validation/run_stage3_validation.py --output-dir custom_results/

# Use custom detector configuration
python validation/run_stage3_validation.py --detector-config config/custom_config.json

# Full example with all options
python validation/run_stage3_validation.py \
    --baseline sample_images/of_jerusalem/img_001.jpg \
    --output-dir validation/results/run_2025_10_26 \
    --detector-config config/detector_config.json
```

### Output Reports

The validation runner generates two comprehensive reports in `validation/results/`:

#### 1. JSON Report (`validation_report.json`)

Machine-readable report containing:
- Validation metadata (date, version, total images processed)
- Detection metrics (accuracy, false positive rate, false negative rate)
- Performance metrics (FPS, memory usage, CPU utilization)
- Per-site breakdown (accuracy per DAF site)
- Go/no-go recommendation with gate criteria results

#### 2. Markdown Report (`validation_report.md`)

Human-readable report with:
- **Executive Summary**: Key findings and production readiness status
- **Validation Metrics**: Confusion matrix and overall performance tables
- **Performance Benchmarks**: System resource utilization vs. targets
- **Per-Site Breakdown**: Accuracy breakdown by DAF site
- **Failure Analysis**: Detailed analysis of false positives/negatives (when applicable)
- **Go/No-Go Recommendation**: Production deployment decision with rationale
- **Next Steps**: Actionable recommendations based on results

### Go/No-Go Gate Criteria

The validation framework uses conservative gate-based logic for production readiness:

| Gate Criterion | Threshold | Status |
|----------------|-----------|--------|
| Detection Accuracy | ≥95% | Pass/Fail |
| False Positive Rate | ≤5% | Pass/Fail |
| Processing Speed (FPS) | ≥0.0167 (1/60 Hz) | Pass/Fail |
| Memory Usage | ≤500 MB | Pass/Fail |

**Decision Logic**: ALL gates must pass for GO recommendation. ANY single gate failure results in NO-GO to ensure production safety.

### Exit Codes

The validation runner uses standard Unix exit codes:

- `0`: Validation successful, reports generated
- `1`: Validation errors detected (check reports for details)
- `2`: System errors (configuration not found, permission denied, etc.)

### Example Workflow

```bash
# 1. Run validation
python validation/run_stage3_validation.py

# 2. Check exit code
echo $?  # 0 = success, 1 = validation errors, 2 = system errors

# 3. Review reports
cat validation/results/validation_report.md

# 4. Parse JSON programmatically
python -c "import json; print(json.load(open('validation/results/validation_report.json'))['go_no_go']['recommendation'])"
```

### Validation Dataset

The validation uses 50 systematically sampled real images from 3 DAF sites:
- **OF_JERUSALEM**: 23 images (site: 9bc4603f-0d21-4f60-afea-b6343d372034)
- **CARMIT**: 17 images (site: e2336087-0143-4895-bc12-59b8b8f97790)
- **GAD**: 10 images (site: f10f17d4-ac26-4c28-b601-06c64b8a22a4)

Ground truth annotations are maintained in `validation/ground_truth/validation_ground_truth.json`.
