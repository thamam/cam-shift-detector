# Camera Shift Detector

Computer vision system for detecting camera movement in time-series imagery using feature matching and transformation analysis.

## Project Status

**Current Version:** v0.2.0 (2025-10-29) - Production-ready with interactive debugging tools

**Ground Truth Validation Complete**: Manual annotation of 50 images reveals detector achieves 66% accuracy with 100% recall (perfect shift detection). See [documentation/ground-truth-annotation-results.md](documentation/ground-truth-annotation-results.md).

**Recent Updates:**
- ✅ **Stage 4 (Epic 4)**: Interactive debugging tools (Modes A/B/C) for visual analysis
- ✅ **Stage 3 (Epic 3)**: Dual detector comparison tool with offline/online modes
- ✅ Stage 2: Test harness with performance profiling
- ✅ Stage 1: Validation dataset and ground truth generation
- ✅ Integration documentation and stub implementation
- ✅ Ground truth annotation tool with corrected validation results

## Tools

### Ground Truth Annotation Tool

Interactive tool for manually verifying camera shifts in image pairs with per-site baselines and shift magnitude levels.

```bash
.venv/bin/python tools/annotation/ground_truth_annotator.py
```

**Features**:
- Per-site baselines (not single global baseline)
- Shift magnitude levels: Aligned / Small (<2%) / Medium (2-4%) / Large (>4%) / Inconclusive
- Alpha blending for visual verification
- Auto-save and resume capability

**Documentation**:
- Tool guide: [tools/annotation/README.md](tools/annotation/README.md)
- Validation results: [documentation/ground-truth-annotation-results.md](documentation/ground-truth-annotation-results.md)
- Corrected summary: [claudedocs/validation-corrected-summary.md](claudedocs/validation-corrected-summary.md)

---

### Interactive Debugging Tools (Stage 4)

Real-time OpenCV-based debugging tools for visual camera shift analysis:

#### Mode A - 4-Quadrant Comparison

Side-by-side comparison of ChArUco and cam-shift detectors with feature overlays.

```bash
.venv/bin/python tools/validation/comparison_tool.py --input-dir sample_images/of_jerusalem
```

**Features**:
- 4-image layout: baseline/current × charuco/csd
- Feature overlay visualization (ChArUco corners + ORB features)
- Manual frame stepping (←/→ arrows)
- Enhanced metrics (Δdx, Δdy, error magnitude)
- CSV/PNG export

#### Mode B - Baseline Correspondence

Motion vector visualization with baseline pinning for drift analysis.

```bash
.venv/bin/python tools/validation/baseline_correspondence_tool.py --input-dir sample_images/of_jerusalem
```

**Features**:
- Motion vector arrows showing feature displacement
- Inlier/outlier coloring (green/red)
- Baseline pinning mechanism
- Match quality metrics (RMSE, inlier percentage)
- Difference heatmap overlay
- Diagnostic mode (D key) with keypoint density

**Keyboard shortcuts**:
- `←/→` - Navigate frames
- `b` - Set baseline frame
- `d` - Toggle diagnostic mode
- `s` - Save snapshot
- `q` - Quit

#### Mode C - Enhanced Alpha Blending

Transform computation with pre-warp and blink mode for visual alignment verification.

```bash
.venv/bin/python tools/validation/alpha_blending_tool.py --input-dir sample_images/of_jerusalem
```

**Features**:
- Transform computation using CSD homography
- Pre-warp toggle (W key) for alignment verification
- Blink mode (Space) with 500ms A/B alternation
- Frame selector for arbitrary frame pairs
- Alpha blending with adjustable transparency
- 10×10 alignment grid overlay
- Descriptive file naming with metadata

**Keyboard shortcuts**:
- `a/b` - Select Frame A/B
- `←/→` - Navigate frames (during selection)
- `Enter` - Confirm frame selection
- `↑/↓` - Adjust alpha value
- `w` - Toggle pre-warp
- `Space` - Toggle blink mode
- `g` - Toggle grid overlay
- `s` - Save snapshot with metadata
- `q` - Quit

**Documentation**: [tools/validation/README.md](tools/validation/README.md)

---

### Dual Detector Comparison Tool (Stage 3 / Epic 3)

Compare ChArUco and cam-shift detectors in offline and online modes.

```bash
# Offline mode (directory)
.venv/bin/python tools/validation/comparison_tool.py --mode offline --input-dir sample_images/of_jerusalem

# Online mode (live camera)
.venv/bin/python tools/validation/comparison_tool.py --mode online
```

**Features**:
- Dual OpenCV display windows
- Real-time displacement comparison
- Agreement flagging (red/green) with 3% threshold
- JSON logging and MSE graph generation
- Performance: 8.4 FPS offline processing

**Documentation**: [tools/validation/README.md](tools/validation/README.md)

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

- **Integration Guide:** `documentation/integration-guide.md` - Comprehensive integration documentation
- **Installation Guide:** `documentation/installation.md` - Step-by-step installation instructions
- **Integration Cheat Sheet:** `documentation/integration-cheat-sheet.md` - Quick reference for meetings

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
