# cam-shift-detector - Technical Specification
## Stage 3 Validation Framework

**Author:** Tomer
**Date:** 2025-10-25
**Project Level:** 1 (Coherent Feature)
**Project Type:** software
**Development Context:** Brownfield - Extending existing detection system

---

## Source Tree Structure

```
cam-shift-detector/
├── validation/                           # Stage 3 validation framework
│   ├── __init__.py
│   ├── real_data_loader.py              # NEW: Load real DAF imagery
│   ├── stage3_test_harness.py           # NEW: Execute detector + ground truth comparison
│   ├── performance_profiler.py          # NEW: FPS, memory, CPU measurement
│   ├── run_stage3_validation.py         # NEW: Orchestrate complete validation suite
│   ├── ground_truth/                    # NEW: Ground truth annotations
│   │   ├── ground_truth.json            # Manual annotations for 50 samples
│   │   └── annotation_schema.json       # Schema definition
│   └── results/                         # NEW: Validation outputs
│       ├── validation_report.json       # Machine-readable metrics
│       ├── validation_report.md         # Human-readable report
│       └── failure_analysis/            # Failed detection examples
├── sample_images/                       # EXISTING: Real DAF imagery
│   ├── of_jerusalem/                    # 23 images
│   ├── carmit/                          # 17 images
│   └── gad/                             # 10 images
├── src/                                 # EXISTING: Core detection system
│   ├── static_region_manager.py         # Reused for validation
│   ├── feature_extractor.py             # Reused for validation
│   ├── movement_detector.py             # Reused for validation
│   ├── result_manager.py                # Reused for validation
│   └── camera_movement_detector.py      # Main API - under test
└── tests/                               # EXISTING: Unit/integration tests
    └── validation/                      # NEW: Stage 3 validation tests
        └── test_validation_framework.py # Framework integrity tests
```

---

## Technical Approach

### Architecture Overview

**Design Philosophy:** Lightweight validation harness that reuses Epic 1 detection components while adding systematic testing infrastructure.

**Core Components:**

1. **Real Data Loader** (`real_data_loader.py`)
   - **Purpose:** Ingest 50 DAF sample images from 3 sites
   - **Responsibilities:**
     - Load images from `sample_images/` directory structure
     - Metadata extraction (site ID, timestamp from filename)
     - Image validation (format, dimensions, quality checks)
   - **Output:** Structured dataset ready for validation

2. **Stage 3 Test Harness** (`stage3_test_harness.py`)
   - **Purpose:** Execute detector and compare against ground truth
   - **Responsibilities:**
     - Invoke `CameraMovementDetector` API for each image
     - Load ground truth annotations
     - Calculate accuracy metrics (TP, TN, FP, FN)
     - Generate per-image and aggregate results
   - **Output:** Detection results + accuracy metrics

3. **Performance Profiler** (`performance_profiler.py`)
   - **Purpose:** Measure system performance on target hardware
   - **Responsibilities:**
     - FPS measurement (frames processed per second)
     - Memory profiling (peak usage, sustained usage)
     - CPU usage tracking (single-core, multi-core if applicable)
   - **Output:** Performance benchmarks

4. **Validation Runner** (`run_stage3_validation.py`)
   - **Purpose:** Orchestrate complete validation workflow
   - **Responsibilities:**
     - Sequential execution: Load data → Run tests → Profile performance → Generate reports
     - Error handling and graceful degradation
     - Progress reporting
   - **Output:** Complete validation report

### Validation Flow

```
1. Initialize
   ├─ Load 50 sample images from sample_images/
   └─ Load ground_truth.json annotations

2. Execute Detection
   ├─ For each image:
   │  ├─ Run CameraMovementDetector.detect()
   │  ├─ Record detection result (shift/no-shift)
   │  └─ Compare with ground truth
   └─ Aggregate results

3. Calculate Metrics
   ├─ Detection accuracy (TP + TN) / Total
   ├─ False positive rate: FP / (FP + TN)
   ├─ False negative rate: FN / (FN + TP)
   └─ Per-site breakdown

4. Profile Performance
   ├─ Measure FPS (target: ≥1/60 Hz = 0.0167 FPS)
   ├─ Measure memory (target: ≤500 MB)
   └─ Measure CPU usage

5. Generate Report
   ├─ JSON report (machine-readable)
   ├─ Markdown report (human-readable)
   └─ Failure analysis (if any)

6. Go/No-Go Recommendation
   └─ Based on gate criteria: accuracy ≥95%, FP ≤5%, performance within limits
```

---

## Implementation Stack

### Core Technology (Existing - Reused)

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV 4.8.0
- **Numerical Computing:** NumPy 1.24.3
- **Feature Detection:** SIFT (OpenCV implementation)
- **Transformation Model:** Affine transformation (existing optimized thresholds from Epic 1)

### New Dependencies (Stage 3 Specific)

- **Performance Profiling:**
  - `psutil==5.9.5` - CPU and memory monitoring
  - `memory_profiler==0.61.0` - Detailed memory profiling
  - Standard library `time` module for FPS measurement

- **Data Management:**
  - `json` (stdlib) - Ground truth storage and report generation
  - `pathlib` (stdlib) - Cross-platform path handling

- **Testing:**
  - `pytest==7.4.0` (existing) - Framework integrity tests

### Platform Requirements

- **OS:** Linux (production environment)
- **Memory:** 500 MB maximum constraint
- **CPU:** CPU-only processing (edge deployment)
- **Python:** 3.8+ runtime environment

---

## Technical Details

### 1. Real Data Loader Implementation

**Class:** `RealDataLoader`

**Key Methods:**
- `load_dataset()` → List[ImageMetadata]
  - Scans `sample_images/` directory structure
  - Returns list of 50 image paths with metadata

- `load_image(path: str)` → np.ndarray
  - Loads image using OpenCV
  - Validates format and dimensions
  - Returns image array

**Data Structure:**
```python
@dataclass
class ImageMetadata:
    image_path: Path
    site_id: str  # 'of_jerusalem', 'carmit', 'gad'
    timestamp: datetime  # Extracted from filename
    has_shift: bool  # From ground truth
```

### 2. Ground Truth Format

**File:** `validation/ground_truth/ground_truth.json`

**Schema:**
```json
{
  "version": "1.0",
  "annotator": "Tomer",
  "annotation_date": "2025-10-25",
  "images": [
    {
      "image_path": "sample_images/of_jerusalem/image_001.jpg",
      "site_id": "of_jerusalem",
      "has_camera_shift": true|false,
      "confidence": "high|medium|low",
      "notes": "Optional annotation notes"
    }
  ]
}
```

**Annotation Process:** Manual review of 50 images to label shift/no-shift

### 3. Test Harness Implementation

**Class:** `Stage3TestHarness`

**Key Methods:**
- `run_validation()` → ValidationResults
  - Executes detector on all 50 images
  - Compares with ground truth
  - Returns comprehensive results

- `calculate_metrics(results: List[Detection])` → Metrics
  - Computes accuracy, FP rate, FN rate
  - Per-site breakdown
  - Confusion matrix

**Detection Result:**
```python
@dataclass
class DetectionResult:
    image_path: Path
    ground_truth: bool
    predicted: bool
    is_correct: bool
    detection_time_ms: float
```

### 4. Performance Profiler Implementation

**Class:** `PerformanceProfiler`

**Key Metrics:**
- **FPS Measurement:**
  ```python
  start_time = time.time()
  detector.detect(image)
  elapsed = time.time() - start_time
  fps = 1.0 / elapsed
  ```

- **Memory Profiling:**
  ```python
  import psutil
  process = psutil.Process()
  memory_info = process.memory_info()
  memory_mb = memory_info.rss / 1024 / 1024
  ```

- **CPU Usage:**
  ```python
  cpu_percent = psutil.cpu_percent(interval=1)
  ```

**Target Benchmarks:**
- FPS: ≥1/60 Hz (0.0167 FPS minimum)
- Memory: ≤500 MB
- CPU: Measured for baseline (no hard target)

### 5. Validation Report Format

**JSON Report** (`validation_report.json`):
```json
{
  "validation_date": "2025-10-25",
  "total_images": 50,
  "metrics": {
    "accuracy": 0.96,
    "false_positive_rate": 0.04,
    "false_negative_rate": 0.00
  },
  "performance": {
    "mean_fps": 0.025,
    "peak_memory_mb": 420,
    "mean_cpu_percent": 45.2
  },
  "site_breakdown": {
    "of_jerusalem": {"accuracy": 0.96, "n_images": 23},
    "carmit": {"accuracy": 0.94, "n_images": 17},
    "gad": {"accuracy": 1.00, "n_images": 10}
  },
  "go_no_go": {
    "recommendation": "GO|NO-GO",
    "gate_criteria_met": true|false,
    "rationale": "Explanation"
  }
}
```

**Markdown Report** (`validation_report.md`):
- Executive summary
- Metrics visualization (accuracy, FP/FN rates)
- Performance benchmarks
- Failure analysis (if applicable)
- Go/no-go recommendation with supporting evidence

---

## Development Setup

### Environment Preparation

1. **Activate existing environment:**
   ```bash
   # Assuming venv already exists from Epic 1
   source venv/bin/activate
   ```

2. **Install Stage 3 dependencies:**
   ```bash
   pip install psutil==5.9.5 memory_profiler==0.61.0
   ```

3. **Verify sample images:**
   ```bash
   ls sample_images/*/*.jpg | wc -l  # Should return 50
   ```

### Ground Truth Creation

**Manual annotation process:**
1. Review each of 50 sample images
2. Compare with adjacent images (if available in dataset)
3. Label as shift/no-shift based on visual inspection
4. Record in `ground_truth.json` following schema
5. Review for quality assurance

**Annotation Guidelines:**
- Shift = visible camera position change (translation, rotation, scale)
- No-shift = stable camera position across time
- Confidence levels: high (clear), medium (subtle), low (uncertain)

---

## Implementation Guide

### Phase 1: Infrastructure Setup

**Duration:** 1 day

**Tasks:**
1. Create `validation/` directory structure
2. Set up ground truth JSON schema
3. Create empty Python files with docstrings
4. Add pytest configuration for validation tests

### Phase 2: Data Loading + Ground Truth

**Duration:** 1 day

**Tasks:**
1. Implement `RealDataLoader` class
   - Directory scanning
   - Metadata extraction
   - Image loading

2. Create ground truth annotations
   - Manual review of 50 images
   - Populate `ground_truth.json`
   - Validate schema

3. Test data loader
   - Verify all 50 images load correctly
   - Validate metadata accuracy

### Phase 3: Test Harness Implementation

**Duration:** 2 days

**Tasks:**
1. Implement `Stage3TestHarness` class
   - Detection execution loop
   - Ground truth comparison
   - Metrics calculation

2. Add confusion matrix generation

3. Per-site breakdown logic

4. Test harness with mock data
   - Verify metrics calculation
   - Test edge cases (all correct, all wrong, etc.)

### Phase 4: Performance Profiling

**Duration:** 1 day

**Tasks:**
1. Implement `PerformanceProfiler` class
   - FPS measurement
   - Memory profiling
   - CPU tracking

2. Integration with test harness

3. Verify measurements on target hardware

### Phase 5: Validation Runner + Reports

**Duration:** 1 day

**Tasks:**
1. Implement `run_stage3_validation.py` orchestration
   - Sequential workflow execution
   - Error handling
   - Progress reporting

2. JSON report generation

3. Markdown report generation

4. Go/no-go decision logic

### Phase 6: Testing + Documentation

**Duration:** 1 day

**Tasks:**
1. Write framework integrity tests
   - Test data loader
   - Test metrics calculation
   - Test report generation

2. Run complete validation suite

3. Review and refine reports

4. Update project documentation

**Total Estimated Duration:** 7 days

---

## Testing Approach

### 1. Framework Integrity Tests

**Location:** `tests/validation/test_validation_framework.py`

**Test Coverage:**
- **Data Loader Tests:**
  - Test loading all 50 images
  - Test metadata extraction
  - Test invalid image handling

- **Metrics Calculation Tests:**
  - Test accuracy calculation (100% correct, 100% wrong, mixed)
  - Test FP/FN rate calculation
  - Test confusion matrix generation

- **Report Generation Tests:**
  - Test JSON report structure
  - Test markdown report formatting
  - Test go/no-go decision logic

### 2. Integration Testing

**End-to-End Validation:**
```bash
# Run complete Stage 3 validation suite
python validation/run_stage3_validation.py

# Verify outputs:
# - validation/results/validation_report.json exists
# - validation/results/validation_report.md exists
# - Reports contain all required sections
```

### 3. Performance Validation

**Benchmark Verification:**
- Run validation on production-equivalent hardware
- Verify FPS ≥ 1/60 Hz threshold
- Verify memory usage ≤ 500 MB
- Compare results with targets from product brief

### 4. Manual Review

- Review generated reports for clarity and completeness
- Verify go/no-go recommendation aligns with gate criteria
- Validate failure analysis (if any failures detected)

---

## Deployment Strategy

### Stage 3 Validation Execution

**Environment:** Linux production-equivalent system (500 MB RAM constraint)

**Deployment Steps:**

1. **Pre-Deployment Validation:**
   ```bash
   # Verify environment
   python --version  # Confirm 3.8+
   pip list | grep -E "opencv|numpy|psutil|memory_profiler"

   # Verify sample images
   ls -lh sample_images/*/*.jpg | wc -l  # Should be 50

   # Verify ground truth exists
   cat validation/ground_truth/ground_truth.json | jq '.images | length'  # Should be 50
   ```

2. **Execute Validation:**
   ```bash
   # Run complete Stage 3 validation
   python validation/run_stage3_validation.py

   # Monitor execution
   # - Expected duration: ~30-60 seconds for 50 images
   # - Progress updates printed to console
   ```

3. **Review Results:**
   ```bash
   # Check JSON report
   cat validation/results/validation_report.json | jq '.go_no_go'

   # Review markdown report
   less validation/results/validation_report.md
   ```

4. **Decision Gate:**
   - **GO Criteria Met (accuracy ≥95%, FP ≤5%, performance OK):**
     - Approve Sprint 2 (production deployment)
     - Proceed to Epic 3 planning

   - **NO-GO Criteria Not Met:**
     - Document failure modes
     - Create algorithm refinement epic
     - Iterate on detection parameters

### Production Integration (Post-Validation)

**Note:** Production deployment (24/7 monitoring, alerting) is **Sprint 2** scope, not Stage 3.

Stage 3 deliverable is the **validation report** that enables the production deployment decision.

---

## Acceptance Criteria

**Stage 3 Validation Framework is complete when:**

1. ✅ All 50 sample images processed successfully
2. ✅ Ground truth annotations completed and validated
3. ✅ Detection accuracy calculated and documented
4. ✅ False positive/negative rates calculated
5. ✅ Performance benchmarks measured (FPS, memory, CPU)
6. ✅ JSON and Markdown reports generated
7. ✅ Go/no-go recommendation provided with supporting evidence
8. ✅ Failure modes documented (if any)
9. ✅ Framework integrity tests passing
10. ✅ Workflow status updated to reflect Phase 2 completion

**Go/No-Go Gate Criteria:**

- **Detection Accuracy:** ≥95% on real DAF imagery
- **False Positive Rate:** ≤5%
- **Processing Performance:** ≥1 frame per 60 seconds (1/60 Hz)
- **Memory Usage:** ≤500 MB

**Sprint 2 Unblocked:** Validation report provides quantifiable confidence for production deployment decision.

---

**📋 Technical Specification Complete**

**Next Steps:**
1. Generate Epic and user stories for Level 1 implementation
2. Proceed to Phase 3 (Architecture/Solutioning) OR
3. Begin implementation with story tracking
