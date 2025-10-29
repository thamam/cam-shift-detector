# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL: BMAD Workflow Project

**This is a BMAD (Build-Measure-Adapt-Deploy) managed project.**

**ALWAYS check workflow compliance before starting ANY development work, unless explicitly told otherwise by the user.**

### Before Starting Implementation:

1. **Check Workflow Status**: Run `/bmad:bmm:workflows:workflow-status` or check `docs/bmm-workflow-status.md`
2. **Verify Current Phase**: Ensure you're in the correct phase (Analysis → Planning → Solutioning → Implementation)
3. **Generate Story Context**: For Phase 4 (Implementation), ALWAYS generate story context first:
   - Run `/bmad:bmm:workflows:story-context` for the current story
   - Creates `docs/stories/story-{name}.context.md` with architectural guidance
4. **Load Appropriate Agent**: Use the agent specified in workflow status (pm, dev, analyst, etc.)
5. **Follow Story**: Implement according to acceptance criteria in `docs/stories/story-{name}.md`

### BMAD Workflow Phases:

```
Phase 1: Analysis (Product Brief) → docs/product-brief-*.md
Phase 2: Planning (Tech Spec) → docs/tech-spec.md, docs/epics.md, docs/stories/
Phase 3: Solutioning (Story Context) → docs/stories/*.context.md
Phase 4: Implementation (Development) → src/, tests/, tools/
Phase 5: Release Engineering (Packaging) → dist/, pyproject.toml
```

### Workflow Files:

- **Status Tracker**: `docs/bmm-workflow-status.md` (current phase, next action)
- **Epic**: `docs/epic-{name}.md` (high-level goals)
- **Stories**: `docs/stories/story-{name}.md` (requirements, ACs, tasks)
- **Story Context**: `docs/stories/story-{name}.context.md` (technical design)

### When to Skip BMAD:

- User explicitly says "skip BMAD" or "quick fix"
- Emergency hotfix or debugging
- Documentation-only changes
- Small script or tool outside main codebase

**Default behavior: ALWAYS follow BMAD workflow unless told otherwise.**

---

## Project Overview

Camera shift detection system for DAF (Dissolved Air Flotation) water quality monitoring. Detects camera position changes ≥2 pixels using feature matching and homography analysis to prevent corrupted measurements when camera ROI becomes misaligned.

**Current Status**: Stage 3 validation complete - 66% accuracy, 100% recall on 50 real DAF site images.

## Development Commands

### Environment Setup
```bash
# Python 3.11+ required
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv/Scripts/activate     # Windows

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests
.venv/bin/python -m pytest tests/

# Run specific test file
.venv/bin/python -m pytest tests/test_camera_movement_detector.py

# Run with coverage
.venv/bin/python -m pytest --cov=src --cov-report=html tests/

# Run performance tests
.venv/bin/python -m pytest tests/test_performance.py -v

# Run integration tests
.venv/bin/python -m pytest tests/test_integration.py -v
```

### Validation Framework
```bash
# Stage 1: Generate synthetic test data with known transforms
.venv/bin/python validation/data_generators/generate_stage1_data.py

# Stage 2: ChArUco board validation
.venv/bin/python validation/core/run_stage2_validation.py

# Stage 3: Full validation with real data and comprehensive reporting
.venv/bin/python validation/core/run_stage3_validation.py

# Stage 3 with custom options
.venv/bin/python validation/core/run_stage3_validation.py \
    --baseline sample_images/of_jerusalem/img_001.jpg \
    --output-dir validation/results/custom_run \
    --detector-config config/detector_config.json
```

### Tools
```bash
# Ground truth annotation tool (manual verification)
.venv/bin/python tools/annotation/ground_truth_annotator.py

# ROI selection tool (interactive GUI for defining static regions)
.venv/bin/python tools/select_roi.py

# Manual recalibration
.venv/bin/python tools/recalibrate.py

# Dual detector comparison (cam-shift vs charuco)
.venv/bin/python tools/validation/comparison_tool.py

# ArUco board printer
.venv/bin/python tools/aruco/board_printer.py
```

## Architecture

### Core Components (src/)

**Black-box API entry point:**
- `camera_movement_detector.py` - Main `CameraMovementDetector` class with 5 public methods
- `camera_movement_detector_stub.py` - Stub implementation for parallel integration development

**Internal modules (not exposed):**
- `feature_extractor.py` - ORB feature extraction using OpenCV (FeatureExtractor class)
- `movement_detector.py` - Homography-based displacement detection (MovementDetector class)
- `static_region_manager.py` - ROI mask generation from config (StaticRegionManager class)
- `result_manager.py` - Result dict formatting and history buffer (ResultManager class)

### Validation Framework (validation/)

**Validation runners:**
- `validation/core/run_stage1_validation.py` - Synthetic data validation
- `validation/core/run_stage2_validation.py` - ChArUco board validation
- `validation/core/run_stage3_validation.py` - Full validation with real data

**Test harnesses:**
- `validation/harnesses/stage1_test_harness.py`
- `validation/harnesses/stage2_test_harness.py`
- `validation/harnesses/stage3_test_harness.py`

**Utilities:**
- `validation/utilities/performance_profiler.py` - FPS, memory, CPU profiling
- `validation/utilities/report_generator.py` - JSON/Markdown report generation
- `validation/utilities/real_data_loader.py` - Load sample images from DAF sites
- `validation/utilities/comparison_metrics.py` - Detection accuracy metrics
- `validation/utilities/dual_detector_runner.py` - Parallel detector comparison

### Data Flow

```
Image → StaticRegionManager (ROI mask) → FeatureExtractor (ORB) →
MovementDetector (homography) → ResultManager (history buffer) → Result dict
```

### API Contract

**Input:** NumPy array (H×W×3, uint8, BGR), optional frame_id

**Output:** Result dict:
```python
{
  "status": "VALID" | "INVALID",
  "displacement": float,      # pixels, 2 decimals
  "confidence": float,        # [0.0, 1.0]
  "frame_id": str,
  "timestamp": str            # ISO 8601 UTC
}
```

## Configuration

**Primary config:** `config.json`
```json
{
  "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
  "threshold_pixels": 2.0,
  "history_buffer_size": 100,
  "min_features_required": 50
}
```

**Validation ground truth:** `validation/ground_truth/validation_ground_truth.json`

## Test Data

**Sample images:** 50 real DAF site images in `sample_images/`
- OF_JERUSALEM: 23 images (site: 9bc4603f-0d21-4f60-afea-b6343d372034)
- CARMIT: 17 images (site: e2336087-0143-4895-bc12-59b8b8f97790)
- GAD: 10 images (site: f10f17d4-ac26-4c28-b601-06c64b8a22a4)

Source: `/home/thh3/data/greenpipe/` (systematic sampling)

## Project Conventions

### YAGNI Enforcement
**Backlog location:** `documentation/backlog.md`

**Build only what's explicitly required** - defer sophisticated solutions to backlog:
- ❌ ML-based auto-detection → use simple geometric ROI
- ❌ Full config management → use simple config.json
- ❌ Real-time video → focus on single image detection
- ✅ Directly solves current problem
- ✅ Fixes blocking bug
- ✅ Required for current sprint/stage

### Module Design Principles
- **Black-box API:** Single `CameraMovementDetector` class, no internal state exposure
- **Synchronous execution:** Direct function calls, no async
- **Stateless operations:** Each `process_frame()` call is independent
- **No side effects:** Returns status directly, no flag files/REST calls/DB writes
- **Configuration-driven:** ROI and thresholds in `config.json`

### Documentation Organization
- **Technical specs:** `docs/tech-spec-epic-MVP-001.md`
- **Integration guide:** `documentation/integration-guide.md`
- **Installation:** `documentation/installation.md`
- **Backlog:** `documentation/backlog.md`
- **Claude artifacts:** `claudedocs/` (session summaries, investigations)
- **Validation results:** `validation/results/`

### Testing Strategy
- **Unit tests:** `tests/test_*.py` for each src module
- **Integration tests:** `tests/test_integration.py`, `tests/test_e2e.py`
- **Performance tests:** `tests/test_performance.py`
- **Validation framework:** Three-stage progressive validation (synthetic → calibrated → real)

## Key Constraints

- **Python:** ≥3.11 (uses match statements, type hints)
- **OpenCV:** ≥4.12.0.88 (ORB feature extraction, homography)
- **Deployment:** Single camera only (no multi-camera support in MVP)
- **Frequency:** 5-10 minute intervals (not real-time 1Hz)
- **Threshold:** 2.0 pixels displacement triggers INVALID status
- **Target accuracy:** ≥95% detection accuracy with ≤5% false positive rate

## Integration Context

**Parent system:** DAF water quality monitoring
**Integration point:** Camera interface (already provides NumPy BGR arrays)
**Use case:** Detect camera shifts to prevent neural network ROI misalignment
**Action on INVALID:** Halt data collection, trigger manual correction
**Recalibration:** Manual only (no automatic drift handling in MVP)

## Related Documentation

- README.md - Project status, quick start, validation framework
- documentation/integration-guide.md - Comprehensive integration documentation
- documentation/integration-cheat-sheet.md - Quick reference for meetings
- docs/tech-spec-epic-MVP-001.md - Full technical specification
- tools/annotation/README.md - Ground truth annotation tool guide
- documentation/ground-truth-annotation-results.md - Validation results analysis
