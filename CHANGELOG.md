# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-10-26

### Added

**Epic 1: Core Camera Shift Detection (100% detection on synthetic data)**
- Camera shift detection module using ORB feature matching with homography estimation
- Affine transformation model for improved detection accuracy
- Static ROI (Region of Interest) configuration and selection tool
- Manual recalibration capability for baseline reset
- Result history buffer maintaining last 100 detection results
- Configuration-driven threshold system (default: 2.0 pixels)
- JSON-based configuration file support
- Comprehensive unit test suite (109 tests, 93% coverage)
- Black-box API with 5 public methods:
  - `__init__(config_path)` - Initialize detector with configuration
  - `set_baseline(image_array)` - Capture reference baseline features
  - `process_frame(image_array, frame_id)` - Detect camera movement in frame
  - `recalibrate(image_array)` - Reset baseline features manually
  - `get_history(frame_id, limit)` - Query detection history

**Epic 2: Production Validation Framework**
- Real DAF image data loader with 50 sample images from 3 production sites
  - OF_JERUSALEM: 23 images (site: 9bc4603f-0d21-4f60-afea-b6343d372034)
  - CARMIT: 17 images (site: e2336087-0143-4895-bc12-59b8b8f97790)
  - GAD: 10 images (site: f10f17d4-ac26-4c28-b601-06c64b8a22a4)
- Ground truth annotation system with JSON-based validation data
- Stage 3 validation test harness with automated testing
- Performance profiling capabilities:
  - FPS (frames per second) measurement
  - Memory usage tracking (psutil integration)
  - CPU utilization monitoring
- Automated validation runner (`run_stage3_validation.py`)
- Dual-format reporting system:
  - JSON report (machine-readable metrics)
  - Markdown report (human-readable analysis)
- Go/no-go production readiness assessment with gate criteria:
  - Detection accuracy ≥95%
  - False positive rate ≤5%
  - Processing speed ≥0.0167 FPS (1 frame per 60 seconds)
  - Memory usage ≤500 MB

**Release Preparation**
- Comprehensive integration documentation (`documentation/integration-guide.md`)
- Step-by-step installation guide (`documentation/installation.md`)
- Integration cheat sheet for quick reference (`documentation/integration-cheat-sheet.md`)
- Stub implementation for parallel integration development
- Professional package metadata (pyproject.toml with PEP 621 compliance)
- MIT License for open source distribution
- Complete CHANGELOG following "Keep a Changelog" format

### Known Limitations

- **Single Camera Support**: Multi-camera scenarios not supported in v0.1.0 (deferred to future releases)
- **Manual Ground Truth**: Requires manual annotation for validation (no automatic ground truth generation)
- **Manual Recalibration Only**: No automatic recalibration on lighting changes (operator must trigger manually)
- **Static ROI**: ROI cannot be adjusted at runtime (requires configuration file update and restart)
- **CPU-Only Processing**: No GPU acceleration (sufficient for 1 frame/60s requirement)

### Technical Details

**Detection Algorithm:**
- ORB (Oriented FAST and Rotated BRIEF) feature detection
- FLANN-based feature matching with ratio test
- RANSAC-based affine transformation estimation
- Displacement calculation from transformation matrix

**Performance Targets:**
- Detection accuracy: ≥95% on real DAF imagery
- Processing speed: ≥0.0167 FPS (1 frame per 60 seconds)
- Memory footprint: ≤500 MB
- False positive rate: ≤5%

**Testing Coverage:**
- Unit tests: 109 tests, 93% code coverage
- Integration tests: Stage 3 validation framework
- Real-world validation: 50 DAF production images

For full technical specifications, see [docs/tech-spec-epic-MVP-001.md](docs/tech-spec-epic-MVP-001.md).

For integration guidance, see [documentation/integration-guide.md](documentation/integration-guide.md).

---

## Version History

- **0.1.0** (2025-10-26) - Initial release with core detection and validation framework
