# Epic: Validation Comparison Tool

**Epic Slug**: `comparison-tool`

## Epic Goal

Enable developers and QA personnel to visually compare ChArUco 6-DOF pose estimation against cam-shift detector performance in real-time, supporting both offline validation (recorded sequences) and online debugging (live feeds).

## Epic Scope

**In Scope:**
- Side-by-side OpenCV display windows showing both detectors simultaneously
- Real-time displacement comparison with red/green agreement flagging
- Dual operation modes: offline (directory input) and online (live camera feed)
- Minimal JSON logging for post-analysis
- MSE graph generation and worst match retrieval
- Integration with existing validation system (`validation/utilities/`)
- Standalone tool capability (`tools/validation/`)

**Out of Scope:**
- Automated CI/CD integration (future enhancement)
- 3D visualization of ChArUco poses (future enhancement)
- Video recording/export of comparison sessions (future enhancement)
- Multi-session aggregate analysis (future enhancement)

## Success Criteria

**Functional:**
- ✅ Tool runs in both offline and online modes without errors
- ✅ Dual windows display simultaneously with displacement overlay
- ✅ Comparison metric correctly calculates ||d1-d2||_2 with 3% threshold
- ✅ Green/red status updates correctly based on agreement threshold
- ✅ JSON logs contain all required fields (frame_idx, displacements, comparison)
- ✅ MSE graph generated with correct data and axis labels
- ✅ Worst matches retrieval returns correct top 10 frames

**Performance:**
- ✅ Offline mode: Process 157 frames in <30 seconds (~5 FPS minimum)
- ✅ Online mode: Maintain 15-20 FPS on live camera feed
- ✅ Display latency: <100ms synchronization between detector outputs

**Quality:**
- ✅ Code coverage ≥80% for all new modules
- ✅ Documentation complete in `tools/validation/README.md`
- ✅ Single command execution for both offline and online modes

## Epic Dependencies

**External Dependencies:**
- matplotlib==3.9.2 (new dependency for MSE graph generation)

**Internal Dependencies:**
- `src.camera_movement_detector` - Cam-shift detector API
- `tools.aruco.camshift_annotator` - ChArUco pose estimation utilities
- `validation.utilities.performance_profiler` - FPS profiling

**Pre-requisites:**
- camera.yaml calibration file must exist
- ChArUco board configuration (7×5, 35mm squares, 26mm markers)
- Detector config.json with ROI and thresholds

## Story Map

```
Epic: Validation Comparison Tool (8 points)
├── Story 1: Core Comparison Infrastructure (5 points)
│   ├── DualDetectorRunner orchestration
│   ├── ComparisonMetrics calculations
│   └── ComparisonLogger with analysis
│
└── Story 2: Tool Integration & Testing (3 points)
    ├── Main comparison_tool.py executable
    ├── Offline and online mode implementation
    ├── Display windows with OpenCV
    └── Comprehensive testing and documentation
```

**Total Story Points:** 8
**Estimated Timeline:** 1 sprint (1 week at 1-2 points/day)

## Implementation Sequence

**Story 1 (Core Infrastructure) - First**
- Build dual detector orchestration layer
- Implement comparison metric calculations
- Create logging and analysis framework
- Unit tests for all core modules

**Story 2 (Tool Integration) - Second** (Depends on Story 1)
- Create main executable with CLI
- Implement offline mode (directory processing)
- Implement online mode (live camera feed)
- Create OpenCV display windows
- Integration testing with session_001 data
- Complete documentation

## Story Summaries

### Story 1: Core Comparison Infrastructure (5 points)
**As a** developer,
**I want** dual detector orchestration and comparison metrics,
**so that** I can measure agreement between ChArUco and cam-shift detectors systematically.

**Key Deliverables:**
- `validation/utilities/dual_detector_runner.py` - Parallel detector execution
- `validation/utilities/comparison_metrics.py` - ||d1-d2||_2 calculations
- `validation/utilities/comparison_logger.py` - JSON logging and MSE analysis
- Unit tests with ≥80% coverage

**Time Estimate:** 3-5 days

---

### Story 2: Tool Integration & Testing (3 points)
**As a** QA personnel,
**I want** a standalone comparison tool with dual display windows,
**so that** I can visually validate detector agreement in offline and online modes.

**Key Deliverables:**
- `tools/validation/comparison_tool.py` - Main executable
- `tools/validation/README.md` - Usage documentation
- `comparison_config.json` - Default configuration
- Offline mode: directory processing
- Online mode: live camera feed
- Integration tests with session_001 data

**Time Estimate:** 2-3 days

---

**Epic Status:** Ready for Implementation
**Next Action:** Generate story context for Story 1
**Documents:** tech-spec.md (technical reference), story-comparison-tool-1.md, story-comparison-tool-2.md
