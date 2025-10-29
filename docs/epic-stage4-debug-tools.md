# Epic: Stage 4 - Interactive Debugging Tools

**Epic ID:** STAGE4-DEBUG-TOOLS
**Status:** TODO
**Priority:** High
**Start Date:** 2025-10-28
**Target Date:** 2025-11-01 (4 days)

## Epic Goal

Provide developers and QA engineers with three interactive OpenCV-based debugging tools for real-time camera shift analysis, feature correspondence visualization, and visual alignment verification.

## Problem Statement

Current validation tools operate in batch mode with post-processing analysis. Developers need interactive debugging capabilities to:
- Compare ChArUco and CSD detectors side-by-side with feature overlays
- Visualize motion vectors and match quality in real-time
- Manually verify frame alignment with alpha blending

Without these tools, debugging detector behavior requires manual inspection of JSON logs and static images, making it difficult to understand dynamic behavior and identify edge cases.

## Solution Overview

Build three OpenCV-based desktop tools:
- **Mode A:** 4-image side-by-side comparison with feature overlays
- **Mode B:** Baseline correspondence with motion vectors
- **Mode C:** Enhanced alpha blending with transform pre-warp

All tools support manual frame stepping (FWD/BKWD), live computation, and CSV/PNG export.

## Success Criteria

- ✅ Load 10-200 frames with smooth manual stepping
- ✅ Mode A: 4-image layout with ChArUco/ORB feature overlays
- ✅ Mode B: Motion vectors display with inlier/outlier coloring
- ✅ Mode C: Transform computation with pre-warp and blink toggle
- ✅ CSV export with all displacement metrics
- ✅ PNG snapshot capability

## Technical Approach

**Architecture:** Pure Python + OpenCV (no web UI)
- Direct module calls (no REST API overhead)
- OpenCV window controls (keyboard-driven)
- Reuse existing comparison_tool.py and annotator.py infrastructure
- Extend with new visualization functions

**Implementation Strategy:**
- Day 1: Enhance comparison_tool.py → Mode A
- Day 2: Build baseline_correspondence_tool.py → Mode B
- Day 3: Enhance ground_truth_annotator.py → Mode C
- Day 4: Testing, CSV export, documentation

## Stories

### Story 1: Mode A - 4-Image Comparison with Feature Overlays (5 points)
**File:** `docs/stories/story-stage4-mode-a.md`

Enhance existing comparison_tool.py with:
- 4-quadrant layout (baseline/current × charuco/csd)
- ChArUco corner visualization
- ORB feature visualization
- Manual frame stepping (FWD/BKWD)
- Enhanced metrics (Δdx, Δdy, error magnitude)

### Story 2: Mode B - Baseline Correspondence Tool (3 points)
**File:** `docs/stories/story-stage4-mode-b.md`

New standalone tool for baseline analysis:
- Baseline pinning mechanism
- Motion vector visualization (arrows)
- Inlier/outlier coloring (green/red)
- RMSE and match quality display
- Manual frame stepping

### Story 3: Mode C - Enhanced Alpha Blending (2 points)
**File:** `docs/stories/story-stage4-mode-c.md`

Enhance existing annotator.py with:
- Transform computation using CSD
- Pre-warp toggle
- Blink toggle (rapid A/B switching)
- Frame selector for arbitrary pairs

## Total Story Points: 10 points

## Dependencies

**Existing Infrastructure:**
- ✅ tools/validation/comparison_tool.py (Epic 3)
- ✅ tools/annotation/ground_truth_annotator.py
- ✅ src/camera_movement_detector.py (CSD module)
- ✅ validation/utilities/dual_detector_runner.py
- ✅ ChArUco detection capability

**New Requirements:**
- Feature coordinate extraction from CSD module
- Corner coordinate extraction from ChArUco detector
- Motion vector drawing utilities
- Transform pre-warp implementation

## Out of Scope

❌ Web UI / HTML5 interface
❌ FastAPI backend
❌ REST API endpoints
❌ Real-time streaming (>30 FPS)
❌ Video encoding/decoding (use frames only)
❌ Database storage
❌ Cloud deployment

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Feature extraction performance | Medium | Use existing optimized ORB implementation |
| Display lag with 4-image layout | Low | Resize windows if needed, cache frames |
| ChArUco detection failures | Medium | Fallback to CSD-only mode |
| Memory usage with large sequences | Low | Frame caching with LRU eviction |

## Acceptance Criteria

- All 3 story acceptance criteria met
- Tools tested with 10-200 frame sequences
- Documentation complete with usage examples
- CSV export validated against spec
- No regressions in existing validation tools

## Related Documentation

- Requirements: `docs/Stage_4_prompt.md`
- Gap Analysis: Initial analysis (2025-10-28)
- Existing Tools: `tools/validation/README.md`

---

**Next Steps:**
1. Create story-stage4-mode-a.md
2. Create story-stage4-mode-b.md
3. Create story-stage4-mode-c.md
4. Update workflow status to start Epic 4
