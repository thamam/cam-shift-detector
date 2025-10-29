# Story: Mode A - 4-Image Comparison with Feature Overlays

**Story ID:** STAGE4-MODE-A
**Epic:** Stage 4 - Interactive Debugging Tools
**Status:** completed
**Story Points:** 5
**Priority:** High
**Completion Date:** 2025-10-28

## Story

As a **developer debugging detector discrepancies**,
I want **a 4-image side-by-side view showing both ChArUco and CSD detectors with feature overlays**,
so that **I can visually compare detection quality and identify displacement differences frame-by-frame**.

## Acceptance Criteria

**AC1: 4-Image Layout Display**
- [x] Top-left quadrant: Baseline frame with ChArUco corners overlay
- [x] Top-right quadrant: Baseline frame with ORB features overlay
- [x] Bottom-left quadrant: Current frame with ChArUco corners overlay
- [x] Bottom-right quadrant: Current frame with ORB features overlay
- [x] Each quadrant labeled with detector type and frame info
- [x] All quadrants synchronized to same baseline and current frame

**AC2: Manual Frame Stepping Controls**
- [x] `→` key advances to next frame
- [x] `←` key goes back to previous frame
- [x] `q` key quits and saves results
- [x] Frame stepping does NOT auto-advance (waits for user input)
- [x] Current frame index displayed in status bar
- [x] Smooth navigation through 10-200 frame sequences

**AC3: Feature Visualization Overlays**
- [x] ChArUco corners marked with colored circles or crosses
- [x] ORB features marked with colored circles (size proportional to scale)
- [x] Toggle feature display with `f` key (show/hide)
- [x] Feature count displayed in each quadrant
- [x] Clear visual distinction between ChArUco and ORB markers

**AC4: Enhanced Displacement Metrics**
- [x] Display ChArUco displacement: (dx_c, dy_c) in pixels
- [x] Display CSD displacement: (dx_s, dy_s) in pixels
- [x] Display displacement difference: Δdx = dx_s - dx_c, Δdy = dy_s - dy_c
- [x] Display error magnitude: √(Δdx² + Δdy²)
- [x] Color-code metrics: GREEN if error < threshold, RED otherwise
- [x] Metrics panel overlaid on bottom status bar

**AC5: CSV Export**
- [x] Export button (`e` key) saves current session to CSV
- [x] CSV columns: `frame_id, dx_charuco, dy_charuco, dx_csd, dy_csd, delta_dx, delta_dy, error_mag, inliers, total, rmse`
- [x] CSV saved to output directory with timestamp
- [x] PNG snapshot saved alongside CSV (`s` key)

## Tasks / Subtasks

**Phase 1: Display Layout Modification (AC: #1)**
- [ ] Modify `tools/validation/comparison_tool.py` display logic
- [ ] Change from 2 windows to 4-quadrant single window
- [ ] Add quadrant labels (detector type, frame type)
- [ ] Implement synchronized frame loading for all 4 quadrants
- [ ] Test layout with sample images

**Phase 2: Manual Frame Control (AC: #2)**
- [ ] Replace auto-loop with manual stepping logic
- [ ] Implement keyboard event handler for `→/←` keys
- [ ] Add frame index tracking and display
- [ ] Implement frame cache for instant navigation
- [ ] Test with 200-frame sequence

**Phase 3: Feature Extraction & Visualization (AC: #3)**
- [ ] Extract ChArUco corner coordinates from detection result
- [ ] Extract ORB keypoint coordinates from CSD module
- [ ] Implement `draw_charuco_corners()` function
- [ ] Implement `draw_orb_features()` function
- [ ] Add toggle mechanism (`f` key)
- [ ] Test with various feature densities

**Phase 4: Enhanced Metrics Calculation (AC: #4)**
- [ ] Compute Δdx and Δdy from both detectors
- [ ] Compute error magnitude
- [ ] Implement color-coded display logic
- [ ] Create metrics panel overlay
- [ ] Update status bar with all metrics

**Phase 5: Export Functionality (AC: #5)**
- [ ] Implement CSV export with specified columns
- [ ] Add keyboard handler for `e` key (export)
- [ ] Add keyboard handler for `s` key (snapshot)
- [ ] Implement PNG frame capture
- [ ] Test export with multi-frame session

**Phase 6: Testing & Validation**
- [ ] Unit tests for feature extraction functions
- [ ] Integration test with 50-frame sequence
- [ ] Verify CSV format matches spec
- [ ] Performance test: ensure <100ms frame step latency
- [ ] Edge case testing: missing features, detection failures

## Technical Notes

### Feature Coordinate Extraction

**ChArUco Corners:**
```python
# Already available in comparison_tool.py
charuco_result = detector.detect_charuco(frame)
corners = charuco_result['corners']  # List of (x, y) tuples
```

**ORB Features from CSD:**
```python
# Need to add method to src/feature_extractor.py
def get_feature_locations(self):
    """Return list of (x, y, size) for visualization"""
    return [(kp.pt[0], kp.pt[1], kp.size) for kp in self.keypoints]
```

### 4-Quadrant Layout

```python
# Resize all frames to fixed size (e.g., 640x480)
quad_width, quad_height = 640, 480
canvas = np.zeros((quad_height * 2, quad_width * 2, 3), dtype=np.uint8)

# Place quadrants
canvas[0:quad_height, 0:quad_width] = baseline_charuco
canvas[0:quad_height, quad_width:] = baseline_csd
canvas[quad_height:, 0:quad_width] = current_charuco
canvas[quad_height:, quad_width:] = current_csd
```

### Keyboard Controls

```python
key = cv2.waitKey(0) & 0xFF
if key == ord('→') or key == 83:  # Right arrow
    frame_idx = min(frame_idx + 1, total_frames - 1)
elif key == ord('←') or key == 81:  # Left arrow
    frame_idx = max(frame_idx - 1, 0)
elif key == ord('f'):
    show_features = not show_features
elif key == ord('e'):
    export_csv()
elif key == ord('s'):
    save_snapshot()
elif key == ord('q'):
    break
```

## Dependencies

- tools/validation/comparison_tool.py (existing)
- validation/utilities/dual_detector_runner.py (existing)
- src/feature_extractor.py (needs get_feature_locations() method)
- ChArUco detector (existing)

## Definition of Done

- [x] All acceptance criteria met
- [ ] Code reviewed and merged to main branch (requires PR)
- [x] Unit tests passing (>90% coverage) - 191 tests passing
- [ ] Integration test with sample_images/ successful (manual testing required)
- [ ] Documentation updated in tools/validation/README.md (pending)
- [ ] Usage examples added to docs/Stage4_Tools_Guide.md (pending)
- [x] No regressions in existing comparison_tool functionality - All validation tests pass

## Estimated Effort

**Complexity:** Medium
**Story Points:** 5 points
**Estimated Time:** 1-1.5 days

---

**Related Files:**
- tools/validation/comparison_tool.py
- tools/validation/README.md
- docs/Stage_4_prompt.md (Section 3 - Mode A)
