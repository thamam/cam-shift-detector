# Story: Mode B - Baseline Correspondence Tool

**Story ID:** STAGE4-MODE-B
**Epic:** Stage 4 - Interactive Debugging Tools
**Status:** TODO
**Story Points:** 3
**Priority:** High

## Story

As a **QA engineer investigating feature matching quality**,
I want **a tool that shows motion vectors and match correspondences against a pinned baseline**,
so that **I can visualize inliers, outliers, and match quality for each frame in the sequence**.

## Acceptance Criteria

**AC1: Baseline Pinning Mechanism**
- [ ] User can press `b` key to set current frame as baseline
- [ ] Baseline confirmation message displayed on screen
- [ ] Baseline frame stored and used for all subsequent comparisons
- [ ] Baseline can be changed at any time by pressing `b` again
- [ ] Baseline frame thumbnail displayed in corner of window

**AC2: Motion Vector Visualization**
- [ ] Arrows drawn from baseline feature locations to current frame locations
- [ ] Inlier arrows colored GREEN
- [ ] Outlier arrows colored RED
- [ ] Arrow thickness proportional to match confidence (optional)
- [ ] Toggle motion vectors with `v` key (show/hide)
- [ ] Minimum 50 matches required for visualization

**AC3: Match Quality Metrics Display**
- [ ] Display inlier count / total matches (e.g., "142/210 inliers")
- [ ] Display RMSE value in pixels
- [ ] Display confidence score from CSD detector
- [ ] Display inlier ratio as percentage (e.g., "67.6%")
- [ ] Color-code metrics: GREEN if ratio > 80%, ORANGE if 50-80%, RED if < 50%
- [ ] Metrics overlaid on top-right corner of frame

**AC4: Manual Frame Stepping**
- [ ] `→` key advances to next frame
- [ ] `←` key goes back to previous frame
- [ ] `q` key quits and saves results
- [ ] Frame index displayed in status bar
- [ ] Baseline always compared against current frame (not sequential pairs)

**AC5: Difference Heatmap (Optional)**
- [ ] `h` key toggles heatmap view
- [ ] Heatmap shows pixel-wise difference between warped baseline and current
- [ ] Color scale: blue (low difference) to red (high difference)
- [ ] Heatmap overlaid with 50% transparency

## Tasks / Subtasks

**Phase 1: New Tool Scaffolding (AC: #1, #4)**
- [ ] Create `tools/validation/baseline_correspondence_tool.py`
- [ ] Implement argument parser (input_dir, config paths, output_dir)
- [ ] Implement frame loader (reuse from comparison_tool.py)
- [ ] Implement manual stepping controls (`→/←` keys)
- [ ] Implement baseline pinning (`b` key)
- [ ] Test with sample_images/ directory

**Phase 2: CSD Match Extraction (AC: #2)**
- [ ] Call CSD detector with baseline and current frame
- [ ] Extract match correspondences (x0, y0, x1, y1, inlier flag)
- [ ] Implement motion vector drawing function
- [ ] Draw arrows with color-coding (green/red)
- [ ] Add toggle mechanism (`v` key)
- [ ] Test with known shifted pairs

**Phase 3: Match Quality Metrics (AC: #3)**
- [ ] Calculate inlier count / total matches
- [ ] Calculate inlier ratio percentage
- [ ] Extract RMSE from CSD detector result
- [ ] Implement metrics overlay panel
- [ ] Add color-coding logic
- [ ] Test with various quality levels

**Phase 4: Difference Heatmap (AC: #5, Optional)**
- [ ] Compute homography from baseline to current
- [ ] Warp baseline frame using homography
- [ ] Calculate pixel-wise absolute difference
- [ ] Apply color map (cv2.applyColorMap)
- [ ] Implement toggle mechanism (`h` key)
- [ ] Test heatmap visualization

**Phase 5: Testing & Export**
- [ ] Unit tests for motion vector drawing
- [ ] Integration test with 50-frame sequence
- [ ] CSV export with match quality metrics
- [ ] PNG snapshot capability
- [ ] Performance test: ensure <100ms per frame

## Technical Notes

### Match Correspondence Extraction

```python
# Modify CSD detector to return match details
result = csd_detector.process_frame(current_frame, frame_id)

# Add new method to CSD module
def get_match_correspondences(self):
    """Return list of match correspondences for visualization"""
    return [
        {
            'x0': baseline_kp.pt[0],
            'y0': baseline_kp.pt[1],
            'x1': current_kp.pt[0],
            'y1': current_kp.pt[1],
            'inlier': is_inlier(match)
        }
        for match in self.matches
    ]
```

### Motion Vector Drawing

```python
def draw_motion_vectors(image, matches):
    """Draw arrows for each match correspondence"""
    for match in matches:
        pt1 = (int(match['x0']), int(match['y0']))
        pt2 = (int(match['x1']), int(match['y1']))
        color = (0, 255, 0) if match['inlier'] else (0, 0, 255)
        cv2.arrowedLine(image, pt1, pt2, color, 2, tipLength=0.3)
    return image
```

### Difference Heatmap

```python
def compute_diff_heatmap(baseline, current, homography):
    """Compute pixel-wise difference heatmap"""
    # Warp baseline to current frame perspective
    h, w = current.shape[:2]
    warped = cv2.warpPerspective(baseline, homography, (w, h))

    # Compute absolute difference
    diff = cv2.absdiff(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(current, cv2.COLOR_BGR2GRAY))

    # Apply color map
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return heatmap
```

### Keyboard Controls

```python
key = cv2.waitKey(0) & 0xFF
if key == ord('b'):
    baseline_frame = current_frame.copy()
    baseline_idx = frame_idx
    print("✅ Baseline set")
elif key == ord('v'):
    show_vectors = not show_vectors
elif key == ord('h'):
    show_heatmap = not show_heatmap
elif key == ord('→') or key == 83:
    frame_idx += 1
elif key == ord('←') or key == 81:
    frame_idx -= 1
elif key == ord('q'):
    break
```

## Dependencies

- src/camera_movement_detector.py (CSD module)
- src/movement_detector.py (homography computation)
- src/feature_extractor.py (feature matching)
- tools/validation/comparison_tool.py (frame loader pattern)

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed and merged to main branch
- [ ] Unit tests passing (>85% coverage)
- [ ] Integration test with sample sequence successful
- [ ] Motion vectors render correctly for 50+ matches
- [ ] Documentation added to docs/Stage4_Tools_Guide.md
- [ ] Usage examples with screenshots

## Estimated Effort

**Complexity:** Medium
**Story Points:** 3 points
**Estimated Time:** 1 day

---

**Related Files:**
- tools/validation/baseline_correspondence_tool.py (NEW)
- src/camera_movement_detector.py
- docs/Stage_4_prompt.md (Section 3 - Mode B)
