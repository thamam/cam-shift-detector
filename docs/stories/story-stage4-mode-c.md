# Story: Mode C - Enhanced Alpha Blending Tool

**Story ID:** STAGE4-MODE-C
**Epic:** Stage 4 - Interactive Debugging Tools
**Status:** TODO
**Story Points:** 2
**Priority:** Medium

## Story

As a **validation engineer verifying frame alignment**,
I want **an enhanced alpha blending tool with transform computation and blink toggle**,
so that **I can quickly verify if two frames are properly aligned and detect subtle camera shifts visually**.

## Acceptance Criteria

**AC1: Transform Computation & Pre-Warp**
- [ ] Compute homography transform between selected frames using CSD
- [ ] `w` key toggles pre-warp mode (warp Frame B to align with Frame A before blending)
- [ ] Display shows "Pre-warp: ON" or "Pre-warp: OFF" indicator
- [ ] Pre-warped blend shows better alignment for shifted frames
- [ ] Transform computation completes in <500ms

**AC2: Blink Toggle for A/B Comparison**
- [ ] `Space` key toggles rapid A/B switching (blink mode)
- [ ] Blink alternates between Frame A and Frame B every 500ms
- [ ] Blink mode indicator displayed on screen ("Blink: ON")
- [ ] Press `Space` again to return to blended view
- [ ] Blink helps identify subtle shifts

**AC3: Frame Selector for Arbitrary Pairs**
- [ ] `a` key enters Frame A selection mode
- [ ] `b` key enters Frame B selection mode
- [ ] In selection mode, `→/←` keys navigate frames
- [ ] Press `Enter` to confirm frame selection
- [ ] Selected frame indices displayed in status bar
- [ ] Default: Frame A = first, Frame B = current

**AC4: Alpha Blending with Grid Overlay**
- [ ] `↑/↓` keys adjust alpha value (already exists)
- [ ] Alpha value displayed as percentage (e.g., "Alpha: 50%")
- [ ] `g` key toggles alignment grid overlay
- [ ] Grid shows 10×10 reference lines for alignment checking
- [ ] Grid color: cyan with 50% transparency

**AC5: Export and Snapshot**
- [ ] `s` key saves current blended view as PNG
- [ ] Filename includes Frame A and Frame B indices
- [ ] Export includes metadata (alpha, pre-warp status)
- [ ] CSV export with transform parameters

## Tasks / Subtasks

**Phase 1: Transform Computation (AC: #1)**
- [ ] Modify `tools/annotation/ground_truth_annotator.py`
- [ ] Add CSD module integration for transform computation
- [ ] Implement pre-warp toggle mechanism (`w` key)
- [ ] Implement frame warping using computed homography
- [ ] Add pre-warp status indicator overlay
- [ ] Test with known shifted frame pairs

**Phase 2: Blink Toggle (AC: #2)**
- [ ] Implement blink mode state machine
- [ ] Add timer-based alternation (500ms intervals)
- [ ] Implement `Space` key handler
- [ ] Add blink mode indicator overlay
- [ ] Test with subtle shift detection

**Phase 3: Frame Selector (AC: #3)**
- [ ] Implement frame selection mode state machine
- [ ] Add keyboard handlers for `a` and `b` keys
- [ ] Implement frame navigation during selection
- [ ] Add confirmation mechanism (`Enter` key)
- [ ] Display selected frame indices
- [ ] Test with arbitrary frame pairs

**Phase 4: Grid Overlay (AC: #4)**
- [ ] Implement grid drawing function (10×10 lines)
- [ ] Add toggle mechanism (`g` key)
- [ ] Overlay grid on blended view
- [ ] Test grid visibility on various image content

**Phase 5: Export Enhancement (AC: #5)**
- [ ] Update snapshot function with metadata
- [ ] Generate meaningful filenames (frameA_frameB_alpha_prewarp.png)
- [ ] Export transform parameters to CSV
- [ ] Test export functionality

**Phase 6: Testing**
- [ ] Unit tests for transform computation
- [ ] Integration test with frame pairs
- [ ] Verify blink timing accuracy
- [ ] Verify pre-warp alignment improvement
- [ ] Edge case testing: detection failures, invalid frames

## Technical Notes

### Transform Computation

```python
# Integrate CSD module
from src.camera_movement_detector import CameraMovementDetector

detector = CameraMovementDetector(config_path)
detector.set_baseline(frame_a)
result = detector.process_frame(frame_b, frame_id="frame_b")

# Get homography matrix (need to expose from movement_detector)
homography = detector.get_last_homography()  # NEW METHOD NEEDED
```

### Pre-Warp Implementation

```python
def apply_prewarp(frame_b, homography):
    """Warp frame B to align with frame A"""
    h, w = frame_b.shape[:2]
    warped = cv2.warpPerspective(frame_b, homography, (w, h))
    return warped

# Blend logic
if prewarp_enabled:
    frame_b_for_blend = apply_prewarp(frame_b, homography)
else:
    frame_b_for_blend = frame_b

blended = cv2.addWeighted(frame_a, 1-alpha, frame_b_for_blend, alpha, 0)
```

### Blink Toggle

```python
blink_mode = False
blink_state = 0  # 0 = show frame A, 1 = show frame B
last_blink_time = time.time()

while True:
    if blink_mode:
        current_time = time.time()
        if current_time - last_blink_time > 0.5:  # 500ms
            blink_state = 1 - blink_state
            last_blink_time = current_time

        display = frame_a if blink_state == 0 else frame_b
    else:
        display = blended_frame

    cv2.imshow("Mode C", display)
    key = cv2.waitKey(1)

    if key == ord(' '):
        blink_mode = not blink_mode
```

### Grid Overlay

```python
def draw_alignment_grid(image, rows=10, cols=10):
    """Draw reference grid for alignment verification"""
    h, w = image.shape[:2]
    overlay = image.copy()

    # Vertical lines
    for i in range(1, cols):
        x = int(w * i / cols)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 0), 1)

    # Horizontal lines
    for i in range(1, rows):
        y = int(h * i / rows)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 0), 1)

    # Blend with original
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
```

### Enhanced Keyboard Controls

```python
key = cv2.waitKey(1) & 0xFF

if key == ord('w'):
    prewarp_enabled = not prewarp_enabled
elif key == ord(' '):  # Space
    blink_mode = not blink_mode
elif key == ord('a'):
    selection_mode = 'A'
    print("Select Frame A (→/← to navigate, Enter to confirm)")
elif key == ord('b'):
    selection_mode = 'B'
    print("Select Frame B (→/← to navigate, Enter to confirm)")
elif key == ord('g'):
    show_grid = not show_grid
elif key == ord('s'):
    save_snapshot_with_metadata()
elif key == 13:  # Enter
    confirm_frame_selection()
```

## Dependencies

- tools/annotation/ground_truth_annotator.py (existing)
- src/camera_movement_detector.py (CSD module)
- src/movement_detector.py (needs get_last_homography() method)

**New Requirements:**
- Expose homography matrix from movement_detector.py
- Add get_last_homography() method to CameraMovementDetector

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed and merged to main branch
- [ ] Unit tests passing (>85% coverage)
- [ ] Integration test with frame pairs successful
- [ ] Pre-warp improves alignment visually
- [ ] Blink timing accurate (±50ms)
- [ ] Documentation updated in tools/annotation/README.md
- [ ] Usage examples with screenshots

## Estimated Effort

**Complexity:** Low-Medium
**Story Points:** 2 points
**Estimated Time:** 0.5-1 day

---

**Related Files:**
- tools/annotation/ground_truth_annotator.py (MODIFY)
- src/movement_detector.py (ADD get_last_homography method)
- docs/Stage_4_prompt.md (Section 3 - Mode C)
