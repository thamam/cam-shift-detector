# ChArUco Calibration & Pose-Tracking Checklist

Use this to calibrate once with **ChArUco**, then track pose daily with **GridBoard**.

---

## 0) Prep
- [ ] Install OpenCV-contrib: `pip install opencv-contrib-python`
- [ ] Confirm camera model and lens are final for this setup
- [ ] Fix lens settings for the session: focus, zoom/focal length, aperture
- [ ] Disable auto‑exposure if possible, or keep lighting constant

## 1) Print Boards at True Scale
- [ ] Generate PNGs with the provided tool (`board_printer.py`) using your metric sizes
- [ ] Print at **100% scale** (no fit-to-page, no shrink)
- [ ] Measure the **50 mm check square** → should be **50.0 mm ±0.5 mm**
- [ ] Mount the board on a **flat, rigid** surface (no warping)

## 2) Scene Setup
- [ ] Place ChArUco board in the workspace where the camera will operate
- [ ] Ensure good, even lighting; avoid glare on markers
- [ ] Set the camera at its **operational height and orientation**
- [ ] Lock exposure and focus if possible

## 3) Capture “Diverse” Views (15–30 frames)
Strive to span the camera’s working envelope so intrinsics and distortion are well observed.

**Distances**
- [ ] ~0.5× normal working distance
- [ ] ~1.0× normal working distance
- [ ] ~2.0× normal working distance

**Angles**
- [ ] Yaw variety: left/right up to about ±60°
- [ ] Pitch variety: up/down up to about ±60°
- [ ] Roll variety: rotate board up to about ±30°

**Positions in Frame**
- [ ] Center
- [ ] Near each edge (top, bottom, left, right)
- [ ] Near each corner (TL, TR, BL, BR)

**Scale in View**
- [ ] Small: board ≈ 20–30% of image area
- [ ] Medium: board ≈ 40–60% of image area
- [ ] Large: board ≈ 70–80% of image area

**Quality**
- [ ] Sharp focus, no motion blur
- [ ] Good contrast
- [ ] Many detected ChArUco corners (≥50% of possible)

> Tip: Don’t capture near-similar consecutive frames. Move the board meaningfully between shots.

## 4) Run ChArUco Calibration
- [ ] Load the calibration tool or `camshift_annotator.py --mode charuco`
- [ ] Press **capture** only on sharp, well-framed views
- [ ] Collect **15–30** frames that meet the diversity criteria
- [ ] Complete calibration → obtain **camera.yaml** containing `camera_matrix (K)` and `dist_coeffs`

## 5) Validate Calibration
- [ ] RMS reprojection error ≤ **0.5 px** (≤0.3 px is excellent)
- [ ] Inspect per‑view residuals for outliers
- [ ] If some views are bad, remove them and recalibrate
- [ ] Confirm undistortion looks correct across the frame

## 6) Deploy for Pose Tracking
- [ ] Mount and fix the **GridBoard** in the scene (or use ChArUco if fully visible)
- [ ] In runtime code, **load `camera.yaml`**
- [ ] Detect board → compute SE(3) pose and deltas
- [ ] Keep board rigid and well lit for stability

## 7) Maintenance
- [ ] Re‑calibrate if camera/lens/focus changes, or if temperature/impact may have shifted intrinsics
- [ ] Reprint boards if damaged or warped
- [ ] Periodically re‑verify the 50 mm check square on paper

## 8) Troubleshooting
- [ ] **Pose jitter:** Check lighting, increase board size, verify `K, dist` match this camera
- [ ] **False motion at edges:** Ensure undistortion is enabled and `dist_coeffs` are correct
- [ ] **Few detections:** Use higher‑contrast print, larger board, or get closer
- [ ] **Scale off:** Viewer/printer scaled the PNG; reprint at 100%

## 9) Recordkeeping
- [ ] Save: printer model, paper type, DPI, margins, dictionary, sizes, date
- [ ] Store: `camera.yaml` with version tag and backup
- [ ] Optionally attach a photo of a ruler against the 50 mm square

---

### Minimal Workflow Recap
1. **Calibrate once** with ChArUco → save `camera.yaml`.
2. **Track daily** with GridBoard using the same `camera.yaml`.
3. Re‑calibrate after any optical change.

