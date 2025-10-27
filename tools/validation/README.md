# ChArUco vs Cam-Shift Comparison Tool

Visual comparison tool for validating detector agreement between ChArUco 6-DOF pose estimation and cam-shift feature-based detection through dual OpenCV display windows.

## Features

- **Offline Mode**: Batch processing of image directories for systematic validation
- **Online Mode**: Real-time camera capture for interactive debugging
- **Dual Display Windows**: Side-by-side comparison (ChArUco left, Cam-shift right)
- **Real-Time Metrics**: Frame-by-frame displacement comparison with visual feedback
- **Automated Analysis**: MSE calculation, worst matches identification, and JSON logging

## Installation

### Dependencies

The tool requires the following Python packages (automatically installed with the project):

```bash
# Core dependencies
opencv-python==4.10.0.84
opencv-contrib-python>=4.12.0.88
numpy==1.26.4

# Story 1 dependencies
matplotlib==3.9.2
psutil>=5.9.0
```

### Setup

1. **Install project dependencies** (if not already installed):
   ```bash
   pip install -e .
   ```

2. **Verify configuration files**:
   - `camera.yaml` - Camera calibration parameters (required)
   - `comparison_config.json` - ChArUco board and display settings (provided)
   - `config.json` - Cam-shift detector configuration (required)

## Usage

### Offline Mode (Batch Processing)

Process a directory of images for systematic validation:

```bash
python tools/validation/comparison_tool.py \
    --mode offline \
    --input-dir session_001/frames \
    --camera-yaml camera.yaml \
    --charuco-config comparison_config.json \
    --camshift-config config.json \
    --output-dir comparison_results
```

**Workflow**:
1. Tool loads all images from `--input-dir` (sorted by filename)
2. Sets baseline from first image (requires ChArUco detection)
3. Processes all frames sequentially
4. Displays dual windows for each frame (ChArUco left, Cam-shift right)
5. Logs all results to JSON in `--output-dir`
6. Generates MSE graph on completion
7. Generates worst matches report on completion

**Performance**:
- Expected FPS: ~5+ FPS for 157 frames (<30 seconds total)
- Display latency: <100ms window synchronization

**Output Files**:
- `{session_name}_comparison.json` - Structured JSON log with all frame results
- `{session_name}_mse_graph.png` - MSE visualization with threshold line
- `{session_name}_worst_matches.txt` - Top 10 frames with largest displacement differences

### Online Mode (Live Camera)

Real-time camera capture for interactive debugging:

```bash
python tools/validation/comparison_tool.py \
    --mode online \
    --camera-id 0 \
    --camera-yaml camera.yaml \
    --charuco-config comparison_config.json \
    --camshift-config config.json \
    --output-dir comparison_results
```

**Workflow**:
1. Tool opens camera (default: camera ID 0)
2. Displays live preview with instructions
3. User presses **'s'** to set baseline (captures current frame for both detectors)
4. Continuous processing begins with dual display updates
5. User presses **'q'** to quit
6. Tool generates MSE graph and worst matches report

**Performance**:
- Expected FPS: 15-20 FPS on live camera feed
- Baseline setting: Requires ChArUco board in view

**Output Files** (same as offline mode):
- `online_session_{timestamp}_comparison.json`
- `online_session_{timestamp}_mse_graph.png`
- `online_session_{timestamp}_worst_matches.txt`

### Display Windows

#### Dual Window Layout

```
┌─────────────────────────┬─────────────────────────┐
│  ChArUco Detector       │  Cam-Shift Detector     │
│  (640x480)              │  (640x480)              │
│                         │                         │
│  Disp: 12.34px          │  Disp: 12.50px          │
│  Status: DETECTED       │  Status: VALID          │
│  Confidence: 8          │  Confidence: 0.92       │
└─────────────────────────┴─────────────────────────┘
│  ||d1-d2||_2 = 0.16px [GREEN]                     │
│  Threshold: 14.40px (3% of 480px)                 │
│  Frame: 42 | FPS: 18.5                            │
└───────────────────────────────────────────────────┘
```

#### ChArUco Window (Left)
- **Frame**: Original frame with pose axes (if detected)
- **Displacement**: ChArUco 2D displacement in pixels
- **Status**: "DETECTED" (green) or "NOT DETECTED" (red)
- **Confidence**: Number of detected ChArUco corners

#### Cam-Shift Window (Right)
- **Frame**: Original frame with ORB features (if `--show-features` enabled)
- **Displacement**: Cam-shift 2D displacement in pixels
- **Status**: "VALID" (green) or "INVALID" (orange)
- **Confidence**: Cam-shift confidence score (0.0-1.0)

#### Status Bar (Bottom)
- **Comparison Metric**: `||d1-d2||_2` - L2 norm of displacement difference
- **Agreement Status**: GREEN (diff ≤ threshold) or RED (diff > threshold)
- **Threshold**: 3% of minimum image dimension (typically ~14.4px for 480p)
- **Frame Info**: Current frame index and real-time FPS

### Configuration

#### comparison_config.json

```json
{
  "charuco_board": {
    "squares_x": 7,
    "squares_y": 5,
    "square_len_m": 0.035,
    "marker_len_m": 0.026,
    "dict_name": "DICT_4X4_50"
  },
  "comparison_settings": {
    "threshold_percent": 0.03,
    "default_z_distance_m": 1.15
  },
  "display_settings": {
    "window_width": 640,
    "window_height": 480,
    "show_axes": true,
    "show_features": true
  },
  "logging_settings": {
    "output_dir": "comparison_results",
    "log_format": "json",
    "save_frames": false
  }
}
```

**Parameters**:
- `squares_x/y`: ChArUco board dimensions (7×5 squares)
- `square_len_m`: Physical square size in meters (0.035m = 35mm)
- `marker_len_m`: Physical marker size in meters (0.026m = 26mm)
- `dict_name`: ArUco dictionary ("DICT_4X4_50")
- `threshold_percent`: Comparison threshold as percentage of min dimension (0.03 = 3%)
- `default_z_distance_m`: Camera-to-board distance for 3D-to-2D projection (1.15m)
- `window_width/height`: Display window dimensions (640×480)

## Output Structure

After running the tool, the output directory will contain:

```
comparison_results/
├── session_001_comparison.json           # Structured JSON log
├── session_001_mse_graph.png             # MSE visualization
└── session_001_worst_matches.txt         # Top 10 worst matches
```

### JSON Log Structure

```json
{
  "session_name": "session_001",
  "timestamp": "2025-10-27T14:30:00.123456",
  "threshold_px": 14.4,
  "total_frames": 157,
  "results": [
    {
      "frame_idx": 0,
      "timestamp_ns": 1234567890123456789,
      "charuco_detected": true,
      "charuco_displacement_px": 12.34,
      "charuco_confidence": 8.0,
      "camshift_status": "VALID",
      "camshift_displacement_px": 12.50,
      "camshift_confidence": 0.92,
      "displacement_diff": 0.16,
      "agreement_status": "GREEN",
      "threshold_px": 14.4
    },
    ...
  ]
}
```

## Troubleshooting

### Issue: "ChArUco board not detected in baseline image"

**Cause**: Baseline image doesn't contain a visible ChArUco board.

**Solution**:
- Ensure ChArUco board is clearly visible in the first image (offline mode)
- Press 's' when ChArUco board is in camera view (online mode)
- Check board configuration matches physical board (squares, size, dictionary)
- Verify camera calibration file is correct (`camera.yaml`)

### Issue: "No images found in {directory}"

**Cause**: Input directory is empty or contains no .jpg/.png files.

**Solution**:
- Verify `--input-dir` path is correct
- Check directory contains image files with .jpg or .png extension
- Use absolute path or ensure current directory is correct

### Issue: "Failed to open camera {id}"

**Cause**: Camera device not available or already in use.

**Solution**:
- Verify camera is connected and recognized by system
- Try different `--camera-id` values (0, 1, 2, etc.)
- Check no other application is using the camera
- On Linux, verify user has camera permissions

### Issue: Display windows freeze or lag

**Cause**: Processing too slow for real-time display.

**Solution**:
- Reduce image resolution (resize before processing)
- Disable `--show-features` flag to improve performance
- Check CPU/memory usage with `htop` or `top`
- Offline mode: Increase `cv.waitKey()` delay in code

### Issue: MSE graph generation failed

**Cause**: No valid comparison data (all ChArUco detections failed).

**Solution**:
- Ensure ChArUco board is visible in most frames
- Check camera calibration accuracy
- Verify ChArUco board configuration matches physical board
- Review individual frame results in JSON log

### Issue: FPS too low in online mode

**Cause**: Processing bottleneck or camera hardware limitation.

**Solution**:
- Reduce camera resolution in camera driver settings
- Disable unnecessary processing (e.g., `--show-features`)
- Check camera supports desired frame rate (some webcams cap at 15 FPS)
- Consider hardware upgrade for higher FPS requirements

## Examples

### Example 1: Quick offline validation with session_001

```bash
python tools/validation/comparison_tool.py \
    --mode offline \
    --input-dir session_001/frames \
    --camera-yaml camera.yaml \
    --charuco-config comparison_config.json \
    --camshift-config config.json \
    --output-dir comparison_results
```

Expected output:
```
🎬 Starting offline comparison mode
📁 Found 157 images
🎯 Setting baseline from first image...
✅ Baseline set successfully
🔄 Processing frames...
   Processed 10/157 frames (5.2 FPS)
   Processed 20/157 frames (5.3 FPS)
   ...
✅ Processed 157 frames in 28.5s (5.51 FPS)
💾 Saving results...
   Log saved: comparison_results/session_001_comparison.json
   MSE graph saved: comparison_results/session_001_mse_graph.png
   Worst matches retrieved: 10
   Worst matches report saved: comparison_results/session_001_worst_matches.txt
🎉 Offline comparison complete!
```

### Example 2: Live camera debugging

```bash
python tools/validation/comparison_tool.py \
    --mode online \
    --camera-id 0 \
    --camera-yaml camera.yaml \
    --charuco-config comparison_config.json \
    --camshift-config config.json \
    --output-dir comparison_results
```

Expected workflow:
1. Camera window opens with "Press 's' to set baseline, 'q' to quit"
2. Position ChArUco board in view
3. Press **'s'** → "Baseline set successfully"
4. Dual windows show real-time comparison
5. Press **'q'** → Tool saves results and exits

## Technical Details

### Performance Benchmarks

**Offline Mode**:
- Target: Process 157 frames in <30 seconds
- Expected FPS: ~5.2 FPS minimum
- Actual: ~5.5 FPS on typical hardware (Intel i5, 16GB RAM)

**Online Mode**:
- Target: Maintain 15-20 FPS on live camera feed
- Display latency: <100ms window synchronization
- Actual: 18-20 FPS on 640×480 webcam

### Comparison Algorithm

1. **Baseline Capture**:
   - ChArUco: 6-DOF pose estimation with `solvePnP`
   - Cam-shift: ORB feature extraction and matching

2. **Frame Processing**:
   - ChArUco: 3D translation vector → 2D displacement projection
   - Cam-shift: Affine transformation → 2D translation displacement

3. **Comparison Metric**:
   - `||d1-d2||_2` - L2 norm of displacement difference
   - Threshold: 3% of min(image_width, image_height)
   - Agreement: GREEN if diff ≤ threshold, RED otherwise

## References

- Story 1: Core Comparison Infrastructure (`docs/stories/story-comparison-tool-1.md`)
- Tech Spec: Section 5.4 comparison_tool.py Implementation (`tech-spec.md`)
- Epic Overview: Validation Comparison Tool (`epics.md`)

## License

This tool is part of the cam-shift-detector project. See project LICENSE for details.
