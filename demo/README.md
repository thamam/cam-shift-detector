# Camera Movement Detection - Demo Scripts

This directory contains demonstration scripts showcasing the Camera Movement Detector API.

## Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure you're in the project root directory
cd /home/thh3/personal/cam-shift-detector
```

## Demo Scripts

### Demo 1: Basic Detection API

**File:** `demo1_basic_detection.py`

**Demonstrates:**
- Initializing detector with configuration
- Setting baseline reference
- Processing a single test frame
- Displaying detection results

**Run:**
```bash
python demo/demo1_basic_detection.py
```

**Expected Output:**
```
======================================================================
DEMO 1: Basic Camera Movement Detection API
======================================================================

1. Initializing detector with config: config/config_session_001.json
   ✓ Detector initialized

2. Loading baseline image: sample_images/of_jerusalem/frame_00000000.jpg
   ✓ Baseline loaded: 640×480 pixels

3. Setting baseline reference...
   ✓ Baseline set successfully

4. Loading test frame: sample_images/of_jerusalem/frame_00000100.jpg
   ✓ Test frame loaded: 640×480 pixels

5. Processing frame and detecting movement...

======================================================================
DETECTION RESULTS
======================================================================
Status:       VALID
Displacement: 1.23 pixels
Frame ID:     demo_frame_100
Timestamp:    2025-10-28T15:30:45.123Z

✓  Camera position is STABLE
   → Action: Proceed with water quality analysis
```

---

### Demo 2: History Buffer Query

**File:** `demo2_history_buffer.py`

**Demonstrates:**
- Processing multiple frames
- Building detection history
- Querying last N results
- Retrieving specific frame results
- Generating statistics from history

**Run:**
```bash
python demo/demo2_history_buffer.py
```

**Expected Output:**
```
======================================================================
DEMO 2: History Buffer Query
======================================================================

1. Initializing detector...
   ✓ Detector initialized

2. Found 10 images in sample_images/of_jerusalem

3. Setting baseline from first image...
   ✓ Baseline set: frame_00000000.jpg

4. Processing frames and building history...
   Frame  1: VALID   -  0.85px - frame_00000010.jpg
   Frame  2: VALID   -  1.23px - frame_00000020.jpg
   Frame  3: VALID   -  0.67px - frame_00000030.jpg
   [...]

======================================================================
HISTORY BUFFER QUERIES
======================================================================

5. Query: Last 5 detections
----------------------------------------------------------------------
   [Results displayed]

6. Query: Specific frame (demo_frame_003)
----------------------------------------------------------------------
   [Frame details displayed]

7. Query: Complete history buffer
----------------------------------------------------------------------
   Total entries in buffer: 9
   VALID entries:   9
   INVALID entries: 0
   Average displacement: 0.95px
   Maximum displacement: 1.45px
```

---

### Demo 3: Manual Recalibration

**File:** `demo3_recalibration.py`

**Demonstrates:**
- Setting initial baseline
- Detecting camera movement
- Manual recalibration workflow
- Verifying detection after recalibration

**Run:**
```bash
python demo/demo3_recalibration.py
```

**Expected Output:**
```
======================================================================
DEMO 3: Manual Recalibration
======================================================================

SCENARIO: Camera repositioned between sites
          Simulate lighting change or camera maintenance

[... setup steps ...]

4. Simulating camera movement (switching to Site 2)...
   Image: frame_00000050.jpg
   Status:       INVALID
   Displacement: 125.34px
   ⚠️  Movement detected! Camera position changed.

======================================================================
MANUAL RECALIBRATION
======================================================================

5. Operator action: Recalibrating with new baseline...
   New baseline: Site 2 (Carmit)
   ✓ Recalibration successful - new baseline set

6. Testing with same Site 2 image after recalibration...
   Status:       VALID
   Displacement: 0.42px
   ✓ Camera position stable - measurements can resume

======================================================================
RECALIBRATION SUMMARY
======================================================================

Before recalibration:
  Site 1 test: VALID   (1.23px)
  Site 2 test: INVALID (125.34px) ← Movement!

After recalibration:
  Site 2 test: VALID   (0.42px) ← Stable!

✓ Recalibration workflow validated
```

---

### Demo 4: Multi-Frame Sequence

**File:** `demo4_multi_frame_sequence.py`

**Demonstrates:**
- Processing image sequences
- Tracking displacement over time
- Identifying movement events
- Generating sequence statistics

**Run:**
```bash
python demo/demo4_multi_frame_sequence.py
```

**Expected Output:**
```
======================================================================
DEMO 4: Multi-Frame Sequence Processing
======================================================================

[... setup steps ...]

4. Processing image sequence...
======================================================================
 Frame   Status  Displacement  Image
----------------------------------------------------------------------
     1 ✓  VALID        0.85px  frame_00000010.jpg
     2 ✓  VALID        1.23px  frame_00000020.jpg
     3 ✓  VALID        0.67px  frame_00000030.jpg
   [...]
    18 ⚠️  INVALID     5.42px  frame_00000180.jpg  ← Movement!
    19 ✓  VALID        1.05px  frame_00000190.jpg
======================================================================

5. Sequence Statistics
----------------------------------------------------------------------
   Total frames processed:  19
   VALID frames:            18 (94.7%)
   INVALID frames:          1 (5.3%)

   Average displacement:    1.18px
   Minimum displacement:    0.42px
   Maximum displacement:    5.42px

6. Movement Events
----------------------------------------------------------------------
   1 movement event(s) detected:

   Frame: seq_frame_018
   Displacement: 5.42px
   Timestamp: 2025-10-28T15:32:45.789Z
```

---

## Notes

- All demos use actual detector components from `src/`
- No logic duplication - imports real modules
- Demos assume sample images exist in `sample_images/` directory
- Config file `config/config_session_001.json` must exist

## Integration Example

For DAF system integration, see `demo/integration_example.py` (if needed) or refer to Appendix B in the stakeholder presentation.

---

**Created:** 2025-10-28
**Version:** 1.0
