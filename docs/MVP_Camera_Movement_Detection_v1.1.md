# MVP Specification: Camera Movement Detection System
## Minimal Viable Product for DAF Water Quality Monitoring

**Document Version:** 1.1-MVP (RANSAC Update)  
**Date:** October 16, 2025  
**Parent Document:** PRD_Camera_Movement_Detection_v1.1.md  
**Status:** Ready for Implementation

---

## 1. Executive Summary

### MVP Philosophy

**Goal**: Validate core detection capability and operator workflow with simplest possible implementation that handles dynamic DAF environments

**Success Criteria**: 
- Detect camera movement ≥2 pixels within 1 second
- **Robust to dynamic scene elements (water, bubbles, flocs)**
- Alert operator clearly
- Prevent use of corrupted measurement data
- Enable manual recovery
- **Single camera, single operator, local deployment only**

### What's In Scope for MVP

✅ **RANSAC-based movement detection** (robust to dynamic scenes)  
✅ **Static region masking UI** (operator-defined ROI during setup)  
✅ Visual alarm display (desktop application)  
✅ Data validity flag (simple file-based)  
✅ Manual recalibration wizard  
✅ Local event logging (SQLite)  
✅ Single camera support  
✅ Local deployment only  
✅ **Scene dynamics diagnostics** (inlier ratio, feature stability)

### Explicitly Out of Scope for MVP

❌ Multi-camera support  
❌ SCADA/PLC integration  
❌ Email/SMS notifications  
❌ Multi-site deployment  
❌ Advanced classification (sudden/gradual/vibration)  
❌ Historical analytics dashboard  
❌ Cloud connectivity  
❌ User management and authentication  
❌ Remote access  
❌ Mobile app  

### Key Changes in v1.1

**Critical Updates:**
1. **RANSAC homography** replaces naive feature matching
2. **Outlier rejection** separates static features (camera) from dynamic (water/bubbles)
3. **Static region masking UI** during initial setup
4. **Inlier ratio monitoring** to detect scene quality issues
5. **Dynamic scene test cases** added to validation

**Why**: Original approach would fail in DAF environments due to water surface motion, bubbles, and floating flocs causing false positives.

---

## 2. MVP Architecture

```
┌─────────────────────────────────────────────────────┐
│              CAMERA LAYER                           │
│   ┌─────────────────┐                              │
│   │  Single Camera  │  (USB or GigE)               │
│   │   DAF Tank      │                              │
│   └────────┬────────┘                              │
└────────────┼────────────────────────────────────────┘
             │ Video Stream
┌────────────▼────────────────────────────────────────┐
│         DETECTION ENGINE (UPDATED)                  │
│   ┌─────────────────────────────────────────────────┐
│   │  Movement Detection Module                      │
│   │  - ORB feature extraction                       │
│   │  - Feature matching                             │
│   │  - RANSAC homography (NEW)                      │
│   │  - Inlier/outlier separation (NEW)              │
│   │  - Static feature displacement only             │
│   └────────────────┬───────────────────────────────┘
└────────────────────┼────────────────────────────────┘
                     │ Movement Event + Diagnostics
┌────────────────────▼────────────────────────────────┐
│           SIMPLE UI LAYER                           │
│   ┌─────────────────────────────────────────────────┐
│   │  Status Display                                 │
│   │  - Camera view (1 fps)                          │
│   │  - Status: OK / MOVEMENT DETECTED               │
│   │  - Movement magnitude (pixels)                  │
│   │  - Inlier ratio (scene quality) (NEW)           │
│   │  - [Recalibrate] button                         │
│   │  - [Resume Operation] button                    │
│   └─────────────────────────────────────────────────┘
│                                                     │
│   ┌─────────────────────────────────────────────────┐
│   │  Static Region Masking UI (NEW)                 │
│   │  - Define ROI during setup                      │
│   │  - Exclude water surface, dynamic areas         │
│   └─────────────────────────────────────────────────┘
│                                                     │
│   ┌─────────────────────────────────────────────────┐
│   │  Local Event Log (SQLite)                       │
│   │  - Timestamp, magnitude, inlier_ratio (NEW)     │
│   └─────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────┘
```

### Data Flow

**Normal Operation**:
```
Camera → Frame Acquisition → Feature Extraction 
→ RANSAC Homography Estimation → Separate Inliers/Outliers
→ Calculate displacement from INLIERS only
→ Movement = 0 pixels → Continue monitoring
```

**Movement Detected**:
```
Camera → Frame Acquisition → Feature Extraction 
→ RANSAC Homography → Inlier displacement > 2 pixels
→ Trigger Alarm → Update UI (Red)
→ Set data_validity.flag = INVALID → Log Event (with inlier_ratio)
→ Wait for operator recalibration
```

**Recovery**:
```
Operator → Click Recalibrate → Wizard (4 steps - NEW: includes ROI)
→ Define Static Regions → Capture New Reference → Stability Test (30 sec)
→ If Stable: Clear Alarm → Set data_validity.flag = VALID
```

---

## 3. Core MVP Features (UPDATED)

### Feature 1: RANSAC-Based Movement Detection

**Implementation**: ORB feature tracking with RANSAC homography for outlier rejection

**Algorithm**:
```python
1. During setup:
   - Operator defines static region mask (ROI)
   - Capture reference frame
   - Extract ORB features from static regions (target: 50+ features)

2. Every 1 second:
   - Capture current frame
   - Extract ORB features from static regions
   - Match features between reference and current
   - Use RANSAC to fit homography transformation
     → Inliers: static features (represent camera movement)
     → Outliers: dynamic features (water, bubbles - ignored)
   - Calculate median displacement of INLIERS only
   - If displacement > 2 pixels: TRIGGER ALARM
   - Monitor inlier ratio for scene quality
```

**Key Innovation**: RANSAC separates camera motion from scene motion

**Configuration Parameters**:
```json
{
  "detection_threshold_pixels": 2,
  "analysis_interval_seconds": 1,
  "min_features_required": 50,
  "ransac_threshold_pixels": 2.0,
  "min_inlier_ratio": 0.3,
  "static_region_mask_enabled": true
}
```

**Acceptance Criteria**:
- Detects 2-pixel camera movement in lab testing (>95% accuracy)
- <5% false positive rate with active water/bubble motion
- Detection latency < 2 seconds
- Runs at ≥1 Hz on standard laptop (Intel i5 or equivalent)
- CPU usage < 10% average (slightly higher due to RANSAC)
- Inlier ratio >30% in all normal conditions

---

### Feature 1a: Static Region Masking UI (NEW)

**Implementation**: Interactive ROI selection during initial setup

**UI Flow**:
```
┌─────────────────────────────────────────────────────┐
│  Initial Setup - Define Static Regions             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Step 1: Define Static Monitoring Region           │
│                                                     │
│  ┌───────────────────┐                            │
│  │                   │   Instructions:             │
│  │   [Live Camera]   │   Draw a rectangle around   │
│  │                   │   STATIC elements only:     │
│  │   ▓▓▓▓▓▓▓▓▓▓▓    │   ✓ Tank walls              │
│  │   ▓ Static  ▓    │   ✓ Pipes                   │
│  │   ▓ Region  ▓    │   ✓ Equipment               │
│  │   ▓▓▓▓▓▓▓▓▓▓▓    │                             │
│  │   ░░░░░░░░░░░    │   ✗ Water surface           │
│  │   ░ Exclude ░    │   ✗ Bubble zone             │
│  │   ░░░░░░░░░░░    │   ✗ Foam accumulation       │
│  └───────────────────┘                            │
│                                                     │
│  Click and drag to define rectangle                │
│  [Clear Selection]  [Preview Features]             │
│                                                     │
│  Features detected in static region: 68 ✓          │
│  (Minimum 50 required)                             │
│                                                     │
│  [< Back]  [Continue to Calibration >]             │
└─────────────────────────────────────────────────────┘
```

**Operator Actions**:
1. **Click and drag** to draw rectangle on live camera view
2. **Preview features**: Shows detected ORB keypoints in selected region
3. **Adjust** rectangle if feature count < 50 or too many on water
4. **Save** static region mask for all future detections

**Validation**:
- Mask must contain ≥50 features
- Display warning if mask includes bottom 30% of frame (likely water)
- Allow manual override if operator confirms region is static

**Storage**:
```python
# Mask saved with reference frame
static_mask = {
    "x": 100,
    "y": 50,
    "width": 400,
    "height": 300,
    "features_detected": 68,
    "timestamp": "2025-10-16T10:15:00Z"
}
```

**Acceptance Criteria**:
- Operator can define ROI in <2 minutes
- Feature count displayed in real-time
- Mask persists across application restarts
- Can be redefined during recalibration

---

### Feature 2: Visual Alarm & Status Display (UPDATED)

**Implementation**: Simple desktop application (Python + Qt5 or Web-based)

**UI Layout (UPDATED)**:
```
┌─────────────────────────────────────────────────────┐
│  Camera Movement Detection - MVP v1.1               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────┐   STATUS: [OK / ALARM]   │
│  │                     │                           │
│  │   Camera View       │   Movement: 0.0 px       │
│  │   (320x240, 1fps)   │                           │
│  │                     │   Features: 78            │
│  │   ▓▓▓▓▓▓▓▓▓        │   Inliers: 72 (92%) ✓    │
│  │   ▓Static ▓        │   ⚠ Low inliers if <30%  │
│  │   ▓Region ▓        │                           │
│  │   ▓▓▓▓▓▓▓▓▓        │   Scene Quality: GOOD     │
│  └─────────────────────┘                           │
│                                                     │
│  [ Recalibrate Camera ]  [ Resume Operation ]      │
│                                                     │
│  Recent Events:                                     │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 2025-10-16 14:32:18 - Movement: 7.2px - ALARM  │ │
│  │   Inliers: 45/78 (58%)                         │ │
│  │ 2025-10-16 14:35:42 - Recalibrated - OK        │ │
│  │ 2025-10-16 10:15:00 - System started - OK      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**New Display Elements**:
- **Inlier Count**: "72/78 (92%)" - shows how many features are static
- **Scene Quality Indicator**:
  - EXCELLENT: Inliers >70%
  - GOOD: Inliers 50-70%
  - FAIR: Inliers 30-50%
  - ⚠️ POOR: Inliers <30% (consider redefining static region)

**Status Indicators**:
- **OK (Green)**: Camera stable, inlier ratio >30%
- **ALARM (Red)**: Movement detected
- **⚠️ WARNING (Yellow)**: Inlier ratio <30% (scene too dynamic)

**Alarm Behavior (UNCHANGED)**:
- Status changes from "OK" (green) to "ALARM" (red)
- Movement magnitude displayed prominently
- Audible beep (system beep, single notification)
- "Resume Operation" button disabled until recalibration
- Recent events list updated with inlier ratio

**Acceptance Criteria**:
- Alarm visible within 2 seconds of detection
- Inlier ratio updates every 1 second
- Scene quality warning triggers at <30% inliers
- Status persists until operator action
- UI remains responsive during detection
- Event list shows last 10 events with diagnostics

---

### Feature 3: Data Validity Flag (UNCHANGED)

**Implementation**: Simple text file that external systems can read

**File Location**: `./data_validity.flag`

**File Content**:
```
VALID      # When camera is OK
INVALID    # When movement detected
```

**Behavior**:
- File created at startup with "VALID"
- Set to "INVALID" immediately upon movement detection
- Set to "VALID" only after successful recalibration and operator confirmation
- External measurement system (Flocky) reads this file before using vision data

**Integration Pattern**:
```python
# External system (Flocky) integration example
def should_use_vision_data():
    try:
        with open('./data_validity.flag', 'r') as f:
            status = f.read().strip()
            return status == 'VALID'
    except:
        return False  # Fail-safe: don't use data if flag unreadable
```

**Acceptance Criteria**:
- Flag changes within 1 second of detection
- File is atomic (no partial writes)
- Persists across application restarts
- Readable by external processes

---

### Feature 4: Recalibration Wizard (UPDATED - 4 Steps)

**Implementation**: 4-step wizard process (added static region definition)

**Step 1: Operator Verification (UNCHANGED)**
```
┌─────────────────────────────────────────────────────┐
│  Recalibration Required                             │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Camera movement detected.                          │
│                                                     │
│  Please verify:                                     │
│  • Camera mount is physically secure                │
│  • No loose bolts or connections                    │
│  • Camera position is stable                        │
│                                                     │
│  [Mount is Secure - Proceed]  [Cancel]              │
└─────────────────────────────────────────────────────┘
```

**Step 2: Redefine Static Region (NEW)**
```
┌─────────────────────────────────────────────────────┐
│  Redefine Static Monitoring Region                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Current view has changed. Please redefine the      │
│  static region or confirm existing region is OK.    │
│                                                     │
│  ┌───────────────────┐                            │
│  │   [Live Camera]   │   Previous Region (dotted)  │
│  │   ┄┄┄┄┄┄┄┄┄┄┄    │   ✓ Use existing region     │
│  │   ┆ Old     ┆    │   OR                         │
│  │   ┆ Region  ┆    │   Draw new rectangle         │
│  │   ┄┄┄┄┄┄┄┄┄┄┄    │                             │
│  └───────────────────┘                            │
│                                                     │
│  [Use Existing Region]  [Define New Region]         │
└─────────────────────────────────────────────────────┘
```

**Step 3: Capture New Reference**
```
┌─────────────────────────────────────────────────────┐
│  Capturing New Reference                            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Current camera view:                               │
│  ┌───────────────────┐                            │
│  │                   │                            │
│  │   [Live Camera]   │                            │
│  │                   │                            │
│  │   ▓▓▓▓▓▓▓▓▓▓▓    │                            │
│  │   ▓ Static  ▓    │                            │
│  │   ▓ Region  ▓    │                            │
│  │   ▓▓▓▓▓▓▓▓▓▓▓    │                            │
│  └───────────────────┘                            │
│                                                     │
│  Features detected in static region: 68             │
│  Inlier ratio (last 5 seconds): 85% ✓              │
│  Status: ✓ Sufficient features                      │
│                                                     │
│  [Capture Reference]                                │
└─────────────────────────────────────────────────────┘
```

**Step 4: Stability Test (30 seconds)**
```
┌─────────────────────────────────────────────────────┐
│  Stability Test in Progress                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Testing camera stability...                        │
│                                                     │
│  Time remaining: 23 seconds                         │
│                                                     │
│  Movement detected: 0.3 pixels (OK)                 │
│  Inlier ratio: 88% ✓                                │
│                                                     │
│  Progress: ████████░░░░░░░░░ 45%                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Success/Failure Messages (UPDATED)**:
```
SUCCESS:
┌─────────────────────────────────────────────────────┐
│  ✓ Calibration Successful                           │
│                                                     │
│  Camera is stable and ready for operation.          │
│  Static region features: 68                         │
│  Average inlier ratio: 87%                          │
│                                                     │
│  Click "Resume Operation" to continue.              │
│                                                     │
│  [OK]                                               │
└─────────────────────────────────────────────────────┘

FAILURE (Low Inliers):
┌─────────────────────────────────────────────────────┐
│  ✗ Calibration Failed - Scene Too Dynamic          │
│                                                     │
│  Inlier ratio: 18% (need ≥30%)                      │
│  Too many dynamic features in static region.        │
│                                                     │
│  Suggestions:                                       │
│  • Redefine static region (exclude more water)      │
│  • Adjust camera angle away from water surface      │
│  • Wait for bubbling to stabilize                   │
│  • Add reference markers to tank walls              │
│                                                     │
│  [Redefine Region]  [Try Again]  [Cancel]           │
└─────────────────────────────────────────────────────┘

FAILURE (Unstable):
┌─────────────────────────────────────────────────────┐
│  ✗ Calibration Failed                               │
│                                                     │
│  Camera is still unstable.                          │
│  Movement detected: 2.5 pixels                      │
│  Inlier ratio: 75% (good scene quality)             │
│                                                     │
│  Suggestions:                                       │
│  • Check camera mount for vibration                 │
│  • Verify all connections are tight                 │
│  • Wait for vibration source to stop                │
│                                                     │
│  [Try Again]  [Cancel]                              │
└─────────────────────────────────────────────────────┘
```

**Acceptance Criteria**:
- Wizard completes in <3 minutes when successful (1 min longer for ROI step)
- Clear instructions at each step
- Can reuse existing static region or define new one
- Validates new reference quality (≥50 features in static region)
- 30-second stability test (movement <1 pixel, inliers >30%)
- Updates reference frame and mask on success
- Provides actionable error messages with diagnostics on failure

---

### Feature 5: Local Event Logging (UPDATED)

**Implementation**: SQLite database with enhanced schema

**Database Schema (UPDATED)**:
```sql
CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,  -- ISO 8601 format: 2025-10-16T14:32:18.456Z
  event_type TEXT NOT NULL, -- 'movement_detected', 'recalibration_success', 
                            -- 'recalibration_failed', 'system_start', 'system_stop'
  magnitude_pixels REAL,    -- Movement magnitude (NULL for non-movement events)
  features_tracked INTEGER, -- Total number of features matched
  inliers_count INTEGER,    -- NEW: Number of inlier features (static)
  inlier_ratio REAL,        -- NEW: Ratio of inliers to total (0.0-1.0)
  scene_quality TEXT,       -- NEW: 'excellent', 'good', 'fair', 'poor'
  notes TEXT                -- Additional context
);

CREATE INDEX idx_timestamp ON events(timestamp DESC);
CREATE INDEX idx_event_type ON events(event_type);
CREATE INDEX idx_inlier_ratio ON events(inlier_ratio);  -- NEW: for diagnostics
```

**Logged Events (UPDATED)**:

| Event Type | When | Data Captured |
|------------|------|---------------|
| `system_start` | Application starts | Timestamp, initial feature count, inlier_ratio |
| `movement_detected` | Movement >2px detected | Timestamp, magnitude, features, **inliers, inlier_ratio, scene_quality** |
| `recalibration_success` | Recalibration completes successfully | Timestamp, new feature count, **final inlier_ratio** |
| `recalibration_failed` | Recalibration fails stability test | Timestamp, reason, **inlier_ratio at failure** |
| `scene_quality_warning` | **NEW**: Inlier ratio <30% | Timestamp, inlier_ratio, scene_quality='poor' |
| `resume_operation` | Operator clicks "Resume Operation" | Timestamp |
| `system_stop` | Application exits | Timestamp |

**Query Interface in UI**:
- Display last 10 events in scrollable list
- Export all events to CSV via button click
- CSV columns: Timestamp, Event Type, Magnitude (px), Features, **Inliers, Inlier Ratio (%), Scene Quality**, Notes

**Example CSV Export (UPDATED)**:
```csv
Timestamp,Event Type,Magnitude (px),Features,Inliers,Inlier Ratio (%),Scene Quality,Notes
2025-10-16T14:32:18.456Z,movement_detected,7.2,142,78,54.9,fair,
2025-10-16T14:35:42.123Z,recalibration_success,,68,62,91.2,excellent,Operator confirmed stable
2025-10-16T14:36:01.789Z,resume_operation,,,,,,
2025-10-16T10:15:00.000Z,system_start,,75,68,90.7,excellent,
```

**Acceptance Criteria**:
- All events logged with accurate timestamp and diagnostics
- Inlier ratio and scene quality captured for all detections
- Database file survives application restart
- Export generates valid CSV with new columns
- No log entry failures (with error handling)
- Database size <10MB for typical operation (auto-cleanup after 1000 events)

---

## 4. MVP User Workflows (UPDATED)

### 4.1 Initial Setup Workflow (UPDATED - Includes Static Region)

```
┌─────────────────────────────────────────────────────┐
│  First Time Setup                                   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  No reference frame found.                          │
│                                                     │
│  Steps:                                             │
│  1. Position camera to view DAF tank                │
│  2. Ensure stable mounting                          │
│  3. Verify adequate lighting (>500 lux)             │
│  4. Define static monitoring region (NEW)           │
│                                                     │
│  Current view:                                      │
│  ┌───────────────────┐                            │
│  │   [Live Camera]   │   Draw rectangle around     │
│  │                   │   STATIC elements:          │
│  │   ▓▓▓▓▓▓▓▓▓      │   ✓ Tank walls, pipes       │
│  │   ▓Static ▓      │   ✗ Water, bubbles          │
│  │   ▓Region ▓      │                             │
│  │   ▓▓▓▓▓▓▓▓▓      │   Click and drag to select  │
│  │   ░░░░░░░░░      │                             │
│  │   ░Exclude░      │                             │
│  │   ░░░░░░░░░      │                             │
│  └───────────────────┘                            │
│                                                     │
│  Features detected in static region: 82  ✓          │
│  (Minimum 50 required)                             │
│                                                     │
│  [Capture Initial Reference]                        │
└─────────────────────────────────────────────────────┘
```

**Steps**:
1. Launch application
2. System detects no reference frame exists
3. Displays setup wizard
4. Operator positions camera
5. **Operator defines static region** (click and drag rectangle)
6. System validates feature count in region (≥50)
7. Operator clicks "Capture Initial Reference"
8. System saves reference + static mask
9. System starts monitoring

---

### 4.2 Normal Operation Workflow (UPDATED)

```
START
  │
  ├─> System loads reference frame + static mask
  │
  ├─> Set data_validity.flag = VALID
  │
  ├─> Begin monitoring (1 Hz)
  │
  ├─> Every 1 second:
  │     │
  │     ├─> Capture frame
  │     ├─> Extract features from static region only
  │     ├─> Match to reference features
  │     ├─> RANSAC: Fit homography, separate inliers/outliers
  │     ├─> Calculate inlier_ratio
  │     ├─> Calculate displacement from INLIERS only
  │     │
  │     ├─> If inlier_ratio < 0.3:
  │     │     └─> Log scene_quality_warning
  │     │         Display "⚠ Scene Quality: POOR"
  │     │         (Continue monitoring but operator should review)
  │     │
  │     ├─> If displacement < 2 pixels:
  │     │     └─> Update UI (status OK, magnitude 0.X px, inliers XX%)
  │     │
  │     └─> If displacement ≥ 2 pixels:
  │           └─> Go to ALARM workflow
  │
  └─> Continue until stopped
```

**UI During Normal Operation (UPDATED)**:
- Status: OK (green background)
- Movement: 0.0 - 1.9 pixels (small fluctuations normal)
- Features: 50-200 (varies by scene)
- **Inliers: 45-150 (typically 60-90% of features)**
- **Scene Quality: EXCELLENT / GOOD / FAIR / ⚠️ POOR**
- Recent events: Shows system start, any past events

---

### 4.3 Movement Detection & Alarm Workflow (UNCHANGED)

```
MOVEMENT DETECTED (≥2 pixels in INLIERS)
  │
  ├─> Set data_validity.flag = INVALID
  │
  ├─> Update UI:
  │     ├─> Status: ALARM (red background)
  │     ├─> Movement: X.X pixels (actual value)
  │     ├─> Inliers: XX/YY (ZZ%)
  │     ├─> Play system beep (single notification)
  │     └─> Disable "Resume Operation" button
  │
  ├─> Log event to database:
  │     └─> Timestamp, magnitude, features, inliers, inlier_ratio
  │
  ├─> Wait for operator action
  │
  └─> Operator options:
        │
        ├─> Click "Recalibrate Camera"
        │     └─> Go to RECALIBRATION workflow (4 steps now)
        │
        └─> Ignore (not recommended)
              └─> Alarm persists, data remains invalid
```

---

### 4.4 Recalibration Workflow (UPDATED)

```
OPERATOR CLICKS "RECALIBRATE CAMERA"
  │
  ├─> STEP 1: Operator Verification
  │     │
  │     ├─> Display: "Please verify camera mount is secure"
  │     ├─> Show current camera view
  │     ├─> Operator physically checks mount
  │     │
  │     └─> Click "Mount is Secure - Proceed" or "Cancel"
  │           │
  │           └─> If Cancel: Return to alarm state
  │
  ├─> STEP 2: Redefine Static Region (NEW)
  │     │
  │     ├─> Show previous static region (dotted outline)
  │     ├─> Options:
  │     │     ├─> Use existing region
  │     │     └─> Define new region (click and drag)
  │     │
  │     ├─> If new region:
  │     │     ├─> Extract features from new region
  │     │     ├─> Validate count ≥50
  │     │     ├─> Preview inlier ratio
  │     │     └─> Save new mask
  │     │
  │     └─> Proceed to Step 3
  │
  ├─> STEP 3: Capture New Reference
  │     │
  │     ├─> Capture current frame
  │     ├─> Extract ORB features from static region
  │     │
  │     ├─> Validate:
  │     │     ├─> Feature count ≥ 50?
  │     │     ├─> Inlier ratio preview ≥ 30% (5 sec test)?
  │     │     │
  │     │     └─> If validation fails:
  │     │           └─> ERROR with specific guidance
  │     │
  │     └─> Display: "Features: XX, Inliers: YY (ZZ%)  ✓"
  │
  ├─> STEP 4: Stability Test (30 seconds)
  │     │
  │     ├─> Monitor camera for movement using RANSAC
  │     ├─> Track both movement AND inlier ratio
  │     ├─> Update progress bar
  │     │
  │     ├─> If movement >1 pixel detected:
  │     │     └─> FAIL: "Camera is unstable"
  │     │
  │     ├─> If inlier_ratio <30% during test:
  │     │     └─> FAIL: "Scene too dynamic"
  │     │          Suggest redefining static region
  │     │
  │     └─> If stable for full 30 seconds AND inliers ≥30%:
  │           └─> SUCCESS: "Calibration successful!"
  │
  ├─> On SUCCESS:
  │     │
  │     ├─> Store new reference frame + static mask
  │     ├─> Clear alarm state
  │     ├─> Enable "Resume Operation" button
  │     ├─> Log: recalibration_success with inlier_ratio
  │     │
  │     └─> Display: "Click 'Resume Operation' to continue"
  │
  └─> On FAILURE:
        │
        ├─> Keep alarm state
        ├─> Log: recalibration_failed with reason and inlier_ratio
        │
        └─> Operator options:
              ├─> [Redefine Region]: Go back to Step 2
              ├─> [Try Again]: Restart from Step 3
              └─> [Cancel]: Return to alarm state
```

---

### 4.5 Resume Operation Workflow (UNCHANGED)

```
OPERATOR CLICKS "RESUME OPERATION"
(Only enabled after successful recalibration)
  │
  ├─> Set data_validity.flag = VALID
  │
  ├─> Update UI:
  │     ├─> Status: OK (green background)
  │     ├─> Movement: 0.0 px (reset)
  │     └─> Disable "Resume Operation" button (no longer needed)
  │
  ├─> Log event: resume_operation
  │
  ├─> Add event to recent events list
  │
  └─> Return to normal monitoring
```

---

### 4.6 Error Handling Workflows (UPDATED)

**Camera Disconnected (UNCHANGED)**:
```
CAMERA LOST
  │
  ├─> Detect: No frames received for 5 seconds
  │
  ├─> Update UI:
  │     ├─> Status: ERROR (yellow background)
  │     └─> Message: "Camera disconnected. Reconnecting..."
  │
  ├─> Attempt reconnection every 5 seconds
  │
  ├─> If reconnected:
  │     └─> Display: "Camera reconnected. Please recalibrate."
  │          └─> Require recalibration before resuming
  │
  └─> If not reconnected after 60 seconds:
        └─> Display: "Camera connection failed. Check cable and restart application."
```

**Insufficient Lighting (UNCHANGED)**:
```
LOW FEATURE COUNT (<50) IN STATIC REGION
  │
  ├─> During initial setup OR recalibration
  │
  ├─> Display ERROR:
  │     "Insufficient features detected for reliable tracking"
  │     
  │     Features in static region: XX (need ≥50)
  │     
  │     Suggestions:
  │     • Increase lighting (target >500 lux)
  │     • Enlarge static region to include more textured areas
  │     • Add high-contrast reference markers to tank walls
  │     • Adjust camera to view more textured area
  │     
  │     [Adjust Region]  [Try Again]  [Add Markers Guide]
  │
  └─> Wait for operator action
```

**Scene Too Dynamic (NEW)**:
```
LOW INLIER RATIO (<30%)
  │
  ├─> During recalibration stability test or normal operation
  │
  ├─> Display WARNING:
  │     "Scene Too Dynamic for Reliable Detection"
  │     
  │     Inlier ratio: XX% (need ≥30%)
  │     Too many features moving (water, bubbles, flocs)
  │     
  │     Current static region may include dynamic elements.
  │     
  │     Suggestions:
  │     • Redefine static region (exclude more water surface)
  │     • Adjust camera angle upward (focus on tank walls)
  │     • Wait for bubbling to stabilize
  │     • Reduce bubble generation temporarily
  │     • Add static reference markers (ArUco or printed patterns)
  │     
  │     [Redefine Static Region]  [Adjust Camera]  [Continue Anyway]
  │
  └─> Operator decision:
        ├─> Redefine Region: Go to static region UI
        ├─> Continue Anyway: Monitoring continues with low confidence
        └─> Adjust Camera: Physical camera adjustment needed
```

---

## 5. MVP Technical Specifications (UPDATED)

### 5.1 Hardware Requirements (UNCHANGED)

**Minimum Specifications**:
- **Computer**: Laptop or PC with Intel i5 (6th gen) or equivalent
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB for application + logs
- **Camera**: 
  - USB webcam (USB 2.0 minimum, USB 3.0 preferred)
  - OR Industrial GigE/USB3 camera
  - Resolution: 640x480 minimum (1920x1080 recommended)
  - Frame rate: 10 fps minimum
- **OS**: Ubuntu 22.04 LTS (primary) or Windows 10/11

**Recommended Setup for Field Deployment**:
- Industrial PC or ruggedized laptop
- Industrial USB camera with M12 connector
- Stable mounting (tripod or wall mount)
- Supplemental LED lighting (500-1000 lux)

---

### 5.2 Software Stack (UPDATED)

**Core Technologies**:
```
Language:     Python 3.10+
Vision:       OpenCV 4.8+ (cv2)
Features:     ORB (built into OpenCV)
Homography:   cv2.findHomography() with RANSAC (NEW)
UI:           PyQt5 (option 1) OR Flask + HTML/CSS/JS (option 2)
Database:     SQLite 3.40+
Logging:      Python logging module
```

**Python Dependencies** (`requirements.txt`):
```
opencv-python>=4.8.0
numpy>=1.24.0
PyQt5>=5.15.0        # If using Qt UI
Flask>=2.3.0         # If using web UI
```

**Installation**:
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install python3-pip python3-opencv libqt5gui5

# Install Python packages
pip3 install -r requirements.txt
```

---

### 5.3 Performance Targets (UPDATED)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Detection Latency** | <2 seconds (95th percentile) | Timestamp difference: movement → alarm |
| **CPU Usage** | <10% average (increased from 5% due to RANSAC) | System monitor during 1-hour run |
| **Memory Usage** | <250 MB (increased from 200 MB) | Process memory monitoring |
| **Frame Processing Rate** | ≥1 Hz | Frames analyzed per second |
| **Feature Extraction Time** | <100ms per frame | Profiling with OpenCV |
| **RANSAC Homography Time** | <50ms per frame (NEW) | Profiling with cv2.findHomography |
| **Database Write Time** | <50ms per event | Database profiling |
| **Startup Time** | <10 seconds | From launch to monitoring active |
| **UI Responsiveness** | No lag during detection | Manual testing |
| **Inlier Ratio (Normal)** | >60% typical, >30% minimum (NEW) | Real-time monitoring |
| **False Positive Rate** | <5% with dynamic scene (NEW) | Field testing with active bubbling |

---

### 5.4 Configuration File (UPDATED)

**File**: `config.json`

**Default Configuration (UPDATED)**:
```json
{
  "camera": {
    "device_id": 0,
    "resolution_width": 640,
    "resolution_height": 480,
    "fps": 10
  },
  "detection": {
    "threshold_pixels": 2.0,
    "analysis_interval_seconds": 1.0,
    "min_features_required": 50,
    "max_features": 500,
    "orb_parameters": {
      "n_features": 500,
      "scale_factor": 1.2,
      "n_levels": 8
    },
    "ransac_parameters": {
      "enabled": true,
      "reprojection_threshold": 2.0,
      "confidence": 0.995,
      "max_iterations": 2000,
      "min_inlier_ratio": 0.3
    },
    "static_region": {
      "enabled": true,
      "x": null,
      "y": null,
      "width": null,
      "height": null
    }
  },
  "recalibration": {
    "stability_test_duration_seconds": 30,
    "stability_threshold_pixels": 1.0,
    "min_inlier_ratio_during_test": 0.3
  },
  "ui": {
    "display_fps": 1,
    "display_width": 320,
    "display_height": 240,
    "enable_sound": true,
    "show_inlier_diagnostics": true,
    "scene_quality_thresholds": {
      "excellent": 0.7,
      "good": 0.5,
      "fair": 0.3
    }
  },
  "logging": {
    "database_path": "./events.db",
    "log_level": "INFO",
    "max_events": 1000,
    "log_inlier_ratio": true
  },
  "data_validity": {
    "flag_file_path": "./data_validity.flag"
  }
}
```

**Tunable Parameters**:
- `threshold_pixels`: Adjust based on site vibration (1.5-5.0 typical range)
- `ransac_parameters.reprojection_threshold`: Increase to 3.0-5.0 for noisier scenes
- `ransac_parameters.min_inlier_ratio`: Lower to 0.2 for very dynamic scenes (with caution)
- `analysis_interval_seconds`: Increase to 2-5 for lower CPU usage
- `min_features_required`: Lower to 30 for low-texture scenes
- `stability_test_duration_seconds`: Shorten to 15 for faster recalibration

---

### 5.5 Project Structure (UPDATED)

```
camera-movement-mvp/
│
├── README.md                       # Installation and usage guide
├── requirements.txt                # Python dependencies
├── config.json                     # Configuration (see above)
├── LICENSE                         # MIT or appropriate license
│
├── src/
│   ├── __init__.py
│   ├── main.py                     # Application entry point
│   ├── detection_engine.py         # Core movement detection logic (UPDATED: RANSAC)
│   ├── camera_manager.py           # Camera I/O and frame capture
│   ├── reference_manager.py        # Reference frame + static mask storage (UPDATED)
│   ├── alarm_manager.py            # Alarm state management
│   ├── database.py                 # SQLite event logging (UPDATED: inlier ratio)
│   ├── flag_manager.py             # Data validity flag file I/O
│   ├── ransac_detection.py         # NEW: RANSAC homography implementation
│   ├── static_region_manager.py    # NEW: Static region mask management
│   └── ui/
│       ├── __init__.py
│       ├── main_window.py          # Main UI (UPDATED: inlier display)
│       ├── recalibration_wizard.py # Recalibration workflow UI (UPDATED: 4 steps)
│       ├── static_region_ui.py     # NEW: Interactive ROI selection
│       └── styles.css              # UI styling (if web-based)
│
├── data/
│   ├── reference_frame.pkl         # Stored reference (created at runtime)
│   ├── static_mask.pkl             # NEW: Static region mask (created at runtime)
│   ├── events.db                   # SQLite database (created at runtime)
│   └── data_validity.flag          # Flag file (created at runtime)
│
├── tests/
│   ├── __init__.py
│   ├── test_detection.py           # Unit tests for detection logic (UPDATED: RANSAC)
│   ├── test_ransac.py              # NEW: RANSAC-specific tests
│   ├── test_dynamic_scene.py       # NEW: Dynamic scene test cases
│   ├── test_database.py            # Unit tests for logging (UPDATED: inlier ratio)
│   ├── test_integration.py         # End-to-end integration tests
│   ├── sample_images/
│   │   ├── reference.jpg           # Test reference image
│   │   ├── stable_1.jpg            # No movement
│   │   ├── moved_2px.jpg           # 2-pixel shift
│   │   ├── moved_5px.jpg           # 5-pixel shift
│   │   ├── with_bubbles.jpg        # NEW: Active bubbling
│   │   ├── with_water_motion.jpg   # NEW: Water surface movement
│   │   └── with_foam.jpg           # NEW: Foam accumulation
│   └── conftest.py                 # Pytest configuration
│
└── docs/
    ├── USER_GUIDE.md               # Operator manual (UPDATED)
    ├── DEVELOPER_GUIDE.md          # Code documentation (UPDATED)
    ├── TESTING_REPORT.md           # Lab test results (generated)
    ├── DEPLOYMENT_GUIDE.md         # Installation instructions
    └── RANSAC_EXPLANATION.md       # NEW: How RANSAC works for operators
```

---

## 6. MVP Testing Strategy (UPDATED)

### 6.1 Unit Tests (UPDATED)

**Test Suite**: `tests/test_detection.py` and `tests/test_ransac.py`

**Test 1: Feature Extraction in Static Region**
```python
def test_feature_extraction_with_mask():
    """Verify ORB extracts sufficient features from masked region"""
    image = cv2.imread('tests/sample_images/reference.jpg')
    # Define static region (top 60% of image, excluding water)
    static_mask = create_mask(image.shape, y_start=0, y_end=0.6)
    
    features = extract_orb_features(image, mask=static_mask)
    assert len(features) >= 50, "Insufficient features in static region"
```

**Test 2: RANSAC Inlier Detection**
```python
def test_ransac_separates_static_from_dynamic():
    """Verify RANSAC correctly identifies static features"""
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    # Simulate: 70% static features (0px movement) + 30% dynamic (10px random)
    moved_image = create_synthetic_scene(
        ref_image, 
        camera_shift=0,  # No camera movement
        dynamic_ratio=0.3,
        dynamic_displacement=10
    )
    
    displacement, inlier_ratio, inliers = detect_movement_ransac(
        ref_image, moved_image
    )
    
    # Should detect no camera movement despite dynamic features
    assert displacement < 0.5, f"Expected ~0px, got {displacement}"
    # Should identify ~70% as inliers (static)
    assert inlier_ratio > 0.6, f"Expected >60% inliers, got {inlier_ratio*100}%"
```

**Test 3: Movement Detection with Dynamic Scene**
```python
def test_movement_detection_2px_with_bubbles():
    """Verify 2-pixel camera movement is detected despite bubbles"""
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    # Simulate: 2px camera shift + 40% features on bubbles (random motion)
    moved_image = create_synthetic_scene(
        ref_image,
        camera_shift=2,
        dynamic_ratio=0.4,
        dynamic_displacement=8
    )
    
    displacement, inlier_ratio, _ = detect_movement_ransac(
        ref_image, moved_image
    )
    
    assert displacement >= 1.5 and displacement <= 2.5, \
        f"Expected ~2px, got {displacement}"
    assert inlier_ratio > 0.3, "Need >30% inliers for reliable detection"
```

**Test 4: Scene Quality Warning**
```python
def test_low_inlier_ratio_warning():
    """Verify system warns when scene is too dynamic"""
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    # Simulate: 80% dynamic features (unrealistic but tests edge case)
    moved_image = create_synthetic_scene(
        ref_image,
        camera_shift=0,
        dynamic_ratio=0.8,
        dynamic_displacement=10
    )
    
    displacement, inlier_ratio, _ = detect_movement_ransac(
        ref_image, moved_image
    )
    
    scene_quality = assess_scene_quality(inlier_ratio)
    assert scene_quality == 'poor', f"Expected 'poor', got {scene_quality}"
    # Should trigger warning but not alarm (no camera movement)
    assert should_trigger_alarm(displacement, threshold=2.0) == False
```

**Test 5: RANSAC Robustness to Outlier Ratio**
```python
def test_ransac_with_varying_outlier_ratios():
    """Test RANSAC performance with different outlier percentages"""
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    camera_shift = 3.0  # Ground truth
    
    for outlier_ratio in [0.1, 0.3, 0.5, 0.7]:
        moved_image = create_synthetic_scene(
            ref_image,
            camera_shift=camera_shift,
            dynamic_ratio=outlier_ratio,
            dynamic_displacement=10
        )
        
        displacement, inlier_ratio, _ = detect_movement_ransac(
            ref_image, moved_image
        )
        
        # Should detect camera movement accurately up to 50% outliers
        if outlier_ratio <= 0.5:
            assert abs(displacement - camera_shift) < 0.5, \
                f"At {outlier_ratio*100}% outliers: expected ~{camera_shift}px, got {displacement}"
        
        # Should warn if >70% outliers
        if outlier_ratio > 0.7:
            assert inlier_ratio < 0.3, "Should warn about poor scene quality"
```

---

### 6.2 Integration Tests (UPDATED)

**Test Suite**: `tests/test_integration.py`

**Test 1: End-to-End Detection Flow with Dynamic Scene**
```python
def test_e2e_movement_detection_with_bubbles():
    """Test complete flow: stable → movement → alarm in dynamic DAF scene"""
    # Initialize system with reference + static mask
    system = MovementDetectionSystem()
    
    # Load reference image with static region defined
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    static_mask = create_mask(ref_image.shape, y_start=0, y_end=0.6)
    system.load_reference(ref_image, static_mask)
    
    # Process stable frame with bubbles
    stable_with_bubbles = cv2.imread('tests/sample_images/with_bubbles.jpg')
    status1 = system.process_frame(stable_with_bubbles)
    assert status1 == 'OK'
    assert system.data_validity_flag == 'VALID'
    assert system.last_inlier_ratio > 0.3  # Adequate scene quality
    
    # Process moved frame (camera shifted 5px) with bubbles
    moved_with_bubbles = shift_image_with_bubbles(
        stable_with_bubbles, 
        camera_shift=5,
        bubble_motion=True
    )
    status2 = system.process_frame(moved_with_bubbles)
    assert status2 == 'ALARM'
    assert system.data_validity_flag == 'INVALID'
    assert system.last_event_type == 'movement_detected'
    assert system.last_magnitude >= 4.5 and system.last_magnitude <= 5.5
```

**Test 2: Recalibration with Static Region Redefinition**
```python
def test_recalibration_with_region_change():
    """Test recalibration when static region needs adjustment"""
    system = MovementDetectionSystem()
    
    # Initial setup with suboptimal static region (includes water)
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    bad_mask = create_mask(ref_image.shape, y_start=0, y_end=1.0)  # Includes all
    system.load_reference(ref_image, bad_mask)
    
    # Simulate low inlier ratio due to water in region
    bubbling_frame = cv2.imread('tests/sample_images/with_bubbles.jpg')
    system.process_frame(bubbling_frame)
    initial_inlier_ratio = system.last_inlier_ratio
    assert initial_inlier_ratio < 0.5, "Expected low inliers with bad mask"
    
    # Recalibrate with improved static region (excludes bottom 40%)
    better_mask = create_mask(ref_image.shape, y_start=0, y_end=0.6)
    result = system.recalibrate(ref_image, better_mask)
    
    assert result == 'SUCCESS'
    
    # Verify improved scene quality
    system.process_frame(bubbling_frame)
    improved_inlier_ratio = system.last_inlier_ratio
    assert improved_inlier_ratio > 0.6, "Expected higher inliers with better mask"
    assert improved_inlier_ratio > initial_inlier_ratio + 0.2, \
        "Inlier ratio should improve significantly"
```

**Test 3: Database Logging with Diagnostics**
```python
def test_event_logging_with_diagnostics():
    """Verify all events include inlier ratio and scene quality"""
    system = MovementDetectionSystem()
    
    # Generate various events
    system.start()
    system.trigger_alarm(magnitude=7.2, inliers=45, total_features=78)
    system.log_scene_quality_warning(inlier_ratio=0.25)
    system.recalibrate_success(inlier_ratio=0.85)
    system.resume_operation()
    
    # Query database
    events = system.db.get_all_events()
    
    assert len(events) == 5
    
    # Check movement_detected event
    movement_event = events[1]
    assert movement_event['event_type'] == 'movement_detected'
    assert movement_event['magnitude_pixels'] == 7.2
    assert movement_event['inliers_count'] == 45
    assert movement_event['inlier_ratio'] == pytest.approx(0.577, rel=0.01)
    assert movement_event['scene_quality'] == 'good'
    
    # Check scene_quality_warning event
    warning_event = events[2]
    assert warning_event['event_type'] == 'scene_quality_warning'
    assert warning_event['inlier_ratio'] == 0.25
    assert warning_event['scene_quality'] == 'poor'
```

---

### 6.3 Dynamic Scene Test Cases (NEW)

**Test Suite**: `tests/test_dynamic_scene.py`

**Test Scenario 1: Water Surface Ripples**
```python
def test_water_ripples_no_false_alarm():
    """Water surface ripples should not trigger alarm"""
    system = MovementDetectionSystem()
    
    # Reference: calm water
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    static_mask = create_mask(ref_image.shape, y_start=0, y_end=0.6)  # Exclude water
    system.load_reference(ref_image, static_mask)
    
    # Current: same camera position but water has ripples
    rippled_water = simulate_water_ripples(
        ref_image, 
        amplitude=5,  # 5 pixel waves
        frequency=0.2
    )
    
    status = system.process_frame(rippled_water)
    
    assert status == 'OK', "Water ripples should not trigger alarm"
    assert system.last_magnitude < 1.0, "Should detect minimal camera movement"
    assert system.last_inlier_ratio > 0.3, "Static region should have good inliers"
```

**Test Scenario 2: Active Bubbling**
```python
def test_active_bubbling_no_false_alarm():
    """Active DAF bubbles should not trigger alarm"""
    system = MovementDetectionSystem()
    
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    static_mask = create_mask(ref_image.shape, y_start=0, y_end=0.6)
    system.load_reference(ref_image, static_mask)
    
    # Simulate: 100+ bubbles rising through water (not in static region)
    bubbling = simulate_daf_bubbles(
        ref_image,
        bubble_count=100,
        bubble_velocity=5,  # pixels/frame
        region='bottom_40%'  # Outside static region
    )
    
    # Process 10 frames with continuous bubbling
    false_alarms = 0
    for i in range(10):
        bubbling_frame = simulate_daf_bubbles(ref_image, frame_number=i)
        status = system.process_frame(bubbling_frame)
        if status == 'ALARM':
            false_alarms += 1
    
    # Allow max 1 false alarm in 10 frames (10% rate, target <5% over longer period)
    assert false_alarms <= 1, f"Too many false alarms: {false_alarms}/10 frames"
```

**Test Scenario 3: Foam Accumulation**
```python
def test_foam_accumulation_no_false_alarm():
    """Foam layer accumulation should not trigger alarm"""
    system = MovementDetectionSystem()
    
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    static_mask = create_mask(ref_image.shape, y_start=0, y_end=0.6)
    system.load_reference(ref_image, static_mask)
    
    # Simulate: foam layer building up on water surface
    with_foam = simulate_foam_accumulation(
        ref_image,
        foam_height=50,  # pixels from bottom
        texture_variation='high'
    )
    
    status = system.process_frame(with_foam)
    
    assert status == 'OK', "Foam accumulation should not trigger alarm"
    # Inlier ratio might be slightly lower but should still be adequate
    assert system.last_inlier_ratio > 0.3, "Should maintain adequate inliers"
```

**Test Scenario 4: Camera Movement + Bubbles (True Positive)**
```python
def test_camera_movement_with_bubbles_detected():
    """Camera movement should be detected even with active bubbles"""
    system = MovementDetectionSystem()
    
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    static_mask = create_mask(ref_image.shape, y_start=0, y_end=0.6)
    system.load_reference(ref_image, static_mask)
    
    # Simulate: 3px camera shift + active bubbling
    moved_with_bubbles = create_synthetic_scene(
        ref_image,
        camera_shift=3,  # True movement
        dynamic_ratio=0.4,  # 40% of features on bubbles
        dynamic_displacement=8
    )
    
    status = system.process_frame(moved_with_bubbles)
    
    assert status == 'ALARM', "Should detect camera movement despite bubbles"
    assert system.last_magnitude >= 2.5 and system.last_magnitude <= 3.5, \
        f"Expected ~3px, got {system.last_magnitude}"
    assert system.last_inlier_ratio > 0.3, "Should have adequate inliers"
```

**Test Scenario 5: Extreme Dynamic Scene (Poor Quality)**
```python
def test_extreme_dynamic_scene_warning():
    """Very dynamic scene should trigger scene quality warning"""
    system = MovementDetectionSystem()
    
    ref_image = cv2.imread('tests/sample_images/reference.jpg')
    # Intentionally bad mask that includes water
    bad_mask = create_mask(ref_image.shape, y_start=0, y_end=0.9)
    system.load_reference(ref_image, bad_mask)
    
    # Extreme: heavy bubbling + foam + water waves
    extreme_dynamic = simulate_extreme_daf_conditions(ref_image)
    
    status = system.process_frame(extreme_dynamic)
    
    # Should not false alarm but should warn about scene quality
    assert status == 'OK' or status == 'WARNING', \
        "Should not false alarm but may warn"
    assert system.last_inlier_ratio < 0.3, "Expected low inliers"
    
    # Check that scene quality warning was logged
    events = system.db.get_recent_events(limit=5)
    assert any(e['event_type'] == 'scene_quality_warning' for e in events), \
        "Should log scene quality warning"
```

---

### 6.4 Lab Validation Testing (UPDATED)

**Test Environment**: Controlled lab setup

**Equipment Needed**:
- Camera mounted on adjustable tripod or XY stage
- Calibrated movement mechanism (micrometer stage or known reference grid)
- **Water tank with pump** (simulate water surface motion)
- **Air pump** (simulate bubbles)
- Lux meter for lighting measurement
- Stopwatch for latency testing

**Test 1: Detection Accuracy Curve (UPDATED)**
```
Procedure:
1. Set up reference frame with camera stable
2. Define static region (top 60% of view, excluding water)
3. Move camera in 0.5-pixel increments from 0 to 10 pixels
4. Test in 4 directions: Up, Down, Left, Right
5. WHILE water surface is moving and bubbles are active
6. Record detection rate at each magnitude

Expected Results:
- 0.0-1.5 pixels: 0% detection (below threshold)
- 1.5-2.5 pixels: >90% detection (at threshold)
- 2.5+ pixels: >99% detection (above threshold)
- Inlier ratio: 50-80% typical with bubbles

Pass Criteria:
- Detection rate >95% for movements ≥2 pixels WITH dynamic scene
- False positive rate <5% with water/bubble motion
- Inlier ratio >30% in all test conditions
```

**Test 2: Dynamic Scene Robustness (NEW)**
```
Procedure:
1. Set up camera viewing water tank with DAF simulation
2. Define static region on tank walls (not water)
3. Activate water circulation (5-10 cm/s flow)
4. Activate air pump (bubble rate: 50-100 bubbles/sec)
5. Run system for 1 hour continuous operation
6. Log all detections

Test Conditions:
- Lighting: 500 lux
- Camera stable (no actual movement)
- Water surface moving continuously
- Bubbles rising through frame

Expected Results:
- Zero camera movements detected (camera is actually stable)
- Inlier ratio: 40-70% (due to bubbles/water as outliers)
- Scene quality: GOOD or FAIR

Pass Criteria:
- False positive rate <5% (max 3 false alarms in 1 hour)
- Inlier ratio remains >30% throughout test
- System continues operating without crashes
```

**Test 3: Static Region Effectiveness (NEW)**
```
Procedure:
1. Capture same scene with 3 different static region definitions:
   a) Full frame (includes water) - BASELINE
   b) Top 60% (excludes most water)
   c) Top 40% (excludes all water, minimal region)

2. For each configuration:
   - Capture reference
   - Introduce 3px camera movement
   - Activate bubbles
   - Measure: detection accuracy, false positive rate, inlier ratio

Expected Results:
Configuration (a): High false positives (20-40%), low inliers (20-40%)
Configuration (b): Low false positives (<5%), good inliers (50-70%)
Configuration (c): Low false positives (<5%), excellent inliers (70-90%)
                   BUT may have low feature count (<50) if region too small

Pass Criteria:
- Configuration (b) or (c) meets <5% false positive target
- Inlier ratio improves by 30%+ vs. full frame
- Demonstrates value of static region masking
```

**Test 4: Lighting Variation with Dynamic Scene (UPDATED)**
```
Procedure:
1. Test at 100 lux (very dim)
2. Test at 500 lux (typical indoor)
3. Test at 1000 lux (bright)
4. For each lighting level:
   - Measure feature count in static region
   - Test 2-pixel movement detection
   - WITH active water motion and bubbles
   - Record success rate and inlier ratio

Expected Results:
- 100 lux: 30-50 features, inliers 40-60% (marginal)
- 500 lux: 50-100 features, inliers 50-70% (acceptable)
- 1000 lux: 100+ features, inliers 60-80% (excellent)

Pass Criteria:
- ≥50 features in static region at 500+ lux
- >90% detection accuracy at 500+ lux with dynamic scene
- Inlier ratio >40% at 500+ lux
```

**Test 5: Long-Duration Stability with Dynamic Scene (UPDATED)**
```
Procedure:
1. Set up camera in stable position
2. Define static region (tank walls)
3. Run DAF simulation (water + bubbles) continuously
4. Run system for 8 hours continuous operation
5. Log all detected movements
6. Manually verify any alarms (video recording)

Expected Results:
- Zero false positives in stable environment WITH dynamic scene
- CPU usage remains <10% throughout
- Inlier ratio varies but stays >30% throughout
- No memory leaks (constant memory usage)

Pass Criteria:
- <1 false alarm per 8 hours (<0.125/hour) WITH bubbles
- CPU and memory usage stable
- Application remains responsive
- Inlier ratio never drops below 25% for >1 minute
```

**Test 6: Detection Latency with RANSAC (UPDATED)**
```
Procedure:
1. Set up high-speed camera (30 fps) as reference
2. Introduce movement (push camera mount)
3. WITH active bubbling
4. Record time from movement to alarm display
5. Repeat 20 times

Expected Results:
- Mean latency: 1.2-1.8 seconds (slightly higher due to RANSAC)
- 95th percentile: <2 seconds
- Maximum: <3 seconds

Pass Criteria:
- 95th percentile latency <2 seconds WITH RANSAC processing
```

---

### 6.5 Acceptance Testing (Field Trial) - UPDATED

**Location**: Jerusalem site, DAF Tank 1

**Duration**: 2 weeks parallel operation

**Procedure**:
1. **Week 1: Installation and Calibration**
   - Install MVP on laptop connected to existing camera
   - **Operator defines static region** during setup (tank walls, pipes - NOT water)
   - Verify static region has ≥50 features and >60% inlier ratio
   - Capture initial reference frame
   - Run in "monitoring only" mode (alarms logged but not actionable)
   - Operator familiarization with static region concept

2. **Week 2: Active Operation**
   - Enable alarms (actionable)
   - Operator responds to alarms per workflow
   - Log all events (movements, recalibrations, operator actions, **inlier ratios**)
   - Monitor scene quality (inlier ratio trends)
   - Collect operator feedback via daily survey

**Data Collection**:
- Movement detection events (timestamp, magnitude, **inlier ratio, scene quality**)
- False positive events (operator manually confirms no real movement)
- **Scene quality warnings** (inlier ratio <30%)
- Recalibration attempts (success/failure, time taken, **static region adjustments**)
- Operator feedback (ease of use, clarity of instructions, **static region definition**)
- System performance (CPU, memory, uptime, **RANSAC processing time**)

**Success Criteria (UPDATED)**:
- Zero missed movements (confirmed by manual inspection)
- False positive rate <5% (no more than 7 false alarms in 2 weeks) **WITH active DAF operation**
- Recalibration success rate >80% on first attempt
- **Inlier ratio >40% average** during normal operation
- **Scene quality warnings <10%** of operating time
- Operator satisfaction ≥4/5 on survey
- **Operator can define effective static region in <5 minutes**
- System uptime >99% (continuous operation except during testing)

**Go/No-Go Decision**:
- **GO**: If all success criteria met → Proceed to multi-camera development
- **NO-GO**: If critical issues found → Iterate on MVP, extend trial period

**Specific Validation for Dynamic Scene Handling**:
- Compare detection accuracy on days with high vs. low bubble generation
- Analyze correlation between bubble rate and inlier ratio
- Verify false alarm rate remains <5% even during peak bubbling
- Operator assessment: "Static region helped reduce false alarms" ≥4/5

---

## 7. MVP Deliverables (UPDATED)

### 7.1 Software Package

**Deliverable**: `camera-movement-mvp-v1.1.zip`

**Contents**:
- Complete source code (including RANSAC implementation)
- Configuration file (config.json with RANSAC parameters)
- Requirements.txt with pinned versions
- Sample test images (including dynamic scene samples)
- All documentation (including RANSAC explanation)

**Installation Package**:
```
camera-movement-mvp-v1.1/
├── README.md                   # Quick start guide (UPDATED: mentions RANSAC)
├── INSTALL.md                  # Detailed installation
├── LICENSE
├── requirements.txt
├── config.json
├── src/                        # Source code (UPDATED with RANSAC)
├── tests/                      # Test suite (UPDATED with dynamic scene tests)
└── docs/                       # Full documentation (UPDATED)
```

---

### 7.2 Documentation (UPDATED)

**User Guide** (`docs/USER_GUIDE.md`):
- Installation instructions (step-by-step with screenshots)
- First-time setup (capturing initial reference + **defining static region**)
- **Understanding static vs. dynamic regions** (with DAF examples)
- Normal operation overview
- **Interpreting inlier ratio and scene quality**
- How to respond to movement alarms
- Recalibration procedure (4 steps including region adjustment)
- Troubleshooting common issues:
  - Camera not detected
  - Insufficient features **in static region**
  - **Low inlier ratio** (scene too dynamic)
  - Recalibration failures
  - False alarms
  - **When to redefine static region**
- FAQ (including RANSAC-related questions)

**Developer Guide** (`docs/DEVELOPER_GUIDE.md`):
- Architecture overview
- Code structure and organization
- Key classes and methods
- **RANSAC implementation details**
- **Static region masking implementation**
- Configuration options (including RANSAC tuning)
- How to extend (adding new detection methods)
- Running tests (including dynamic scene tests)
- Debugging tips

**RANSAC Explanation for Operators** (`docs/RANSAC_EXPLANATION.md` - NEW):
- What is RANSAC in simple terms
- Why it's needed for DAF environments
- How it separates camera movement from scene motion
- What inlier ratio means
- When to worry about low inliers
- Illustrated with DAF-specific examples

**Testing Report** (`docs/TESTING_REPORT.md`):
- Lab test results (detection accuracy curve)
- **Dynamic scene test results** (water, bubbles, foam)
- Performance benchmarks (CPU, memory, latency, **RANSAC overhead**)
- **Inlier ratio analysis** across different conditions
- Known limitations
- Recommendations for deployment

**Deployment Guide** (`docs/DEPLOYMENT_GUIDE.md`):
- Hardware requirements and recommendations
- Camera selection guide
- Lighting requirements
- Mounting best practices
- **Static region definition guidelines** (DAF-specific)
- Network/connectivity (not applicable for MVP, but mention for future)
- Integration with external systems (data validity flag)

---

### 7.3 Demo Setup (UPDATED)

**Physical Setup**:
- Laptop with MVP software installed
- USB webcam mounted on tripod
- **Water tank with pump** (simulate water surface motion)
- **Air pump with tubing** (simulate bubbles in lower portion of view)
- Sample surface with texture (printed pattern or actual DAF tank view)
- Lighting (desk lamp providing 500+ lux)

**Demo Script** (20 minutes - extended from 15):
1. **Introduction** (2 min)
   - Overview of camera movement detection problem
   - Challenge: DAF has water, bubbles, flocs (dynamic elements)
   - MVP goals and scope

2. **Static Region Definition** (3 min - NEW)
   - Show live camera view with water and bubbles
   - Demonstrate interactive ROI selection
   - Draw rectangle around static elements (tank walls, above water line)
   - Show feature count in static region (50-100 features)
   - **Key point**: "We ignore water and bubbles by focusing on static parts"

3. **Normal Operation** (4 min)
   - Show status: OK (green)
   - Point out feature tracking in static region
   - Show inlier count: "72/78 features (92%)"
   - Activate water pump and air pump
   - **Key point**: "Water and bubbles move, but system ignores them"
   - Movement magnitude stays 0-1 pixels despite water motion

4. **Movement Detection** (4 min)
   - Deliberately push camera/tripod
   - Alarm triggers: Status → ALARM (red)
   - Movement magnitude displayed (e.g., 7.2 pixels)
   - **Show inlier ratio**: Still good (60-80%) because static region worked
   - Data validity flag changes to INVALID
   - Event logged with diagnostics

5. **Recalibration** (5 min)
   - Click "Recalibrate Camera"
   - Step 1: Operator verification
   - **Step 2: Redefine static region** (show can use existing or adjust)
   - Step 3: Capture new reference
   - Step 4: Stability test (30 seconds with live progress)
   - Success: Status returns to OK
   - Data validity flag returns to VALID

6. **Event History** (2 min)
   - Show logged events with diagnostics
   - Point out inlier ratios for each event
   - Export events to CSV
   - Open CSV in spreadsheet, show all columns

**Q&A Topics to Prepare For**:
- **What is RANSAC and why do we need it?**
  - RANSAC finds consensus among features
  - Separates camera motion (all features move together) from scene motion (random)
  - Essential for DAF because water/bubbles would cause false alarms otherwise
  
- **What is inlier ratio?**
  - Percentage of features that are static (not moving randomly)
  - Higher is better (60-90% is good)
  - <30% means scene is too dynamic, may need to redefine static region
  
- **How does static region masking work?**
  - Operator tells system which parts of view are static (tank walls, pipes)
  - System only looks at those parts for detecting camera movement
  - Water and bubbles are excluded, so they can't cause false alarms
  
- **What if lighting is poor?**
  - Need ≥50 features in static region
  - Add lighting or enlarge static region
  - Add reference markers if needed
  
- **How does this integrate with existing SCADA?**
  - MVP: Simple flag file that external systems read
  - Production: OPC UA integration (Phase 2)
  
- **Can it handle multiple cameras?**
  - MVP: Single camera only
  - Multi-camera support in Phase 1 (post-MVP)
  
- **What about other water treatment processes (not DAF)?**
  - RANSAC approach works for any process with dynamic elements
  - Static region can be adapted to clarifiers, filters, etc.
  - May need tuning for different types of motion

---

## 8. MVP Success Metrics (UPDATED)

### 8.1 Technical Performance Metrics

| Metric | Target | Measurement Method | Pass/Fail Criteria |
|--------|--------|-------------------|-------------------|
| Detection Accuracy (≥2px) | >95% | Lab controlled tests (100 trials) | Pass: ≥95% |
| **False Positive Rate (Static Scene)** | <1% | 8-hour stability test, no dynamics | Pass: <1 false alarm |
| **False Positive Rate (Dynamic Scene)** | <5% (NEW) | 8-hour with water+bubbles | Pass: <4 false alarms |
| Detection Latency | <2 sec (95th percentile) | Timestamp analysis (50 events) | Pass: ≥95% under 2 sec |
| CPU Usage | <10% average (increased) | System monitoring (1 hour) | Pass: Average <10% |
| Memory Usage | <250 MB (increased) | Process monitoring (1 hour) | Pass: <250 MB |
| **RANSAC Processing Time** | <50 ms per frame (NEW) | Profiling with cv2.findHomography | Pass: 95% < 50ms |
| Feature Count (Static Region) | ≥50 features | Feature extraction test | Pass: ≥50 at 500 lux |
| **Inlier Ratio (Normal Operation)** | >60% typical (NEW) | Real-time monitoring, dynamic scene | Pass: Mean >60% |
| **Inlier Ratio (Minimum)** | >30% (NEW) | Warning threshold | Pass: >95% of time >30% |
| Recalibration Time | <3 minutes (increased) | User observation (20 trials) | Pass: Median <3 min |
| System Uptime | >99% | Continuous operation (2 weeks) | Pass: <3.4 hours downtime |

---

### 8.2 User Experience Metrics (UPDATED)

| Metric | Target | Measurement Method | Pass/Fail Criteria |
|--------|--------|-------------------|-------------------|
| Operator Satisfaction | ≥4/5 | Post-trial survey (5-point scale) | Pass: Average ≥4 |
| UI Clarity | ≥4/5 | Survey: "Instructions are clear" | Pass: Average ≥4 |
| **Static Region Definition Time** | <5 min (NEW) | Operator observation during setup | Pass: 90% complete <5 min |
| **Static Region Effectiveness** | ≥4/5 (NEW) | Survey: "Static region helped reduce false alarms" | Pass: Average ≥4 |
| Recalibration Success (1st attempt) | >80% | Field trial data | Pass: ≥80% |
| Time to Respond to Alarm | <5 min | Alarm timestamp to acknowledgment | Pass: Median <5 min |
| Training Time Required | <45 min (increased) | Operator training observations | Pass: <45 min to proficiency |

**Survey Questions (UPDATED)** (5-point scale: 1=Strongly Disagree, 5=Strongly Agree):
1. The alarm clearly indicates when camera movement occurs
2. The recalibration procedure is easy to follow
3. **Defining the static region during setup was straightforward** (NEW)
4. **The static region helps prevent false alarms from water and bubbles** (NEW)
5. **I understand what inlier ratio means and when to worry about it** (NEW)
6. I understand when to recalibrate vs. when to call maintenance
7. The system helps me trust the vision data
8. I would recommend using this system

---

### 8.3 Business Value Metrics (UPDATED - with RANSAC benefits)

**Note**: These are forward-looking estimates to validate with field trial data

| Metric | Baseline (Without MVP) | Target (With MVP) | Estimated Impact |
|--------|----------------------|-------------------|------------------|
| Undetected Camera Movements | 1-2 per month | <1 per year | 95% reduction |
| **False Alarms (from dynamic scene)** | N/A (no system) | <5% rate with RANSAC (NEW) | Prevents alarm fatigue |
| Chemical Waste from False Dosing | $15K/year | <$5K/year | $10K savings |
| Manual Data Reviews | 80 hours/quarter | <20 hours/quarter | 60 hour savings |
| Treatment Failures | 2 per quarter | <1 per quarter | 50% reduction |
| Time to Identify Camera Issues | 4-24 hours (reactive) | <1 minute (proactive) | 95% faster detection |
| **Operator Training Time** | N/A | <45 min (NEW) | Quick adoption |

**ROI Calculation (UPDATED)**:
```
MVP Development Cost: ~$25K (5 weeks × $5K/week, increased from $20K)
  - Additional $5K for RANSAC implementation + dynamic scene testing

Annual Savings:
  - Chemical waste prevention: $10K
  - Labor savings (60 hrs × $50/hr × 4 quarters): $12K
  - Treatment failure prevention: $5K (conservative)
  - Reduced false alarm investigation: $2K (NEW - less wasted time on false alarms)
Total Annual Savings: ~$29K

Simple Payback Period: $25K / $29K = 0.86 years ≈ 10.3 months
```

**RANSAC-Specific Benefits**:
- Enables deployment in DAF environments (otherwise infeasible)
- Prevents alarm fatigue from false positives
- No need for operators to manually filter out water/bubble motion
- Quantifiable via: False alarm rate with vs. without RANSAC

---

## 9. MVP Limitations & Assumptions (UPDATED)

### 9.1 Known Limitations

**Technical Limitations**:
1. **Single Camera Only**: Cannot monitor multiple cameras simultaneously
2. **Manual Static Region Definition**: Operator must define region (no auto-detection)
3. **No Movement Classification**: Does not categorize sudden vs. gradual movements
4. **Manual Recalibration**: No automatic recalibration capability
5. **Limited Lighting Range**: Requires 500+ lux for reliable feature detection in static region
6. **No Historical Analytics**: Only displays last 10 events in UI
7. **Local Deployment Only**: No remote access or cloud connectivity
8. **RANSAC Performance Dependency**: Higher CPU usage (~10% vs 5% baseline)

**Operational Limitations**:
1. **Single Operator**: No multi-user support or authentication
2. **No SCADA Integration**: Only basic flag file interface
3. **No Alarm Routing**: No email, SMS, or escalation
4. **Basic Logging**: SQLite only, no enterprise historian integration
5. **Limited Configuration**: Minimal tuning options compared to production system
6. **Static Region Fixed**: Must redefine manually if camera view changes significantly

**Integration Limitations**:
1. **Flag File Polling**: External systems must poll file (not push notification)
2. **No API**: No programmatic interface for external systems
3. **No Redundancy**: Single point of failure (no backup/failover)

**RANSAC-Specific Limitations (NEW)**:
1. **Minimum Inlier Requirement**: Needs >30% static features (may fail in extremely dynamic scenes)
2. **Computational Overhead**: ~5-10% more CPU usage than naive matching
3. **Static Region Dependency**: Effectiveness depends on operator's static region definition

---

### 9.2 Key Assumptions (UPDATED)

**Hardware Assumptions** (🚩 Require validation):
1. 🚩 Camera provides ≥10 fps at 640x480 resolution
2. 🚩 Laptop/PC has Intel i5 (6th gen) or better CPU (for RANSAC processing)
3. 🚩 Lighting in DAF tank area is ≥500 lux in static region or can be supplemented
4. 🚩 Camera mounting is rigid (not flexible stand)
5. 🚩 USB cable length is <5 meters (USB 2.0 limit) or <3 meters (USB 3.0 recommended)

**Environmental Assumptions** (🚩 Require site survey):
1. 🚩 Sufficient texture in static region for feature detection (tank walls, pipes, equipment visible)
2. 🚩 Static region can be defined that excludes water surface and bubble zone
3. 🚩 Static region contains ≥50 trackable features (NEW)
4. 🚩 Stable lighting (no rapid on/off cycling, flickering)
5. 🚩 Vibration is low-frequency (<1 Hz) or can be isolated from camera
6. 🚩 Temperature is within 0-40°C range (standard PC operating range)
7. 🚩 No water spray directly on camera lens
8. 🚩 DAF bubble rate is typical (50-200 bubbles/sec) not extreme (NEW)

**Operational Assumptions** (🚩 Require validation):
1. 🚩 Operator is available to respond to alarms within 5 minutes
2. 🚩 Operator can physically access camera mount for inspection
3. 🚩 Operator can identify static elements in scene (tank walls vs. water) (NEW)
4. 🚩 Operator can define effective static region in <5 minutes with guidance (NEW)
5. 🚩 Maintenance can respond to mount issues within 1-2 hours
6. 🚩 Flocky coagulant system can read and respect flag file
7. 🚩 Recalibration can be performed during normal operation (no shutdown required)
8. 🚩 Inlier ratio >30% is achievable in normal DAF operation with proper static region (NEW)

**Performance Assumptions** (🚩 Require lab testing):
1. 🚩 RANSAC provides sufficient accuracy for 2-pixel detection (NEW)
2. 🚩 RANSAC processing time <50ms per frame on target hardware (NEW)
3. 🚩 Inlier ratio >30% is adequate for reliable detection (NEW)
4. 🚩 1 Hz analysis frequency is sufficient for detection within 2 seconds
5. 🚩 30-second stability test reliably confirms camera is stable
6. 🚩 False positive rate <5% is acceptable for pilot deployment with dynamic scenes (NEW)
7. 🚩 Static region masking reduces false positives by >50% vs. full frame (NEW)

---

### 9.3 Out of Scope (Post-MVP Features) - UNCHANGED

The following features are **explicitly deferred** to post-MVP development:

**Phase 1 (Post-MVP Validation)**:
- Multi-camera support (2-5 cameras)
- OPC UA integration for SCADA
- Email notifications
- Movement classification (sudden/gradual/vibration)
- Enhanced UI with graphs and analytics

**Phase 2 (Production Features)**:
- Remote access and monitoring
- User authentication and roles
- SMS and escalation
- Historical analytics dashboard
- Mobile app for alarms
- **Automatic static region suggestion** (ML-based)

**Phase 3 (Enterprise Scale)**:
- Multi-site support (central monitoring)
- Cloud connectivity
- Predictive maintenance (vibration pattern analysis)
- Advanced diagnostics
- Auto-recalibration
- **Adaptive RANSAC parameters** based on scene conditions

---

## 10. MVP Development Timeline (UPDATED)

**Total Duration**: 5 weeks (increased from 4 weeks)

### Week 1: Core Detection Engine with RANSAC (UPDATED)

**Deliverables**:
- Feature extraction (ORB) implemented
- **RANSAC homography implementation** (NEW)
- **Inlier/outlier separation** (NEW)
- Frame-to-frame matching implemented
- Threshold checking logic
- Reference frame management
- Unit tests passing (>90% coverage)

**Tasks**:
```
Day 1-2: Project setup, OpenCV integration, basic frame capture
Day 3-4: ORB feature extraction and RANSAC homography algorithm
Day 5: Threshold detection, displacement calculation, inlier ratio tracking
```

**Exit Criteria**:
- Can detect 2-pixel movement in test images using RANSAC
- Inlier/outlier separation working correctly
- Feature extraction extracts ≥50 features from sample images
- Unit tests pass (including RANSAC tests)
- Code committed to version control

---

### Week 2: Static Region Masking + UI (UPDATED)

**Deliverables**:
- **Static region masking UI** (interactive ROI selection) (NEW)
- Basic UI (PyQt5 or Flask)
- Camera integration (live video display)
- Alarm visualization with diagnostics
- Flag file mechanism
- SQLite logging (with inlier ratio)
- Configuration file loading

**Tasks**:
```
Day 1-2: Static region selection UI (click and drag rectangle)
Day 3: UI framework setup, main window layout with diagnostics
Day 4: Camera integration, live video display with mask overlay
Day 5: Alarm visualization with inlier ratio, flag file + database logging
```

**Exit Criteria**:
- Operator can define static region interactively
- UI displays live camera feed with static region overlay
- Status changes OK → ALARM when movement detected
- Inlier ratio displayed in real-time
- Flag file updates correctly
- Events logged to database with diagnostics
- Application runs without crashes

---

### Week 3: Recalibration & Dynamic Scene Testing (UPDATED)

**Deliverables**:
- Recalibration wizard (4 steps including static region adjustment)
- Stability test implementation with inlier monitoring
- **Dynamic scene test cases** (water, bubbles, foam) (NEW)
- Integration testing complete
- Lab testing with controlled movements
- **Lab testing with simulated dynamic scene** (NEW)
- Performance profiling and optimization

**Tasks**:
```
Day 1-2: Recalibration wizard UI and logic (4 steps)
Day 3: Stability test with inlier ratio monitoring
Day 4: Dynamic scene test cases (simulate bubbles, water motion)
Day 5: Lab testing with camera + water tank + air pump
```

**Exit Criteria**:
- Recalibration wizard completes successfully with region adjustment
- Stability test correctly identifies stable/unstable camera
- Integration tests pass
- **Lab test shows >90% detection accuracy with dynamic scene**
- **False positive rate <5% with water/bubble motion**
- **Inlier ratio >40% average in dynamic scene tests**
- CPU usage <10%, memory <250 MB

---

### Week 4: Documentation & Polish (UPDATED)

**Deliverables**:
- User Guide (with static region and RANSAC explanations)
- Developer Guide (with RANSAC implementation details)
- **RANSAC Explanation for Operators** (NEW)
- Testing Report (with dynamic scene results)
- Deployment Guide (with static region guidelines)
- README.md with quickstart
- Bug fixes and polish
- Demo setup prepared (with water + bubbles)

**Tasks**:
```
Day 1: Write User Guide (installation, operation, static region)
Day 2: Write Developer Guide, RANSAC Explanation, Testing Report
Day 3: Bug fixes, UI polish, error handling
Day 4: Demo preparation with water tank, final testing
Day 5: Package release, hand-off
```

**Exit Criteria**:
- All documentation complete and reviewed
- Known bugs fixed or documented
- Demo runs successfully with dynamic scene
- Package ready for deployment
- Acceptance testing plan prepared

---

### Week 5: Field Validation Prep (NEW)

**Deliverables**:
- Field validation test plan
- Operator training materials (with static region focus)
- Site survey checklist
- Data collection templates
- Troubleshooting guide for field issues
- Backup/recovery procedures

**Tasks**:
```
Day 1-2: Create field validation test plan and checklists
Day 3: Develop operator training materials with hands-on exercises
Day 4: Create data collection templates and analysis scripts
Day 5: Final system testing and validation readiness review
```

**Exit Criteria**:
- Field validation plan approved
- Training materials tested with 2-3 operators
- Data collection process defined
- System ready for 2-week field trial
- Support escalation procedure documented

---

### Resource Requirements (UPDATED)

**Team**:
- 1× Computer Vision Engineer (full-time, 5 weeks - increased from 4)
- 0.5× QA Engineer (part-time, weeks 3-5)
- 0.25× Technical Writer (part-time, weeks 4-5)
- 0.25× Project Manager (oversight, weekly check-ins)

**Equipment**:
- Development laptop (provided)
- USB webcam for testing (~$50)
- Adjustable tripod/stand (~$100)
- **Small water tank** (~$50) (NEW)
- **Air pump with tubing** (~$30) (NEW)
- Printed pattern for testing (~$20)
- Lux meter (~$50)

**Total MVP Budget**: ~$25,000 (increased from $20,000)
- Labor (5 weeks × $5K/week): $25,000 (increased from 4 weeks × $4.5K = $18,000)
- Equipment: $300 (increased from $220)
- Contingency: Absorbed in labor estimate

**Justification for increased timeline**:
- RANSAC implementation: +3 days
- Static region masking UI: +2 days
- Dynamic scene test cases: +2 days
- Enhanced documentation: +2 days
- Field validation prep: +5 days
- **Total**: +14 days ≈ +1 week (rounded to full week for planning)

---

## 11. Risk Management (UPDATED)

### 11.1 Technical Risks

**Risk 1: ORB Features Insufficient in Dark/Uniform Scenes**
- **Likelihood**: Medium
- **Impact**: Critical (cannot detect movement without features)
- **Mitigation**:
  - Document minimum lighting requirements (500+ lux **in static region**)
  - Provide clear error message: "Insufficient features in static region"
  - Suggest adding reference markers (ArUco or printed pattern) to static areas
  - **Suggest enlarging static region** to include more textured areas
  - Test with sample images during week 1 to identify early
- **Contingency**: If ORB fails even in static region, implement ArUco marker detection as fallback

**Risk 2: RANSAC Insufficient for Extremely Dynamic Scenes** (UPDATED - replaced "False Positives from Water")
- **Likelihood**: Low-Medium
- **Impact**: Medium (system may issue scene quality warnings frequently)
- **Mitigation**:
  - **Static region masking** is primary defense (exclude dynamic areas)
  - Make RANSAC parameters tunable (reprojection threshold, min inliers)
  - Display clear guidance when inlier ratio <30%
  - Operator can adjust static region to improve inlier ratio
  - Document best practices for static region definition
  - Test at actual DAF tank during field trial
- **Contingency**: If inlier ratio consistently <20%, add feature stability filtering (track features over multiple frames, use only stable ones)

**Risk 3: Static Region Definition Too Difficult for Operators** (NEW)
- **Likelihood**: Medium
- **Impact**: High (incorrect region defeats purpose of RANSAC)
- **Mitigation**:
  - Provide clear visual guidance during setup (show examples)
  - Display feature count in real-time as operator adjusts region
  - Warn if region likely includes water (bottom X% of frame)
  - **Provide "suggested region" default** (top 60% of frame)
  - Allow quick iteration: preview features → adjust → preview again
  - Include in operator training with hands-on practice
- **Contingency**: Implement ML-based static region suggestion in Phase 1 (outside MVP scope)

**Risk 4: RANSAC Processing Time Too Slow** (NEW)
- **Likelihood**: Low
- **Impact**: High (cannot meet 1 Hz analysis target)
- **Mitigation**:
  - Profile RANSAC performance in week 1
  - Use efficient cv2.findHomography() implementation
  - Reduce max features if needed (from 500 to 300)
  - Consider optimized RANSAC parameters (fewer iterations)
  - Test on target hardware early (Intel i5 minimum)
- **Contingency**: Reduce analysis frequency to 0.5 Hz (2 second intervals) if needed

**Risk 5: USB Camera Disconnects or Unstable Connection** (UNCHANGED)
- **Likelihood**: Low-Medium
- **Impact**: High (system non-functional)
- **Mitigation**:
  - Implement camera reconnection logic (retry every 5 seconds)
  - Display clear error: "Camera lost, reconnecting..."
  - Log camera disconnection events
  - Recommend industrial USB camera with locking connector
- **Contingency**: Switch to GigE camera if USB proves unreliable

---

### 11.2 Operational Risks (UPDATED)

**Risk 6: Operator Ignores Scene Quality Warnings** (NEW)
- **Likelihood**: Medium
- **Impact**: Medium (poor detection reliability, but not false alarms)
- **Mitigation**:
  - Make scene quality warnings visible but not alarming (yellow, not red)
  - Provide clear guidance: "Consider redefining static region"
  - Track frequency of warnings in logs for post-trial analysis
  - Operator training emphasizes when warnings are actionable
- **Contingency**: Add persistent warning banner if inlier ratio <30% for >1 hour

**Risk 7: Operator Defines Ineffective Static Region**
- **Likelihood**: Medium-High initially (improves with training)
- **Impact**: High (defeats purpose of RANSAC, false alarms return)
- **Mitigation**:
  - Suggest default region (top 60%) as starting point
  - Real-time feedback during definition (feature count, preview)
  - Warning if region includes bottom 30% of frame
  - Operator training with examples of good/bad regions
  - Allow quick iteration without restarting system
  - **Post-trial analysis**: Compare inlier ratios with region definitions
- **Contingency**: Provide "region templates" for common DAF tank configurations

**Risk 8: Recalibration Procedure Too Complex** (UPDATED - now 4 steps)
- **Likelihood**: Low-Medium
- **Impact**: Medium (delays recovery, operator frustration)
- **Mitigation**:
  - Keep wizard to 4 simple steps (added region adjustment is optional)
  - Allow "use existing region" quick path
  - Test with 5 operators during week 4
  - Iterate based on usability feedback
  - Provide clear visual guidance at each step
- **Contingency**: Simplify to 3-step process (auto-reuse existing region unless operator explicitly requests change)

**Risk 9: Field Trial Environment Too Challenging** (UNCHANGED)
- **Likelihood**: Medium
- **Impact**: High (MVP fails validation)
- **Mitigation**:
  - Conduct site survey before installation
  - Validate lighting, camera specs, mounting
  - **Validate that static region can be defined with ≥50 features**
  - Install supplemental lighting if needed
  - Test with sample images from site if possible
  - Have backup air pump control (reduce bubble rate temporarily for initial setup)
- **Contingency**: Choose easier site for MVP trial, defer challenging site to production

---

### 11.3 Schedule Risks (UPDATED)

**Risk 10: Development Delays Due to RANSAC Complexity** (NEW)
- **Likelihood**: Low-Medium
- **Impact**: Medium (delayed field trial)
- **Mitigation**:
  - Use proven cv2.findHomography() implementation (not custom)
  - Front-load RANSAC work (week 1)
  - Have fallback: naive matching if RANSAC fails by day 3
  - Leverage existing RANSAC examples and tutorials
  - Daily stand-ups to identify blockers early
- **Contingency**: Descope static region UI if needed to hit RANSAC implementation deadline

**Risk 11: Field Trial Delayed Due to Site Access** (UNCHANGED)
- **Likelihood**: Low-Medium
- **Impact**: Medium (delayed validation)
- **Mitigation**:
  - Coordinate with site contact (Ofer) early
  - Identify backup trial site
  - Prepare demo setup for lab validation as alternative
  - **Ensure site has DAF with active bubbling** for realistic test
- **Contingency**: Extend lab testing with simulated dynamic scene, defer field trial by 2 weeks

---

### Risk Matrix (UPDATED)

| Risk | Likelihood | Impact | Priority | Mitigation Owner |
|------|-----------|--------|----------|------------------|
| ORB features insufficient | Medium | Critical | **P0** | CV Engineer |
| RANSAC insufficient (extreme dynamic) | Low-Medium | Medium | **P1** | CV Engineer |
| Static region definition too hard | Medium | High | **P1** | CV Engineer + UX |
| RANSAC processing too slow | Low | High | **P1** | CV Engineer |
| USB camera unstable | Low-Medium | High | **P1** | CV Engineer |
| Scene quality warnings ignored | Medium | Medium | **P2** | PM / Training |
| Ineffective static region defined | Medium-High | High | **P1** | Training / CV Engineer |
| Recalibration too complex | Low-Medium | Medium | **P2** | QA / CV Engineer |
| Field site challenging | Medium | High | **P1** | PM |
| RANSAC development delays | Low-Medium | Medium | **P1** | PM |
| Field trial delayed | Low-Medium | Medium | **P2** | PM |

---

## 12. Post-MVP Roadmap (UNCHANGED)

### 12.1 Validation Phase (2 weeks after MVP)

**Goal**: Validate MVP at Jerusalem site with real operators

**Activities**:
1. **Installation** (Day 1-2):
   - Install MVP on laptop at Jerusalem site
   - Connect to existing camera (or install new USB camera)
   - **Operator defines static region** with guidance
   - Capture initial reference frame
   - Verify lighting and feature count in static region
   - **Verify inlier ratio >40%** during initial monitoring

2. **Parallel Operation** (Day 3-7):
   - Run MVP in "monitoring only" mode
   - Log all detections but don't make actionable
   - Compare with manual inspection logs
   - **Monitor inlier ratio trends**
   - Tune threshold and parameters if needed

3. **Active Operation** (Day 8-14):
   - Enable alarms (actionable)
   - Operators respond per workflow
   - Collect usage data and feedback
   - **Track scene quality warnings and static region adjustments**
   - Document any issues or suggestions

**Validation Criteria (UPDATED)**:
- ✅ Zero missed movements (confirmed by manual checks)
- ✅ False positive rate <5% **with active DAF bubbling**
- ✅ **Inlier ratio >40% average** during normal operation
- ✅ **Scene quality warnings <10%** of operating time
- ✅ Recalibration success rate >80%
- ✅ Operator satisfaction ≥4/5
- ✅ **Operator can define effective static region in <5 minutes**
- ✅ System uptime >99%

**Deliverable**: Validation Report with go/no-go decision

---

### 12.2 Phase 1: Multi-Camera Support (6 weeks)

**Goal**: Support 2-5 cameras at a single site

**New Features**:
- Multi-camera management (add/remove cameras)
- Per-camera configuration and static regions
- Grid view UI (2×2 or 3×3 layout)
- Consolidated alarm panel with diagnostics
- Multi-camera event logging
- **Aggregated inlier ratio statistics**

**Acceptance Criteria**:
- Support 5 cameras with <20% CPU usage (increased from 15% due to RANSAC)
- Independent alarm/recalibration per camera
- Single UI shows all cameras with diagnostics

---

### 12.3 Phase 2: SCADA Integration (4 weeks)

**Goal**: Replace flag file with OPC UA integration

**New Features**:
- OPC UA server implementation
- Per-camera data validity tags
- Alarm status tags
- Movement magnitude tags
- **Inlier ratio tags** for diagnostics
- Configuration via SCADA HMI

**Acceptance Criteria**:
- OPC UA tags update <1 second
- Compatible with Rockwell and Siemens PLCs
- Backward compatible with flag file

---

### 12.4 Phase 3: Enhanced Detection (4 weeks)

**Goal**: Improve detection robustness and classification

**New Features**:
- Movement classification (sudden/gradual/vibration)
- Adaptive thresholding (learn baseline vibration)
- Multiple detection methods (ORB + Homography + Markers)
- Confidence scoring
- **ML-based static region suggestion** (train on labeled examples)
- **Adaptive RANSAC parameters** based on scene dynamics

**Acceptance Criteria**:
- Classification accuracy >95%
- False positive rate <1% (down from 5%)
- Handles high-vibration environments
- **Auto-suggests effective static region with 80% accuracy**

---

### 12.5 Phase 4: Enterprise Features (8 weeks)

**Goal**: Production-ready system for multi-site deployment

**New Features**:
- Email and SMS notifications
- Alarm escalation
- User authentication and roles
- Historical analytics dashboard (including inlier ratio trends)
- Remote monitoring
- Multi-site central dashboard
- Cloud connectivity
- **Scene quality analytics and recommendations**

**Acceptance Criteria**:
- Support 100+ cameras across 10+ sites
- Role-based access control
- 99.9%+ uptime SLA
- Mobile app for notifications
- **Predictive maintenance based on inlier ratio degradation**

---

### Estimated Timeline to Production (UPDATED)

```
MVP Development:           5 weeks  ████████████████████  (increased from 4)
Validation Phase:          2 weeks  ████████
Phase 1 (Multi-Camera):    6 weeks  ████████████████████████
Phase 2 (SCADA):           4 weeks  ████████████████
Phase 3 (Enhanced Detect): 4 weeks  ████████████████
Phase 4 (Enterprise):      8 weeks  ████████████████████████████████

Total:                    29 weeks (~7.25 months)
```

**Comparison to Original PRD**: Original estimated 26 weeks. This approach adds 3 weeks (1 week for RANSAC MVP, 2 weeks distributed across later phases) but provides critical dynamic scene handling.

---

## 13. Acceptance Criteria Summary (UPDATED)

The MVP is **ACCEPTED** if all of the following criteria are met:

### 13.1 Technical Performance ✅

| Criterion | Target | Method |
|-----------|--------|--------|
| Detection accuracy | >95% for ≥2px | Lab tests (100 trials) WITH dynamic scene |
| False positive rate (static) | <1% | 8-hour stability test, no dynamics |
| **False positive rate (dynamic)** | <5% (NEW) | **8-hour test with water+bubbles** |
| Detection latency | <2 sec (95th) | Timestamp analysis |
| CPU usage | <10% average | System monitoring |
| Memory usage | <250 MB | Process monitoring |
| **RANSAC processing time** | <50ms (95th) (NEW) | **Profiling cv2.findHomography** |
| Feature count (static region) | ≥50 at 500 lux | Feature extraction test |
| **Inlier ratio (normal)** | >60% typical (NEW) | **Real-time monitoring, dynamic scene** |
| **Inlier ratio (minimum)** | >30% (NEW) | **Warning threshold, 95% of time** |
| Recalibration time | <3 min | User observation |
| System uptime | >99% | 2-week field trial |

### 13.2 Functional Requirements ✅

- ✅ Detects camera movement ≥2 pixels **using RANSAC**
- ✅ **Separates static features (camera) from dynamic (scene)** (NEW)
- ✅ **Operator can define static region** interactively (NEW)
- ✅ **Inlier ratio displayed and tracked** (NEW)
- ✅ Displays visual alarm within 2 seconds
- ✅ Updates data validity flag correctly
- ✅ Logs all events to database **with diagnostics (inlier ratio)**
- ✅ Recalibration wizard completes successfully **(4 steps including region adjustment)**
- ✅ Resume operation clears alarm and restores valid flag
- ✅ UI remains responsive during detection
- ✅ Configuration loads from config.json **(including RANSAC parameters)**
- ✅ **Scene quality warning triggers at <30% inliers** (NEW)

### 13.3 User Experience ✅

- ✅ 80%+ untrained operators can respond to alarm correctly
- ✅ **80%+ operators can define effective static region in <5 minutes** (NEW)
- ✅ Recalibration success rate >80% on first attempt
- ✅ Operator satisfaction ≥4/5 (survey)
- ✅ **Static region effectiveness ≥4/5** (survey: "helped reduce false alarms") (NEW)
- ✅ Clear error messages for common issues
- ✅ Training time <45 minutes to proficiency (increased from 30 min)

### 13.4 Documentation ✅

- ✅ User Guide covers installation, operation, troubleshooting, **static region definition**
- ✅ **RANSAC Explanation for Operators** (simple, non-technical) (NEW)
- ✅ Developer Guide covers architecture and code structure, **RANSAC implementation**
- ✅ Testing Report demonstrates acceptance criteria met, **dynamic scene results**
- ✅ Deployment Guide covers hardware and installation, **static region guidelines**
- ✅ README provides quickstart instructions

### 13.5 Field Validation ✅

- ✅ 2-week field trial completed at Jerusalem site
- ✅ Zero missed movements (confirmed by manual inspection)
- ✅ False positive rate <5% in field conditions **with active DAF operation**
- ✅ **Inlier ratio >40% average** during trial (NEW)
- ✅ **Scene quality warnings <10%** of operating time (NEW)
- ✅ System operates continuously without intervention
- ✅ Operator feedback is positive (≥4/5 satisfaction)
- ✅ **Static region definition feedback positive** (≥4/5) (NEW)

**Decision**: If all criteria met → **PROCEED** to Phase 1 development. If any critical criterion fails → **ITERATE** on MVP.

---

## 14. Appendix A: Sample Code Snippets (UPDATED)

### RANSAC-Based Detection Engine

```python
import cv2
import numpy as np
from typing import Tuple, Optional

class MovementDetectionEngine:
    def __init__(self, 
                 threshold_pixels: float = 2.0, 
                 min_features: int = 50,
                 min_inlier_ratio: float = 0.3,
                 ransac_threshold: float = 2.0):
        self.threshold_pixels = threshold_pixels
        self.min_features = min_features
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold = ransac_threshold
        
        self.orb = cv2.ORB_create(nfeatures=500)
        
        self.reference_frame = None
        self.reference_keypoints = None
        self.reference_descriptors = None
        self.static_mask = None
        
    def set_static_mask(self, mask: np.ndarray):
        """Set static region mask for feature extraction"""
        self.static_mask = mask
        
    def set_reference(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Capture and validate reference frame with static masking"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features only from static region
        keypoints, descriptors = self.orb.detectAndCompute(
            gray, 
            mask=self.static_mask
        )
        
        if len(keypoints) < self.min_features:
            return False, f"Insufficient features in static region: {len(keypoints)} (need {self.min_features})"
        
        self.reference_frame = gray
        self.reference_keypoints = keypoints
        self.reference_descriptors = descriptors
        return True, f"Reference set with {len(keypoints)} features in static region"
    
    def detect_movement(self, frame: np.ndarray) -> Tuple[float, int, float, int]:
        """
        Detect movement using RANSAC-based homography
        
        Returns:
            displacement: Median displacement of inlier features (pixels)
            total_features: Total number of matched features
            inlier_ratio: Ratio of inliers to total (0.0-1.0)
            inlier_count: Number of inlier features
        """
        if self.reference_descriptors is None:
            raise ValueError("Reference frame not set")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features from static region only
        keypoints, descriptors = self.orb.detectAndCompute(
            gray, 
            mask=self.static_mask
        )
        
        if descriptors is None or len(keypoints) < 10:
            return 0.0, 0, 0.0, 0  # Not enough features
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(self.reference_descriptors, descriptors)
        
        if len(matches) < 10:
            return 0.0, len(matches), 0.0, 0  # Not enough matches
        
        # Extract matched point coordinates
        ref_pts = np.float32([
            self.reference_keypoints[m.queryIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        
        curr_pts = np.float32([
            keypoints[m.trainIdx].pt for m in matches
        ]).reshape(-1, 1, 2)
        
        # RANSAC: Fit homography and separate inliers from outliers
        H, inlier_mask = cv2.findHomography(
            ref_pts, 
            curr_pts,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=0.995,
            maxIters=2000
        )
        
        if H is None:
            # Homography failed (degenerate case)
            return 0.0, len(matches), 0.0, 0
        
        # Separate inliers (static features) from outliers (dynamic features)
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = np.sum(inliers)
        inlier_ratio = inlier_count / len(inliers)
        
        if inlier_count < 3:
            # Not enough inliers for reliable estimate
            return 0.0, len(matches), inlier_ratio, inlier_count
        
        # Calculate displacement using INLIERS only
        static_ref_pts = ref_pts[inliers].reshape(-1, 2)
        static_curr_pts = curr_pts[inliers].reshape(-1, 2)
        
        displacements = np.linalg.norm(
            static_curr_pts - static_ref_pts, 
            axis=1
        )
        
        median_displacement = np.median(displacements)
        
        return median_displacement, len(matches), inlier_ratio, inlier_count
    
    def should_trigger_alarm(self, movement: float, inlier_ratio: float) -> bool:
        """
        Check if movement exceeds threshold AND scene quality is adequate
        """
        movement_exceeds = movement >= self.threshold_pixels
        scene_quality_ok = inlier_ratio >= self.min_inlier_ratio
        
        # Only alarm if BOTH conditions met
        return movement_exceeds and scene_quality_ok
    
    def assess_scene_quality(self, inlier_ratio: float) -> str:
        """Assess scene quality based on inlier ratio"""
        if inlier_ratio >= 0.7:
            return 'excellent'
        elif inlier_ratio >= 0.5:
            return 'good'
        elif inlier_ratio >= 0.3:
            return 'fair'
        else:
            return 'poor'
```

### Static Region Mask Creation

```python
import cv2
import numpy as np

def create_rectangular_mask(image_shape: tuple, 
                             x: int, y: int, 
                             width: int, height: int) -> np.ndarray:
    """
    Create rectangular mask for static region
    
    Args:
        image_shape: (height, width) of image
        x, y: Top-left corner of rectangle
        width, height: Size of rectangle
    
    Returns:
        Binary mask (255 inside rectangle, 0 outside)
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    mask[y:y+height, x:x+width] = 255
    return mask

def create_top_region_mask(image_shape: tuple, 
                             exclude_bottom_fraction: float = 0.4) -> np.ndarray:
    """
    Create mask excluding bottom portion of image (typical for DAF)
    
    Args:
        image_shape: (height, width) of image
        exclude_bottom_fraction: Fraction of image height to exclude (0.0-1.0)
    
    Returns:
        Binary mask
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Include top (1 - exclude_bottom_fraction) of image
    include_height = int(height * (1.0 - exclude_bottom_fraction))
    mask[0:include_height, :] = 255
    
    return mask

# Example usage in static region UI
class StaticRegionSelector:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.rect_start = None
        self.rect_end = None
        self.mask = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for rectangle selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.rect_end = (x, y)
            self.create_mask()
    
    def create_mask(self):
        """Create mask from selected rectangle"""
        if self.rect_start and self.rect_end:
            x1, y1 = self.rect_start
            x2, y2 = self.rect_end
            
            # Ensure x1 < x2 and y1 < y2
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            self.mask = create_rectangular_mask(
                self.frame.shape,
                x1, y1,
                x2 - x1, y2 - y1
            )
    
    def count_features(self, orb) -> int:
        """Count ORB features in selected region"""
        if self.mask is None:
            return 0
        
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        keypoints, _ = orb.detectAndCompute(gray, mask=self.mask)
        return len(keypoints) if keypoints else 0
    
    def visualize(self) -> np.ndarray:
        """Draw rectangle and feature points on frame"""
        vis = self.frame.copy()
        
        if self.rect_start and self.rect_end:
            cv2.rectangle(vis, self.rect_start, self.rect_end, (0, 255, 0), 2)
        
        if self.mask is not None:
            # Show features in masked region
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=500)
            keypoints, _ = orb.detectAndCompute(gray, mask=self.mask)
            
            if keypoints:
                for kp in keypoints:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    cv2.circle(vis, (x, y), 3, (0, 255, 255), -1)
        
        return vis
```

### Flag File Manager (UNCHANGED)

```python
import os
from pathlib import Path

class FlagManager:
    def __init__(self, flag_path: str = "./data_validity.flag"):
        self.flag_path = Path(flag_path)
        self.ensure_file_exists()
    
    def ensure_file_exists(self):
        """Create flag file if it doesn't exist"""
        if not self.flag_path.exists():
            self.set_valid()
    
    def set_valid(self):
        """Set data validity flag to VALID"""
        self.flag_path.write_text("VALID")
    
    def set_invalid(self):
        """Set data validity flag to INVALID"""
        self.flag_path.write_text("INVALID")
    
    def is_valid(self) -> bool:
        """Check if data is currently valid"""
        try:
            content = self.flag_path.read_text().strip()
            return content == "VALID"
        except:
            return False
```

### Database Logger (UPDATED with inlier ratio)

```python
import sqlite3
from datetime import datetime
from typing import Optional

class EventLogger:
    def __init__(self, db_path: str = "./events.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                magnitude_pixels REAL,
                features_tracked INTEGER,
                inliers_count INTEGER,
                inlier_ratio REAL,
                scene_quality TEXT,
                notes TEXT
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_inlier_ratio ON events(inlier_ratio)')
        conn.commit()
        conn.close()
    
    def log_event(self, 
                  event_type: str, 
                  magnitude: Optional[float] = None,
                  features: Optional[int] = None, 
                  inliers: Optional[int] = None,
                  inlier_ratio: Optional[float] = None,
                  scene_quality: Optional[str] = None,
                  notes: Optional[str] = None):
        """Log an event to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, magnitude_pixels, 
                               features_tracked, inliers_count, inlier_ratio,
                               scene_quality, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.utcnow().isoformat(), event_type, magnitude, features,
              inliers, inlier_ratio, scene_quality, notes))
        conn.commit()
        conn.close()
    
    def get_recent_events(self, limit: int = 10):
        """Get most recent events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, event_type, magnitude_pixels, features_tracked, 
                   inliers_count, inlier_ratio, scene_quality, notes
            FROM events
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        events = cursor.fetchall()
        conn.close()
        return events
    
    def get_inlier_ratio_stats(self, hours: int = 24):
        """Get inlier ratio statistics for last N hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get events from last N hours
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        cursor.execute('''
            SELECT AVG(inlier_ratio), MIN(inlier_ratio), MAX(inlier_ratio),
                   COUNT(*) as count
            FROM events
            WHERE timestamp > ? AND inlier_ratio IS NOT NULL
        ''', (cutoff_time.isoformat(),))
        
        stats = cursor.fetchone()
        conn.close()
        
        if stats and stats[3] > 0:  # If count > 0
            return {
                'mean': stats[0],
                'min': stats[1],
                'max': stats[2],
                'count': stats[3]
            }
        else:
            return None
```

### Synthetic Test Data Generation (for dynamic scene tests)

```python
import cv2
import numpy as np
from typing import Tuple

def create_synthetic_scene(reference_image: np.ndarray,
                             camera_shift: float = 0,
                             dynamic_ratio: float = 0.3,
                             dynamic_displacement: float = 10) -> np.ndarray:
    """
    Create synthetic scene with camera movement + dynamic elements
    
    Args:
        reference_image: Base image
        camera_shift: Camera movement in pixels (applied globally)
        dynamic_ratio: Fraction of image that has dynamic motion (0.0-1.0)
        dynamic_displacement: Magnitude of dynamic motion in pixels
    
    Returns:
        Modified image with camera shift and dynamic regions
    """
    height, width = reference_image.shape[:2]
    
    # Apply camera shift (global translation)
    M = np.float32([[1, 0, camera_shift], [0, 1, camera_shift]])
    shifted = cv2.warpAffine(reference_image, M, (width, height))
    
    # Add dynamic motion to bottom portion (simulating water/bubbles)
    if dynamic_ratio > 0:
        dynamic_height = int(height * dynamic_ratio)
        dynamic_region = shifted[height - dynamic_height:, :]
        
        # Apply random local displacements
        for _ in range(50):  # 50 random "bubbles"
            bubble_x = np.random.randint(0, width - 20)
            bubble_y = np.random.randint(0, dynamic_height - 20)
            bubble_dx = np.random.randint(-dynamic_displacement, dynamic_displacement)
            bubble_dy = np.random.randint(-dynamic_displacement, 0)  # Upward motion
            
            # Shift small patch
            patch = dynamic_region[bubble_y:bubble_y+20, bubble_x:bubble_x+20]
            M_bubble = np.float32([[1, 0, bubble_dx], [0, 1, bubble_dy]])
            shifted_patch = cv2.warpAffine(patch, M_bubble, (20, 20))
            dynamic_region[bubble_y:bubble_y+20, bubble_x:bubble_x+20] = shifted_patch
        
        shifted[height - dynamic_height:, :] = dynamic_region
    
    return shifted

def simulate_water_ripples(image: np.ndarray,
                             amplitude: int = 5,
                             frequency: float = 0.2) -> np.ndarray:
    """Simulate water surface ripples"""
    height, width = image.shape[:2]
    
    # Apply sinusoidal distortion to bottom portion
    water_height = int(height * 0.3)
    water_region = image[height - water_height:, :]
    
    rows, cols = water_region.shape[:2]
    
    # Create displacement map
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(rows):
        for j in range(cols):
            offset_x = amplitude * np.sin(frequency * i)
            offset_y = amplitude * np.sin(frequency * j)
            map_x[i, j] = j + offset_x
            map_y[i, j] = i + offset_y
    
    rippled = cv2.remap(water_region, map_x, map_y, cv2.INTER_LINEAR)
    
    result = image.copy()
    result[height - water_height:, :] = rippled
    return result

def simulate_daf_bubbles(image: np.ndarray,
                          bubble_count: int = 100,
                          bubble_velocity: int = 5,
                          frame_number: int = 0) -> np.ndarray:
    """
    Simulate rising DAF bubbles
    
    Args:
        image: Base image
        bubble_count: Number of bubbles to simulate
        bubble_velocity: Upward velocity in pixels/frame
        frame_number: Current frame number (for animation)
    """
    result = image.copy()
    height, width = result.shape[:2]
    
    # Generate consistent bubble positions based on frame number
    np.random.seed(42)  # Fixed seed for consistency
    
    for i in range(bubble_count):
        # Initial bubble position (bottom of frame)
        bubble_x = np.random.randint(10, width - 10)
        bubble_y_init = height - 10
        
        # Current position based on frame number and velocity
        bubble_y = bubble_y_init - (bubble_velocity * ((frame_number + i) % (height // bubble_velocity)))
        
        if bubble_y < 0 or bubble_y > height - 10:
            continue
        
        # Draw bubble (white circle with transparency)
        bubble_radius = np.random.randint(2, 6)
        cv2.circle(result, (bubble_x, bubble_y), bubble_radius, (255, 255, 255), -1)
        
        # Add slight transparency effect (blend with background)
        overlay = result.copy()
        cv2.circle(overlay, (bubble_x, bubble_y), bubble_radius, (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    
    return result
```

---

## 15. Appendix B: Configuration Examples (UPDATED)

### Default Configuration (`config.json`)

```json
{
  "camera": {
    "device_id": 0,
    "resolution_width": 640,
    "resolution_height": 480,
    "fps": 10,
    "backend": "auto"
  },
  "detection": {
    "threshold_pixels": 2.0,
    "analysis_interval_seconds": 1.0,
    "min_features_required": 50,
    "max_features": 500,
    "orb_parameters": {
      "n_features": 500,
      "scale_factor": 1.2,
      "n_levels": 8,
      "edge_threshold": 31,
      "first_level": 0,
      "wta_k": 2,
      "patch_size": 31
    },
    "ransac_parameters": {
      "enabled": true,
      "reprojection_threshold": 2.0,
      "confidence": 0.995,
      "max_iterations": 2000,
      "min_inlier_ratio": 0.3
    },
    "static_region": {
      "enabled": true,
      "x": null,
      "y": null,
      "width": null,
      "height": null,
      "auto_suggest": true,
      "suggested_exclude_bottom_fraction": 0.4
    }
  },
  "recalibration": {
    "stability_test_duration_seconds": 30,
    "stability_threshold_pixels": 1.0,
    "min_inlier_ratio_during_test": 0.3,
    "allow_region_adjustment": true
  },
  "ui": {
    "display_fps": 1,
    "display_width": 320,
    "display_height": 240,
    "enable_sound": true,
    "show_inlier_diagnostics": true,
    "show_static_region_overlay": true,
    "scene_quality_thresholds": {
      "excellent": 0.7,
      "good": 0.5,
      "fair": 0.3
    }
  },
  "logging": {
    "database_path": "./data/events.db",
    "log_level": "INFO",
    "max_events": 1000,
    "log_inlier_ratio": true,
    "log_scene_quality": true
  },
  "data_validity": {
    "flag_file_path": "./data/data_validity.flag"
  },
  "reference": {
    "reference_file_path": "./data/reference_frame.pkl",
    "static_mask_path": "./data/static_mask.pkl"
  }
}
```

### High-Vibration Site Configuration (UPDATED)

```json
{
  "detection": {
    "threshold_pixels": 4.0,
    "analysis_interval_seconds": 2.0,
    "ransac_parameters": {
      "reprojection_threshold": 3.0,
      "min_inlier_ratio": 0.25
    }
  }
}
```

### Low-Light Site Configuration (UPDATED)

```json
{
  "detection": {
    "min_features_required": 30,
    "orb_parameters": {
      "n_features": 1000,
      "edge_threshold": 20
    },
    "static_region": {
      "suggested_exclude_bottom_fraction": 0.3
    }
  }
}
```

### Extremely Dynamic Scene Configuration (NEW)

```json
{
  "detection": {
    "threshold_pixels": 3.0,
    "ransac_parameters": {
      "reprojection_threshold": 3.0,
      "confidence": 0.99,
      "max_iterations": 3000,
      "min_inlier_ratio": 0.2
    },
    "static_region": {
      "suggested_exclude_bottom_fraction": 0.5
    }
  },
  "ui": {
    "scene_quality_thresholds": {
      "excellent": 0.6,
      "good": 0.4,
      "fair": 0.2
    }
  }
}
```

---

## 16. Appendix C: Glossary (UPDATED)

**ArUco Markers**: Machine-readable fiducial markers for camera pose estimation

**DAF (Dissolved Air Flotation)**: Water treatment process using fine bubbles to float suspended solids

**Dynamic Elements**: Moving scene features (water, bubbles, flocs) that should be ignored for camera movement detection

**False Positive**: System triggers alarm when no real camera movement occurred

**Feature**: Distinctive point in an image that can be reliably tracked (e.g., corner, edge)

**Flag File**: Simple text file used for inter-process communication

**Flocculation (Flocky)**: Process of clumping suspended particles into larger aggregates (flocs) for easier removal

**Homography**: Mathematical transformation describing relationship between two images of the same plane

**Inlier**: Feature that follows the consensus model (static, moves with camera). Opposite of outlier.

**Inlier Ratio**: Percentage of features that are static (0.0-1.0). Higher is better. <0.3 indicates poor scene quality.

**Lux**: Unit of illumination (1 lux = 1 lumen per square meter)

**NTU (Nephelometric Turbidity Units)**: Standard unit for measuring water turbidity

**ORB (Oriented FAST and Rotated BRIEF)**: Fast feature detection algorithm for computer vision

**Outlier**: Feature that doesn't follow the consensus model (dynamic, moving independently). Opposite of inlier.

**RANSAC (Random Sample Consensus)**: Algorithm for fitting a model to data with outliers. Separates inliers from outliers.

**Recalibration**: Process of capturing a new reference frame after camera movement

**Reference Frame**: Baseline image used for movement detection comparison

**Scene Quality**: Assessment of how suitable the scene is for detection (based on inlier ratio). Excellent >70%, Good 50-70%, Fair 30-50%, Poor <30%.

**Static Region**: Area of camera view containing only static elements (tank walls, pipes). Defined by operator during setup.

**Threshold**: Minimum movement magnitude (in pixels) that triggers an alarm

**Turbidity**: Measure of water clarity indicating suspended particle concentration

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Manager | [Your Name] | _________ | _______ |
| Engineering Lead | [Name TBD] | _________ | _______ |
| Customer Sponsor | Ofer | _________ | _______ |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0-MVP | 2025-10-16 | Product Management | Initial MVP specification derived from full PRD |
| 1.1-MVP | 2025-10-16 | Product Management | **MAJOR UPDATE: Added RANSAC implementation, static region masking, dynamic scene handling, updated all sections** |

---

**END OF MVP SPECIFICATION v1.1**
