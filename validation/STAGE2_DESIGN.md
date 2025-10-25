# Stage 2 Validation Design: Temporal Sequence Simulation

## Overview

Stage 2 validation bridges the gap between Stage 1 (single-frame synthetic transformations) and Stage 3 (live deployment) by introducing **temporal dynamics** - realistic camera movement patterns over time.

**Generation Date**: 2025-10-24
**Design Status**: APPROVED - High-Fidelity Simulation Approach
**Target AC**: AC-1.9.2 (0% false negatives / 100% detection rate)

## Design Rationale

### Why Temporal Sequences?

Stage 1 validated **spatial accuracy** (can we detect a 2px shift?), but real-world deployment requires **temporal reliability**:

1. **Gradual vs Sudden Movements**: Real camera shifts don't happen instantaneously
2. **Detection Latency**: How quickly does the system respond to movement onset?
3. **Sustained Displacement**: Can the system maintain INVALID status during prolonged shifts?
4. **Recovery Dynamics**: How does the system behave when camera returns to baseline?
5. **False Negative Criticality**: Missing even ONE movement event in a sequence is unacceptable

### High-Fidelity Simulation Approach

Since real recorded footage with known ground truth is unavailable, we'll create **synthetic temporal sequences** that model realistic camera movement dynamics:

**Advantages**:
- Perfect ground truth: Every frame's movement is precisely known
- Repeatable: Can regenerate sequences for debugging/tuning
- Comprehensive: Can test edge cases impossible to capture naturally
- Controllable: Can isolate specific movement patterns for analysis

**Limitations** (vs real footage):
- No real-world noise sources (vibration, wind, thermal expansion)
- No optical artifacts (lens distortion, chromatic aberration, motion blur)
- No environmental factors (lighting changes, moving vegetation)

**Mitigation**: Stage 3 (live deployment) will provide real-world validation

## Temporal Movement Patterns

### Pattern 1: Gradual Onset (Slow Drift)
**Scenario**: Camera mounting slowly loosens over 60 seconds
**Dynamics**: Linear accumulation from 0px → 5px over 60 frames (1fps)

```
Frame:  0    10   20   30   40   50   60
Shift:  0px  0.8  1.7  2.5  3.3  4.2  5.0px
Status: V    V    I    I    I    I    I
```

**Critical Test**: Does detector transition from VALID → INVALID at 2px threshold?

### Pattern 2: Sudden Onset (Impact/Bump)
**Scenario**: Camera mount receives sudden impact
**Dynamics**: Instantaneous shift from 0px → 5px, sustained for 30 frames

```
Frame:  0    1    2    3    ...  30
Shift:  0px  5px  5px  5px  ...  5px
Status: V    I    I    I    ...  I
```

**Critical Test**: Does detector respond within 1-2 frames to sudden movement?

### Pattern 3: Progressive Displacement (Incremental Creep)
**Scenario**: Thermal expansion causes slow camera drift
**Dynamics**: Small incremental shifts (0.5px steps) over 20 frames

```
Frame:  0    4    8    12   16   20
Shift:  0px  0.5  1.0  1.5  2.0  2.5px
Status: V    V    V    V    I    I
```

**Critical Test**: Accumulation detection at threshold boundary

### Pattern 4: Oscillation (Vibration)
**Scenario**: Wind-induced camera vibration
**Dynamics**: Sinusoidal oscillation ±3px around baseline over 20 frames

```
Frame:  0    5    10   15   20
Shift:  0px  +3   0    -3   0px
Status: V    I    V    I    V
```

**Critical Test**: Rapid status transitions during oscillation

### Pattern 5: Recovery Sequence (Return to Baseline)
**Scenario**: Camera shift corrected by automatic mechanism
**Dynamics**: Gradual return from 5px → 0px over 30 frames

```
Frame:  0    10   20   30
Shift:  5px  3.3  1.7  0px
Status: I    I    V    V
```

**Critical Test**: Proper INVALID → VALID transition as displacement falls below threshold

### Pattern 6: Multi-Axis Movement (Diagonal Drift)
**Scenario**: Camera mount loosens non-uniformly
**Dynamics**: Independent X and Y drift rates

```
Frame:  0        10       20       30
X:      0px      +1.5     +3.0     +4.5
Y:      0px      +2.0     +4.0     +6.0
Total:  0px      2.5      5.0      7.5px
Status: V        I        I        I
```

**Critical Test**: Vector displacement calculation accuracy

## Dataset Structure

### Sequence Organization

```
validation/stage2_data/
├── README.md                          # Methodology documentation
├── ground_truth_sequences.json        # All sequence ground truth labels
├── pattern_1_gradual_onset/
│   ├── sequence_metadata.json         # Pattern-specific metadata
│   ├── baseline_*.jpg → frame_000.jpg # Baseline image
│   ├── frame_001.jpg ... frame_060.jpg # 60-frame sequence
│   └── frame_annotations.json         # Per-frame ground truth
├── pattern_2_sudden_onset/
│   ├── sequence_metadata.json
│   ├── frame_000.jpg ... frame_030.jpg # 30-frame sequence
│   └── frame_annotations.json
├── pattern_3_progressive/
│   └── ...
├── pattern_4_oscillation/
│   └── ...
├── pattern_5_recovery/
│   └── ...
└── pattern_6_multi_axis/
    └── ...
```

### Ground Truth Schema

**sequence_metadata.json**:
```json
{
  "sequence_id": "pattern_1_gradual_onset_baseline_001",
  "pattern_type": "gradual_onset",
  "baseline_image": "baseline_001.jpg",
  "total_frames": 60,
  "frame_rate": 1.0,
  "duration_seconds": 60.0,
  "movement_description": "Linear drift 0→5px over 60 seconds",
  "expected_detections": 42,
  "expected_first_invalid_frame": 18,
  "critical_frames": [17, 18, 19]
}
```

**frame_annotations.json** (per-frame ground truth):
```json
{
  "frames": [
    {
      "frame_number": 0,
      "frame_id": "pattern_1_gradual_onset_baseline_001_frame_000",
      "timestamp": 0.0,
      "cumulative_shift_px": 0.0,
      "shift_from_previous_px": 0.0,
      "expected_status": "VALID",
      "expected_displacement_range": [0.0, 0.3],
      "movement_vector": {"x": 0.0, "y": 0.0},
      "is_critical_transition": false
    },
    {
      "frame_number": 18,
      "frame_id": "pattern_1_gradual_onset_baseline_001_frame_018",
      "timestamp": 18.0,
      "cumulative_shift_px": 1.5,
      "shift_from_previous_px": 0.083,
      "expected_status": "VALID",
      "expected_displacement_range": [1.2, 1.8],
      "movement_vector": {"x": 1.06, "y": 1.06},
      "is_critical_transition": true,
      "transition_notes": "Last VALID frame before threshold crossing"
    },
    {
      "frame_number": 19,
      "frame_id": "pattern_1_gradual_onset_baseline_001_frame_019",
      "timestamp": 19.0,
      "cumulative_shift_px": 1.58,
      "shift_from_previous_px": 0.083,
      "expected_status": "INVALID",
      "expected_displacement_range": [1.26, 1.9],
      "movement_vector": {"x": 1.12, "y": 1.12},
      "is_critical_transition": true,
      "transition_notes": "First INVALID frame - threshold crossing (1.5px)"
    }
  ]
}
```

## Generation Parameters

### Baseline Images
- **Count**: 10 sequences per pattern × 6 patterns = 60 total sequences
- **Source**: Sample from existing 50 baseline images in sample_images/
- **Selection**: Diverse scenes across of_jerusalem, carmit, gad datasets

### Frame Counts per Pattern
- Pattern 1 (Gradual Onset): 60 frames (0→5px linear drift)
- Pattern 2 (Sudden Onset): 30 frames (instant 5px shift sustained)
- Pattern 3 (Progressive): 20 frames (0.5px incremental steps)
- Pattern 4 (Oscillation): 20 frames (sinusoidal ±3px)
- Pattern 5 (Recovery): 30 frames (5px→0 gradual return)
- Pattern 6 (Multi-Axis): 30 frames (independent X/Y drift)

**Total Frames**: (60 + 30 + 20 + 20 + 30 + 30) × 10 baselines = **1,900 frames**

### Movement Directions
Each pattern tests multiple directional variations:
- **Gradual/Sudden/Progressive**: 4 directions (right, up, diagonal_ur, diagonal_dr)
- **Oscillation**: 2 axes (horizontal, vertical)
- **Recovery**: 4 directions (matching drift directions)
- **Multi-Axis**: 4 quadrants (++, +-, -+, --)

## Acceptance Criteria Validation

### AC-1.9.2: 0% False Negatives (100% Detection Rate)

**Validation Approach**:

1. **Critical Frame Analysis**: Every frame marked `is_critical_transition: true` must be correctly classified
   - Threshold crossing frames (VALID → INVALID)
   - Recovery frames (INVALID → VALID)
   - Zero tolerance for misclassification

2. **Sequence Continuity**: Once movement exceeds threshold:
   - System must maintain INVALID status for all subsequent frames with displacement ≥1.5px
   - No "flickering" between VALID/INVALID during sustained displacement

3. **Latency Requirements**: Sudden onset movements must be detected within 2 frames
   - Frame N: First movement occurs (0→5px)
   - Frame N+1 or N+2: System must report INVALID

4. **False Negative Categories**:
   ```yaml
   missed_onset:
     definition: "Movement starts but system remains VALID"
     criticality: CRITICAL
     tolerance: 0%

   late_detection:
     definition: "Movement detected >2 frames after onset"
     criticality: HIGH
     tolerance: 0%

   premature_recovery:
     definition: "System returns to VALID while displacement ≥1.5px"
     criticality: CRITICAL
     tolerance: 0%

   intermittent_detection:
     definition: "System flickers VALID during sustained displacement"
     criticality: HIGH
     tolerance: 0%
   ```

### Metrics Calculation

```python
# Per-Pattern Metrics
for pattern in patterns:
    sequences = load_sequences(pattern)
    for seq in sequences:
        results = run_detector_on_sequence(seq)

        # Critical Frame Analysis
        critical_frames = [f for f in seq.frames if f.is_critical_transition]
        for frame in critical_frames:
            if result[frame.number].status != frame.expected_status:
                record_false_negative(pattern, seq, frame)

        # Latency Analysis (Pattern 2: Sudden Onset)
        if pattern == "sudden_onset":
            onset_frame = 1  # Movement occurs at frame 1
            detection_frame = first_invalid_frame(results)
            latency = detection_frame - onset_frame
            if latency > 2:
                record_late_detection(seq, latency)

        # Continuity Analysis (sustained displacement)
        sustained_frames = [f for f in seq.frames
                            if f.cumulative_shift_px >= 1.5]
        for frame in sustained_frames:
            if result[frame.number].status == "VALID":
                record_premature_recovery(pattern, seq, frame)

# Overall Stage 2 Metrics
total_frames = sum(len(seq.frames) for seq in all_sequences)
total_expected_invalid = sum(1 for f in all_frames
                              if f.expected_status == "INVALID")
total_false_negatives = count_false_negatives(results)

false_negative_rate = total_false_negatives / total_expected_invalid
detection_rate = 1.0 - false_negative_rate

# AC-1.9.2 Pass Criterion
assert detection_rate == 1.0, f"FAIL: Detection rate {detection_rate*100:.2f}% < 100%"
```

## Implementation Components

### 1. Stage2TestHarness Class

```python
class Stage2TestHarness:
    """Temporal sequence generation and validation"""

    def generate_movement_trajectory(
        self,
        pattern: str,
        total_frames: int,
        max_displacement: float,
        direction: str
    ) -> List[Tuple[float, float]]:
        """Generate frame-by-frame movement trajectory"""

    def apply_temporal_transformation(
        self,
        baseline_image: np.ndarray,
        trajectory: List[Tuple[float, float]],
        output_dir: Path
    ) -> List[str]:
        """Generate frame sequence following trajectory"""

    def create_sequence_ground_truth(
        self,
        sequence_id: str,
        pattern: str,
        trajectory: List[Tuple[float, float]],
        threshold_px: float
    ) -> Dict:
        """Create per-frame annotations with critical transitions"""

    def validate_sequence_detection(
        self,
        results: List[DetectionResult],
        ground_truth: Dict,
        pattern: str
    ) -> SequenceMetrics:
        """Validate temporal detection correctness"""
```

### 2. run_stage2_validation.py Script

```python
def main():
    """Execute Stage 2 validation on temporal sequences"""

    # Load sequences
    sequences = load_stage2_sequences("validation/stage2_data/")

    # Initialize detector (persistent baseline for each sequence)
    for sequence in sequences:
        detector = CameraMovementDetector(config_path='config.json')
        baseline_image = cv2.imread(sequence.frames[0])
        detector.set_baseline(baseline_image)

        # Process frame sequence
        results = []
        for frame_path in sequence.frames[1:]:
            frame = cv2.imread(frame_path)
            result = detector.process_frame(frame, frame_id=frame_path)
            results.append(result)

        # Validate against ground truth
        metrics = validate_sequence(results, sequence.ground_truth)

        # Check for false negatives
        if metrics.false_negatives > 0:
            analyze_false_negative_pattern(sequence, results, metrics)

    # Overall Stage 2 assessment
    overall_detection_rate = calculate_overall_detection_rate(all_metrics)

    if overall_detection_rate < 1.0:
        print(f"❌ FAIL: Detection rate {overall_detection_rate*100:.2f}% < 100%")
        return 1
    else:
        print(f"✅ PASS: Detection rate 100% - AC-1.9.2 SATISFIED")
        return 0
```

## Expected Outcomes

### Success Criteria (AC-1.9.2)

```yaml
overall_performance:
  detection_rate: 100%
  false_negative_count: 0
  critical_frame_accuracy: 100%

per_pattern_requirements:
  gradual_onset:
    threshold_crossing_accuracy: 100%
    sustained_detection: 100%

  sudden_onset:
    response_latency: ≤2 frames
    immediate_detection: 100%

  progressive:
    accumulation_detection: 100%
    threshold_sensitivity: 100%

  oscillation:
    rapid_transition_accuracy: 100%
    no_missed_peaks: true

  recovery:
    valid_transition_accuracy: 100%
    no_premature_recovery: true

  multi_axis:
    vector_calculation_accuracy: 100%
    directional_detection: 100%
```

### Failure Analysis

If AC-1.9.2 is not satisfied:

1. **Pattern-Specific Failures**: Which temporal patterns cause false negatives?
2. **Latency Issues**: Is detection delayed beyond acceptable limits?
3. **Threshold Sensitivity**: Does threshold need further adjustment?
4. **Algorithm Limitations**: Are there fundamental detector architecture issues?

## Advantages Over Real Footage

1. **Perfect Ground Truth**: Every frame's true displacement known precisely
2. **Repeatable**: Can regenerate exact sequences for debugging
3. **Comprehensive**: Tests edge cases impossible to capture naturally
4. **Controlled Variables**: Isolates temporal dynamics from environmental noise
5. **Scalable**: Can generate unlimited sequences for robustness testing

## Transition to Stage 3

Stage 2 validates **temporal detection capability** under controlled conditions. Stage 3 (live deployment) will introduce:

- Real environmental noise (vibration, wind, thermal effects)
- Optical artifacts (motion blur, lighting changes)
- Unpredictable movement patterns
- False positive testing (<5% threshold)

Stage 2 success (100% detection rate) provides confidence that the detector can reliably identify camera movements in real-time operational scenarios.

---

**Next Steps**: Implement Stage2TestHarness class with trajectory generation and sequence validation logic.
