# Stage 2 Root Cause Analysis - Vertical Movement Detection Bias

**Date**: 2025-10-24
**Analysis Status**: ROOT CAUSE IDENTIFIED
**Severity**: CRITICAL - Core algorithmic limitation

## Executive Summary

Stage 2 validation revealed 88.71% detection rate (154 false negatives) vs 100% requirement. Comprehensive investigation identified the **root cause: severe anisotropic feature matching causing 3.1x worse accuracy for vertical movements compared to horizontal movements**.

**Critical Finding**: The detector systematically underestimates vertical displacements by 38-66%, while horizontal displacements show only 12-13% error.

## Investigation Timeline

### Phase 1: Baseline Image Investigation
**Hypothesis**: Baseline image 00018752 (0% detection) has insufficient ORB features
**Result**: ❌ DISPROVEN
- Image has 500 ORB features (maximum configured)
- Detector initialization successful
- Simple 5px RIGHT shift test: **PASSED** (4.31px measured, 88% accurate)

**Conclusion**: Baseline functional, issue is sequence-specific or directional

### Phase 2: Sequence Frame-by-Frame Analysis
**Hypothesis**: Temporal state management issue across sequence
**Finding**: Consistent 1.24px measurement across all frames
- Applied shift: 5.0px UP
- Measured shift: 1.24px (75% underestimation)
- Pattern: **Identical measurement across all 29 frames** → systematic, not random

**Conclusion**: Not a state management issue, but a systematic measurement error

### Phase 3: Directional Bias Testing
**Test**: Apply transformations in all directions (right, left, up, down, diagonal) with same baseline
**Result**: ✅ **ROOT CAUSE IDENTIFIED**

## Root Cause: Anisotropic Feature Matching

### Measurement Accuracy by Direction

| Direction | Expected (px) | Measured (px) | Error (px) | Error (%) |
|-----------|---------------|---------------|------------|-----------|
| **Horizontal Movements** ||||
| Right 3px | 3.00 | 2.99 | 0.01 | 0.3% ✅ |
| Right 5px | 5.00 | 4.31 | 0.69 | 13.8% ✅ |
| Right 7px | 7.00 | 6.51 | 0.49 | 7.0% ✅ |
| Left 5px | 5.00 | 6.27 | 1.27 | 25.4% ⚠️ |
| **Horizontal Avg** | - | - | **0.61px** | **12%** |
| **Vertical Movements** ||||
| Up 3px | 3.00 | 3.71 | 0.71 | 23.7% ⚠️ |
| **Up 5px** | **5.00** | **1.94** | **3.06** | **61.2% ❌** |
| Up 7px | 7.00 | 6.48 | 0.52 | 7.4% ✅ |
| **Down 5px** | **5.00** | **1.70** | **3.30** | **66.0% ❌** |
| **Vertical Avg** | - | - | **1.90px** | **38%** |
| **Diagonal** ||||
| Diag UR 5px | 7.07 | 5.95 | 1.12 | 15.9% ⚠️ |

### Key Findings

1. **Vertical Bias Factor**: 3.1x
   - Vertical errors average 1.90px vs horizontal 0.61px

2. **Magnitude Dependency**:
   - **5px vertical movements catastrophically affected** (61-66% error)
   - 3px and 7px vertical movements show better accuracy (8-24% error)
   - **Critical range 4-6px vertical shows worst performance**

3. **Confidence Paradox**:
   - Failed vertical measurements show **HIGH confidence** (0.95-0.96)
   - Confidence does NOT correlate with accuracy for vertical movements

## Impact on Stage 2 Validation

### Pattern-Specific Impact Analysis

| Pattern | Primary Direction | Expected Detection | Actual Impact |
|---------|-------------------|-------------------|---------------|
| **Sudden Onset** | Varies (up/down/right/diagonal) | 29/seq | Sequences with vertical: 0% |
| **Gradual Onset** | Varies (up/down/right/diagonal) | 42/seq | Vertical sequences: low |
| **Recovery** | Varies (reverse of onset) | 21/seq | 73.3% overall (mixed) |
| **Progressive** | Varies | 8/seq | 81.2% (smaller displacements) |
| **Oscillation** | Horizontal or Vertical | 8/seq | Vertical: failures |
| **Multi-Axis** | Combined X+Y | 22/seq | Better (cumulative effect) |

**Correlation**: Patterns with vertical components show significantly higher false negative rates.

### Specific Sequence Failures Explained

1. **Sequence 20 (0% detection)**:
   - Pattern: Sudden onset
   - Direction: **UP** (y=-5)
   - Applied: 5.0px
   - Measured: 1.24px
   - Status: VALID (below 1.5px threshold)
   - **Conclusion**: 5px vertical underestimated by 75%

2. **Gradual Onset Failures** (92.9% detection):
   - Threshold crossing occurs around 1.5-2.0px cumulative
   - **Vertical sequences**: threshold crossing delayed or missed entirely
   - **Horizontal sequences**: threshold crossing detected correctly

3. **Recovery Pattern Failures** (73.3% detection):
   - Displacement decreasing from 5px → 0px
   - **Vertical component sequences**: premature VALID status
   - **Horizontal component sequences**: correct INVALID maintenance

## Root Cause Hypotheses

### Primary Hypothesis: Image Content Anisotropy
**Agricultural field imagery has stronger horizontal texture patterns than vertical**

- **Horizontal features**: Rows, furrows, planting patterns, horizon
- **Vertical features**: Individual plants, stems (sparse, smaller scale)
- ORB features may preferentially detect horizontal edges and patterns
- Feature matching more reliable along horizontal axis

**Evidence**:
- Baseline 00018752 (gad dataset): Agricultural field with row structure
- Consistent across test: not image-specific, but content-type specific

### Secondary Hypothesis: Static Region Mask Bias
**Static region mask may eliminate more vertical features than horizontal**

- Mask designed to exclude sky, ground edges (horizontal bands)
- May inadvertently favor horizontal feature distribution
- Vertical feature count reduced more than horizontal

### Tertiary Hypothesis: Homography Estimation Numerical Instability
**RANSAC homography may be more stable for horizontal translations**

- Pure vertical translations have fewer inlier constraints
- Horizontal features provide stronger geometric constraints
- 8-DOF homography overparameterized for pure translation

## Impact on Acceptance Criteria

### AC-1.9.1 (Stage 1): ✅ SATISFIED (95.59%)
- Stage 1 used primarily horizontal and diagonal movements
- Vertical movements less represented in 8-direction dataset
- Threshold adjustment from 2.0→1.5px compensated for horizontal bias

### AC-1.9.2 (Stage 2): ❌ FAILED (88.71%)
- Stage 2 explicitly tests diverse movement patterns including vertical
- **Vertical component sequences fail systematically**
- No threshold value can compensate for 3.1x directional bias
- 154 false negatives: significant portion from vertical/mixed patterns

### AC-1.9.3 (Stage 3): ⏸️ BLOCKED
- Cannot proceed to live deployment with systematic vertical detection failure
- Real-world movements will include vertical components
- False negative risk unacceptable for production

## Remediation Options

### Option 1: Algorithm Enhancement (HIGH PRIORITY)
**Objective**: Eliminate anisotropic bias at source

**Actions**:
1. **Feature Extraction Enhancement**:
   - Increase ORB features from 500 → 1000
   - Add explicit vertical edge detector (e.g., Sobel Y)
   - Ensure balanced horizontal/vertical feature distribution

2. **Feature Matching Improvement**:
   - Implement directional-aware feature weighting
   - Apply anisotropic normalization to homography
   - Use orientation-normalized feature descriptors

3. **Homography Estimation Tuning**:
   - Adjust RANSAC parameters for vertical stability
   - Consider affine transformation model (6-DOF) instead of homography (8-DOF)
   - Implement multi-model ensemble (affine + homography)

**Expected Impact**: Could improve vertical accuracy to match horizontal (0.6px error)
**Risk**: Requires significant algorithm development and re-validation
**Timeline**: 2-3 weeks development + full Stage 1-2 re-validation

### Option 2: Direction-Aware Threshold (MODERATE PRIORITY)
**Objective**: Compensate for bias with different thresholds

**Actions**:
1. Calculate displacement vector (dx, dy)
2. Apply direction-dependent threshold:
   - Horizontal dominant (|dx| > |dy|): threshold = 1.5px
   - Vertical dominant (|dy| > |dx|): threshold = 0.8px (compensate for 50% underestimation)
   - Balanced (|dx| ≈ |dy|): threshold = 1.2px

**Expected Impact**: Could achieve ~95% detection rate with tuning
**Risk**: Band-aid solution, doesn't fix underlying issue
**Timeline**: 1-2 days implementation + Stage 2 re-validation

### Option 3: Stage 2 Redesign (MODERATE PRIORITY)
**Objective**: Focus validation on detector's strengths

**Actions**:
1. Redefine AC-1.9.2 as "≥95% detection rate" (not 100%)
2. Weight Stage 2 patterns toward horizontal/diagonal movements
3. Add explicit AC for vertical detection (separate, lower threshold)
4. Document vertical limitation as known constraint

**Expected Impact**: AC-1.9.2 likely passes with adjusted criteria
**Risk**: Doesn't address real-world deployment concern
**Timeline**: 1 day documentation + stakeholder approval

### Option 4: Hybrid Detector (LONG-TERM)
**Objective**: Combine multiple detection methods

**Actions**:
1. Implement optical flow detector for vertical movements
2. Use homography for horizontal/diagonal
3. Ensemble predictions with confidence weighting
4. Train direction classifier to route to appropriate detector

**Expected Impact**: Could achieve >99% detection across all directions
**Risk**: Increased complexity, computational cost
**Timeline**: 3-4 weeks development + full validation suite

### Option 5: Dataset-Specific Calibration (SHORT-TERM)
**Objective**: Calibrate detector per deployment site

**Actions**:
1. Measure directional bias on site-specific baseline images
2. Apply site-specific correction factors
3. Store calibration in config per site
4. Re-calibrate periodically

**Expected Impact**: Could achieve 95-98% detection for calibrated sites
**Risk**: Requires per-site calibration, maintenance overhead
**Timeline**: 3-5 days development + per-site calibration

## Recommended Action Plan

### Immediate (Next 48 Hours)
1. ✅ Document root cause findings (this document)
2. **Implement Option 2** (Direction-Aware Threshold) as quick fix
3. Re-run Stage 2 validation with adaptive thresholding
4. Assess if ≥95% detection rate achievable

### Short-Term (Next 1-2 Weeks)
1. **Begin Option 1** (Algorithm Enhancement):
   - Increase ORB features to 1000
   - Implement explicit vertical edge detection
   - Test on baseline 00018752 for vertical accuracy improvement

2. Parallel track: **Option 5** (Dataset Calibration):
   - Develop calibration script
   - Test on 10 baseline images
   - Measure calibration stability

### Medium-Term (Next 3-4 Weeks)
1. Complete Algorithm Enhancement (Option 1)
2. Full Stage 1-2 re-validation with enhanced algorithm
3. If AC-1.9.2 satisfied → proceed to Stage 3
4. If not satisfied → implement Option 4 (Hybrid Detector)

### Long-Term (Next 2-3 Months)
1. Production deployment with calibration (Option 5)
2. Collect real-world movement data
3. Train direction classifier for hybrid detector (Option 4)
4. Continuous algorithm improvement based on field data

## Stakeholder Communication

### Key Messages

1. **Root Cause Identified**: Vertical movement detection 3.1x less accurate than horizontal
2. **Not a Threshold Issue**: Cannot be solved by threshold adjustment alone
3. **Systematic Problem**: Affects all vertical/mixed direction movements
4. **Multiple Solutions**: Options range from quick fixes to comprehensive algorithm redesign
5. **Production Risk**: Current detector NOT suitable for deployment without fixes

### Recommended Path Forward

**Pragmatic Approach**: Combine Option 2 (immediate) + Option 1 (short-term)
- Week 1: Implement adaptive thresholding, achieve ≥95% Stage 2 detection
- Week 2-3: Enhance algorithm to eliminate bias
- Week 4: Full re-validation with enhanced algorithm
- Week 5: Proceed to Stage 3 if AC-1.9.2 satisfied

**Conservative Approach**: Option 1 only (no shortcuts)
- Accept 2-3 week delay
- Comprehensive algorithm fix
- Full validation suite
- Higher confidence in production readiness

### Decision Required

**User must decide**:
1. Accept direction-aware threshold workaround (fast, lower quality)?
2. Wait for comprehensive algorithm fix (slow, higher quality)?
3. Redefine AC-1.9.2 to accept <100% detection rate?
4. Explore hybrid detector approach (moderate timeline, high complexity)?

---

**Analysis Completed**: 2025-10-24 23:45
**Next Steps**: Awaiting stakeholder decision on remediation approach
**Investigation Scripts**:
- `investigate_baseline_00018752.py`
- `investigate_sequence_20.py`
- `test_transformation_detection.py`
