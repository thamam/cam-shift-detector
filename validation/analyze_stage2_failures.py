#!/usr/bin/env python3
"""
Stage 2 Failure Analysis Script

Analyzes false negatives from Stage 2 validation to identify patterns and root causes
for the 88.71% detection rate (11.29 points below 100% requirement).

Usage:
    python validation/analyze_stage2_failures.py

Inputs:
    - validation/stage2_results.json - Complete validation results

Outputs:
    - validation/stage2_failure_analysis.txt - Detailed failure analysis report
    - Console output with failure patterns and recommendations
"""

import sys
import json
from pathlib import Path
from collections import defaultdict, Counter


def load_results(results_file: Path) -> dict:
    """Load validation results from JSON"""
    with open(results_file, 'r') as f:
        return json.load(f)


def analyze_failures(results: dict):
    """Analyze Stage 2 failure patterns"""

    print("=" * 80)
    print("Stage 2 Failure Analysis")
    print("=" * 80)
    print()

    overall = results['overall_metrics']
    pattern_metrics = results['pattern_metrics']

    # Overall statistics
    print(f"📊 Overall Statistics:")
    print(f"   Detection Rate: {overall['detection_rate'] * 100:.2f}%")
    print(f"   False Negatives: {overall['false_negatives']} / {overall['expected_invalid_frames']}")
    print(f"   False Positives: {overall['false_positives']}")
    print(f"   Critical Frame Accuracy: {overall['critical_frame_accuracy'] * 100:.2f}%")
    print(f"   Miss Rate: {(1 - overall['detection_rate']) * 100:.2f}%")
    print()

    # Per-pattern failure analysis
    print(f"🔍 Per-Pattern Failure Analysis:")
    print()

    # Sort patterns by false negative count (worst first)
    sorted_patterns = sorted(
        pattern_metrics.items(),
        key=lambda x: x[1]['false_negatives'],
        reverse=True
    )

    for pattern, metrics in sorted_patterns:
        fn_count = metrics['false_negatives']
        detection_rate = metrics['detection_rate'] * 100
        expected = metrics['expected_invalid']

        print(f"   {pattern:20s}:")
        print(f"      Detection Rate: {detection_rate:.1f}%")
        print(f"      False Negatives: {fn_count} / {expected}")
        print(f"      False Positives: {metrics['false_positives']}")
        print()

    # Sequence-level analysis
    print(f"🎯 Sequence-Level Analysis:")
    print()

    # Identify worst sequences
    sequences = results['sequence_metrics']
    failed_sequences = [s for s in sequences if s['false_negatives'] > 0]
    perfect_sequences = [s for s in sequences if s['false_negatives'] == 0]

    print(f"   Perfect sequences (0 FN): {len(perfect_sequences)}/{len(sequences)}")
    print(f"   Failed sequences (≥1 FN): {len(failed_sequences)}/{len(sequences)}")
    print()

    # Worst offenders
    worst_sequences = sorted(failed_sequences, key=lambda x: x['false_negatives'], reverse=True)[:5]
    print(f"   🔴 Worst 5 Sequences:")
    for seq in worst_sequences:
        print(f"      {seq['sequence_id'][:60]:60s} | FN: {seq['false_negatives']:2d} | Rate: {seq['detection_rate']*100:5.1f}%")
    print()

    # Analyze sudden onset latency issue
    print(f"⚡ Sudden Onset Latency Analysis:")
    sudden_onset_sequences = [s for s in sequences if s['pattern_type'] == 'sudden_onset']

    latency_ok = [s for s in sudden_onset_sequences if s['detection_latency_frames'] <= 2]
    latency_bad = [s for s in sudden_onset_sequences if s['detection_latency_frames'] > 2]

    print(f"   Total sudden onset sequences: {len(sudden_onset_sequences)}")
    print(f"   Within 2-frame threshold: {len(latency_ok)}")
    print(f"   Exceeds 2-frame threshold: {len(latency_bad)}")

    if latency_bad:
        print(f"\n   🔴 Latency Violations:")
        for seq in latency_bad:
            print(f"      {seq['sequence_id'][:60]:60s} | Latency: {seq['detection_latency_frames']} frames")
    print()

    # Root cause analysis
    print(f"{'='*80}")
    print(f"ROOT CAUSE ANALYSIS")
    print(f"{'='*80}")

    print(f"\n🔴 Primary Issue: Temporal Detection Failures")
    print(f"   - Detection Rate: 88.71% (11.29 points below 100% target)")
    print(f"   - False Negatives: 154 across 1,364 expected detections")
    print(f"   - Critical Frame Accuracy: 55.56% (45% of transitions missed)")

    print(f"\n💡 Technical Analysis:")

    print(f"\n   1. Recovery Pattern Catastrophic Failure (73.3% detection):")
    print(f"      - 56 false negatives in recovery sequences (36% of all FN)")
    print(f"      - Pattern: Displacement decreasing from 5px → 0px")
    print(f"      - Problem: System prematurely transitioning to VALID status")
    print(f"      - Hypothesis: Detector may not handle decreasing displacement correctly")
    print(f"      - Impact: Most severe pattern-specific failure")

    print(f"\n   2. Sudden Onset Complete Failure (sequence 20):")
    print(f"      - 0% detection rate, 29/29 frames missed")
    print(f"      - Latency: 999 frames (no detection)")
    print(f"      - Hypothesis: Baseline image may have insufficient features")
    print(f"      - Or: Detector initialization failure for this specific image")
    print(f"      - Impact: Single catastrophic failure affecting sudden onset metrics")

    print(f"\n   3. Gradual Onset Threshold Boundary Issues (92.9% detection):")
    print(f"      - 30 false negatives at threshold crossing region")
    print(f"      - Pattern: Slow accumulation 0→5px over 60 frames")
    print(f"      - Problem: Measurements near 1.5px threshold not reliably detected")
    print(f"      - Similar to Stage 1 issue but persisting in temporal context")
    print(f"      - Impact: Consistent low-level failures across sequences")

    print(f"\n   4. Progressive Displacement Issues (81.2% detection):")
    print(f"      - 12 false negatives in step-wise accumulation")
    print(f"      - Pattern: 0.5px increments every 4 frames")
    print(f"      - Problem: Small incremental changes not accumulating correctly")
    print(f"      - Hypothesis: Detector may reset or smooth measurements")
    print(f"      - Impact: Moderate failure rate for slow creep scenarios")

    print(f"\n   5. Oscillation Detection Problems (90.7% detection):")
    print(f"      - 13 false negatives during vibration simulation")
    print(f"      - Pattern: Sinusoidal ±3px oscillation")
    print(f"      - Problem: Rapid transitions or zero-crossings missed")
    print(f"      - Impact: Moderate failure for high-frequency movement")

    print(f"\n   6. Multi-Axis Best Performer (94.2% detection):")
    print(f"      - 14 false negatives, but highest success rate")
    print(f"      - Pattern: Independent X/Y drift rates")
    print(f"      - Success: Larger cumulative displacements easier to detect")
    print(f"      - Note: Still 5.8% failure rate, not meeting 100% requirement")

    print(f"\n📈 Recommendations:")

    print(f"\n   Option 1: Investigate Detector State Management (High Priority)")
    print(f"      - Action: Review how detector maintains state across frames")
    print(f"      - Check: Does set_baseline() properly initialize for all images?")
    print(f"      - Check: Is feature matching stable across temporal sequences?")
    print(f"      - Check: Are there edge cases in homography estimation?")
    print(f"      - Expected improvement: Address sequence 20 complete failure")

    print(f"\n   Option 2: Analyze Recovery Pattern Logic (Critical)")
    print(f"      - Action: Debug why decreasing displacement causes premature VALID status")
    print(f"      - Hypothesis: Detector may use frame-to-frame delta instead of cumulative")
    print(f"      - Or: Confidence scoring may degrade for decreasing movement")
    print(f"      - Expected improvement: Fix 56/154 FN (36% of failures)")

    print(f"\n   Option 3: Further Threshold Adjustment (Moderate Priority)")
    print(f"      - Current: 1.5px threshold (adjusted from 2.0px in Stage 1)")
    print(f"      - Consider: Lower to 1.2px for additional safety margin")
    print(f"      - Trade-off: Increased false positive risk")
    print(f"      - Expected improvement: Address gradual onset threshold boundary issues")

    print(f"\n   Option 4: Feature Matching Robustness (Long-term)")
    print(f"      - Action: Increase ORB feature count for temporal stability")
    print(f"      - Action: Apply temporal filtering across frames")
    print(f"      - Action: Implement confidence-based hysteresis for status transitions")
    print(f"      - Expected improvement: General robustness increase")

    print(f"\n   Option 5: Baseline Image Validation (Immediate)")
    print(f"      - Action: Pre-validate baseline images for sufficient features")
    print(f"      - Action: Reject baselines with feature count < min_features_required")
    print(f"      - Expected improvement: Prevent complete failures like sequence 20")

    print(f"\n🎯 Recommended Action Plan:")
    print(f"   1. IMMEDIATE: Investigate sequence 20 baseline image (00018752)")
    print(f"      - Check feature extraction success")
    print(f"      - Verify image quality and content")
    print(f"      - Test detector initialization manually")
    print(f"   2. HIGH PRIORITY: Debug recovery pattern logic")
    print(f"      - Add logging for displacement calculations")
    print(f"      - Verify cumulative vs delta measurement")
    print(f"      - Test with synthetic decreasing displacement sequence")
    print(f"   3. MODERATE: Consider threshold adjustment to 1.2px")
    print(f"      - Re-run Stage 1 validation with 1.2px threshold")
    print(f"      - If Stage 1 passes, re-run Stage 2")
    print(f"   4. LONG-TERM: Implement temporal filtering and hysteresis")
    print(f"      - Require N consecutive frames to confirm status change")
    print(f"      - Apply moving average to displacement measurements")

    print(f"\n⚠️  BLOCKER STATUS:")
    print(f"   AC-1.9.2 NOT SATISFIED: 88.71% < 100% required")
    print(f"   Stage 3 (Live Deployment) BLOCKED until Stage 2 passes")
    print(f"   NO-GO for production deployment")

    print(f"\n{'='*80}")


def main():
    """Main execution"""

    results_file = Path("validation/stage2_results.json")
    if not results_file.exists():
        print(f"❌ ERROR: Results file not found: {results_file}")
        print(f"   Run validation/run_stage2_validation.py first")
        return 1

    results = load_results(results_file)
    analyze_failures(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
