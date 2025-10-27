"""
Report Generation Module - JSON and Markdown Validation Reports

Generates comprehensive validation reports in both machine-readable (JSON)
and human-readable (Markdown) formats, including go/no-go production
readiness recommendations.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dataclasses import asdict

from validation.stage3_test_harness import Metrics
from validation.performance_profiler import PerformanceMetrics


logger = logging.getLogger(__name__)


class GoNoGoDecisionMaker:
    """Evaluates production readiness based on gate criteria.

    Gate Criteria:
    - Detection accuracy ≥95%
    - False positive rate ≤5%
    - FPS ≥1/60 Hz (0.0167 FPS)
    - Memory ≤500 MB

    Conservative Logic: ANY gate failure → NO-GO recommendation
    """

    # Gate criteria thresholds
    ACCURACY_THRESHOLD = 0.95
    FPR_THRESHOLD = 0.05
    FPS_THRESHOLD = 1.0 / 60.0  # 0.0167 FPS
    MEMORY_THRESHOLD_MB = 500.0

    def evaluate(self, metrics: Metrics, performance_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate production readiness and generate recommendation.

        Args:
            metrics: Validation metrics from test harness
            performance_metrics: Performance metrics from profiler

        Returns:
            Dictionary with recommendation, gate_criteria_met flags, and rationale
        """
        # Evaluate each gate criterion
        accuracy_pass = metrics.accuracy >= self.ACCURACY_THRESHOLD
        fpr_pass = metrics.false_positive_rate <= self.FPR_THRESHOLD
        fps_pass = performance_metrics.meets_fps_target
        memory_pass = performance_metrics.meets_memory_target

        # Conservative logic: ALL gates must pass for GO recommendation
        all_gates_pass = accuracy_pass and fpr_pass and fps_pass and memory_pass

        # Generate recommendation
        recommendation = "GO" if all_gates_pass else "NO-GO"

        # Build rationale explaining decision
        rationale_parts = []

        if accuracy_pass:
            rationale_parts.append(f"✓ Accuracy: {metrics.accuracy:.4f} (≥{self.ACCURACY_THRESHOLD})")
        else:
            rationale_parts.append(f"✗ Accuracy: {metrics.accuracy:.4f} (<{self.ACCURACY_THRESHOLD}) - GATE FAILED")

        if fpr_pass:
            rationale_parts.append(f"✓ False Positive Rate: {metrics.false_positive_rate:.4f} (≤{self.FPR_THRESHOLD})")
        else:
            rationale_parts.append(f"✗ False Positive Rate: {metrics.false_positive_rate:.4f} (>{self.FPR_THRESHOLD}) - GATE FAILED")

        if fps_pass:
            rationale_parts.append(f"✓ FPS: {performance_metrics.fps:.4f} (≥{self.FPS_THRESHOLD:.4f})")
        else:
            rationale_parts.append(f"✗ FPS: {performance_metrics.fps:.4f} (<{self.FPS_THRESHOLD:.4f}) - GATE FAILED")

        if memory_pass:
            rationale_parts.append(f"✓ Memory: {performance_metrics.memory_peak_mb:.2f} MB (≤{self.MEMORY_THRESHOLD_MB} MB)")
        else:
            rationale_parts.append(f"✗ Memory: {performance_metrics.memory_peak_mb:.2f} MB (>{self.MEMORY_THRESHOLD_MB} MB) - GATE FAILED")

        rationale = " | ".join(rationale_parts)

        return {
            "recommendation": recommendation,
            "gate_criteria_met": {
                "accuracy": accuracy_pass,
                "false_positive_rate": fpr_pass,
                "fps": fps_pass,
                "memory": memory_pass,
                "all_gates": all_gates_pass
            },
            "rationale": rationale,
            "thresholds": {
                "accuracy_threshold": self.ACCURACY_THRESHOLD,
                "fpr_threshold": self.FPR_THRESHOLD,
                "fps_threshold": self.FPS_THRESHOLD,
                "memory_threshold_mb": self.MEMORY_THRESHOLD_MB
            }
        }


class JSONReportGenerator:
    """Generates machine-readable JSON validation reports.

    Output Format:
    {
      "validation_date": "ISO-8601 timestamp",
      "total_images": integer,
      "metrics": {...},
      "performance": {...},
      "site_breakdown": {...},
      "go_no_go": {...}
    }
    """

    def generate(self, metrics: Metrics, performance_metrics: PerformanceMetrics,
                 go_no_go: Dict[str, Any], output_path: Path) -> None:
        """Generate JSON validation report.

        Args:
            metrics: Validation metrics from test harness
            performance_metrics: Performance metrics from profiler
            go_no_go: Go/no-go decision from decision maker
            output_path: Path to save JSON report
        """
        report = {
            "validation_date": datetime.now().isoformat(),
            "version": "1.0.0",
            "total_images": metrics.total_images,
            "metrics": {
                "accuracy": metrics.accuracy,
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "confusion_matrix": metrics.confusion_matrix,
                "true_positives": metrics.true_positives,
                "true_negatives": metrics.true_negatives,
                "false_positives": metrics.false_positives,
                "false_negatives": metrics.false_negatives,
                "errors_count": metrics.errors_count
            },
            "performance": {
                "mean_fps": performance_metrics.fps,
                "fps_min": performance_metrics.fps_min,
                "fps_max": performance_metrics.fps_max,
                "peak_memory_mb": performance_metrics.memory_peak_mb,
                "mean_memory_mb": performance_metrics.memory_mean_mb,
                "memory_stddev_mb": performance_metrics.memory_stddev_mb,
                "mean_cpu_percent": performance_metrics.cpu_percent_mean,
                "max_cpu_percent": performance_metrics.cpu_percent_max,
                "detection_time_mean_ms": performance_metrics.detection_time_mean_ms,
                "detection_time_stddev_ms": performance_metrics.detection_time_stddev_ms,
                "meets_fps_target": performance_metrics.meets_fps_target,
                "meets_memory_target": performance_metrics.meets_memory_target
            },
            "site_breakdown": metrics.site_breakdown,
            "go_no_go": go_no_go,
            "execution_time_seconds": metrics.total_time_seconds
        }

        # Write JSON report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"JSON report generated: {output_path}")

        # Validate JSON structure (ensure it's parseable)
        self._validate_json(output_path)

    def _validate_json(self, json_path: Path) -> None:
        """Validate JSON report structure.

        Args:
            json_path: Path to JSON report file

        Raises:
            ValueError: If JSON is invalid or missing required fields
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Verify required top-level fields
        required_fields = [
            "validation_date", "total_images", "metrics",
            "performance", "site_breakdown", "go_no_go"
        ]

        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            raise ValueError(f"JSON report missing required fields: {missing_fields}")

        logger.info("JSON report structure validated successfully")


class MarkdownReportGenerator:
    """Generates human-readable Markdown validation reports.

    Output Sections:
    - Executive Summary
    - Validation Metrics (confusion matrix table)
    - Performance Benchmarks (vs targets)
    - Per-Site Breakdown
    - Failure Analysis (if applicable)
    - Go/No-Go Recommendation
    """

    def generate(self, metrics: Metrics, performance_metrics: PerformanceMetrics,
                 go_no_go: Dict[str, Any], output_path: Path) -> None:
        """Generate Markdown validation report.

        Args:
            metrics: Validation metrics from test harness
            performance_metrics: Performance metrics from profiler
            go_no_go: Go/no-go decision from decision maker
            output_path: Path to save Markdown report
        """
        sections = []

        # Header
        sections.append(self._generate_header())

        # Executive Summary
        sections.append(self._generate_executive_summary(metrics, performance_metrics, go_no_go))

        # Validation Metrics
        sections.append(self._generate_metrics_section(metrics))

        # Performance Benchmarks
        sections.append(self._generate_performance_section(performance_metrics))

        # Per-Site Breakdown
        sections.append(self._generate_site_breakdown(metrics))

        # Failure Analysis (if applicable)
        if metrics.false_positives > 0 or metrics.false_negatives > 0:
            sections.append(self._generate_failure_analysis(metrics))

        # Go/No-Go Recommendation
        sections.append(self._generate_go_no_go_section(go_no_go))

        # Next Steps
        sections.append(self._generate_next_steps(go_no_go))

        # Write Markdown report
        report_content = "\n\n".join(sections)
        with open(output_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Markdown report generated: {output_path}")

    def _generate_header(self) -> str:
        """Generate report header."""
        return f"""# Stage 3 Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Version:** 1.0.0"""

    def _generate_executive_summary(self, metrics: Metrics, performance_metrics: PerformanceMetrics,
                                    go_no_go: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        recommendation = go_no_go['recommendation']
        recommendation_emoji = "✅" if recommendation == "GO" else "❌"

        return f"""## Executive Summary

{recommendation_emoji} **Production Readiness: {recommendation}**

### Key Findings

- **Detection Accuracy:** {metrics.accuracy*100:.2f}% ({metrics.true_positives + metrics.true_negatives}/{metrics.total_images} correct predictions)
- **Performance:** {performance_metrics.fps:.4f} FPS, {performance_metrics.memory_peak_mb:.2f} MB peak memory
- **Error Rate:** {metrics.false_positive_rate*100:.2f}% false positives, {metrics.false_negative_rate*100:.2f}% false negatives
- **Sites Tested:** {len(metrics.site_breakdown)} DAF sites ({metrics.total_images} total images)
- **Execution Time:** {metrics.total_time_seconds:.2f} seconds"""

    def _generate_metrics_section(self, metrics: Metrics) -> str:
        """Generate validation metrics section with confusion matrix."""
        cm = metrics.confusion_matrix

        return f"""## Validation Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| **Total Images** | {metrics.total_images} |
| **Accuracy** | {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%) |
| **False Positive Rate** | {metrics.false_positive_rate:.4f} ({metrics.false_positive_rate*100:.2f}%) |
| **False Negative Rate** | {metrics.false_negative_rate:.4f} ({metrics.false_negative_rate*100:.2f}%) |
| **Detection Errors** | {metrics.errors_count} |

### Confusion Matrix

|                    | **Predicted: No Shift** | **Predicted: Shift** |
|--------------------|------------------------|---------------------|
| **Actual: No Shift** | {cm['TN']} (TN) | {cm['FP']} (FP) |
| **Actual: Shift**    | {cm['FN']} (FN) | {cm['TP']} (TP) |"""

    def _generate_performance_section(self, perf: PerformanceMetrics) -> str:
        """Generate performance benchmarks section."""
        fps_status = "✓ PASS" if perf.meets_fps_target else "✗ FAIL"
        memory_status = "✓ PASS" if perf.meets_memory_target else "✗ FAIL"

        fps_target = 1.0 / 60.0
        memory_target = 500.0

        return f"""## Performance Benchmarks

### System Resource Utilization

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **FPS (Mean)** | {perf.fps:.4f} fps | ≥{fps_target:.4f} fps | {fps_status} |
| **FPS Range** | {perf.fps_min:.4f} - {perf.fps_max:.4f} fps | - | - |
| **Peak Memory** | {perf.memory_peak_mb:.2f} MB | ≤{memory_target:.0f} MB | {memory_status} |
| **Mean Memory** | {perf.memory_mean_mb:.2f} MB | - | - |
| **Mean CPU** | {perf.cpu_percent_mean:.2f}% | - | - |
| **Peak CPU** | {perf.cpu_percent_max:.2f}% | - | - |

### Detection Time Statistics

- **Mean Detection Time:** {perf.detection_time_mean_ms:.2f} ms
- **Std Deviation:** {perf.detection_time_stddev_ms:.2f} ms
- **Total Images Profiled:** {perf.total_images}"""

    def _generate_site_breakdown(self, metrics: Metrics) -> str:
        """Generate per-site breakdown section."""
        site_rows = []
        for site_id, site_metrics in metrics.site_breakdown.items():
            accuracy_pct = site_metrics['accuracy'] * 100
            site_rows.append(
                f"| **{site_id}** | {site_metrics['accuracy']:.4f} ({accuracy_pct:.2f}%) | "
                f"{site_metrics['correct']}/{site_metrics['total']} |"
            )

        site_table = "\n".join(site_rows)

        return f"""## Per-Site Breakdown

| Site | Accuracy | Correct/Total |
|------|----------|---------------|
{site_table}"""

    def _generate_failure_analysis(self, metrics: Metrics) -> str:
        """Generate failure analysis section."""
        return f"""## Failure Analysis

### False Positives ({metrics.false_positives} cases)
Camera shifts detected where none occurred. This may indicate:
- Overly sensitive feature matching thresholds
- Environmental factors (lighting changes, camera vibration)
- Need for stricter validation criteria

### False Negatives ({metrics.false_negatives} cases)
Camera shifts missed by the detector. This may indicate:
- Insufficient feature density in ROI
- Subtle shifts below detection threshold
- Need for more sensitive matching parameters

**Recommendation:** Review individual failure cases to identify patterns and refine detection parameters."""

    def _generate_go_no_go_section(self, go_no_go: Dict[str, Any]) -> str:
        """Generate go/no-go recommendation section."""
        recommendation = go_no_go['recommendation']
        rationale = go_no_go['rationale']
        gates = go_no_go['gate_criteria_met']

        recommendation_emoji = "✅ GO" if recommendation == "GO" else "❌ NO-GO"
        recommendation_style = "**APPROVED FOR PRODUCTION**" if recommendation == "GO" else "**NOT READY FOR PRODUCTION**"

        gate_status = []
        for gate_name, passed in gates.items():
            if gate_name == "all_gates":
                continue
            status_emoji = "✓" if passed else "✗"
            gate_status.append(f"- {status_emoji} {gate_name.replace('_', ' ').title()}")

        gate_status_str = "\n".join(gate_status)

        return f"""## Go/No-Go Recommendation

### {recommendation_emoji} {recommendation_style}

**Rationale:** {rationale}

### Gate Criteria Status

{gate_status_str}

### Decision Logic

All gate criteria must pass for GO recommendation (conservative approach).
Any single gate failure results in NO-GO to ensure production safety."""

    def _generate_next_steps(self, go_no_go: Dict[str, Any]) -> str:
        """Generate next steps section."""
        if go_no_go['recommendation'] == "GO":
            return """## Next Steps

1. **Proceed with Production Deployment**
   - Review deployment checklist
   - Schedule deployment window
   - Prepare rollback procedures

2. **Monitoring Setup**
   - Configure production monitoring dashboards
   - Set up alerting for detection anomalies
   - Plan regular validation cycles

3. **Documentation**
   - Update production runbook
   - Document deployment procedures
   - Archive validation results"""
        else:
            gates = go_no_go['gate_criteria_met']
            failed_gates = [k for k, v in gates.items() if not v and k != "all_gates"]

            return f"""## Next Steps

1. **Address Failed Gate Criteria**
   - Failed gates: {', '.join(failed_gates)}
   - Review failure root causes
   - Implement corrective actions

2. **Re-validate After Fixes**
   - Run Stage 3 validation again
   - Verify all gates pass
   - Document improvements

3. **Production Hold**
   - Do NOT deploy to production
   - Schedule fix review meeting
   - Plan re-validation timeline"""
