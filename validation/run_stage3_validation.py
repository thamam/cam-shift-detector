#!/usr/bin/env python3
"""
Stage 3 Validation Runner - Complete Workflow Orchestration

Executes the complete Stage 3 validation workflow:
1. Load validation dataset (Story 1)
2. Run test harness with performance profiling (Story 2)
3. Generate comprehensive reports (JSON + Markdown)
4. Provide go/no-go production readiness recommendation

Usage:
    python validation/run_stage3_validation.py [--baseline PATH] [--output-dir DIR]

Exit Codes:
    0: Validation successful, reports generated
    1: Validation errors detected
    2: System errors (file not found, permission denied, etc.)
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# Import Story 2 components (no modifications to existing code)
# Note: RealDataLoader from Story 1 is used internally by Stage3TestHarness
from validation.stage3_test_harness import Stage3TestHarness, Metrics
from validation.performance_profiler import PerformanceMetrics


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationRunner:
    """Orchestrates complete Stage 3 validation workflow.

    Sequential execution:
    1. Load data (RealDataLoader from Story 1)
    2. Run harness (Stage3TestHarness from Story 2)
    3. Generate reports (JSON + Markdown)

    Implements graceful degradation: log errors but continue when possible.
    """

    def __init__(self, detector_config_path: str,
                 baseline_image_path: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        """Initialize validation runner.

        Args:
            detector_config_path: Path to detector configuration JSON
            baseline_image_path: Optional baseline image (default: first image)
            output_dir: Output directory for reports (default: validation/results/)
        """
        self.detector_config_path = detector_config_path
        self.baseline_image_path = baseline_image_path
        self.output_dir = output_dir or Path("validation/results")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ValidationRunner initialized")
        logger.info(f"  Detector config: {detector_config_path}")
        logger.info(f"  Output directory: {self.output_dir}")

    def run(self) -> Tuple[int, Optional[Metrics], Optional[PerformanceMetrics]]:
        """Execute complete validation workflow.

        Returns:
            Tuple of (exit_code, metrics, performance_metrics)

        Exit Codes:
            0: Success
            1: Validation errors
            2: System errors
        """
        try:
            logger.info("=" * 60)
            logger.info("STAGE 3 VALIDATION WORKFLOW")
            logger.info("=" * 60)

            # Step 1: Initialize test harness
            logger.info("\n[Step 1/3] Initializing test harness...")
            harness = self._initialize_harness()
            if harness is None:
                return 2, None, None  # System error

            # Step 2: Execute validation
            logger.info("\n[Step 2/3] Executing validation workflow...")
            metrics, performance_metrics = self._execute_validation(harness)
            if metrics is None or performance_metrics is None:
                return 1, None, None  # Validation error

            # Step 3: Generate reports
            logger.info("\n[Step 3/3] Generating validation reports...")
            success = self._generate_reports(metrics, performance_metrics)
            if not success:
                logger.warning("Report generation failed, but validation completed")
                return 1, metrics, performance_metrics

            logger.info("\n" + "=" * 60)
            logger.info("VALIDATION WORKFLOW COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Reports saved to: {self.output_dir}")

            return 0, metrics, performance_metrics

        except KeyboardInterrupt:
            logger.warning("\nValidation interrupted by user")
            return 2, None, None
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}", exc_info=True)
            return 2, None, None

    def _initialize_harness(self) -> Optional[Stage3TestHarness]:
        """Initialize test harness with error handling.

        Returns:
            Stage3TestHarness instance or None on error
        """
        try:
            harness = Stage3TestHarness(
                detector_config_path=self.detector_config_path
            )
            logger.info("Test harness initialized successfully")
            return harness
        except FileNotFoundError as e:
            logger.error(f"Configuration file not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize test harness: {e}", exc_info=True)
            return None

    def _execute_validation(self, harness: Stage3TestHarness) -> Tuple[Optional[Metrics], Optional[PerformanceMetrics]]:
        """Execute validation workflow with progress reporting.

        Args:
            harness: Initialized test harness

        Returns:
            Tuple of (Metrics, PerformanceMetrics) or (None, None) on error
        """
        try:
            # Run validation (this executes Stories 1 & 2 logic)
            metrics, performance_metrics = harness.run_validation(
                baseline_image_path=self.baseline_image_path
            )

            logger.info("\n✅ Validation execution complete")
            logger.info(f"  Processed: {metrics.total_images} images")
            logger.info(f"  Accuracy: {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)")
            logger.info(f"  Errors: {metrics.errors_count}")

            return metrics, performance_metrics

        except Exception as e:
            logger.error(f"Validation execution failed: {e}", exc_info=True)
            return None, None

    def _generate_reports(self, metrics: Metrics, performance_metrics: PerformanceMetrics) -> bool:
        """Generate JSON and Markdown reports.

        Args:
            metrics: Validation metrics from test harness
            performance_metrics: Performance metrics from profiler

        Returns:
            True if reports generated successfully, False otherwise
        """
        try:
            # Import report generators (will be implemented in next phase)
            from validation.report_generator import JSONReportGenerator, MarkdownReportGenerator, GoNoGoDecisionMaker

            # Make go/no-go decision
            decision_maker = GoNoGoDecisionMaker()
            go_no_go = decision_maker.evaluate(metrics, performance_metrics)

            # Generate JSON report
            json_generator = JSONReportGenerator()
            json_path = self.output_dir / "validation_report.json"
            json_generator.generate(metrics, performance_metrics, go_no_go, json_path)
            logger.info(f"✅ JSON report: {json_path}")

            # Generate Markdown report
            md_generator = MarkdownReportGenerator()
            md_path = self.output_dir / "validation_report.md"
            md_generator.generate(metrics, performance_metrics, go_no_go, md_path)
            logger.info(f"✅ Markdown report: {md_path}")

            return True

        except ImportError:
            logger.warning("Report generators not yet implemented - skipping report generation")
            return False
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            return False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Stage 3 Validation Runner - Execute complete validation workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validation/run_stage3_validation.py
  python validation/run_stage3_validation.py --baseline sample_images/of_jerusalem/img_001.jpg
  python validation/run_stage3_validation.py --output-dir custom_results/
        """
    )

    parser.add_argument(
        '--baseline',
        type=Path,
        default=None,
        help='Baseline image path (default: use first image from dataset)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("validation/results"),
        help='Output directory for reports (default: validation/results/)'
    )

    parser.add_argument(
        '--detector-config',
        type=str,
        default='config/detector_config.json',
        help='Detector configuration path (default: config/detector_config.json)'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for validation runner.

    Returns:
        Exit code (0=success, 1=validation error, 2=system error)
    """
    args = parse_arguments()

    # Initialize and run validation
    runner = ValidationRunner(
        detector_config_path=args.detector_config,
        baseline_image_path=args.baseline,
        output_dir=args.output_dir
    )

    exit_code, metrics, performance_metrics = runner.run()

    # Log final status
    if exit_code == 0:
        logger.info("\n✅ Validation completed successfully")
    elif exit_code == 1:
        logger.warning("\n⚠️ Validation completed with errors")
    else:
        logger.error("\n❌ Validation failed due to system errors")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
