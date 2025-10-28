"""
Comparison Logger - JSON Logging and MSE Analysis

Logs dual detector results to JSON, calculates MSE metrics, identifies worst
matches, and generates matplotlib visualizations for analysis.
"""

import csv
import json
import logging
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from validation.utilities.dual_detector_runner import DualDetectionResult
from validation.utilities.comparison_metrics import calculate_mse


logger = logging.getLogger(__name__)


class ComparisonLogger:
    """Logs and analyzes dual detector comparison results.

    Provides JSON logging, MSE calculation, worst match retrieval,
    and matplotlib visualization for detector agreement analysis.
    """

    def __init__(self, output_dir: str, session_name: str):
        """Initialize comparison logger.

        Args:
            output_dir: Directory for output files (logs, graphs)
            session_name: Session identifier for file naming

        Note:
            Creates output_dir if it doesn't exist
        """
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        self.results: List[DualDetectionResult] = []

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ComparisonLogger:")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Session: {session_name}")

    def log_frame(self, result: DualDetectionResult) -> None:
        """Log a single frame's dual detection result.

        Args:
            result: DualDetectionResult from process_frame()

        Note:
            Results are stored in memory until save_log() is called
        """
        self.results.append(result)

    def save_log(self, filename: Optional[str] = None) -> Path:
        """Persist logged results to structured JSON file.

        Args:
            filename: Optional custom filename (default: {session_name}_comparison.json)

        Returns:
            Path to saved JSON file

        JSON Structure:
            {
                "session_name": str,
                "timestamp": str (ISO format),
                "threshold_px": float,
                "total_frames": int,
                "results": [
                    {
                        "frame_idx": int,
                        "timestamp_ns": int,
                        "charuco_detected": bool,
                        "charuco_displacement_px": float,
                        "charuco_confidence": float,
                        "camshift_status": str,
                        "camshift_displacement_px": float,
                        "camshift_confidence": float,
                        "displacement_diff": float,
                        "agreement_status": str or null,
                        "threshold_px": float
                    },
                    ...
                ]
            }
        """
        if filename is None:
            filename = f"{self.session_name}_comparison.json"

        output_path = self.output_dir / filename

        # Build structured log
        log_data = {
            "session_name": self.session_name,
            "timestamp": datetime.now().isoformat(),
            "threshold_px": self.results[0].threshold_px if self.results else 0.0,
            "total_frames": len(self.results),
            "results": [asdict(result) for result in self.results]
        }

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Saved comparison log: {output_path} ({len(self.results)} frames)")
        return output_path

    def save_csv(self) -> Path:
        """Export comparison results to CSV format.

        CSV columns: frame_id, dx_charuco, dy_charuco, dx_csd, dy_csd, 
                     delta_dx, delta_dy, error_mag, inliers, total, rmse

        Returns:
            Path to saved CSV file

        Raises:
            ValueError: If no results have been logged
        """
        if not self.results:
            raise ValueError("No results to export")

        # Generate timestamp-based filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{self.session_name}_comparison_{timestamp}.csv"

        # Calculate session RMSE
        valid_results = [
            r for r in self.results 
            if not np.isnan(r.charuco_displacement_px) and not np.isnan(r.camshift_displacement_px)
        ]
        
        if valid_results:
            squared_errors = [
                (r.charuco_displacement_px - r.camshift_displacement_px) ** 2 
                for r in valid_results
            ]
            rmse = np.sqrt(np.mean(squared_errors))
        else:
            rmse = np.nan

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'frame_id', 'dx_charuco', 'dy_charuco', 'dx_csd', 'dy_csd',
                'delta_dx', 'delta_dy', 'error_mag', 'inliers', 'total', 'rmse'
            ])

            # Data rows
            for result in self.results:
                # Calculate deltas and error magnitude
                if not np.isnan(result.charuco_dx) and not np.isnan(result.camshift_dx):
                    delta_dx = result.camshift_dx - result.charuco_dx
                    delta_dy = result.camshift_dy - result.charuco_dy
                    error_mag = np.sqrt(delta_dx**2 + delta_dy**2)
                else:
                    delta_dx = np.nan
                    delta_dy = np.nan
                    error_mag = np.nan

                # Extract inliers/total from confidence scores
                # ChArUco confidence = corner count (total)
                # CamShift confidence = inlier ratio
                total = result.charuco_confidence if not np.isnan(result.charuco_confidence) else 0
                inliers = total  # Approximation: assume all ChArUco corners are inliers

                writer.writerow([
                    result.frame_idx,
                    f"{result.charuco_dx:.2f}" if not np.isnan(result.charuco_dx) else "NaN",
                    f"{result.charuco_dy:.2f}" if not np.isnan(result.charuco_dy) else "NaN",
                    f"{result.camshift_dx:.2f}" if not np.isnan(result.camshift_dx) else "NaN",
                    f"{result.camshift_dy:.2f}" if not np.isnan(result.camshift_dy) else "NaN",
                    f"{delta_dx:.2f}" if not np.isnan(delta_dx) else "NaN",
                    f"{delta_dy:.2f}" if not np.isnan(delta_dy) else "NaN",
                    f"{error_mag:.2f}" if not np.isnan(error_mag) else "NaN",
                    f"{inliers:.0f}",
                    f"{total:.0f}",
                    f"{rmse:.2f}" if not np.isnan(rmse) else "NaN"
                ])

        logger.info(f"Saved CSV export: {output_path} ({len(self.results)} frames)")
        return output_path

    def calculate_mse(self) -> float:
        """Calculate Mean Squared Error across all logged frames.

        Returns:
            MSE between ChArUco and cam-shift displacement sequences

        Raises:
            ValueError: If no valid comparison data available (all ChArUco failures)

        Note:
            Only includes frames where ChArUco was detected (displacement_diff is not NaN)
        """
        # Extract valid displacement pairs (ChArUco detected)
        charuco_disps = []
        camshift_disps = []

        for result in self.results:
            if result.charuco_detected and not np.isnan(result.displacement_diff):
                charuco_disps.append(result.charuco_displacement_px)
                camshift_disps.append(result.camshift_displacement_px)

        if not charuco_disps:
            raise ValueError("No valid comparison data (all ChArUco detections failed)")

        mse = calculate_mse(charuco_disps, camshift_disps)
        logger.info(f"MSE calculated: {mse:.4f} (across {len(charuco_disps)} valid frames)")
        return mse

    def get_worst_matches(self, n: int = 10) -> List[DualDetectionResult]:
        """Retrieve top N frames with largest displacement differences.

        Args:
            n: Number of worst matches to retrieve (default: 10)

        Returns:
            List of DualDetectionResult sorted by displacement_diff (descending)

        Note:
            Only includes frames where ChArUco was detected.
            If n > available frames, returns all available frames.
        """
        # Filter valid results (ChArUco detected)
        valid_results = [
            result for result in self.results
            if result.charuco_detected and not np.isnan(result.displacement_diff)
        ]

        # Sort by displacement_diff descending
        sorted_results = sorted(
            valid_results,
            key=lambda r: r.displacement_diff,
            reverse=True
        )

        # Return top N (or all if fewer than N)
        worst_matches = sorted_results[:n]

        logger.info(f"Retrieved {len(worst_matches)} worst matches (requested: {n})")
        return worst_matches

    def generate_mse_graph(
        self,
        output_path: Optional[str] = None,
        highlight_worst_n: int = 10
    ) -> Path:
        """Generate matplotlib MSE graph with threshold line and highlighted worst matches.

        Args:
            output_path: Optional custom output path (default: {session_name}_mse_graph.png)
            highlight_worst_n: Number of worst matches to highlight in red (default: 10)

        Returns:
            Path to saved PNG file

        Graph Features:
            - X-axis: frame_idx
            - Y-axis: displacement_diff (pixels)
            - Blue dots: All comparison points
            - Red dots: Worst N matches
            - Horizontal line: Threshold (GREEN/RED boundary)
            - Legend and axis labels
        """
        if output_path is None:
            output_path = self.output_dir / f"{self.session_name}_mse_graph.png"
        else:
            output_path = Path(output_path)

        # Extract data for plotting (only valid comparisons)
        frame_indices = []
        displacement_diffs = []
        threshold_px = None

        for result in self.results:
            if result.charuco_detected and not np.isnan(result.displacement_diff):
                frame_indices.append(result.frame_idx)
                displacement_diffs.append(result.displacement_diff)
                if threshold_px is None:
                    threshold_px = result.threshold_px

        if not frame_indices:
            raise ValueError("No valid comparison data for graph generation")

        # Get worst matches for highlighting
        worst_matches = self.get_worst_matches(n=highlight_worst_n)
        worst_frame_indices = {result.frame_idx for result in worst_matches}

        # Separate worst matches for red highlighting
        normal_indices = []
        normal_diffs = []
        worst_indices = []
        worst_diffs = []

        for idx, diff in zip(frame_indices, displacement_diffs):
            if idx in worst_frame_indices:
                worst_indices.append(idx)
                worst_diffs.append(diff)
            else:
                normal_indices.append(idx)
                normal_diffs.append(diff)

        # Create figure
        plt.figure(figsize=(12, 6))

        # Plot normal points
        if normal_indices:
            plt.scatter(normal_indices, normal_diffs, c='blue', alpha=0.6,
                       label='Comparison Points', s=30)

        # Plot worst matches
        if worst_indices:
            plt.scatter(worst_indices, worst_diffs, c='red', alpha=0.8,
                       label=f'Worst {len(worst_matches)} Matches', s=50, marker='x')

        # Plot threshold line
        if threshold_px is not None:
            plt.axhline(y=threshold_px, color='green', linestyle='--',
                       linewidth=2, label=f'Threshold ({threshold_px:.2f}px)')

        # Labels and title
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Displacement Difference (pixels)', fontsize=12)
        plt.title(f'ChArUco vs Cam-Shift Comparison: {self.session_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        logger.info(f"Generated MSE graph: {output_path}")
        return output_path
