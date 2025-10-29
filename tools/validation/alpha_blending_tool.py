#!/usr/bin/env python3
"""
Mode C - Enhanced Alpha Blending Tool

Interactive debugging tool for visual frame alignment verification with:
- Transform computation using CSD
- Pre-warp toggle for alignment testing
- Blink mode for A/B comparison
- Frame selection for arbitrary pairs
- Grid overlay for alignment reference
- Export with metadata

Controls:
    a       - Enter Frame A selection mode
    b       - Enter Frame B selection mode
    →/←     - Navigate frames (in selection mode)
    Enter   - Confirm frame selection
    w       - Toggle pre-warp mode
    Space   - Toggle blink mode (alternates A/B)
    g       - Toggle alignment grid overlay
    ↑/↓     - Adjust alpha value
    s       - Save snapshot (PNG + CSV)
    q       - Quit

Usage:
    python tools/validation/alpha_blending_tool.py --input-dir sample_images/of_jerusalem
    python tools/validation/alpha_blending_tool.py --input-dir sample_images --config config.json
"""

import cv2
import numpy as np
import argparse
import sys
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.camera_movement_detector import CameraMovementDetector


class AlphaBlendingTool:
    """Mode C - Enhanced Alpha Blending Tool for visual alignment verification."""

    def __init__(self, image_dir: str, config_path: Optional[str] = None):
        """Initialize Mode C tool.

        Args:
            image_dir: Directory containing image sequence
            config_path: Optional path to CSD config file
        """
        self.image_dir = Path(image_dir)
        self.config_path = config_path or "config.json"

        # Load image sequence
        self.images = self._load_images()
        if len(self.images) < 2:
            raise ValueError(f"Need at least 2 images, found {len(self.images)}")

        # Initialize CSD
        self.detector = CameraMovementDetector(self.config_path)

        # Frame selection state
        self.frame_a_idx = 0  # Default: first frame
        self.frame_b_idx = 1  # Default: second frame
        self.selection_mode: Optional[str] = None  # 'A', 'B', or None
        self.temp_selection_idx = 0

        # Visual state
        self.alpha = 0.5
        self.prewarp_enabled = False
        self.blink_mode = False
        self.blink_state = 0  # 0 = show frame A, 1 = show frame B
        self.last_blink_time = time.time()
        self.show_grid = False

        # Homography state
        self.homography: Optional[np.ndarray] = None
        self.transform_computed = False
        self.transform_time_ms: Optional[float] = None

        print(f"✓ Loaded {len(self.images)} images from {self.image_dir}")
        print(f"✓ CSD config: {self.config_path}")
        print(f"✓ Frame A (baseline): #{self.frame_a_idx} - {self.images[self.frame_a_idx].name}")
        print(f"✓ Frame B (compare): #{self.frame_b_idx} - {self.images[self.frame_b_idx].name}")

    def _load_images(self) -> List[Path]:
        """Load all images from directory."""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in extensions:
            images.extend(sorted(self.image_dir.glob(f'*{ext}')))
            images.extend(sorted(self.image_dir.glob(f'*{ext.upper()}')))
        return sorted(set(images))  # Remove duplicates and sort

    def _compute_transform(self):
        """Compute homography between Frame A and Frame B using CSD."""
        start_time = time.time()

        frame_a = cv2.imread(str(self.images[self.frame_a_idx]))
        frame_b = cv2.imread(str(self.images[self.frame_b_idx]))

        if frame_a is None or frame_b is None:
            print("ERROR: Failed to load frames")
            return

        # Set baseline and process frame
        self.detector.set_baseline(frame_a)
        result = self.detector.process_frame(frame_b, frame_id=f"frame_{self.frame_b_idx}")

        # Get homography
        self.homography = self.detector.get_last_homography()
        self.transform_computed = True

        # Measure time
        elapsed_ms = (time.time() - start_time) * 1000
        self.transform_time_ms = elapsed_ms

        status = "✓" if result.get('status') == 'VALID' else "⚠"
        displacement = result.get('displacement', 0.0)
        print(f"{status} Transform computed in {elapsed_ms:.1f}ms (displacement: {displacement:.2f}px)")

    def _apply_prewarp(self, frame: np.ndarray) -> np.ndarray:
        """Warp Frame B to align with Frame A using computed homography."""
        if self.homography is None:
            return frame

        h, w = frame.shape[:2]
        warped = cv2.warpPerspective(frame, self.homography, (w, h))
        return warped

    def _draw_alignment_grid(self, image: np.ndarray, rows: int = 10, cols: int = 10) -> np.ndarray:
        """Draw reference grid for alignment verification.

        Args:
            image: Input image
            rows: Number of horizontal lines
            cols: Number of vertical lines

        Returns:
            Image with grid overlay (50% transparency)
        """
        h, w = image.shape[:2]
        overlay = image.copy()

        # Grid color: cyan (BGR: 255, 255, 0)
        color = (255, 255, 0)
        thickness = 1

        # Vertical lines
        for i in range(1, cols):
            x = int(w * i / cols)
            cv2.line(overlay, (x, 0), (x, h), color, thickness)

        # Horizontal lines
        for i in range(1, rows):
            y = int(h * i / rows)
            cv2.line(overlay, (0, y), (w, y), color, thickness)

        # Blend with original (50% transparency)
        blended = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
        return blended

    def _create_blended_view(self) -> np.ndarray:
        """Create blended view of Frame A and Frame B."""
        frame_a = cv2.imread(str(self.images[self.frame_a_idx]))
        frame_b = cv2.imread(str(self.images[self.frame_b_idx]))

        if frame_a is None or frame_b is None:
            # Return blank frame on error
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Compute transform if not done yet
        if not self.transform_computed:
            self._compute_transform()

        # Apply pre-warp if enabled
        if self.prewarp_enabled and self.homography is not None:
            frame_b = self._apply_prewarp(frame_b)

        # Blend
        blended = cv2.addWeighted(frame_a, 1 - self.alpha, frame_b, self.alpha, 0)

        # Apply grid overlay if enabled
        if self.show_grid:
            blended = self._draw_alignment_grid(blended)

        return blended

    def _add_ui_overlay(self, image: np.ndarray) -> np.ndarray:
        """Add status indicators and controls overlay."""
        h, w = image.shape[:2]
        display = image.copy()

        # Semi-transparent background for text
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - 100), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Title
        cv2.putText(display, "Mode C - Enhanced Alpha Blending Tool",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Frame info
        frame_a_name = self.images[self.frame_a_idx].stem
        frame_b_name = self.images[self.frame_b_idx].stem
        cv2.putText(display, f"Frame A (baseline): #{self.frame_a_idx} - {frame_a_name}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(display, f"Frame B (compare): #{self.frame_b_idx} - {frame_b_name}",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

        # Alpha value as percentage
        alpha_pct = int(self.alpha * 100)
        cv2.putText(display, f"Alpha: {alpha_pct}% (Up/Down to adjust)",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Status indicators
        y_pos = 155
        line_height = 30

        # Pre-warp indicator
        prewarp_status = "ON" if self.prewarp_enabled else "OFF"
        prewarp_color = (0, 255, 0) if self.prewarp_enabled else (100, 100, 100)
        cv2.putText(display, f"Pre-warp: {prewarp_status} (w)",
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, prewarp_color, 1)

        # Blink mode indicator
        blink_status = "ON" if self.blink_mode else "OFF"
        blink_color = (0, 255, 255) if self.blink_mode else (100, 100, 100)
        cv2.putText(display, f"Blink: {blink_status} (Space)",
                    (20, y_pos + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 1)

        # Grid indicator
        grid_status = "ON" if self.show_grid else "OFF"
        grid_color = (255, 255, 0) if self.show_grid else (100, 100, 100)
        cv2.putText(display, f"Grid: {grid_status} (g)",
                    (20, y_pos + line_height * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, grid_color, 1)

        # Transform time
        if self.transform_time_ms is not None:
            time_color = (0, 255, 0) if self.transform_time_ms < 500 else (0, 165, 255)
            cv2.putText(display, f"Transform: {self.transform_time_ms:.1f}ms",
                        (20, y_pos + line_height * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, time_color, 1)

        # Controls (bottom)
        controls1 = "[a]Frame-A  [b]Frame-B  [w]Pre-warp  [Space]Blink  [g]Grid"
        controls2 = "Up/Down: Alpha  |  [s]Snapshot  |  [q]Quit"

        cv2.putText(display, controls1,
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, controls2,
                    (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Selection mode overlay
        if self.selection_mode:
            sel_overlay = display.copy()
            cv2.rectangle(sel_overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 255, 255), -1)
            cv2.addWeighted(sel_overlay, 0.3, display, 0.7, 0, display)

            sel_text = f"Select Frame {self.selection_mode}"
            sel_info = f"Current: #{self.temp_selection_idx} - {self.images[self.temp_selection_idx].stem}"
            sel_controls = "Use ←/→ arrows, press Enter to confirm"

            cv2.putText(display, sel_text,
                        (w//4 + 20, h//3 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(display, sel_info,
                        (w//4 + 20, h//3 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display, sel_controls,
                        (w//4 + 20, h//3 + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        return display

    def _save_snapshot(self):
        """Save current blended view as PNG with metadata and CSV export."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prewarp_str = "prewarp" if self.prewarp_enabled else "noprew arp"
        alpha_str = f"alpha{int(self.alpha * 100)}"

        # Generate filename
        filename_base = f"mode_c_frameA{self.frame_a_idx}_frameB{self.frame_b_idx}_{alpha_str}_{prewarp_str}_{timestamp}"
        png_path = self.image_dir / f"{filename_base}.png"
        csv_path = self.image_dir / f"{filename_base}.csv"

        # Get current blended view (without UI overlay)
        blended = self._create_blended_view()

        # Save PNG
        cv2.imwrite(str(png_path), blended)
        print(f"✓ Saved snapshot: {png_path.name}")

        # Save CSV with transform parameters
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Parameter', 'Value'])
            writer.writerow(['timestamp', timestamp])
            writer.writerow(['frame_a_index', self.frame_a_idx])
            writer.writerow(['frame_b_index', self.frame_b_idx])
            writer.writerow(['frame_a_path', self.images[self.frame_a_idx].name])
            writer.writerow(['frame_b_path', self.images[self.frame_b_idx].name])
            writer.writerow(['alpha', self.alpha])
            writer.writerow(['prewarp_enabled', self.prewarp_enabled])
            writer.writerow(['grid_overlay', self.show_grid])
            writer.writerow(['transform_time_ms', self.transform_time_ms or 'N/A'])

            # Write homography matrix if available
            if self.homography is not None:
                writer.writerow([])
                writer.writerow(['Homography Matrix'])
                for i, row in enumerate(self.homography):
                    writer.writerow([f'H_row_{i}'] + row.tolist())

        print(f"✓ Saved metadata: {csv_path.name}")

    def run(self):
        """Run Mode C interactive tool."""
        print("\n" + "=" * 80)
        print("Mode C - Enhanced Alpha Blending Tool")
        print("=" * 80)
        print("\nControls:")
        print("  a         - Enter Frame A selection mode")
        print("  b         - Enter Frame B selection mode")
        print("  →/←       - Navigate frames (in selection mode)")
        print("  Enter     - Confirm frame selection")
        print("  w         - Toggle pre-warp mode")
        print("  Space     - Toggle blink mode (alternates A/B)")
        print("  g         - Toggle alignment grid overlay")
        print("  ↑/↓       - Adjust alpha value")
        print("  s         - Save snapshot (PNG + CSV)")
        print("  q         - Quit")
        print("=" * 80)
        print("\nStarting Mode C...")

        cv2.namedWindow("Mode C", cv2.WINDOW_NORMAL)

        while True:
            # Handle blink mode timing
            if self.blink_mode:
                current_time = time.time()
                if current_time - self.last_blink_time > 0.5:  # 500ms
                    self.blink_state = 1 - self.blink_state
                    self.last_blink_time = current_time

            # Generate display
            if self.blink_mode:
                # Blink mode: alternate between Frame A and Frame B
                frame_idx = self.frame_a_idx if self.blink_state == 0 else self.frame_b_idx
                frame = cv2.imread(str(self.images[frame_idx]))
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                display = frame.copy()
            else:
                # Normal mode: show blended view
                display = self._create_blended_view()

            # Add UI overlay
            display = self._add_ui_overlay(display)

            # Show
            cv2.imshow("Mode C", display)

            # Handle input (1ms wait for smooth blink timing)
            key = cv2.waitKey(1) & 0xFF

            # Selection mode handling
            if self.selection_mode:
                if key == 13:  # Enter
                    # Confirm selection
                    if self.selection_mode == 'A':
                        self.frame_a_idx = self.temp_selection_idx
                        print(f"✓ Frame A set to #{self.frame_a_idx}")
                    elif self.selection_mode == 'B':
                        self.frame_b_idx = self.temp_selection_idx
                        print(f"✓ Frame B set to #{self.frame_b_idx}")

                    # Reset selection mode and recompute transform
                    self.selection_mode = None
                    self.transform_computed = False
                    self._compute_transform()

                elif key == 83 or key == 3:  # Right arrow
                    self.temp_selection_idx = (self.temp_selection_idx + 1) % len(self.images)
                elif key == 81 or key == 2:  # Left arrow
                    self.temp_selection_idx = (self.temp_selection_idx - 1) % len(self.images)
                elif key == 27 or key == ord('q'):  # ESC or Q - cancel selection
                    self.selection_mode = None
                    print("Selection cancelled")

                continue  # Skip other key handling during selection

            # Normal mode key handling
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nQuitting Mode C...")
                break

            elif key == ord('a'):
                # Enter Frame A selection mode
                self.selection_mode = 'A'
                self.temp_selection_idx = self.frame_a_idx
                print("Entering Frame A selection mode (→/← to navigate, Enter to confirm)")

            elif key == ord('b'):
                # Enter Frame B selection mode
                self.selection_mode = 'B'
                self.temp_selection_idx = self.frame_b_idx
                print("Entering Frame B selection mode (→/← to navigate, Enter to confirm)")

            elif key == ord('w'):
                # Toggle pre-warp
                self.prewarp_enabled = not self.prewarp_enabled
                status = "enabled" if self.prewarp_enabled else "disabled"
                print(f"Pre-warp {status}")

            elif key == ord(' '):  # Space
                # Toggle blink mode
                self.blink_mode = not self.blink_mode
                self.last_blink_time = time.time()
                status = "enabled" if self.blink_mode else "disabled"
                print(f"Blink mode {status}")

            elif key == ord('g'):
                # Toggle grid
                self.show_grid = not self.show_grid
                status = "enabled" if self.show_grid else "disabled"
                print(f"Grid overlay {status}")

            elif key == 82 or key == 0:  # Up arrow
                self.alpha = min(1.0, self.alpha + 0.05)
                print(f"Alpha: {int(self.alpha * 100)}%")

            elif key == 84 or key == 1:  # Down arrow
                self.alpha = max(0.0, self.alpha - 0.05)
                print(f"Alpha: {int(self.alpha * 100)}%")

            elif key == ord('s'):
                # Save snapshot
                self._save_snapshot()

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Mode C - Enhanced Alpha Blending Tool for visual alignment verification"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing image sequence"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to CSD config file (default: config.json)"
    )

    args = parser.parse_args()

    try:
        tool = AlphaBlendingTool(args.input_dir, args.config)
        tool.run()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
