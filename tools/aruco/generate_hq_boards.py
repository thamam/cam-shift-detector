#!/usr/bin/env python3
"""
Generate high-quality 600 DPI ChArUco and ArUco boards for high-quality printers.
These boards have twice the resolution of standard 300 DPI boards for sharper marker detection.
"""

from board_printer import save_charuco_png_scaled, save_gridboard_png_scaled

if __name__ == "__main__":
    print("\n=== Generating 600 DPI boards (high quality) ===\n")

    # ChArUco board for calibration - 600 DPI
    save_charuco_png_scaled(
        path="charuco_7x5_35mm_26mm_A4_landscape_600dpi.png",
        squares_x=7, squares_y=5,
        square_len_m=0.035, marker_len_m=0.026,
        dpi=600.0, margin_mm=10.0)

    # GridBoard for pose tracking - 600 DPI
    save_gridboard_png_scaled(
        path="grid_5x7_30mm_sep6mm_A4_portrait_600dpi.png",
        markers_x=5, markers_y=7,
        marker_len_m=0.030, sep_len_m=0.006,
        dpi=600.0, margin_mm=10.0)

    print("\n✓ High-quality boards generated!")
    print("\nPrinting instructions:")
    print("  1. Use 600 DPI or 'High Quality' print settings")
    print("  2. Print at 100% scale (no 'Fit to page')")
    print("  3. Use photo-quality paper for best results")
    print("  4. Verify: Measure the 50mm check square with a ruler")
