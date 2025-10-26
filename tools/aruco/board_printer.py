"""
Helper module for generating ChArUco and ArUco grid board calibration patterns.
Generates PNG files at specified physical scales for printing.
"""
import cv2 as cv
import math
import numpy as np
from typing import Tuple
from PIL import Image

INCH: float = 0.0254

def _ppm(dpi: float) -> float:
    """Convert DPI to pixels per meter."""
    return dpi / INCH


def _save_png(path: str, img: np.ndarray, dpi: float = 300.0) -> None:
    """Save image to PNG file with proper DPI metadata."""
    # Convert BGR to RGB for PIL
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    # Save with DPI metadata (pixels per inch)
    pil_img.save(path, dpi=(dpi, dpi))
    print(f"  → Saved with DPI metadata: {dpi} DPI")


def _draw_mm_check_square(canvas: np.ndarray, dpi: float, mm: float = 50.0, pad_px: int = 20) -> None:
    """Draw a reference square of specified size in millimeters."""
    px = int(round((mm / 1000.0) * _ppm(dpi)))
    h, w = canvas.shape[:2]
    x0, y0 = pad_px, h - pad_px - px
    cv.rectangle(canvas, (x0, y0), (x0 + px, y0 + px), (0, 0, 0), 3)
    cv.putText(canvas, f"{int(mm)} mm check",
               (x0 + px + 10, y0 + px - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv.LINE_AA)


def save_charuco_png_scaled(
    path: str,
    squares_x: int,
    squares_y: int,
    square_len_m: float,
    marker_len_m: float,
    dict_name: str = "DICT_4X4_50",
    dpi: float = 300.0,
    margin_mm: float = 10.0,
    border_bits: int = 1,
) -> None:
    """
    Create a ChArUco PNG where printed square_len equals the metric size.

    Args:
        path: Output PNG file path
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_len_m: Square side length in meters
        marker_len_m: Marker side length in meters
        dict_name: ArUco dictionary name
        dpi: Print resolution in dots per inch
        margin_mm: Margin size in millimeters
        border_bits: Border size in bits
    """
    ar = cv.aruco
    dictionary = ar.getPredefinedDictionary(getattr(ar, dict_name))
    board = ar.CharucoBoard((squares_x, squares_y), square_len_m, marker_len_m, dictionary)

    # Compute total board size in meters (width along X, height along Y)
    board_w_m = squares_x * square_len_m
    board_h_m = squares_y * square_len_m

    ppm = _ppm(dpi)
    margin_px = int(round((margin_mm / 1000.0) * ppm))

    # Target image size in pixels so that the drawable area (img - 2*margin) equals board size in pixels
    board_w_px = int(round(board_w_m * ppm))
    board_h_px = int(round(board_h_m * ppm))
    img_w_px = board_w_px + 2 * margin_px
    img_h_px = board_h_px + 2 * margin_px

    # Generate marker image; aruco fills the drawable area when given marginSize
    img = board.generateImage((img_w_px, img_h_px), marginSize=margin_px, borderBits=border_bits)

    # Draw a 50 mm check square
    _draw_mm_check_square(img, dpi, mm=50.0)

    _save_png(path, img, dpi=dpi)
    print(f"[ChArUco] Saved {path}. Board area: {board_w_m:.3f}×{board_h_m:.3f} m. "
          f"Image: {img_w_px}×{img_h_px} px. Margin: {margin_mm} mm. Print at 100%.")


def save_gridboard_png_scaled(
    path: str,
    markers_x: int,
    markers_y: int,
    marker_len_m: float,
    sep_len_m: float,
    dict_name: str = "DICT_4X4_50",
    dpi: float = 300.0,
    margin_mm: float = 10.0,
    border_bits: int = 1,
) -> None:
    """
    Create an ArUco GridBoard PNG at true physical scale.

    Args:
        path: Output PNG file path
        markers_x: Number of markers in X direction
        markers_y: Number of markers in Y direction
        marker_len_m: Marker side length in meters
        sep_len_m: Separation between markers in meters
        dict_name: ArUco dictionary name
        dpi: Print resolution in dots per inch
        margin_mm: Margin size in millimeters
        border_bits: Border size in bits
    """
    ar = cv.aruco
    dictionary = ar.getPredefinedDictionary(getattr(ar, dict_name))
    board = ar.GridBoard((markers_x, markers_y), marker_len_m, sep_len_m, dictionary)

    # Board outer size in meters per OpenCV definition
    board_w_m = markers_x * marker_len_m + (markers_x - 1) * sep_len_m
    board_h_m = markers_y * marker_len_m + (markers_y - 1) * sep_len_m

    ppm = _ppm(dpi)
    margin_px = int(round((margin_mm / 1000.0) * ppm))
    board_w_px = int(round(board_w_m * ppm))
    board_h_px = int(round(board_h_m * ppm))
    img_w_px = board_w_px + 2 * margin_px
    img_h_px = board_h_px + 2 * margin_px

    img = board.generateImage((img_w_px, img_h_px), marginSize=margin_px, borderBits=border_bits)

    _draw_mm_check_square(img, dpi, mm=50.0)
    _save_png(path, img, dpi=dpi)
    print(f"[GridBoard] Saved {path}. Board area: {board_w_m:.3f}×{board_h_m:.3f} m. "
          f"Image: {img_w_px}×{img_h_px} px. Margin: {margin_mm} mm. Print at 100%.")


if __name__ == "__main__":
    
    # Examples:
    # A4-friendly ChArUco at 300 DPI, 10 mm margins  
    # ChArUco (calibration)
    save_charuco_png_scaled(
        path="charuco_7x5_35mm_26mm_A4_landscape.png",
        squares_x=7, squares_y=5,
        square_len_m=0.035, marker_len_m=0.026,
        dpi=300.0, margin_mm=10.0)

    # GridBoard (pose tracking)
    save_gridboard_png_scaled(
        path="grid_5x7_30mm_sep6mm_A4_portrait.png",
        markers_x=5, markers_y=7,
        marker_len_m=0.030, sep_len_m=0.006,
        dpi=300.0, margin_mm=10.0)