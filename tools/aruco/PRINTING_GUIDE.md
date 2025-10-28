# ChArUco Board Printing Guide

## Problem Fixed ✓

**Issue**: PNG files were being saved **without DPI metadata**, causing printers to default to 72 DPI instead of 300 DPI, making boards print ~4x larger than intended.

**Solution**: Updated `board_printer.py` to embed DPI metadata using PIL/Pillow when saving PNG files.

## Available Boards

### Standard Quality (300 DPI)
- `charuco_7x5_35mm_26mm_A4_landscape.png` - ChArUco calibration board
- `grid_5x7_30mm_sep6mm_A4_portrait.png` - GridBoard for pose tracking

### High Quality (600 DPI) - For your printer!
- `charuco_7x5_35mm_26mm_A4_landscape_600dpi.png` - ChArUco calibration board
- `grid_5x7_30mm_sep6mm_A4_portrait_600dpi.png` - GridBoard for pose tracking

## Printing Instructions

### For 600 DPI / High Quality Printers

1. **Open the 600dpi PNG file** in your preferred viewer
2. **Printer Settings**:
   - Quality: **600 DPI** or **High Quality** or **Best**
   - Scale: **100%** (CRITICAL - no auto-fit, no shrink-to-fit)
   - Paper: Photo-quality paper recommended for sharpest markers
   - Color: Black & white is fine
3. **Print**
4. **Verify**: Use a ruler to measure the 50mm check square in the bottom-left
   - It should measure **exactly 50mm**
   - If not, check printer settings or adjust board parameters

### For Standard Printers (300 DPI)

Use the non-600dpi versions with the same instructions but select 300 DPI or "Standard Quality"

## Board Specifications

### ChArUco Board (Landscape A4)
- **Board area**: 245mm × 175mm (fits A4 landscape: 297mm × 210mm)
- **Squares**: 7 × 5
- **Square size**: 35mm
- **Marker size**: 26mm
- **Margins**: 10mm on all sides
- **Dictionary**: DICT_4X4_50

### GridBoard (Portrait A4)
- **Board area**: 174mm × 246mm (fits A4 portrait: 210mm × 297mm)
- **Markers**: 5 × 7
- **Marker size**: 30mm
- **Separation**: 6mm
- **Margins**: 10mm on all sides
- **Dictionary**: DICT_4X4_50

## Verification Checklist

After printing:

- [ ] 50mm check square measures exactly 50mm
- [ ] Board is completely flat (no warping)
- [ ] All markers are sharp and clear
- [ ] No smudging or bleed
- [ ] No glare when photographed

## Generating Custom Boards

### For 600 DPI boards:
```bash
python tools/aruco/generate_hq_boards.py
```

### For 300 DPI boards:
```bash
python tools/aruco/board_printer.py
```

### Custom sizes (Python):
```python
from tools.aruco.board_printer import save_charuco_png_scaled

# Example: Larger board for 600 DPI printer
save_charuco_png_scaled(
    path="custom_board_600dpi.png",
    squares_x=10, squares_y=7,
    square_len_m=0.040,  # 40mm squares
    marker_len_m=0.030,  # 30mm markers
    dpi=600.0,
    margin_mm=10.0
)
```

## Technical Details

### DPI Metadata
All PNG files now include proper DPI metadata:
- 300 DPI files: 299.9994 DPI (floating point precision)
- 600 DPI files: 599.9988 DPI (floating point precision)

### File Sizes
- 300 DPI ChArUco: ~31 KB
- 600 DPI ChArUco: ~98 KB (3.2x larger)
- 300 DPI GridBoard: ~33 KB
- 600 DPI GridBoard: ~102 KB (3.1x larger)

### Image Dimensions
**300 DPI ChArUco**: 3130 × 2303 pixels
**600 DPI ChArUco**: 6259 × 4606 pixels (exactly 2x in each dimension)

**300 DPI GridBoard**: 2291 × 3142 pixels
**600 DPI GridBoard**: 4582 × 6283 pixels (exactly 2x in each dimension)

## Troubleshooting

### Board prints too large
- Check that scale is set to **100%**
- Verify DPI setting matches the file (600 DPI for *_600dpi.png files)
- Try printing from a different application (e.g., GIMP, Adobe Reader)

### Board prints too small
- Check that "Fit to page" is **disabled**
- Verify you're using the correct paper size (A4)

### Markers are blurry
- Use photo-quality paper
- Ensure printer is set to highest quality/600 DPI
- Clean printer heads
- Use the 600 DPI versions for sharper results

### Physical verification fails
If the 50mm square doesn't measure 50mm:
1. Measure the actual size (e.g., 47mm)
2. Calculate scale factor: actual/expected (e.g., 47/50 = 0.94)
3. When using the board, multiply all measurements by this factor
4. OR regenerate with adjusted size: `square_len_m=0.035*0.94`
