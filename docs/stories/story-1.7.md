# Story 1.7: Recalibration Script

Status: Done

## Story

As a **DAF system operator**,
I want to **manually recalibrate the camera movement detector baseline using a CLI tool**,
so that **I can quickly reset the baseline features after lighting changes, maintenance, or camera repositioning without restarting the entire detection system**.

**Design Principle**: This is a **thin CLI wrapper** around the existing intrinsic `CameraMovementDetector.recalibrate()` method (Story 1.5). ALL validation and logic remain in the detector class. The script only handles: argument parsing, image loading, calling the method, and reporting results.

## Acceptance Criteria

1. **AC-1.7.1: CLI Interface** - Tool provides command-line interface with `--config` (config file path) and `--image` (source image path) arguments

2. **AC-1.7.2: Config Loading** - Tool loads existing `config.json` to access ROI coordinates and detection parameters

3. **AC-1.7.3: Image Loading** - Tool validates and loads the new reference image for baseline reset

4. **AC-1.7.4: Baseline Reset** - Tool uses CameraMovementDetector.recalibrate() to set new baseline features from the provided image

5. **AC-1.7.5: Feature Validation** - Tool validates ≥50 features detected in ROI before accepting new baseline (same validation as initial setup)

6. **AC-1.7.6: Success/Failure Reporting** - Tool reports recalibration status (success/failure) with clear messages and appropriate exit codes (0=success, 1=failure)

7. **AC-1.7.7: Error Handling** - Tool handles missing config file, invalid image paths, and insufficient features gracefully with clear error messages

8. **AC-1.7.8: History Buffer Handling** - Tool provides `--clear-history` flag to optionally clear the detection history buffer after recalibration

## Tasks / Subtasks

- [x] **Task 1: Create CLI interface and argument parsing** (AC: #1.7.1)
  - [x] 1.1: Define `recalibrate.py` script with argparse for --config, --image, and --clear-history arguments
  - [x] 1.2: Validate --config points to valid config.json file
  - [x] 1.3: Validate --image points to valid image file
  - [x] 1.4: Add --help documentation describing usage and parameters
  - [x] 1.5: Add --clear-history optional flag (default: False)

- [x] **Task 2: Implement config and image loading** (AC: #1.7.2, #1.7.3, #1.7.7)
  - [x] 2.1: Load config.json using existing validation (leverage CameraMovementDetector init)
  - [x] 2.2: Load reference image from file path using cv2.imread()
  - [x] 2.3: Validate image loaded successfully (not None)
  - [x] 2.4: Handle missing config file with clear FileNotFoundError
  - [x] 2.5: Handle missing/corrupted image file with clear error messages

- [x] **Task 3: Call intrinsic recalibration method** (AC: #1.7.4, #1.7.5)
  - [x] 3.1: Initialize CameraMovementDetector with existing config
  - [x] 3.2: Call detector.recalibrate(new_image) - ALL logic handled by method
  - [x] 3.3: Capture boolean return value (True=success, False=failure)
  - [x] 3.4: NO validation logic in script - detector.recalibrate() handles everything

- [x] **Task 4: Implement history buffer clearing** (AC: #1.7.8)
  - [x] 4.1: Check if --clear-history flag is set
  - [x] 4.2: Access detector.result_manager.history buffer
  - [x] 4.3: Call .clear() on the deque to empty history
  - [x] 4.4: Confirm history cleared with message to operator

- [x] **Task 5: Implement success/failure reporting** (AC: #1.7.6)
  - [x] 5.1: On success: Print "Recalibration successful" with feature count
  - [x] 5.2: On success: Exit with code 0
  - [x] 5.3: On failure: Print "Recalibration failed" with reason (insufficient features/invalid image)
  - [x] 5.4: On failure: Exit with code 1
  - [x] 5.5: Include timestamp in success/failure messages

- [x] **Task 6: Error handling and edge cases** (AC: #1.7.7)
  - [x] 6.1: Handle missing config.json with clear error message
  - [x] 6.2: Handle invalid config.json (schema validation errors)
  - [x] 6.3: Handle missing/corrupted image file
  - [x] 6.4: Handle permission errors (file access denied)
  - [x] 6.5: Add try/except blocks for unexpected errors

- [x] **Task 7: Testing and documentation** (AC: All)
  - [x] 7.1: Create unit tests for CLI argument parsing
  - [x] 7.2: Create integration test with valid recalibration scenario
  - [x] 7.3: Create test for recalibration failure (insufficient features)
  - [x] 7.4: Test error cases (missing config, missing image, permission errors)
  - [x] 7.5: Verify exit codes (0 for success, 1 for failure)
  - [x] 7.6: Add usage instructions to tool --help and README

## Dev Notes

### Architecture & Design Patterns

**Module Responsibility** (Tech-Spec Section: Services and Modules):
- **THIN CLI WRAPPER** - no business logic
- Delegates all validation/recalibration to `detector.recalibrate()` (intrinsic method from Story 1.5)
- Script responsibilities: CLI arg parsing, image I/O, method invocation, result reporting
- Uses existing config.json (no config generation)
- Operator-driven workflow for maintenance scenarios

**Integration with Existing Components**:
- **Story 1.1 (StaticRegionManager)**: Uses existing config.json for ROI coordinates
- **Story 1.2 (FeatureExtractor)**: Validation logic via CameraMovementDetector.recalibrate()
- **Story 1.4 (ResultManager)**: Optional history buffer clearing via --clear-history flag
- **Story 1.5 (CameraMovementDetector)**: Primary integration - uses recalibrate() method

**Workflow Integration** (Tech-Spec Section: Workflows and Sequencing):
- **Manual Recalibration Workflow**:
  1. Trigger: Lighting change, maintenance, operator decision
  2. Operator runs: `python tools/recalibrate.py --config config.json --image current_frame.jpg`
  3. Tool validates ≥50 features detected
  4. On success: Baseline replaced, detector ready for use
  5. On failure: Clear error message, operator selects different image

### Implementation Guidance

**Tool Structure**:
```python
#!/usr/bin/env python3
"""Recalibration Script for Camera Movement Detection

Manual baseline reset helper for DAF water quality monitoring systems.

Usage:
    python tools/recalibrate.py --config config.json --image current_frame.jpg
    python tools/recalibrate.py --config config.json --image current_frame.jpg --clear-history
"""

import argparse
import cv2
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_movement_detector import CameraMovementDetector


def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Manual recalibration tool for camera movement detection"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json file"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to new reference image for baseline reset"
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear detection history buffer after successful recalibration"
    )
    return parser.parse_args()


def load_image(image_path: str):
    """Load and validate reference image."""
    # Check file exists
    # Load with cv2.imread()
    # Validate not None
    # Return image


def main():
    """Main execution flow - THIN WRAPPER around detector.recalibrate()."""
    args = parse_arguments()

    # Simple validation: config file exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Load reference image (simple I/O)
    try:
        image = load_image(args.image)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Initialize detector (delegates to CameraMovementDetector)
    try:
        detector = CameraMovementDetector(args.config)
    except Exception as e:
        print(f"Error: Failed to initialize detector: {e}")
        sys.exit(1)

    # Call intrinsic method - ALL validation happens here
    print(f"Recalibrating baseline from: {args.image}")
    print(f"Using config: {args.config}")

    success = detector.recalibrate(image)  # ← ALL LOGIC IN THIS METHOD

    # Report result (simple output formatting)
    timestamp = datetime.utcnow().isoformat() + "Z"

    if success:
        print(f"✓ Recalibration successful at {timestamp}")

        # Optional: clear history (simple flag check)
        if args.clear_history:
            detector.result_manager.history.clear()
            print("✓ Detection history buffer cleared")

        sys.exit(0)
    else:
        print(f"✗ Recalibration failed at {timestamp}")
        print("Reason: Insufficient features detected in ROI (<50 required)")
        print("Action: Try a different image with more texture/features in the ROI")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**CLI Usage Examples**:
```bash
# Basic recalibration
python tools/recalibrate.py --config config.json --image current_frame.jpg

# Recalibration with history clearing
python tools/recalibrate.py --config config.json --image current_frame.jpg --clear-history

# Expected success output:
# Recalibrating baseline from: current_frame.jpg
# Using config: config.json
# ✓ Recalibration successful at 2025-10-23T15:30:42.123Z

# Expected failure output:
# Recalibrating baseline from: current_frame.jpg
# Using config: config.json
# ✗ Recalibration failed at 2025-10-23T15:30:42.123Z
# Reason: Insufficient features detected in ROI (<50 required)
# Action: Try a different image with more texture/features in the ROI
```

**Error Messages**:
- Missing config: `"Error: Config file not found: {path}"`
- Missing image: `"Error: Image file not found: {path}"`
- Invalid image: `"Error: Failed to load image from {path}. File may be corrupted."`
- Recalibration failure: `"Recalibration failed: Insufficient features detected in ROI (<50 required)"`

**Integration with CameraMovementDetector** (ALL LOGIC HERE):
```python
# CameraMovementDetector.recalibrate() method (from Story 1.5):
# ← THIS METHOD CONTAINS ALL VALIDATION AND LOGIC
def recalibrate(self, image_array: np.ndarray) -> bool:
    """Manually reset baseline features.

    This intrinsic method:
    - Validates image format
    - Generates mask using existing config
    - Extracts features with FeatureExtractor
    - Validates ≥50 features detected
    - Replaces baseline if validation passes

    Returns:
        True if recalibration successful, False if insufficient features
    """
    try:
        self.set_baseline(image_array)  # ← Handles ALL validation internally
        return True
    except ValueError:
        # Insufficient features or invalid format
        return False
```

**Script Design**: The recalibrate.py script is intentionally shallow:
- **~50 lines total** (mostly boilerplate)
- **Zero validation logic** (detector handles it)
- **Zero feature extraction** (detector handles it)
- Just: parse args → load image → call method → print result

### Project Structure Notes

**File Location**: `tools/recalibrate.py` (sibling to `tools/select_roi.py`)

**Directory Structure**:
```
cam-shift-detector/
├── src/
│   └── camera_movement_detector.py   ← Story 1.5 (provides recalibrate() method)
├── tools/
│   ├── select_roi.py                 ← Story 1.6 (generates config.json)
│   └── recalibrate.py                ← This story (uses config.json to reset baseline)
├── config.json                        ← Generated by select_roi.py, consumed by recalibrate.py
└── sample_images/                     ← Test images for recalibration
```

**Dependencies**:
- Story 1.5 (CameraMovementDetector) - provides recalibrate() method
- Python stdlib: argparse, sys, datetime, pathlib
- OpenCV (cv2) - for image loading
- NumPy - for image array handling

**Workflow Sequence**:
1. **Initial Setup**: `tools/select_roi.py` generates `config.json`
2. **Baseline Capture**: `CameraMovementDetector.set_baseline()`
3. **Runtime Detection**: `CameraMovementDetector.process_frame()`
4. **Recalibration** (this story): `tools/recalibrate.py` resets baseline when needed

### Testing Standards

**Test Framework**: pytest

**Test Categories**:
1. **CLI Interface**: Argument parsing, validation, help text
2. **Config Loading**: Valid config, missing config, invalid schema
3. **Image Loading**: Valid images, missing files, corrupted files
4. **Recalibration Success**: Valid image with ≥50 features
5. **Recalibration Failure**: Image with <50 features
6. **History Clearing**: --clear-history flag functionality
7. **Exit Codes**: 0 for success, 1 for failure
8. **Error Handling**: Missing files, permission errors, invalid inputs

**Test Ideas**:
```python
def test_recalibrate_success_with_valid_image():
    """AC-1.7.4, AC-1.7.5: Test successful recalibration"""
    # Run tool with valid config and feature-rich image
    # Verify exit code 0
    # Verify success message printed

def test_recalibrate_failure_insufficient_features():
    """AC-1.7.5, AC-1.7.6: Test recalibration failure"""
    # Run tool with blank/feature-poor image
    # Verify exit code 1
    # Verify failure message with reason

def test_clear_history_flag():
    """AC-1.7.8: Test history buffer clearing"""
    # Initialize detector with history
    # Run recalibrate with --clear-history
    # Verify history buffer is empty after recalibration

def test_missing_config_file():
    """AC-1.7.7: Test error handling for missing config"""
    # Run tool with nonexistent config path
    # Verify exit code 1
    # Verify clear error message

def test_exit_codes():
    """AC-1.7.6: Test exit codes match spec"""
    # Success scenario → exit code 0
    # Failure scenario → exit code 1
```

### References

- [Source: tech-spec-epic-MVP-001.md#Services and Modules] - Recalibration Script module specification
- [Source: tech-spec-epic-MVP-001.md#Workflows and Sequencing] - Manual Recalibration Workflow
- [Source: story-1.5.md] - CameraMovementDetector.recalibrate() method (dependency)
- [Source: story-1.6.md] - ROI Selection Tool (generates config.json consumed by this tool)

## Dev Agent Record

### Context Reference

- [Story Context XML](story-context-1.1.7.xml) - Generated 2025-10-23
- Contains: Tech spec workflows, CameraMovementDetector.recalibrate() method reference, thin wrapper design constraints, testing standards
- **Design Principle**: THIN CLI WRAPPER - ALL logic delegated to intrinsic method

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

**Implementation Plan** (2025-10-23):
- Created thin CLI wrapper following Story Context constraints (ZERO validation logic)
- ALL recalibration logic delegated to CameraMovementDetector.recalibrate() intrinsic method
- Script responsibilities limited to: CLI arg parsing, image I/O, method invocation, result reporting
- Achieved ~122 lines total (within target scope for thin wrapper)
- Reused patterns from select_roi.py for consistency (argparse, load_image, error handling)

**Test Coverage Achievement**:
- Created 21 comprehensive unit tests covering all 8 acceptance criteria
- Test categories: CLI interface (5 tests), image loading (3 tests), success/failure paths (4 tests), history clearing (3 tests), error handling (4 tests), delegation verification (2 tests)
- 100% test pass rate, full regression suite passed (195 tests)
- Tests verify thin wrapper pattern with NO validation logic in script

### Completion Notes List

**Completed:** 2025-10-23
**Definition of Done:** All acceptance criteria met, code reviewed, tests passing (195 tests, 0 failures), story approved by user

**Implementation Summary** (2025-10-23):
Successfully implemented Story 1.7 as a thin CLI wrapper around CameraMovementDetector.recalibrate() intrinsic method. The script is intentionally minimal (~122 lines) with ALL validation and recalibration logic delegated to the detector class, adhering to the strict design principle of ZERO business logic in the script.

**Key Design Adherence**:
- ✅ THIN WRAPPER: Script contains NO validation logic (all delegated to detector.recalibrate())
- ✅ CLI interface with --config, --image, and --clear-history arguments
- ✅ Proper error handling for missing files and corrupted images
- ✅ Success/failure reporting with timestamps and clear exit codes
- ✅ Optional history buffer clearing functionality

**Test Results**:
- All 21 unit tests passing (100% coverage of acceptance criteria)
- Full regression suite: 195 passed, 5 skipped
- No regressions introduced
- All error paths tested and verified

**Integration Points**:
- Uses CameraMovementDetector.recalibrate() (Story 1.5) for ALL logic
- Consumes config.json generated by select_roi.py (Story 1.6)
- Accesses result_manager.history for optional clearing (Story 1.4)

**Operator Usage**:
```bash
# Basic recalibration
python tools/recalibrate.py --config config.json --image current_frame.jpg

# With history clearing
python tools/recalibrate.py --config config.json --image current_frame.jpg --clear-history
```

### File List

**New Files**:
- `tools/recalibrate.py` - Recalibration CLI script (122 lines, thin wrapper)
- `tests/test_recalibrate.py` - Comprehensive test suite (21 tests, 589 lines)

## Change Log

**2025-10-23** - Story 1.7 Implementation Complete
- Created `tools/recalibrate.py` as thin CLI wrapper (~122 lines)
- Implemented argparse interface with --config, --image, and --clear-history arguments
- All validation and recalibration logic delegated to CameraMovementDetector.recalibrate() intrinsic method
- Added comprehensive error handling for missing/corrupted files
- Implemented success/failure reporting with ISO timestamps and exit codes
- Created test suite with 21 tests covering all acceptance criteria (100% pass rate)
- Full regression suite passing (195 tests, 0 failures)
- Script follows thin wrapper pattern with ZERO business logic (design principle met)
