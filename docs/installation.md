# Camera Shift Detector - Installation Guide

**Version:** 0.1.0
**Last Updated:** 2025-10-26

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Dependency Installation](#dependency-installation)
4. [Quick Verification Test](#quick-verification-test)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Python Version

- **Required:** Python 3.11 or higher
- **Recommended:** Python 3.11 or 3.12

### Operating Systems

- **Linux:** Fully supported (production deployment target)
- **macOS:** Supported for development
- **Windows:** Supported for development

### Hardware Requirements

- **CPU:** Any modern CPU (CPU-only processing, no GPU required)
- **RAM:** Minimum 500 MB for detector process
- **Disk Space:** ~10 MB for codebase + dependencies (~200 MB total with OpenCV)

### System Dependencies

- **None required** - OpenCV bundles all necessary binaries
- **GUI support** for ROI selection tool:
  - Linux: X11 or Wayland
  - macOS: Native window support
  - Windows: Native window support

---

## Installation Methods

### Method 1: Development Installation (Recommended)

Install the package in editable mode for development and testing.

```bash
# Navigate to project directory
cd /path/to/cam-shift-detector

# Install in editable mode
pip install -e .
```

**Advantages:**
- Changes to source code immediately reflected
- No reinstallation needed during development
- Easy debugging and testing

**Use When:**
- Active development or testing
- Integration development
- Debugging issues

---

### Method 2: Installation from Source

Install the package from source (non-editable).

```bash
# Navigate to project directory
cd /path/to/cam-shift-detector

# Install from source
pip install .
```

**Advantages:**
- Standard installation method
- Suitable for production deployment
- Package installed in site-packages

**Use When:**
- Production deployment
- Final release installation
- No development work needed

---

### Method 3: Installation from Built Distribution

Build and install from wheel file.

```bash
# Navigate to project directory
cd /path/to/cam-shift-detector

# Build distribution (requires build package)
pip install build
python -m build

# Install from built wheel
pip install dist/cam_shift_detector-0.1.0-*.whl
```

**Advantages:**
- Distribution-ready package
- Can be deployed without source code
- Standard Python packaging

**Use When:**
- Deploying to multiple systems
- Creating distributable package
- Production environments without source access

---

## Step-by-Step Installation

### Step 1: Verify Python Version

```bash
python --version
```

**Expected Output:**
```
Python 3.11.x
```

**If Python version is too old:**
- Linux: Install Python 3.11+ via package manager
- macOS: Install via Homebrew: `brew install python@3.11`
- Windows: Download from python.org

---

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**Why use virtual environment:**
- Isolates project dependencies
- Prevents conflicts with system Python
- Easy to reproduce environment

---

### Step 3: Upgrade pip (Optional but Recommended)

```bash
pip install --upgrade pip
```

---

### Step 4: Install cam-shift-detector

**Development Installation:**
```bash
pip install -e .
```

**Production Installation:**
```bash
pip install .
```

**Expected Output:**
```
Processing /path/to/cam-shift-detector
Installing build dependencies ... done
Getting requirements to build wheel ... done
Installing backend dependencies ... done
Preparing metadata (pyproject.toml) ... done
Collecting opencv-contrib-python>=4.12.0.88
  ...
Successfully installed cam-shift-detector-0.1.0 opencv-contrib-python-4.12.0.88 opencv-python-4.12.0.88 numpy-...
```

---

## Dependency Installation

### Automatic Dependency Installation

All dependencies are automatically installed when you install the package.

**Core Dependencies:**
- `opencv-contrib-python>=4.12.0.88` - Computer vision library
- `opencv-python>=4.12.0.88` - OpenCV core functionality
- `numpy` (installed as OpenCV dependency)

### Manual Dependency Installation (If Needed)

```bash
# Install dependencies manually
pip install opencv-contrib-python>=4.12.0.88
pip install opencv-python>=4.12.0.88
```

### Verify Dependencies

```bash
# Check installed packages
pip list | grep opencv

# Expected output:
# opencv-contrib-python  4.12.0.88
# opencv-python          4.12.0.88
```

---

## Quick Verification Test

### Test 1: Import Module

```python
# Test basic import
python -c "from src.camera_movement_detector import CameraMovementDetector; print('Import successful!')"
```

**Expected Output:**
```
Import successful!
```

---

### Test 2: Initialize Detector

Create a test script `test_install.py`:

```python
import cv2
import numpy as np
from src.camera_movement_detector import CameraMovementDetector

# Create dummy config for testing
config = {
    "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
    "threshold_pixels": 2.0,
    "history_buffer_size": 100,
    "min_features_required": 50
}

import json
with open('test_config.json', 'w') as f:
    json.dump(config, f)

# Initialize detector
try:
    detector = CameraMovementDetector('test_config.json')
    print("✓ Detector initialized successfully")

    # Create test image (640x480, BGR)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Set baseline
    detector.set_baseline(test_image)
    print("✓ Baseline set successfully")

    # Process frame
    result = detector.process_frame(test_image)
    print(f"✓ Frame processed successfully: {result['status']}")

    print("\n✅ Installation verified - all tests passed!")

except Exception as e:
    print(f"❌ Installation test failed: {e}")

# Cleanup
import os
os.remove('test_config.json')
```

Run the test:
```bash
python test_install.py
```

**Expected Output:**
```
✓ Detector initialized successfully
✓ Baseline set successfully
✓ Frame processed successfully: VALID

✅ Installation verified - all tests passed!
```

---

### Test 3: Verify Package Structure

```bash
# Check installed package location
python -c "import src; print(src.__file__)"

# List package contents
pip show -f cam-shift-detector
```

---

## Troubleshooting

### Issue 1: ImportError: No module named 'cv2'

**Problem:** OpenCV not installed or not found.

**Solution:**
```bash
# Reinstall OpenCV
pip install --force-reinstall opencv-contrib-python opencv-python
```

---

### Issue 2: ModuleNotFoundError: No module named 'src'

**Problem:** Package not installed correctly or not in Python path.

**Solution:**
```bash
# Ensure you're in the project directory
cd /path/to/cam-shift-detector

# Reinstall in editable mode
pip install -e .

# Verify installation
pip list | grep cam-shift-detector
```

---

### Issue 3: Permission denied during installation

**Problem:** Insufficient permissions to install packages.

**Solution:**
```bash
# Option 1: Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install -e .

# Option 2: Install for user only
pip install --user -e .
```

---

### Issue 4: SSL Certificate verification failed

**Problem:** Corporate firewall or network security blocking package downloads.

**Solution:**
```bash
# Option 1: Use trusted host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .

# Option 2: Configure pip to use corporate proxy
# Add to ~/.pip/pip.conf (Linux/macOS) or %APPDATA%\pip\pip.ini (Windows):
# [global]
# trusted-host = pypi.org files.pythonhosted.org
```

---

### Issue 5: numpy version conflicts

**Problem:** Incompatible numpy version with OpenCV.

**Solution:**
```bash
# Let pip resolve dependencies
pip install -e . --force-reinstall

# Or manually install compatible version
pip install "numpy>=1.24.0,<2.0.0"
```

---

### Issue 6: OpenCV import fails on Linux

**Problem:** Missing system libraries for OpenCV.

**Solution:**
```bash
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

# Fedora/RHEL:
sudo dnf install -y glib2 libSM libXext libXrender

# Then reinstall OpenCV:
pip install --force-reinstall opencv-contrib-python opencv-python
```

---

### Issue 7: Virtual environment activation fails

**Problem:** Shell doesn't recognize virtual environment commands.

**Solution:**
```bash
# Ensure you're using the correct activation command:

# bash/zsh (Linux/macOS):
source venv/bin/activate

# fish shell:
source venv/bin/activate.fish

# csh/tcsh:
source venv/bin/activate.csh

# Windows PowerShell:
venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat
```

---

## Post-Installation Setup

### 1. Configuration File

Create a `config.json` file with your ROI settings:

```bash
# Use the ROI selection tool to generate config
python tools/select_roi.py --source image --path sample_images/of_jerusalem/image_001.jpg
```

This will create `config.json` with validated ROI coordinates.

---

### 2. Sample Images

Ensure sample images are available:
```bash
# Sample images should be in:
ls sample_images/of_jerusalem/
ls sample_images/carmit/
ls sample_images/gad/
```

---

### 3. Environment Variables (Optional)

Set environment variables for production:

```bash
# Linux/macOS (.bashrc or .zshrc):
export CAM_SHIFT_CONFIG=/path/to/config.json
export CAM_SHIFT_DATA=/path/to/data

# Windows (PowerShell profile):
$env:CAM_SHIFT_CONFIG="C:\path\to\config.json"
$env:CAM_SHIFT_DATA="C:\path\to\data"
```

---

## Uninstallation

To uninstall the package:

```bash
pip uninstall cam-shift-detector
```

To completely remove the environment:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment directory
rm -rf venv
```

---

## Next Steps

After successful installation:

1. **Review Integration Guide:** See `docs/integration-guide.md`
2. **Run Validation:** Execute Stage 3 validation framework
   ```bash
   python validation/run_stage3_validation.py
   ```
3. **Test Integration:** Use integration code examples
4. **Configure for Production:** Set up config.json for your site

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check error messages carefully
2. Verify Python version (>=3.11)
3. Ensure all dependencies installed correctly
4. Try reinstalling in a fresh virtual environment
5. Contact development team with:
   - Python version (`python --version`)
   - OS and version
   - Complete error message
   - Installation command used

---

**Installation Guide Version:** 0.1.0
**Last Updated:** 2025-10-26
