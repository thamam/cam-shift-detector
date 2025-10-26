"""
Test configuration and fixtures for validation tests.

This module provides pytest configuration and fixtures for the validation test suite.
It handles mocking of external dependencies like OpenCV (cv2) to allow tests to run
without requiring the full production environment.
"""

import sys
from unittest.mock import MagicMock


# Mock cv2 (OpenCV) before any validation modules are imported
# This allows tests to run without requiring opencv-python installation
sys.modules['cv2'] = MagicMock()
