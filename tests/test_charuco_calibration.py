#!/usr/bin/env python3
"""
End-to-end test for ChArUco calibration pipeline.
Tests both interactive calibration (with synthetic images) and loading from saved images.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import cv2 as cv
import pytest

# Add parent directory to path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.aruco.camshift_annotator import (
    make_charuco_board,
    run_charuco_calibration_from_images,
    write_yaml_camera,
    read_yaml_camera,
)


def generate_synthetic_charuco_images(output_dir, num_images=10):
    """Generate synthetic images of a ChArUco board at different poses.

    Args:
        output_dir: Directory to save synthetic images
        num_images: Number of images to generate

    Returns:
        Tuple of (detector, board) used to generate the images
    """
    # Create ChArUco board (matching default parameters in script)
    squares_x, squares_y = 7, 5
    square_m, marker_m = 0.035, 0.026
    dict_name = "DICT_4X4_50"

    dictionary, board, detector = make_charuco_board(
        squares_x, squares_y, square_m, marker_m, dict_name
    )

    # Synthetic camera parameters
    img_w, img_h = 1920, 1080
    fx = fy = 1000.0  # Focal length
    cx, cy = img_w / 2.0, img_h / 2.0  # Principal point
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)  # No distortion for synthetic images

    # Create images with simple transformations
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        # Generate a fresh board image each time with different size and position
        # to simulate different viewing angles
        base_size = 1600 + i * 20  # Vary size
        board_img = board.generateImage((base_size, int(base_size * 0.7)), marginSize=40, borderBits=1)

        # Create output image (white background)
        output_img = np.ones((img_h, img_w), dtype=np.uint8) * 255

        # Compute scaling to fit board in image with some variation
        scale = 0.6 + (i % 5) * 0.08
        new_w = int(board_img.shape[1] * scale)
        new_h = int(board_img.shape[0] * scale)

        # Ensure board fits in image
        if new_w > img_w * 0.9:
            new_w = int(img_w * 0.9)
            new_h = int(new_w * board_img.shape[0] / board_img.shape[1])
        if new_h > img_h * 0.9:
            new_h = int(img_h * 0.9)
            new_w = int(new_h * board_img.shape[1] / board_img.shape[0])

        # Resize board
        board_resized = cv.resize(board_img, (new_w, new_h))

        # Position board with some offset variation
        offset_x = (img_w - new_w) // 2 + ((i * 37) % 100) - 50
        offset_y = (img_h - new_h) // 2 + ((i * 51) % 80) - 40

        # Ensure board is within image bounds
        offset_x = max(10, min(offset_x, img_w - new_w - 10))
        offset_y = max(10, min(offset_y, img_h - new_h - 10))

        # Place board in output image
        output_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = board_resized

        # Add minimal noise
        noise = np.random.normal(0, 2, output_img.shape).astype(np.int16)
        output_img = np.clip(output_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Apply slight blur to simulate camera capture
        output_img = cv.GaussianBlur(output_img, (3, 3), 0.5)

        # Save image
        img_path = os.path.join(output_dir, f"calib_{i+1:03d}.png")
        cv.imwrite(img_path, output_img)

    return detector, board


def test_charuco_calibration_from_synthetic_images():
    """Test calibration pipeline using synthetic ChArUco images."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate synthetic calibration images
        images_dir = os.path.join(tmpdir, "calib_images")
        detector, board = generate_synthetic_charuco_images(images_dir, num_images=15)

        # Run calibration from images
        calib_yaml = os.path.join(tmpdir, "test_camera.yaml")
        result = run_charuco_calibration_from_images(
            images_dir, detector, board, calib_yaml
        )

        # Check that calibration succeeded
        assert result is not None, "Calibration failed"
        K, dist, imsize = result

        # Verify output file exists
        assert os.path.exists(calib_yaml), "Calibration YAML not created"

        # Verify calibration parameters are reasonable
        assert K.shape == (3, 3), "Camera matrix has wrong shape"
        assert dist is not None and len(dist.flatten()) >= 4, "Distortion coefficients missing"
        assert imsize == (1920, 1080), "Image size mismatch"

        # Verify we can read back the calibration
        loaded = read_yaml_camera(calib_yaml)
        assert loaded is not None, "Failed to read calibration YAML"
        K_loaded, dist_loaded, size_loaded = loaded

        # Check that loaded values match
        np.testing.assert_array_almost_equal(K, K_loaded, decimal=5)
        np.testing.assert_array_almost_equal(dist.flatten(), dist_loaded.flatten(), decimal=5)
        assert imsize == size_loaded

        print(f"✓ Calibration successful")
        print(f"  Camera matrix:\n{K}")
        print(f"  Distortion: {dist.flatten()}")
        print(f"  Image size: {imsize}")


def test_calibration_insufficient_images():
    """Test that calibration fails gracefully with insufficient images."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate only 2 images (need at least 5)
        images_dir = os.path.join(tmpdir, "calib_images")
        detector, board = generate_synthetic_charuco_images(images_dir, num_images=2)

        # Run calibration from images
        calib_yaml = os.path.join(tmpdir, "test_camera.yaml")
        result = run_charuco_calibration_from_images(
            images_dir, detector, board, calib_yaml
        )

        # Should fail gracefully
        assert result is None, "Calibration should fail with insufficient images"
        assert not os.path.exists(calib_yaml), "YAML should not be created on failure"

        print("✓ Calibration correctly failed with insufficient images")


def test_calibration_empty_directory():
    """Test that calibration handles empty directory gracefully."""

    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = os.path.join(tmpdir, "empty")
        os.makedirs(images_dir)

        # Create board
        dictionary, board, detector = make_charuco_board(7, 5, 0.035, 0.026, "DICT_4X4_50")

        # Run calibration from empty directory
        calib_yaml = os.path.join(tmpdir, "test_camera.yaml")
        result = run_charuco_calibration_from_images(
            images_dir, detector, board, calib_yaml
        )

        # Should fail gracefully
        assert result is None, "Calibration should fail with no images"

        print("✓ Calibration correctly handled empty directory")


def test_yaml_read_write():
    """Test YAML read/write functions for camera parameters."""

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "camera.yaml")

        # Create test camera parameters
        K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float64)
        dist = np.array([0.1, -0.2, 0.001, 0.002, 0.05], dtype=np.float64)
        imsize = (1920, 1080)

        # Write
        write_yaml_camera(yaml_path, K, dist, imsize)
        assert os.path.exists(yaml_path), "YAML file not created"

        # Read back
        loaded = read_yaml_camera(yaml_path)
        assert loaded is not None, "Failed to read YAML"

        K_loaded, dist_loaded, size_loaded = loaded
        np.testing.assert_array_almost_equal(K, K_loaded, decimal=5)
        np.testing.assert_array_almost_equal(dist, dist_loaded.flatten(), decimal=5)
        assert imsize == size_loaded

        print("✓ YAML read/write working correctly")


if __name__ == "__main__":
    print("Running ChArUco calibration tests...\n")

    # Run tests
    try:
        test_yaml_read_write()
        print()

        test_charuco_calibration_from_synthetic_images()
        print()

        test_calibration_insufficient_images()
        print()

        test_calibration_empty_directory()
        print()

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
