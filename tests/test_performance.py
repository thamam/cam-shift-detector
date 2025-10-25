"""Performance Benchmark Tests for Camera Movement Detection Module

Tests verify that the system meets performance requirements specified in
the Tech Spec (AC-1.8.7):
- process_frame() < 500ms (target from Tech Spec)
- set_baseline() < 2s (target from Tech Spec)
- get_history() < 10ms (target from Tech Spec)

Uses pytest-benchmark for accurate performance measurement.
"""

import cv2
import json
import numpy as np
import pytest
from pathlib import Path

from src.camera_movement_detector import CameraMovementDetector


@pytest.fixture
def detector_with_config(tmp_path):
    """Create detector with standard config for benchmarking"""
    config_path = tmp_path / "config.json"
    config = {
        "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
        "threshold_pixels": 2.0,
        "history_buffer_size": 100,
        "min_features_required": 50
    }
    config_path.write_text(json.dumps(config))
    return CameraMovementDetector(str(config_path))


@pytest.fixture
def standard_test_image():
    """Create standard test image with features for benchmarking"""
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # Add structured features
    cv2.rectangle(image, (150, 100), (250, 200), (255, 255, 255), -1)
    cv2.rectangle(image, (350, 150), (450, 250), (0, 0, 0), -1)
    cv2.rectangle(image, (200, 300), (300, 400), (128, 128, 128), -1)
    return image


@pytest.fixture
def real_test_image():
    """Load real sample image for realistic benchmarking"""
    sample_dir = Path('sample_images/of_jerusalem')
    if not sample_dir.exists():
        pytest.skip("Sample images not available")

    images = sorted(sample_dir.glob('*.jpg'))
    if len(images) == 0:
        pytest.skip("No sample images found")

    image = cv2.imread(str(images[0]))
    if image is None:
        pytest.skip("Failed to load sample image")

    return image


class TestProcessFramePerformance:
    """Benchmark process_frame() method (AC-1.8.7: target <500ms)"""

    def test_process_frame_performance_synthetic_image(self, benchmark, detector_with_config, standard_test_image):
        """Benchmark process_frame() with synthetic image"""
        # Setup: Set baseline
        detector_with_config.set_baseline(standard_test_image)

        # Benchmark: process_frame() should execute in <500ms
        result = benchmark(detector_with_config.process_frame, standard_test_image.copy())

        # Verify: Result is valid while benchmarking
        assert 'status' in result
        assert result['status'] in ['VALID', 'INVALID']

        # Performance assertion: <500ms = 0.5 seconds
        # pytest-benchmark will show actual timing
        # Note: This is a target, not a hard requirement for test pass/fail
        assert benchmark.stats['mean'] < 0.5, f"process_frame took {benchmark.stats['mean']:.3f}s (target: <0.5s)"

    def test_process_frame_performance_real_image(self, benchmark, detector_with_config, real_test_image):
        """Benchmark process_frame() with real DAF image"""
        # Setup: Set baseline
        detector_with_config.set_baseline(real_test_image)

        # Benchmark: process_frame() with real image
        result = benchmark(detector_with_config.process_frame, real_test_image.copy())

        # Verify: Result structure
        assert 'status' in result
        assert 'translation_displacement' in result
        assert 'confidence' in result

        # Performance target: <500ms
        assert benchmark.stats['mean'] < 0.5, f"process_frame took {benchmark.stats['mean']:.3f}s (target: <0.5s)"

    def test_process_frame_worst_case_many_features(self, benchmark, tmp_path):
        """Benchmark worst-case scenario with maximum features"""
        # Setup: Config with large ROI to capture many features
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 0, "y": 0, "width": 640, "height": 480},  # Full image
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))
        detector = CameraMovementDetector(str(config_path))

        # Create image with many features (complex scene)
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Add grid of features across entire image
        for i in range(0, 640, 20):
            for j in range(0, 480, 20):
                cv2.circle(image, (i, j), 3, (255, 255, 255), -1)

        detector.set_baseline(image)

        # Benchmark: Worst case with many features
        result = benchmark(detector.process_frame, image.copy())

        assert 'status' in result

        # Even worst case should meet target
        assert benchmark.stats['mean'] < 0.5, f"Worst-case process_frame took {benchmark.stats['mean']:.3f}s (target: <0.5s)"


class TestSetBaselinePerformance:
    """Benchmark set_baseline() method (AC-1.8.7: target <2s)"""

    def test_set_baseline_performance_synthetic_image(self, benchmark, detector_with_config, standard_test_image):
        """Benchmark set_baseline() with synthetic image"""
        # Benchmark: set_baseline() should execute in <2s
        benchmark(detector_with_config.set_baseline, standard_test_image)

        # Performance assertion: <2s
        assert benchmark.stats['mean'] < 2.0, f"set_baseline took {benchmark.stats['mean']:.3f}s (target: <2s)"

    def test_set_baseline_performance_real_image(self, benchmark, detector_with_config, real_test_image):
        """Benchmark set_baseline() with real DAF image"""
        # Benchmark: set_baseline() with real image
        benchmark(detector_with_config.set_baseline, real_test_image)

        # Performance target: <2s
        assert benchmark.stats['mean'] < 2.0, f"set_baseline took {benchmark.stats['mean']:.3f}s (target: <2s)"

    def test_set_baseline_performance_high_resolution(self, benchmark, tmp_path):
        """Benchmark set_baseline() with high-resolution image (1920x1080)"""
        # Setup: Config for HD resolution
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 200, "y": 100, "width": 800, "height": 600},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))
        detector = CameraMovementDetector(str(config_path))

        # Create HD image with features
        hd_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        cv2.rectangle(hd_image, (300, 200), (500, 400), (255, 255, 255), -1)
        cv2.rectangle(hd_image, (700, 300), (900, 500), (0, 0, 0), -1)
        cv2.rectangle(hd_image, (400, 600), (600, 800), (128, 128, 128), -1)

        # Benchmark: HD image baseline
        benchmark(detector.set_baseline, hd_image)

        # Even HD should meet target
        assert benchmark.stats['mean'] < 2.0, f"HD set_baseline took {benchmark.stats['mean']:.3f}s (target: <2s)"


class TestGetHistoryPerformance:
    """Benchmark get_history() method (AC-1.8.7: target <10ms)"""

    def test_get_history_performance_full_buffer(self, benchmark, detector_with_config, standard_test_image):
        """Benchmark get_history() with full buffer"""
        # Setup: Fill history buffer
        detector_with_config.set_baseline(standard_test_image)
        for i in range(100):  # Fill entire buffer
            detector_with_config.process_frame(standard_test_image.copy(), frame_id=f"frame_{i:04d}")

        # Benchmark: get_history() should execute in <10ms
        result = benchmark(detector_with_config.get_history)

        # Verify: Returned all frames
        assert len(result) == 100

        # Performance assertion: <10ms = 0.01 seconds
        assert benchmark.stats['mean'] < 0.01, f"get_history took {benchmark.stats['mean']:.6f}s (target: <0.01s)"

    def test_get_history_performance_with_limit(self, benchmark, detector_with_config, standard_test_image):
        """Benchmark get_history(limit=N) with full buffer"""
        # Setup: Fill history buffer
        detector_with_config.set_baseline(standard_test_image)
        for i in range(100):
            detector_with_config.process_frame(standard_test_image.copy(), frame_id=f"frame_{i:04d}")

        # Benchmark: get_history() with limit parameter
        result = benchmark(detector_with_config.get_history, limit=10)

        # Verify: Returned limited frames
        assert len(result) == 10

        # Performance target: <10ms
        assert benchmark.stats['mean'] < 0.01, f"get_history(limit=10) took {benchmark.stats['mean']:.6f}s (target: <0.01s)"

    def test_get_history_performance_by_frame_id(self, benchmark, detector_with_config, standard_test_image):
        """Benchmark get_history(frame_id=...) query"""
        # Setup: Fill history buffer
        detector_with_config.set_baseline(standard_test_image)
        for i in range(100):
            detector_with_config.process_frame(standard_test_image.copy(), frame_id=f"frame_{i:04d}")

        # Benchmark: get_history() with frame_id search
        result = benchmark(detector_with_config.get_history, frame_id="frame_0050")

        # Verify: Found specific frame
        assert len(result) == 1
        assert result[0]['frame_id'] == "frame_0050"

        # Performance target: <10ms
        assert benchmark.stats['mean'] < 0.01, f"get_history(frame_id) took {benchmark.stats['mean']:.6f}s (target: <0.01s)"


class TestEndToEndPerformance:
    """Benchmark complete workflows (AC-1.8.7)"""

    def test_complete_workflow_performance(self, benchmark, tmp_path, standard_test_image):
        """Benchmark complete workflow: init → set_baseline → process_frame × 10"""
        def complete_workflow():
            # Setup
            config_path = tmp_path / "workflow_config.json"
            config = {
                "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
                "threshold_pixels": 2.0,
                "history_buffer_size": 100,
                "min_features_required": 50
            }
            config_path.write_text(json.dumps(config))

            # Workflow
            detector = CameraMovementDetector(str(config_path))
            detector.set_baseline(standard_test_image)

            results = []
            for i in range(10):
                result = detector.process_frame(standard_test_image.copy())
                results.append(result)

            return results

        # Benchmark: Complete workflow
        results = benchmark(complete_workflow)

        # Verify: All frames processed
        assert len(results) == 10

        # Complete workflow should be fast: init + baseline + 10 frames
        # Estimate: <2s (baseline) + 10 * 0.5s (frames) = <7s
        assert benchmark.stats['mean'] < 7.0, f"Complete workflow took {benchmark.stats['mean']:.3f}s (target: <7s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
