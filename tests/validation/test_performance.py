"""
Unit tests for PerformanceProfiler (validation/performance_profiler.py)

Tests cover acceptance criteria for Story 2:
- AC3: Performance Profiler Implemented
- AC4: Test coverage requirement
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from validation.performance_profiler import (
    PerformanceProfiler,
    PerformanceMetrics
)


class TestPerformanceMetricsDataclass:
    """Test PerformanceMetrics dataclass structure."""

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics object creation with all fields."""
        metrics = PerformanceMetrics(
            fps=2.0,
            fps_min=1.5,
            fps_max=2.5,
            memory_peak_mb=450.0,
            memory_mean_mb=400.0,
            memory_stddev_mb=25.0,
            cpu_percent_mean=45.0,
            cpu_percent_max=60.0,
            detection_time_mean_ms=500.0,
            detection_time_stddev_ms=50.0,
            total_images=50,
            meets_fps_target=True,
            meets_memory_target=True
        )

        assert metrics.fps == 2.0
        assert metrics.memory_peak_mb == 450.0
        assert metrics.meets_fps_target is True
        assert metrics.meets_memory_target is True


class TestPerformanceProfilerInit:
    """Test PerformanceProfiler initialization."""

    def test_profiler_initialization(self):
        """Test profiler initializes with empty measurements."""
        profiler = PerformanceProfiler()

        assert profiler.detection_times == []
        assert profiler.memory_samples == []
        assert profiler.cpu_samples == []
        assert profiler.process is not None


class TestFPSMeasurement:
    """Test measure_fps method (AC3: FPS measurement)."""

    def test_fps_calculation_accuracy(self):
        """Test FPS measurement accuracy by comparing against time.time() baseline."""
        profiler = PerformanceProfiler()

        # Simulate detection times: 0.5s per detection = 2.0 FPS
        detection_times = [0.5, 0.5, 0.5, 0.5, 0.5]
        profiler.detection_times = detection_times

        fps_metrics = profiler.measure_fps()

        # Expected FPS = 1 / 0.5 = 2.0 fps
        assert fps_metrics['fps'] == pytest.approx(2.0, rel=0.01)
        assert fps_metrics['fps_min'] == pytest.approx(2.0, rel=0.01)
        assert fps_metrics['fps_max'] == pytest.approx(2.0, rel=0.01)

    def test_fps_variable_detection_times(self):
        """Test FPS calculation with variable detection times."""
        profiler = PerformanceProfiler()

        # Variable detection times: 0.5s, 1.0s, 0.25s
        profiler.detection_times = [0.5, 1.0, 0.25]

        fps_metrics = profiler.measure_fps()

        # FPS values: 2.0, 1.0, 4.0 → mean=2.33, min=1.0, max=4.0
        assert fps_metrics['fps'] == pytest.approx(2.333, rel=0.01)
        assert fps_metrics['fps_min'] == pytest.approx(1.0, rel=0.01)
        assert fps_metrics['fps_max'] == pytest.approx(4.0, rel=0.01)

    def test_fps_empty_measurements(self):
        """Test FPS measurement with no detection times."""
        profiler = PerformanceProfiler()

        fps_metrics = profiler.measure_fps()

        assert fps_metrics['fps'] == 0.0
        assert fps_metrics['fps_min'] == 0.0
        assert fps_metrics['fps_max'] == 0.0

    def test_fps_target_verification(self):
        """Test FPS meets target (≥1/60 Hz = 0.0167 FPS)."""
        profiler = PerformanceProfiler()

        # Detection time: 50 seconds per frame = 0.02 FPS (exceeds target)
        profiler.detection_times = [50.0] * 10

        fps_metrics = profiler.measure_fps()

        assert fps_metrics['fps'] == pytest.approx(0.02, rel=0.01)
        assert fps_metrics['fps'] >= profiler.FPS_TARGET  # Meets target


class TestMemoryProfiling:
    """Test measure_memory method (AC3: Memory usage profiling)."""

    def test_memory_profiling_peak_detection(self):
        """Test memory profiling captures peak memory correctly."""
        profiler = PerformanceProfiler()

        # Simulate memory usage: peak at 450 MB
        profiler.memory_samples = [400.0, 420.0, 450.0, 430.0, 410.0]

        memory_metrics = profiler.measure_memory()

        assert memory_metrics['memory_peak_mb'] == 450.0
        assert memory_metrics['memory_mean_mb'] == pytest.approx(422.0, rel=0.01)
        assert memory_metrics['memory_stddev_mb'] > 0.0

    def test_memory_target_verification(self):
        """Test memory meets target (≤500 MB)."""
        profiler = PerformanceProfiler()

        # All measurements below 500 MB
        profiler.memory_samples = [400.0, 450.0, 480.0]

        memory_metrics = profiler.measure_memory()

        assert memory_metrics['memory_peak_mb'] <= profiler.MEMORY_TARGET_MB

    def test_memory_empty_measurements(self):
        """Test memory measurement with no samples."""
        profiler = PerformanceProfiler()

        memory_metrics = profiler.measure_memory()

        assert memory_metrics['memory_peak_mb'] == 0.0
        assert memory_metrics['memory_mean_mb'] == 0.0
        assert memory_metrics['memory_stddev_mb'] == 0.0


class TestCPUTracking:
    """Test measure_cpu method (AC3: CPU usage tracking)."""

    def test_cpu_utilization_calculation(self):
        """Test CPU usage tracking with percentage utilization."""
        profiler = PerformanceProfiler()

        # Simulate CPU usage samples
        profiler.cpu_samples = [45.0, 50.0, 55.0, 48.0, 52.0]

        cpu_metrics = profiler.measure_cpu()

        assert cpu_metrics['cpu_percent_mean'] == pytest.approx(50.0, rel=0.01)
        assert cpu_metrics['cpu_percent_max'] == 55.0

    def test_cpu_empty_measurements(self):
        """Test CPU measurement with no samples."""
        profiler = PerformanceProfiler()

        cpu_metrics = profiler.measure_cpu()

        assert cpu_metrics['cpu_percent_mean'] == 0.0
        assert cpu_metrics['cpu_percent_max'] == 0.0


class TestProfileDetection:
    """Test profile_detection method (AC3: Profiled execution wrapper)."""

    @patch('validation.performance_profiler.PerformanceProfiler._measure_memory')
    @patch('validation.performance_profiler.PerformanceProfiler._measure_cpu')
    def test_profile_detection_wraps_function(self, mock_cpu, mock_memory):
        """Test profile_detection wraps detection function correctly."""
        profiler = PerformanceProfiler()

        # Setup mocks
        mock_memory.return_value = 400.0
        mock_cpu.return_value = 50.0

        # Mock detection function
        mock_detector = Mock(return_value={'status': 'VALID', 'confidence': 0.95})

        # Profile detection
        result, detection_time = profiler.profile_detection(
            mock_detector,
            np.zeros((480, 640, 3), dtype=np.uint8),
            frame_id='test_001'
        )

        # Verify detection function was called
        mock_detector.assert_called_once()

        # Verify result returned
        assert result['status'] == 'VALID'

        # Verify timing recorded
        assert detection_time > 0.0
        assert len(profiler.detection_times) == 1

    @patch('validation.performance_profiler.PerformanceProfiler._measure_memory')
    @patch('validation.performance_profiler.PerformanceProfiler._measure_cpu')
    def test_profile_detection_records_measurements(self, mock_cpu, mock_memory):
        """Test profile_detection records memory and CPU samples."""
        profiler = PerformanceProfiler()

        # Setup mocks
        mock_memory.return_value = 450.0
        mock_cpu.return_value = 55.0

        # Mock detection function
        mock_detector = Mock(return_value={'status': 'VALID'})

        # Profile detection
        profiler.profile_detection(mock_detector, "arg1", kwarg1="value1")

        # Verify measurements recorded
        assert len(profiler.memory_samples) == 1
        assert len(profiler.cpu_samples) == 1
        assert profiler.memory_samples[0] == 450.0
        assert profiler.cpu_samples[0] == 55.0


class TestGetMetrics:
    """Test get_metrics method (AC3: Aggregate metrics)."""

    def test_get_metrics_comprehensive(self):
        """Test get_metrics returns comprehensive PerformanceMetrics."""
        profiler = PerformanceProfiler()

        # Populate measurements
        profiler.detection_times = [0.5, 0.6, 0.4]  # Variable detection times
        profiler.memory_samples = [400.0, 420.0, 450.0]
        profiler.cpu_samples = [45.0, 50.0, 55.0]

        metrics = profiler.get_metrics()

        # Verify FPS calculated
        assert metrics.fps > 0.0
        assert metrics.fps_min > 0.0
        assert metrics.fps_max > 0.0

        # Verify memory calculated
        assert metrics.memory_peak_mb == 450.0
        assert metrics.memory_mean_mb == pytest.approx(423.33, rel=0.01)

        # Verify CPU calculated
        assert metrics.cpu_percent_mean == pytest.approx(50.0, rel=0.01)
        assert metrics.cpu_percent_max == 55.0

        # Verify detection time statistics
        assert metrics.detection_time_mean_ms == pytest.approx(500.0, rel=0.01)
        assert metrics.detection_time_stddev_ms > 0.0

        # Verify total images
        assert metrics.total_images == 3

    def test_get_metrics_target_validation(self):
        """Test get_metrics validates performance targets."""
        profiler = PerformanceProfiler()

        # Set measurements that meet targets
        profiler.detection_times = [50.0]  # 0.02 FPS (exceeds 1/60 Hz = 0.0167)
        profiler.memory_samples = [450.0]  # Below 500 MB

        metrics = profiler.get_metrics()

        assert metrics.meets_fps_target == True  # FPS ≥ 0.0167
        assert metrics.meets_memory_target == True  # Memory ≤ 500 MB

    def test_get_metrics_target_failure(self):
        """Test get_metrics detects target failures."""
        profiler = PerformanceProfiler()

        # Set measurements that fail targets
        profiler.detection_times = [100.0]  # 0.01 FPS (below 1/60 Hz = 0.0167)
        profiler.memory_samples = [600.0]  # Above 500 MB

        metrics = profiler.get_metrics()

        assert metrics.meets_fps_target == False  # FPS < 0.0167
        assert metrics.meets_memory_target == False  # Memory > 500 MB


class TestReset:
    """Test reset method."""

    def test_reset_clears_measurements(self):
        """Test reset clears all measurement arrays."""
        profiler = PerformanceProfiler()

        # Populate measurements
        profiler.detection_times = [0.5, 0.6, 0.4]
        profiler.memory_samples = [400.0, 420.0]
        profiler.cpu_samples = [45.0, 50.0]

        # Reset
        profiler.reset()

        # Verify all cleared
        assert profiler.detection_times == []
        assert profiler.memory_samples == []
        assert profiler.cpu_samples == []


class TestMeasureMemoryInternal:
    """Test _measure_memory internal method."""

    @patch('validation.performance_profiler.psutil.Process')
    def test_measure_memory_uses_rss(self, mock_process):
        """Test _measure_memory uses RSS (resident set size)."""
        # Mock memory info
        mock_mem_info = Mock()
        mock_mem_info.rss = 400 * 1024 * 1024  # 400 MB in bytes
        mock_process.return_value.memory_info.return_value = mock_mem_info

        profiler = PerformanceProfiler()
        memory_mb = profiler._measure_memory()

        assert memory_mb == pytest.approx(400.0, rel=0.01)

    @patch('validation.performance_profiler.psutil.Process')
    def test_measure_memory_error_handling(self, mock_process):
        """Test _measure_memory handles errors gracefully."""
        mock_process.return_value.memory_info.side_effect = Exception("Failed")

        profiler = PerformanceProfiler()
        memory_mb = profiler._measure_memory()

        assert memory_mb == 0.0  # Returns 0 on error


class TestMeasureCPUInternal:
    """Test _measure_cpu internal method."""

    @patch('validation.performance_profiler.psutil.Process')
    def test_measure_cpu_percentage(self, mock_process):
        """Test _measure_cpu returns CPU percentage."""
        mock_process.return_value.cpu_percent.return_value = 55.0

        profiler = PerformanceProfiler()
        cpu_percent = profiler._measure_cpu()

        assert cpu_percent == 55.0

    @patch('validation.performance_profiler.psutil.Process')
    def test_measure_cpu_error_handling(self, mock_process):
        """Test _measure_cpu handles errors gracefully."""
        mock_process.return_value.cpu_percent.side_effect = Exception("Failed")

        profiler = PerformanceProfiler()
        cpu_percent = profiler._measure_cpu()

        assert cpu_percent == 0.0  # Returns 0 on error
