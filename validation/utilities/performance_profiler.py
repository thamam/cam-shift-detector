"""
Performance Profiler - System Resource Monitoring for Validation

Profiles detector execution with real-time measurement of FPS, memory usage,
and CPU utilization to validate production-readiness on target hardware.
"""

import time
import psutil
import logging
from dataclasses import dataclass
from typing import Callable, Any, Dict, List
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance measurements for detector execution.

    Attributes:
        fps: Frames per second (average across all detections)
        fps_min: Minimum FPS observed
        fps_max: Maximum FPS observed
        memory_peak_mb: Peak memory usage in megabytes
        memory_mean_mb: Mean memory usage in megabytes
        memory_stddev_mb: Standard deviation of memory usage
        cpu_percent_mean: Mean CPU utilization percentage
        cpu_percent_max: Peak CPU utilization percentage
        detection_time_mean_ms: Mean detection time in milliseconds
        detection_time_stddev_ms: Standard deviation of detection time
        total_images: Total number of images processed
        meets_fps_target: Whether FPS meets target (≥1/60 Hz = 0.0167 FPS)
        meets_memory_target: Whether memory meets target (≤500 MB)
    """
    fps: float
    fps_min: float
    fps_max: float
    memory_peak_mb: float
    memory_mean_mb: float
    memory_stddev_mb: float
    cpu_percent_mean: float
    cpu_percent_max: float
    detection_time_mean_ms: float
    detection_time_stddev_ms: float
    total_images: int
    meets_fps_target: bool
    meets_memory_target: bool


class PerformanceProfiler:
    """Real-time performance profiler for detector validation.

    Measures system resource usage (FPS, memory, CPU) during detector
    execution to validate production-readiness on target hardware.

    Target Specifications:
    - FPS: ≥1/60 Hz (0.0167 FPS) - One frame per minute
    - Memory: ≤500 MB RAM usage
    """

    # Performance targets from tech spec
    FPS_TARGET = 1.0 / 60.0  # 0.0167 FPS (one frame per minute)
    MEMORY_TARGET_MB = 500.0

    def __init__(self):
        """Initialize performance profiler."""
        self.detection_times: List[float] = []  # Detection times in seconds
        self.memory_samples: List[float] = []   # Memory usage in MB
        self.cpu_samples: List[float] = []      # CPU percentage samples

        self.process = psutil.Process()
        logger.info("Performance profiler initialized")

    def profile_detection(self, detection_func: Callable, *args, **kwargs) -> tuple:
        """Profile single detection execution with resource monitoring.

        Wraps detector execution with real-time FPS, memory, and CPU measurement.

        Args:
            detection_func: Detection function to profile (e.g., detector.process_frame)
            *args: Positional arguments to pass to detection function
            **kwargs: Keyword arguments to pass to detection function

        Returns:
            Tuple of (detection_result, detection_time_seconds)
        """
        # Measure memory before detection
        memory_before = self._measure_memory()

        # Measure CPU before detection
        cpu_before = self._measure_cpu()

        # Execute detection with timing
        start_time = time.time()
        result = detection_func(*args, **kwargs)
        detection_time = time.time() - start_time

        # Measure memory after detection
        memory_after = self._measure_memory()

        # Measure CPU after detection
        cpu_after = self._measure_cpu()

        # Record measurements
        self.detection_times.append(detection_time)
        self.memory_samples.append(max(memory_before, memory_after))  # Peak during detection
        self.cpu_samples.append(max(cpu_before, cpu_after))

        return result, detection_time

    def measure_fps(self) -> Dict[str, float]:
        """Calculate frames per second from detection times.

        Returns:
            Dictionary with fps, fps_min, fps_max (frames per second)
        """
        if not self.detection_times:
            return {'fps': 0.0, 'fps_min': 0.0, 'fps_max': 0.0}

        # FPS = 1 / detection_time
        fps_values = [1.0 / dt if dt > 0 else 0.0 for dt in self.detection_times]

        return {
            'fps': np.mean(fps_values),
            'fps_min': np.min(fps_values),
            'fps_max': np.max(fps_values)
        }

    def measure_memory(self) -> Dict[str, float]:
        """Calculate memory usage statistics.

        Returns:
            Dictionary with memory_peak_mb, memory_mean_mb, memory_stddev_mb
        """
        if not self.memory_samples:
            return {'memory_peak_mb': 0.0, 'memory_mean_mb': 0.0, 'memory_stddev_mb': 0.0}

        return {
            'memory_peak_mb': np.max(self.memory_samples),
            'memory_mean_mb': np.mean(self.memory_samples),
            'memory_stddev_mb': np.std(self.memory_samples)
        }

    def measure_cpu(self) -> Dict[str, float]:
        """Calculate CPU utilization statistics.

        Returns:
            Dictionary with cpu_percent_mean, cpu_percent_max
        """
        if not self.cpu_samples:
            return {'cpu_percent_mean': 0.0, 'cpu_percent_max': 0.0}

        return {
            'cpu_percent_mean': np.mean(self.cpu_samples),
            'cpu_percent_max': np.max(self.cpu_samples)
        }

    def _measure_memory(self) -> float:
        """Measure current memory usage in megabytes.

        Returns:
            Memory usage in MB (resident set size)
        """
        try:
            # Get memory info for current process
            mem_info = self.process.memory_info()
            # RSS (Resident Set Size) = actual physical memory used
            memory_mb = mem_info.rss / (1024 * 1024)
            return memory_mb
        except Exception as e:
            logger.warning(f"Failed to measure memory: {e}")
            return 0.0

    def _measure_cpu(self) -> float:
        """Measure current CPU utilization percentage.

        Returns:
            CPU usage percentage [0.0, 100.0+]
        """
        try:
            # Get CPU percentage for current process
            # interval=None uses cached value (no blocking)
            cpu_percent = self.process.cpu_percent(interval=None)
            return cpu_percent
        except Exception as e:
            logger.warning(f"Failed to measure CPU: {e}")
            return 0.0

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate aggregate performance metrics.

        Returns:
            PerformanceMetrics object with all measurements and target validation.
        """
        # Calculate FPS metrics
        fps_metrics = self.measure_fps()

        # Calculate memory metrics
        memory_metrics = self.measure_memory()

        # Calculate CPU metrics
        cpu_metrics = self.measure_cpu()

        # Calculate detection time statistics
        detection_time_mean_ms = np.mean(self.detection_times) * 1000 if self.detection_times else 0.0
        detection_time_stddev_ms = np.std(self.detection_times) * 1000 if self.detection_times else 0.0

        # Validate against targets
        meets_fps_target = fps_metrics['fps'] >= self.FPS_TARGET
        meets_memory_target = memory_metrics['memory_peak_mb'] <= self.MEMORY_TARGET_MB

        return PerformanceMetrics(
            fps=round(fps_metrics['fps'], 4),
            fps_min=round(fps_metrics['fps_min'], 4),
            fps_max=round(fps_metrics['fps_max'], 4),
            memory_peak_mb=round(memory_metrics['memory_peak_mb'], 2),
            memory_mean_mb=round(memory_metrics['memory_mean_mb'], 2),
            memory_stddev_mb=round(memory_metrics['memory_stddev_mb'], 2),
            cpu_percent_mean=round(cpu_metrics['cpu_percent_mean'], 2),
            cpu_percent_max=round(cpu_metrics['cpu_percent_max'], 2),
            detection_time_mean_ms=round(detection_time_mean_ms, 2),
            detection_time_stddev_ms=round(detection_time_stddev_ms, 2),
            total_images=len(self.detection_times),
            meets_fps_target=meets_fps_target,
            meets_memory_target=meets_memory_target
        )

    def reset(self):
        """Reset all measurements for new profiling session."""
        self.detection_times.clear()
        self.memory_samples.clear()
        self.cpu_samples.clear()
        logger.info("Performance profiler reset")

    def log_summary(self):
        """Log performance summary to console."""
        metrics = self.get_metrics()

        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE PROFILING RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nFPS (Frames Per Second):")
        logger.info(f"  Average FPS: {metrics.fps:.4f} fps")
        logger.info(f"  Min FPS: {metrics.fps_min:.4f} fps")
        logger.info(f"  Max FPS: {metrics.fps_max:.4f} fps")
        logger.info(f"  Target (≥1/60 Hz): {self.FPS_TARGET:.4f} fps")
        logger.info(f"  Status: {'✓ PASS' if metrics.meets_fps_target else '✗ FAIL'}")

        logger.info(f"\nMemory Usage:")
        logger.info(f"  Peak Memory: {metrics.memory_peak_mb:.2f} MB")
        logger.info(f"  Mean Memory: {metrics.memory_mean_mb:.2f} MB")
        logger.info(f"  Std Dev: {metrics.memory_stddev_mb:.2f} MB")
        logger.info(f"  Target (≤500 MB): {self.MEMORY_TARGET_MB:.0f} MB")
        logger.info(f"  Status: {'✓ PASS' if metrics.meets_memory_target else '✗ FAIL'}")

        logger.info(f"\nCPU Utilization:")
        logger.info(f"  Mean CPU: {metrics.cpu_percent_mean:.2f}%")
        logger.info(f"  Peak CPU: {metrics.cpu_percent_max:.2f}%")

        logger.info(f"\nDetection Time:")
        logger.info(f"  Mean: {metrics.detection_time_mean_ms:.2f} ms")
        logger.info(f"  Std Dev: {metrics.detection_time_stddev_ms:.2f} ms")

        logger.info(f"\nTotal Images Processed: {metrics.total_images}")
        logger.info("=" * 60)
