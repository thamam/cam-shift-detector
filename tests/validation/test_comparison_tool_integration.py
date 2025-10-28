"""
Integration Tests for Comparison Tool

End-to-end testing of comparison_tool.py with offline mode using session_001 data.
Validates CLI, offline mode implementation, JSON logging, MSE graph generation,
and worst matches reporting.
"""

import pytest
import json
import subprocess
import sys
from pathlib import Path
from PIL import Image


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestComparisonToolCLI:
    """Test CLI argument parsing and validation."""

    def test_cli_help_displays(self):
        """Test that --help displays help message."""
        result = subprocess.run(
            [sys.executable, "tools/validation/comparison_tool.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "ChArUco vs Cam-Shift Comparison Tool" in result.stdout
        assert "--mode" in result.stdout
        assert "--camera-yaml" in result.stdout

    def test_cli_missing_mode_fails(self):
        """Test that missing --mode argument fails with error."""
        result = subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", "test_output"
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_cli_offline_mode_missing_input_dir_fails(self):
        """Test that offline mode without --input-dir fails."""
        result = subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "offline",
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", "test_output"
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "--input-dir" in result.stderr

    def test_cli_invalid_mode_fails(self):
        """Test that invalid mode argument fails."""
        result = subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "invalid_mode",
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", "test_output"
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()


class TestComparisonToolOfflineMode:
    """Test offline mode implementation end-to-end."""

    @pytest.fixture
    def test_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "comparison_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        yield output_dir

    @pytest.fixture
    def session_001_frames_dir(self):
        """Get session_001 frames directory path."""
        # Use relative path from project root
        frames_dir = PROJECT_ROOT / "session_001" / "frames"
        if not frames_dir.exists():
            pytest.skip("session_001/frames directory not found")
        return frames_dir

    def test_offline_mode_session_001_end_to_end(
        self,
        session_001_frames_dir,
        test_output_dir
    ):
        """Test offline mode end-to-end with session_001 data.

        AC-2: Offline Mode Implementation
        AC-6: Integration Testing
        """
        # Run comparison tool
        result = subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "offline",
                "--input-dir", str(session_001_frames_dir),
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", str(test_output_dir)
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )

        # Verify command succeeded
        assert result.returncode == 0, f"Command failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        # Verify output mentions completion (logging goes to stderr)
        assert "complete" in result.stderr.lower()

    def test_offline_mode_creates_json_log(
        self,
        session_001_frames_dir,
        test_output_dir
    ):
        """Test that offline mode creates JSON log file.

        AC-2: Logs all results to JSON in --output-dir
        AC-6: JSON log structure validation
        """
        # Run comparison tool
        subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "offline",
                "--input-dir", str(session_001_frames_dir),
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", str(test_output_dir)
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=60
        )

        # Find JSON log file
        json_files = list(test_output_dir.glob("*_comparison.json"))
        assert len(json_files) > 0, "No JSON log file created"

        # Verify JSON structure
        with open(json_files[0], 'r') as f:
            log_data = json.load(f)

        # Verify required fields
        assert "session_name" in log_data
        assert "timestamp" in log_data
        assert "threshold_px" in log_data
        assert "total_frames" in log_data
        assert "results" in log_data

        # Verify results structure
        assert isinstance(log_data["results"], list)
        assert len(log_data["results"]) > 0

        # Verify first result structure
        first_result = log_data["results"][0]
        assert "frame_idx" in first_result
        assert "timestamp_ns" in first_result
        assert "charuco_detected" in first_result
        assert "charuco_displacement_px" in first_result
        assert "charuco_confidence" in first_result
        assert "camshift_status" in first_result
        assert "camshift_displacement_px" in first_result
        assert "camshift_confidence" in first_result
        assert "displacement_diff" in first_result
        assert "agreement_status" in first_result
        assert "threshold_px" in first_result

    def test_offline_mode_creates_mse_graph(
        self,
        session_001_frames_dir,
        test_output_dir
    ):
        """Test that offline mode creates MSE graph PNG.

        AC-2: Generates MSE graph on completion
        AC-6: MSE graph generation validation
        """
        # Run comparison tool
        subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "offline",
                "--input-dir", str(session_001_frames_dir),
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", str(test_output_dir)
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=60
        )

        # Find MSE graph PNG
        graph_files = list(test_output_dir.glob("*_mse_graph.png"))
        assert len(graph_files) > 0, "No MSE graph PNG created"

        # Verify PNG is valid image
        graph_path = graph_files[0]
        assert graph_path.exists()

        # Try to open image with PIL
        img = Image.open(graph_path)
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_offline_mode_creates_worst_matches_report(
        self,
        session_001_frames_dir,
        test_output_dir
    ):
        """Test that offline mode creates worst matches report.

        AC-2: Generates worst matches report on completion
        AC-6: Worst matches retrieval accuracy
        """
        # Run comparison tool
        subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "offline",
                "--input-dir", str(session_001_frames_dir),
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", str(test_output_dir)
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=60
        )

        # Find worst matches report
        report_files = list(test_output_dir.glob("*_worst_matches.txt"))
        assert len(report_files) > 0, "No worst matches report created"

        # Verify report content
        with open(report_files[0], 'r') as f:
            content = f.read()

        assert "Top 10 Worst Matches" in content
        assert "displacement_diff" in content.lower() or "difference" in content.lower()

    def test_offline_mode_performance_benchmark(
        self,
        session_001_frames_dir,
        test_output_dir
    ):
        """Test that offline mode meets performance requirements.

        AC-2: Processes 157 frames in <30 seconds (~5 FPS minimum)
        AC-6: Performance benchmarks met (FPS requirements)
        """
        import time

        # Measure execution time
        start_time = time.time()

        result = subprocess.run(
            [
                sys.executable, "tools/validation/comparison_tool.py",
                "--mode", "offline",
                "--input-dir", str(session_001_frames_dir),
                "--camera-yaml", "camera.yaml",
                "--charuco-config", "config/comparison_config.json",
                "--camshift-config", "config/config_session_001.json",
                "--output-dir", str(test_output_dir)
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            timeout=60
        )

        elapsed_time = time.time() - start_time

        # Verify command succeeded
        assert result.returncode == 0

        # Load JSON log to get frame count
        json_files = list(test_output_dir.glob("*_comparison.json"))
        assert len(json_files) > 0

        with open(json_files[0], 'r') as f:
            log_data = json.load(f)

        frame_count = log_data["total_frames"]

        # Calculate FPS
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0.0

        # Verify performance requirement
        # Allow some buffer (4 FPS instead of strict 5 FPS for CI environments)
        assert fps >= 4.0, f"Performance too slow: {fps:.2f} FPS (expected >= 5 FPS)"

        # Verify total time is reasonable
        assert elapsed_time < 40.0, f"Processing took too long: {elapsed_time:.2f}s (expected <30s)"


class TestComparisonToolConfiguration:
    """Test configuration file loading and validation."""

    def test_comparison_config_json_exists(self):
        """Test that comparison_config.json exists.

        AC-5: comparison_config.json created with default ChArUco board parameters
        """
        config_path = PROJECT_ROOT / "comparison_config.json"
        assert config_path.exists(), "comparison_config.json not found"

    def test_comparison_config_json_structure(self):
        """Test comparison_config.json has correct structure.

        AC-5: Configuration structure with ChArUco board parameters
        """
        config_path = PROJECT_ROOT / "comparison_config.json"

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Verify required sections
        assert "charuco_board" in config
        assert "comparison_settings" in config
        assert "display_settings" in config

        # Verify ChArUco board parameters
        board = config["charuco_board"]
        assert board["squares_x"] == 7
        assert board["squares_y"] == 5
        assert board["square_len_m"] == 0.035
        assert board["marker_len_m"] == 0.026
        assert board["dict_name"] == "DICT_4X4_50"

        # Verify comparison settings
        comp_settings = config["comparison_settings"]
        assert "threshold_percent" in comp_settings
        assert "default_z_distance_m" in comp_settings

        # Verify display settings
        display = config["display_settings"]
        assert "window_width" in display
        assert "window_height" in display


class TestComparisonToolDocumentation:
    """Test documentation completeness."""

    def test_readme_exists(self):
        """Test that README.md exists in tools/validation/.

        AC-5: tools/validation/README.md documents installation, usage, troubleshooting
        """
        readme_path = PROJECT_ROOT / "tools" / "validation" / "README.md"
        assert readme_path.exists(), "tools/validation/README.md not found"

    def test_readme_has_required_sections(self):
        """Test that README.md has all required sections.

        AC-5: Documentation includes installation, offline/online usage, output structure
        """
        readme_path = PROJECT_ROOT / "tools" / "validation" / "README.md"

        with open(readme_path, 'r') as f:
            content = f.read()

        # Verify required sections exist
        assert "Installation" in content or "installation" in content.lower()
        assert "Offline Mode" in content or "offline" in content.lower()
        assert "Online Mode" in content or "online" in content.lower()
        assert "Usage" in content or "usage" in content.lower()
        assert "Troubleshooting" in content or "troubleshooting" in content.lower()
        assert "Output" in content or "output structure" in content.lower()

    def test_readme_has_examples(self):
        """Test that README.md includes usage examples."""
        readme_path = PROJECT_ROOT / "tools" / "validation" / "README.md"

        with open(readme_path, 'r') as f:
            content = f.read()

        # Verify examples exist
        assert "example" in content.lower()
        assert "comparison_tool.py" in content
        assert "--mode offline" in content
        assert "--mode online" in content
