"""Integration tests for Camera Movement Detection Module

Tests multi-component interactions and workflows to validate API contracts
and component integration. Focus on end-to-end pipelines without mocking
internal components.
"""

import cv2
import json
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.camera_movement_detector import CameraMovementDetector


class TestDetectionPipeline:
    """Test complete detection pipeline integration (AC-1.8.2)"""

    def test_complete_detection_workflow(self, tmp_path):
        """Verify full pipeline: config → init → baseline → process_frame"""
        # Setup: Create config file
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Create test images with structured content for features
        baseline_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Add structured features (similar to existing test pattern)
        cv2.rectangle(baseline_image, (150, 100), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(baseline_image, (350, 150), (450, 250), (0, 0, 0), -1)
        cv2.rectangle(baseline_image, (200, 300), (300, 400), (128, 128, 128), -1)

        test_image = baseline_image.copy()

        # Execute: Complete workflow
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(baseline_image)
        result = detector.process_frame(test_image, frame_id="test_001")

        # Verify: Result structure and content
        assert 'status' in result
        assert result['status'] == 'VALID'  # Same image should be VALID
        assert 'translation_displacement' in result
        assert result['translation_displacement'] < 2.0
        assert 'confidence' in result
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'frame_id' in result
        assert result['frame_id'] == "test_001"
        assert 'timestamp' in result

    def test_detection_with_simulated_camera_shift(self, tmp_path):
        """Verify detection pipeline detects simulated camera movement"""
        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Create baseline with structured features
        baseline_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(baseline_image, (150, 100), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(baseline_image, (350, 150), (450, 250), (0, 0, 0), -1)
        cv2.rectangle(baseline_image, (200, 300), (300, 400), (128, 128, 128), -1)

        # Create shifted image (shift by 5 pixels)
        M = np.float32([[1, 0, 5], [0, 1, 0]])  # 5 pixel horizontal shift
        shifted_image = cv2.warpAffine(baseline_image, M, (640, 480))

        # Execute
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(baseline_image)
        result = detector.process_frame(shifted_image, frame_id="shifted_001")

        # Verify: Movement detected
        assert result['status'] == 'INVALID'
        assert result['translation_displacement'] >= 2.0  # Exceeds threshold

    def test_multiple_frames_with_history_integration(self, tmp_path):
        """Verify history buffer correctly stores results from multiple frames"""
        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 10,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        baseline_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Add structured features
        cv2.rectangle(baseline_image, (150, 100), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(baseline_image, (350, 150), (450, 250), (0, 0, 0), -1)
        cv2.rectangle(baseline_image, (200, 300), (300, 400), (128, 128, 128), -1)

        # Execute: Process multiple frames
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(baseline_image)

        frame_ids = []
        for i in range(5):
            frame_id = f"frame_{i:03d}"
            result = detector.process_frame(baseline_image.copy(), frame_id=frame_id)
            frame_ids.append(frame_id)

        # Verify: History contains all frames
        history = detector.get_history()
        assert len(history) == 5

        # Verify: Frames in chronological order
        for i, result in enumerate(history):
            assert result['frame_id'] == frame_ids[i]

        # Verify: Query by frame_id works
        specific_result = detector.get_history(frame_id="frame_002")
        assert len(specific_result) == 1
        assert specific_result[0]['frame_id'] == "frame_002"

        # Verify: Limit parameter works
        last_2 = detector.get_history(limit=2)
        assert len(last_2) == 2
        assert last_2[0]['frame_id'] == "frame_003"
        assert last_2[1]['frame_id'] == "frame_004"


class TestRecalibrationWorkflow:
    """Test recalibration workflow integration (AC-1.8.2)"""

    def test_recalibration_resets_baseline(self, tmp_path):
        """Verify recalibration updates baseline and affects subsequent detections"""
        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Create images with structured features
        original_baseline = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(original_baseline, (150, 100), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(original_baseline, (350, 150), (450, 250), (0, 0, 0), -1)
        cv2.rectangle(original_baseline, (200, 300), (300, 400), (128, 128, 128), -1)

        # Shifted image (new baseline after recalibration)
        M = np.float32([[1, 0, 5], [0, 1, 0]])
        shifted_baseline = cv2.warpAffine(original_baseline, M, (640, 480))

        # Execute: Initial setup
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(original_baseline)

        # Detect movement with shifted image
        result1 = detector.process_frame(shifted_baseline, frame_id="before_recal")
        assert result1['status'] == 'INVALID'  # Movement detected

        # Recalibrate with shifted image as new baseline
        success = detector.recalibrate(shifted_baseline)
        assert success is True

        # Verify: Same shifted image now returns VALID (new baseline)
        result2 = detector.process_frame(shifted_baseline, frame_id="after_recal")
        assert result2['status'] == 'VALID'
        assert result2['translation_displacement'] < 2.0

        # Verify: Original baseline now returns INVALID (shifted from new baseline)
        result3 = detector.process_frame(original_baseline, frame_id="original_after_recal")
        assert result3['status'] == 'INVALID'

    def test_recalibration_workflow_with_history(self, tmp_path):
        """Verify recalibration integrates correctly with history buffer"""
        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        baseline_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(baseline_image, (150, 100), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(baseline_image, (350, 150), (450, 250), (0, 0, 0), -1)
        cv2.rectangle(baseline_image, (200, 300), (300, 400), (128, 128, 128), -1)

        # Execute
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(baseline_image)

        # Add some frames to history
        for i in range(3):
            detector.process_frame(baseline_image.copy(), frame_id=f"pre_recal_{i}")

        history_before = detector.get_history()
        assert len(history_before) == 3

        # Recalibrate
        new_baseline = baseline_image.copy()
        success = detector.recalibrate(new_baseline)
        assert success is True

        # Verify: History persists after recalibration
        history_after = detector.get_history()
        assert len(history_after) == 3  # History not cleared

        # Add new frames after recalibration
        detector.process_frame(new_baseline, frame_id="post_recal_0")
        history_final = detector.get_history()
        assert len(history_final) == 4  # History continues to grow


class TestErrorPropagation:
    """Test error propagation between components (AC-1.8.2)"""

    def test_insufficient_features_propagates_correctly(self, tmp_path):
        """Verify insufficient features in FeatureExtractor propagates to set_baseline"""
        # Setup: Config requiring many features
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 1000  # Impossibly high
        }
        config_path.write_text(json.dumps(config))

        # Create feature-poor image
        poor_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Execute and verify: Error propagates through components
        detector = CameraMovementDetector(str(config_path))

        with pytest.raises(ValueError) as exc_info:
            detector.set_baseline(poor_image)

        assert "insufficient features" in str(exc_info.value).lower()
        assert "1000" in str(exc_info.value)  # Shows required count

    def test_process_frame_without_baseline_raises_error(self, tmp_path):
        """Verify RuntimeError propagates when baseline not set"""
        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Execute and verify: Error raised without baseline
        detector = CameraMovementDetector(str(config_path))

        with pytest.raises(RuntimeError) as exc_info:
            detector.process_frame(test_image)

        assert "baseline" in str(exc_info.value).lower()

    def test_invalid_image_format_rejected_early(self, tmp_path):
        """Verify invalid image format raises ValueError before processing"""
        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Create valid baseline with structured features
        valid_baseline = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(valid_baseline, (150, 100), (250, 200), (255, 255, 255), -1)
        cv2.rectangle(valid_baseline, (350, 150), (450, 250), (0, 0, 0), -1)
        cv2.rectangle(valid_baseline, (200, 300), (300, 400), (128, 128, 128), -1)

        # Create invalid images (wrong formats)
        invalid_images = [
            np.random.randint(0, 255, (480, 640), dtype=np.uint8),  # 2D instead of 3D
            np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8),  # 4 channels
            np.random.rand(480, 640, 3),  # float instead of uint8
        ]

        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(valid_baseline)

        # Verify: Each invalid format raises ValueError
        for invalid_image in invalid_images:
            with pytest.raises(ValueError):
                detector.process_frame(invalid_image)


class TestComponentInteraction:
    """Test interactions between multiple components (AC-1.8.2)"""

    def test_config_propagates_to_all_components(self, tmp_path):
        """Verify config values correctly initialize all components"""
        # Setup: Config with specific values
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 200, "y": 100, "width": 300, "height": 200},
            "threshold_pixels": 5.0,  # Custom threshold
            "history_buffer_size": 50,  # Custom buffer size
            "min_features_required": 30  # Custom min features
        }
        config_path.write_text(json.dumps(config))

        # Create baseline with features inside custom ROI (200, 100, 300, 200)
        baseline_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.rectangle(baseline_image, (220, 120), (280, 180), (255, 255, 255), -1)
        cv2.rectangle(baseline_image, (350, 150), (410, 210), (0, 0, 0), -1)
        cv2.rectangle(baseline_image, (280, 200), (340, 260), (128, 128, 128), -1)

        # Execute
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(baseline_image)

        # Verify threshold propagation: Same image should be VALID (no movement)
        result = detector.process_frame(baseline_image.copy())
        assert result['status'] == 'VALID'
        assert result['translation_displacement'] < 5.0  # Custom threshold respected

        # Verify large shift is detected as INVALID
        M = np.float32([[1, 0, 10], [0, 1, 0]])  # 10px shift (exceeds 5.0 threshold)
        shifted_10px = cv2.warpAffine(baseline_image, M, (640, 480))
        result_shifted = detector.process_frame(shifted_10px)
        assert result_shifted['status'] == 'INVALID'  # Exceeds 5.0 threshold

        # Verify buffer size: Fill buffer with more than configured size
        for i in range(60):
            detector.process_frame(baseline_image.copy(), frame_id=f"frame_{i}")

        history = detector.get_history()
        assert len(history) == 50  # Respects custom buffer size

    def test_feature_extractor_mask_from_roi_manager(self, tmp_path):
        """Verify StaticRegionManager mask correctly constrains FeatureExtractor"""
        # Setup: Small ROI to constrain features
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 200, "y": 150, "width": 100, "height": 100},  # Small ROI
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 20  # Lower threshold for small ROI
        }
        config_path.write_text(json.dumps(config))

        # Create image with features ONLY in ROI
        baseline_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Black background
        # Add features in ROI
        for i in range(5):
            for j in range(5):
                x = 210 + i * 15
                y = 160 + j * 15
                cv2.circle(baseline_image, (x, y), 4, (255, 255, 255), -1)

        # Execute
        detector = CameraMovementDetector(str(config_path))
        detector.set_baseline(baseline_image)

        # Process same image (should be VALID)
        result = detector.process_frame(baseline_image.copy())

        # Verify: Detection works with ROI-constrained features
        assert result['status'] == 'VALID'
        assert result['translation_displacement'] < 2.0
        # Confidence should be high for identical image
        assert result['confidence'] > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
