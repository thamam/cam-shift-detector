"""End-to-End Workflow Tests for Camera Movement Detection Module

Tests complete user workflows from start to finish using real sample images
from the sample_images/ directory. These tests verify the full operator
experience and system integration patterns.
"""

import cv2
import json
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.camera_movement_detector import CameraMovementDetector


class TestOperatorWorkflow:
    """Test complete operator workflow: ROI → config → baseline → detection (AC-1.8.3)"""

    def test_operator_workflow_with_real_images_of_jerusalem(self, tmp_path):
        """Test full workflow using real images from OF_JERUSALEM dataset"""
        # Setup: Get real sample images
        sample_dir = Path('sample_images/of_jerusalem')
        if not sample_dir.exists():
            pytest.skip("Sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:5]  # Use first 5 images
        if len(images) < 5:
            pytest.skip("Insufficient sample images")

        # Step 1: Simulated ROI selection (normally done via select_roi.py)
        # For static camera setup, ROI should capture stable regions
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Step 2: Initialize detector
        detector = CameraMovementDetector(str(config_path))

        # Step 3: Capture baseline from first image
        baseline_image = cv2.imread(str(images[0]))
        assert baseline_image is not None, "Failed to load baseline image"
        detector.set_baseline(baseline_image)

        # Step 4: Process subsequent frames (simulating real monitoring)
        results = []
        for img_path in images[1:5]:
            image = cv2.imread(str(img_path))
            assert image is not None, f"Failed to load {img_path}"
            result = detector.process_frame(image, frame_id=img_path.stem)
            results.append(result)

        # Verify: All frames processed
        assert len(results) == 4

        # Verify: Each result has correct structure
        for result in results:
            assert 'status' in result
            assert result['status'] in ['VALID', 'INVALID']
            assert 'translation_displacement' in result
            assert 'confidence' in result
            assert 'frame_id' in result
            assert 'timestamp' in result

        # Verify: History captured all frames
        history = detector.get_history()
        assert len(history) == 4

        # Verify: History in chronological order
        for i, result in enumerate(history):
            assert result['frame_id'] == images[i + 1].stem

    def test_operator_workflow_with_carmit_dataset(self, tmp_path):
        """Test workflow using CARMIT dataset images"""
        sample_dir = Path('sample_images/carmit')
        if not sample_dir.exists():
            pytest.skip("CARMIT sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:5]
        if len(images) < 5:
            pytest.skip("Insufficient CARMIT sample images")

        # Setup config
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 150, "y": 100, "width": 350, "height": 250},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Execute: Complete workflow
        detector = CameraMovementDetector(str(config_path))
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Process frames
        valid_count = 0
        invalid_count = 0
        for img_path in images[1:5]:
            image = cv2.imread(str(img_path))
            result = detector.process_frame(image)
            if result['status'] == 'VALID':
                valid_count += 1
            else:
                invalid_count += 1

        # Verify: At least some frames processed successfully
        assert valid_count + invalid_count == 4

    def test_operator_workflow_with_gad_dataset(self, tmp_path):
        """Test workflow using GAD dataset images"""
        sample_dir = Path('sample_images/gad')
        if not sample_dir.exists():
            pytest.skip("GAD sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:5]
        if len(images) < 5:
            pytest.skip("Insufficient GAD sample images")

        # Setup config
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # Execute: Complete workflow
        detector = CameraMovementDetector(str(config_path))
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Process all frames and collect results
        results = []
        for img_path in images[1:5]:
            image = cv2.imread(str(img_path))
            result = detector.process_frame(image, frame_id=img_path.name)
            results.append(result)

        # Verify: All frames have valid structure
        assert all('status' in r for r in results)
        assert all('translation_displacement' in r for r in results)
        assert all('confidence' in r for r in results)


class TestRecalibrationWorkflow:
    """Test recalibration workflow: detect → recalibrate → resume (AC-1.8.3)"""

    def test_recalibration_workflow_end_to_end(self, tmp_path):
        """Test complete recalibration workflow with real images"""
        sample_dir = Path('sample_images/of_jerusalem')
        if not sample_dir.exists():
            pytest.skip("Sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:3]
        if len(images) < 3:
            pytest.skip("Insufficient sample images for recalibration test")

        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        detector = CameraMovementDetector(str(config_path))

        # Workflow Step 1: Initial baseline
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Workflow Step 2: Process frame (may detect movement or not)
        test_image = cv2.imread(str(images[1]))
        result1 = detector.process_frame(test_image, frame_id="before_recal")

        # Workflow Step 3: Recalibrate with new image
        new_baseline_image = cv2.imread(str(images[2]))
        success = detector.recalibrate(new_baseline_image)
        assert success is True

        # Workflow Step 4: Resume processing with new baseline
        result2 = detector.process_frame(new_baseline_image, frame_id="after_recal")

        # Verify: Recalibration worked - new baseline should be VALID
        assert result2['status'] == 'VALID'
        assert result2['translation_displacement'] < 1.0  # Same image as baseline

        # Verify: History preserved across recalibration
        history = detector.get_history()
        assert len(history) >= 2  # At least before_recal and after_recal

    def test_recalibration_preserves_config_settings(self, tmp_path):
        """Verify recalibration maintains threshold and buffer settings"""
        sample_dir = Path('sample_images/of_jerusalem')
        if not sample_dir.exists():
            pytest.skip("Sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:3]
        if len(images) < 3:
            pytest.skip("Insufficient sample images")

        # Setup with custom config
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 5.0,  # Custom threshold
            "history_buffer_size": 10,  # Small buffer
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        detector = CameraMovementDetector(str(config_path))
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Recalibrate
        new_baseline = cv2.imread(str(images[1]))
        detector.recalibrate(new_baseline)

        # Verify: Config settings still apply after recalibration
        # Fill buffer beyond configured size
        test_image = cv2.imread(str(images[2]))
        for i in range(15):
            detector.process_frame(test_image, frame_id=f"frame_{i}")

        history = detector.get_history()
        assert len(history) == 10  # Respects original buffer size


class TestDAFSystemIntegration:
    """Test DAF system integration patterns (AC-1.8.3)"""

    def test_external_caller_integration_pattern(self, tmp_path):
        """Simulate external DAF water quality system calling detector"""
        sample_dir = Path('sample_images/of_jerusalem')
        if not sample_dir.exists():
            pytest.skip("Sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:5]
        if len(images) < 5:
            pytest.skip("Insufficient sample images")

        # Simulate external system setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        # External system initializes detector once
        detector = CameraMovementDetector(str(config_path))
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Simulate external system calling detector for each new frame
        for i, img_path in enumerate(images[1:5]):
            # External system loads image
            frame = cv2.imread(str(img_path))

            # External system calls detector
            result = detector.process_frame(frame, frame_id=f"daf_frame_{i:04d}")

            # External system checks result and takes action
            if result['status'] == 'INVALID':
                # Simulate logging or alert (in real system)
                displacement = result['translation_displacement']
                assert displacement >= 0.0  # Valid displacement value
            else:
                # Frame is valid, continue monitoring
                confidence = result['confidence']
                assert 0.0 <= confidence <= 1.0

        # External system queries history for monitoring
        history = detector.get_history()
        assert len(history) == 4  # All frames recorded

        # External system queries last N frames for dashboard
        last_2 = detector.get_history(limit=2)
        assert len(last_2) == 2
        assert last_2[0]['frame_id'] == "daf_frame_0002"
        assert last_2[1]['frame_id'] == "daf_frame_0003"

    def test_batch_processing_pattern(self, tmp_path):
        """Test batch processing of multiple frames (DAF batch analysis)"""
        sample_dir = Path('sample_images/carmit')
        if not sample_dir.exists():
            pytest.skip("CARMIT sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:10]
        if len(images) < 10:
            pytest.skip("Insufficient sample images for batch processing")

        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        detector = CameraMovementDetector(str(config_path))
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Batch process frames
        batch_results = []
        for img_path in images[1:10]:
            image = cv2.imread(str(img_path))
            result = detector.process_frame(image, frame_id=img_path.stem)
            batch_results.append(result)

        # Verify: All frames processed
        assert len(batch_results) == 9

        # Verify: Can query specific frames from batch
        for frame_id in [images[1].stem, images[5].stem, images[9].stem]:
            specific_results = detector.get_history(frame_id=frame_id)
            if len(specific_results) > 0:  # Frame exists in history
                assert specific_results[0]['frame_id'] == frame_id


class TestErrorHandlingInRealScenarios:
    """Test error handling with real-world edge cases (AC-1.8.3)"""

    def test_handling_corrupted_image_in_sequence(self, tmp_path):
        """Verify system handles corrupted image gracefully in workflow"""
        sample_dir = Path('sample_images/of_jerusalem')
        if not sample_dir.exists():
            pytest.skip("Sample images not available")

        images = sorted(sample_dir.glob('*.jpg'))[:3]
        if len(images) < 3:
            pytest.skip("Insufficient sample images")

        # Setup
        config_path = tmp_path / "config.json"
        config = {
            "roi": {"x": 100, "y": 50, "width": 400, "height": 300},
            "threshold_pixels": 2.0,
            "history_buffer_size": 100,
            "min_features_required": 50
        }
        config_path.write_text(json.dumps(config))

        detector = CameraMovementDetector(str(config_path))
        baseline_image = cv2.imread(str(images[0]))
        detector.set_baseline(baseline_image)

        # Process valid frame
        valid_image = cv2.imread(str(images[1]))
        result1 = detector.process_frame(valid_image)
        assert 'status' in result1

        # Try to process invalid image (wrong format)
        invalid_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)  # 2D
        with pytest.raises(ValueError):
            detector.process_frame(invalid_image)

        # Verify: System still works after error
        valid_image2 = cv2.imread(str(images[2]))
        result2 = detector.process_frame(valid_image2)
        assert 'status' in result2

        # Verify: History only contains valid frames
        history = detector.get_history()
        assert len(history) == 2  # Only the two valid frames


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
