"""Unit tests for ResultManager

Tests cover all acceptance criteria:
- AC-1.4.1: Result Dict Construction
- AC-1.4.2: Status Determination
- AC-1.4.3: Timestamp Generation
- AC-1.4.4: History Buffer Management
- AC-1.4.5: History Query Interface
- AC-1.4.6: Error Handling
"""

from datetime import datetime
import pytest

from src.result_manager import ResultManager


@pytest.fixture
def manager():
    """Create a ResultManager instance with default threshold and buffer size."""
    return ResultManager(threshold_pixels=2.0, history_buffer_size=100)


@pytest.fixture
def manager_small_buffer():
    """Create a ResultManager instance with small buffer for FIFO testing."""
    return ResultManager(threshold_pixels=2.0, history_buffer_size=3)


@pytest.fixture
def manager_custom_threshold():
    """Create a ResultManager instance with custom threshold."""
    return ResultManager(threshold_pixels=5.0, history_buffer_size=100)


class TestInitialization:
    """Test ResultManager initialization (AC-1.4.4, AC-1.4.6)."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        manager = ResultManager()
        assert manager.threshold_pixels == 2.0
        assert manager._buffer.maxlen == 100

    def test_init_custom_parameters(self):
        """Test initialization with custom threshold and buffer size."""
        manager = ResultManager(threshold_pixels=5.0, history_buffer_size=50)
        assert manager.threshold_pixels == 5.0
        assert manager._buffer.maxlen == 50

    def test_init_invalid_threshold_negative(self):
        """Test initialization fails with negative threshold."""
        with pytest.raises(ValueError, match="threshold_pixels must be positive"):
            ResultManager(threshold_pixels=-1.0)

    def test_init_invalid_threshold_zero(self):
        """Test initialization fails with zero threshold."""
        with pytest.raises(ValueError, match="threshold_pixels must be positive"):
            ResultManager(threshold_pixels=0.0)

    def test_init_invalid_buffer_size_negative(self):
        """Test initialization fails with negative buffer size."""
        with pytest.raises(ValueError, match="history_buffer_size must be positive integer"):
            ResultManager(threshold_pixels=2.0, history_buffer_size=-10)

    def test_init_invalid_buffer_size_zero(self):
        """Test initialization fails with zero buffer size."""
        with pytest.raises(ValueError, match="history_buffer_size must be positive integer"):
            ResultManager(threshold_pixels=2.0, history_buffer_size=0)

    def test_init_invalid_buffer_size_non_integer(self):
        """Test initialization fails with non-integer buffer size."""
        with pytest.raises(ValueError, match="history_buffer_size must be positive integer"):
            ResultManager(threshold_pixels=2.0, history_buffer_size=10.5)


class TestCreateResult:
    """Test result dictionary creation (AC-1.4.1, AC-1.4.2, AC-1.4.3, AC-1.4.6)."""

    def test_create_result_valid_status(self, manager):
        """Test result creation with translation_displacement below threshold."""
        result = manager.create_result(
            translation_displacement=1.5,
            confidence=0.95,
            frame_id="test_001"
        )

        # AC-1.4.1: Verify all required fields present
        assert "status" in result
        assert "translation_displacement" in result
        assert "confidence" in result
        assert "frame_id" in result
        assert "timestamp" in result

        # AC-1.4.2: Status should be VALID (1.5 < 2.0)
        assert result["status"] == "VALID"

        # Verify values
        assert result["translation_displacement"] == 1.5
        assert result["confidence"] == 0.95
        assert result["frame_id"] == "test_001"

    def test_create_result_invalid_status(self, manager):
        """Test result creation with translation_displacement above threshold."""
        result = manager.create_result(
            translation_displacement=3.0,
            confidence=0.85,
            frame_id="test_002"
        )

        # AC-1.4.2: Status should be INVALID (3.0 >= 2.0)
        assert result["status"] == "INVALID"
        assert result["translation_displacement"] == 3.0
        assert result["confidence"] == 0.85

    def test_create_result_exactly_at_threshold(self, manager):
        """Test status determination when displacement exactly equals threshold."""
        result = manager.create_result(
            translation_displacement=2.0,
            confidence=0.90,
            frame_id="test_003"
        )

        # AC-1.4.2: displacement >= threshold → INVALID
        assert result["status"] == "INVALID"
        assert result["translation_displacement"] == 2.0

    def test_create_result_custom_threshold(self, manager_custom_threshold):
        """Test status determination with custom threshold (5.0)."""
        # 3.0 < 5.0 → should be VALID
        result = manager_custom_threshold.create_result(
            translation_displacement=3.0,
            confidence=0.88,
            frame_id="test_004"
        )

        assert result["status"] == "VALID"
        assert result["translation_displacement"] == 3.0

    def test_create_result_auto_frame_id(self, manager):
        """Test automatic frame_id generation when None provided."""
        result = manager.create_result(
            translation_displacement=1.0,
            confidence=0.92
        )

        # Should auto-generate UUID frame_id
        assert result["frame_id"] is not None
        assert isinstance(result["frame_id"], str)
        assert len(result["frame_id"]) > 0

    def test_create_result_timestamp_format(self, manager):
        """Test timestamp is ISO 8601 UTC with milliseconds and 'Z' (AC-1.4.3)."""
        result = manager.create_result(
            translation_displacement=1.2,
            confidence=0.87,
            frame_id="test_005"
        )

        timestamp = result["timestamp"]

        # Verify ISO 8601 format
        assert isinstance(timestamp, str)
        assert timestamp.endswith("Z")

        # Verify parseable by datetime
        try:
            # Remove 'Z' and parse
            dt = datetime.fromisoformat(timestamp[:-1])
            assert isinstance(dt, datetime)
        except ValueError:
            pytest.fail(f"Timestamp '{timestamp}' is not valid ISO 8601 format")

    def test_create_result_field_order(self, manager):
        """Test result dictionary maintains exact field order."""
        result = manager.create_result(
            translation_displacement=0.5,
            confidence=0.99,
            frame_id="test_006"
        )

        # Verify field order: status, translation_displacement, confidence, frame_id, timestamp
        keys = list(result.keys())
        assert keys == ["status", "translation_displacement", "confidence", "frame_id", "timestamp"]

    def test_create_result_rounding(self, manager):
        """Test translation_displacement and confidence are rounded to 2 decimals."""
        result = manager.create_result(
            translation_displacement=1.556,
            confidence=0.8888,
            frame_id="test_007"
        )

        # Should round to 2 decimals
        assert result["translation_displacement"] == 1.56
        assert result["confidence"] == 0.89

    def test_create_result_invalid_displacement_negative(self, manager):
        """Test validation: negative translation_displacement raises ValueError (AC-1.4.6)."""
        with pytest.raises(ValueError, match="translation_displacement must be non-negative float"):
            manager.create_result(
                translation_displacement=-1.0,
                confidence=0.5,
                frame_id="test_008"
            )

    def test_create_result_invalid_confidence_above_range(self, manager):
        """Test validation: confidence > 1.0 raises ValueError (AC-1.4.6)."""
        with pytest.raises(ValueError, match="confidence must be in range"):
            manager.create_result(
                translation_displacement=1.0,
                confidence=1.5,
                frame_id="test_009"
            )

    def test_create_result_invalid_confidence_below_range(self, manager):
        """Test validation: confidence < 0.0 raises ValueError (AC-1.4.6)."""
        with pytest.raises(ValueError, match="confidence must be in range"):
            manager.create_result(
                translation_displacement=1.0,
                confidence=-0.1,
                frame_id="test_010"
            )


class TestHistoryBuffer:
    """Test history buffer management (AC-1.4.4)."""

    def test_add_to_history_single_result(self, manager):
        """Test adding single result to history buffer."""
        result = manager.create_result(1.0, 0.9, "frame_001")
        manager.add_to_history(result)

        history = manager.get_history()
        assert len(history) == 1
        assert history[0] == result

    def test_add_to_history_multiple_results(self, manager):
        """Test adding multiple results to history buffer."""
        for i in range(10):
            result = manager.create_result(1.0, 0.9, f"frame_{i:03d}")
            manager.add_to_history(result)

        history = manager.get_history()
        assert len(history) == 10

    def test_history_buffer_fifo_eviction(self, manager_small_buffer):
        """Test FIFO buffer eviction when buffer is full (AC-1.4.4)."""
        # Add 5 results to buffer with size 3
        for i in range(5):
            result = manager_small_buffer.create_result(1.0, 0.9, f"frame_{i}")
            manager_small_buffer.add_to_history(result)

        history = manager_small_buffer.get_history()

        # Only last 3 should remain
        assert len(history) == 3

        # First two (frame_0, frame_1) should be evicted
        # Remaining should be frame_2, frame_3, frame_4
        assert history[0]["frame_id"] == "frame_2"
        assert history[1]["frame_id"] == "frame_3"
        assert history[2]["frame_id"] == "frame_4"

    def test_history_buffer_fifo_large_overflow(self, manager_small_buffer):
        """Test FIFO buffer with large overflow (150 results to buffer size 3)."""
        # Add 150 results to buffer with size 3
        for i in range(150):
            result = manager_small_buffer.create_result(1.0, 0.9, f"frame_{i:03d}")
            manager_small_buffer.add_to_history(result)

        history = manager_small_buffer.get_history()

        # Only last 3 should remain
        assert len(history) == 3
        assert history[0]["frame_id"] == "frame_147"
        assert history[1]["frame_id"] == "frame_148"
        assert history[2]["frame_id"] == "frame_149"

    def test_add_to_history_missing_required_field(self, manager):
        """Test validation: adding result with missing field raises ValueError."""
        invalid_result = {
            "status": "VALID",
            "translation_displacement": 1.0,
            # Missing: confidence, frame_id, timestamp
        }

        with pytest.raises(ValueError, match="result dict missing required field"):
            manager.add_to_history(invalid_result)


class TestHistoryQueryMethods:
    """Test history query interface (AC-1.4.5)."""

    def test_get_history_empty_buffer(self, manager):
        """Test get_history returns empty list when buffer is empty."""
        history = manager.get_history()
        assert history == []
        assert isinstance(history, list)

    def test_get_history_returns_list_not_deque(self, manager):
        """Test get_history returns list (not deque object)."""
        result = manager.create_result(1.0, 0.9, "frame_001")
        manager.add_to_history(result)

        history = manager.get_history()
        assert isinstance(history, list)
        assert not isinstance(history, type(manager._buffer))

    def test_get_history_chronological_order(self, manager):
        """Test get_history returns results in chronological order (oldest to newest)."""
        for i in range(5):
            result = manager.create_result(1.0, 0.9, f"frame_{i}")
            manager.add_to_history(result)

        history = manager.get_history()

        # Should be in order: frame_0, frame_1, frame_2, frame_3, frame_4
        assert history[0]["frame_id"] == "frame_0"
        assert history[4]["frame_id"] == "frame_4"

    def test_get_last_n_basic(self, manager):
        """Test get_last_n returns n most recent results."""
        for i in range(20):
            result = manager.create_result(1.0, 0.9, f"frame_{i:02d}")
            manager.add_to_history(result)

        recent = manager.get_last_n(10)

        assert len(recent) == 10
        # Should be frame_10 through frame_19
        assert recent[0]["frame_id"] == "frame_10"
        assert recent[9]["frame_id"] == "frame_19"

    def test_get_last_n_exceeds_buffer_size(self, manager):
        """Test get_last_n when n > buffer size returns all available."""
        for i in range(5):
            result = manager.create_result(1.0, 0.9, f"frame_{i}")
            manager.add_to_history(result)

        # Request 100, but only 5 available
        recent = manager.get_last_n(100)

        assert len(recent) == 5
        assert recent[0]["frame_id"] == "frame_0"
        assert recent[4]["frame_id"] == "frame_4"

    def test_get_last_n_empty_buffer(self, manager):
        """Test get_last_n on empty buffer returns empty list."""
        recent = manager.get_last_n(10)
        assert recent == []

    def test_get_last_n_invalid_n_negative(self, manager):
        """Test get_last_n with negative n raises ValueError."""
        result = manager.create_result(1.0, 0.9, "frame_001")
        manager.add_to_history(result)

        with pytest.raises(ValueError, match="n must be positive integer"):
            manager.get_last_n(-5)

    def test_get_last_n_invalid_n_zero(self, manager):
        """Test get_last_n with zero n raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive integer"):
            manager.get_last_n(0)

    def test_get_last_n_invalid_n_non_integer(self, manager):
        """Test get_last_n with non-integer n raises ValueError."""
        with pytest.raises(ValueError, match="n must be positive integer"):
            manager.get_last_n(5.5)

    def test_get_by_frame_id_found(self, manager):
        """Test get_by_frame_id finds matching result."""
        for i in range(10):
            result = manager.create_result(1.0, 0.9, f"frame_{i:03d}")
            manager.add_to_history(result)

        # Search for frame_005
        found = manager.get_by_frame_id("frame_005")

        assert found is not None
        assert found["frame_id"] == "frame_005"
        assert found["translation_displacement"] == 1.0

    def test_get_by_frame_id_not_found(self, manager):
        """Test get_by_frame_id returns None when frame_id not found."""
        result = manager.create_result(1.0, 0.9, "frame_001")
        manager.add_to_history(result)

        # Search for non-existent frame_id
        found = manager.get_by_frame_id("nonexistent")

        assert found is None

    def test_get_by_frame_id_empty_buffer(self, manager):
        """Test get_by_frame_id on empty buffer returns None."""
        found = manager.get_by_frame_id("frame_001")
        assert found is None

    def test_get_by_frame_id_invalid_type(self, manager):
        """Test get_by_frame_id with non-string frame_id raises ValueError."""
        result = manager.create_result(1.0, 0.9, "frame_001")
        manager.add_to_history(result)

        with pytest.raises(ValueError, match="frame_id must be string"):
            manager.get_by_frame_id(123)


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_complete_workflow(self, manager):
        """Test complete workflow: create result, add to history, query."""
        # Create result
        result = manager.create_result(
            translation_displacement=1.5,
            confidence=0.95,
            frame_id="integration_test"
        )

        # Verify result structure
        assert result["status"] == "VALID"
        assert result["translation_displacement"] == 1.5
        assert result["confidence"] == 0.95

        # Add to history
        manager.add_to_history(result)

        # Query by frame_id
        found = manager.get_by_frame_id("integration_test")
        assert found == result

        # Query last n
        recent = manager.get_last_n(1)
        assert len(recent) == 1
        assert recent[0] == result

        # Query all history
        history = manager.get_history()
        assert len(history) == 1
        assert history[0] == result

    def test_buffer_maxlen_attribute(self, manager):
        """Test buffer is initialized with correct maxlen from history_buffer_size."""
        assert manager._buffer.maxlen == 100

        manager_custom = ResultManager(threshold_pixels=2.0, history_buffer_size=50)
        assert manager_custom._buffer.maxlen == 50

    def test_zero_displacement(self, manager):
        """Test result creation with zero displacement."""
        result = manager.create_result(
            translation_displacement=0.0,
            confidence=1.0,
            frame_id="zero_disp"
        )

        assert result["status"] == "VALID"
        assert result["translation_displacement"] == 0.0

    def test_confidence_boundary_values(self, manager):
        """Test confidence at boundary values (0.0 and 1.0)."""
        # Confidence = 0.0
        result_min = manager.create_result(1.0, 0.0, "conf_min")
        assert result_min["confidence"] == 0.0

        # Confidence = 1.0
        result_max = manager.create_result(1.0, 1.0, "conf_max")
        assert result_max["confidence"] == 1.0
