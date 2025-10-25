# Story 1.8: Comprehensive Test Coverage & Integration Test Suite

Status: Ready

## Story

As a **QA engineer and development team member**,
I want **comprehensive unit and integration test coverage for all MVP components**,
so that **we can verify >80% code coverage, validate API contracts, and ensure system reliability before Stage 1-3 validation testing**.

## Acceptance Criteria

1. **AC-1.8.1: Unit Test Coverage** - Achieve >80% code coverage across all source modules (StaticRegionManager, FeatureExtractor, MovementDetector, ResultManager, CameraMovementDetector)

2. **AC-1.8.2: Integration Test Suite** - Create integration tests validating CameraMovementDetector API contracts and component interactions

3. **AC-1.8.3: End-to-End Workflow Tests** - Implement E2E tests for complete workflows (setup → baseline capture → detection → recalibration)

4. **AC-1.8.4: Error Path Coverage** - Test all error handling paths (invalid inputs, missing files, insufficient features, initialization failures)

5. **AC-1.8.5: Test Documentation** - Document test framework, test execution instructions, and coverage reporting in project README

6. **AC-1.8.6: CI/CD Readiness** - Ensure all tests can be executed via single command (`pytest`) and return appropriate exit codes

7. **AC-1.8.7: Performance Benchmarks** - Add basic performance tests verifying `process_frame()` execution time <500ms target

8. **AC-1.8.8: Test Data Management** - Organize test fixtures, sample images, and mock data with clear documentation

## Tasks / Subtasks

- [ ] **Task 1: Audit existing test coverage** (AC: #1.8.1)
  - [ ] 1.1: Run coverage report on existing tests (pytest-cov)
  - [ ] 1.2: Identify gaps in unit test coverage (<80% modules)
  - [ ] 1.3: Document coverage baseline and improvement targets
  - [ ] 1.4: Create coverage report in docs/

- [ ] **Task 2: Add missing unit tests** (AC: #1.8.1, #1.8.4)
  - [ ] 2.1: Add tests for uncovered StaticRegionManager edge cases
  - [ ] 2.2: Add tests for uncovered FeatureExtractor error paths
  - [ ] 2.3: Add tests for uncovered MovementDetector boundary conditions
  - [ ] 2.4: Add tests for uncovered ResultManager validation logic
  - [ ] 2.5: Add tests for uncovered CameraMovementDetector workflows
  - [ ] 2.6: Verify >80% coverage achieved after additions

- [ ] **Task 3: Create integration test suite** (AC: #1.8.2)
  - [ ] 3.1: Create tests/test_integration.py module
  - [ ] 3.2: Test complete setup workflow (config → detector init → baseline capture)
  - [ ] 3.3: Test runtime detection workflow (process_frame pipeline)
  - [ ] 3.4: Test recalibration workflow (recalibrate → baseline update)
  - [ ] 3.5: Test history buffer persistence across multiple frames
  - [ ] 3.6: Test error propagation between components

- [ ] **Task 4: Implement E2E workflow tests** (AC: #1.8.3)
  - [ ] 4.1: Create tests/test_e2e.py module
  - [ ] 4.2: Test operator workflow: ROI selection → config → baseline → detection
  - [ ] 4.3: Test recalibration workflow: detect movement → recalibrate → resume
  - [ ] 4.4: Test DAF system integration pattern (mock external caller)
  - [ ] 4.5: Use real sample images from sample_images/ directory

- [ ] **Task 5: Verify error handling coverage** (AC: #1.8.4)
  - [ ] 5.1: Test invalid image formats (wrong dtype, dimensions, channels)
  - [ ] 5.2: Test missing/corrupted config files
  - [ ] 5.3: Test insufficient features scenarios
  - [ ] 5.4: Test baseline not set errors
  - [ ] 5.5: Test invalid API parameter combinations
  - [ ] 5.6: Verify all error paths raise appropriate exceptions

- [ ] **Task 6: Add performance benchmarks** (AC: #1.8.7)
  - [ ] 6.1: Create tests/test_performance.py module
  - [ ] 6.2: Benchmark process_frame() execution time (target <500ms)
  - [ ] 6.3: Benchmark set_baseline() execution time (target <2s)
  - [ ] 6.4: Benchmark get_history() query time (target <10ms)
  - [ ] 6.5: Add pytest-benchmark markers for performance tests

- [ ] **Task 7: Organize test data and fixtures** (AC: #1.8.8)
  - [ ] 7.1: Create tests/fixtures/ directory structure
  - [ ] 7.2: Add sample config files (valid, invalid, edge cases)
  - [ ] 7.3: Add test images (baseline, shifted, feature-poor)
  - [ ] 7.4: Create pytest fixtures for common test setup
  - [ ] 7.5: Document test data organization in tests/README.md

- [ ] **Task 8: Document test framework** (AC: #1.8.5, #1.8.6)
  - [ ] 8.1: Add "Testing" section to main README.md
  - [ ] 8.2: Document test execution commands (pytest, coverage)
  - [ ] 8.3: Document test categories (unit, integration, e2e, performance)
  - [ ] 8.4: Add CI/CD integration instructions
  - [ ] 8.5: Document coverage reporting and thresholds

- [ ] **Task 9: Validate test suite completeness** (AC: All)
  - [ ] 9.1: Run full test suite and verify 100% pass rate
  - [ ] 9.2: Generate coverage report and verify >80% coverage
  - [ ] 9.3: Verify all acceptance criteria from AC-001 to AC-011 (Tech Spec) are testable
  - [ ] 9.4: Run tests on clean environment to verify dependencies
  - [ ] 9.5: Document any remaining test gaps for post-MVP

## Dev Notes

### Architecture & Design Patterns

**Testing Philosophy** (Tech-Spec Section: Test Strategy Summary):
- **Test Pyramid**: Strong unit test foundation (>80% coverage) + focused integration tests + minimal E2E tests
- **Fast Feedback**: Unit tests execute in <10s, full suite in <60s
- **Deterministic**: No flaky tests; all tests use fixtures/mocks for consistency
- **Maintainable**: Tests document expected behavior; serve as living specification

**Test Framework Stack**:
- **pytest**: Primary test runner (already in use)
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance benchmarking (optional)
- **unittest.mock**: Mocking for isolation
- **fixtures**: Shared test setup via pytest fixtures

**Current Test Status** (From existing implementation):
- ✅ `tests/test_static_region_manager.py` - 30 tests (comprehensive)
- ✅ `tests/test_feature_extractor.py` - 20 tests (comprehensive)
- ✅ `tests/test_movement_detector.py` - 25 tests (comprehensive)
- ✅ `tests/test_result_manager.py` - 40 tests (comprehensive)
- ✅ `tests/test_camera_movement_detector.py` - 35 tests (comprehensive)
- ✅ `tests/test_select_roi.py` - 20 tests (tool testing)
- ✅ `tests/test_recalibrate.py` - 21 tests (tool testing)
- **Total**: 191 existing tests

**Gaps to Address**:
- Integration tests for multi-component workflows
- End-to-end tests using real sample images
- Performance benchmarks for process_frame()
- Test documentation and coverage reporting

### Implementation Guidance

**Test Organization**:
```
tests/
├── __init__.py
├── conftest.py                    # Shared pytest fixtures
├── fixtures/                      # Test data
│   ├── configs/                   # Sample config.json files
│   ├── images/                    # Test images (baseline, shifted)
│   └── features/                  # Serialized feature data
├── test_static_region_manager.py  ✅ (existing)
├── test_feature_extractor.py      ✅ (existing)
├── test_movement_detector.py      ✅ (existing)
├── test_result_manager.py         ✅ (existing)
├── test_camera_movement_detector.py ✅ (existing)
├── test_select_roi.py             ✅ (existing)
├── test_recalibrate.py            ✅ (existing)
├── test_integration.py            🆕 (new - component interactions)
├── test_e2e.py                    🆕 (new - full workflows)
├── test_performance.py            🆕 (new - benchmarks)
└── README.md                      🆕 (new - test documentation)
```

**Integration Test Examples**:
```python
# tests/test_integration.py
def test_complete_detection_pipeline():
    """Verify full pipeline: config → init → baseline → process_frame"""
    # Setup
    detector = CameraMovementDetector('tests/fixtures/configs/valid.json')
    baseline_image = load_image('tests/fixtures/images/baseline.jpg')
    test_image = load_image('tests/fixtures/images/shifted_2px.jpg')

    # Execute
    detector.set_baseline(baseline_image)
    result = detector.process_frame(test_image, frame_id="test_001")

    # Verify
    assert result['status'] == 'INVALID'
    assert result['displacement'] >= 2.0
    assert 'frame_id' in result


def test_recalibration_resets_baseline():
    """Verify recalibration updates baseline and clears history"""
    detector = CameraMovementDetector('tests/fixtures/configs/valid.json')
    detector.set_baseline(original_baseline)

    # Detect movement
    result1 = detector.process_frame(shifted_image)
    assert result1['status'] == 'INVALID'

    # Recalibrate with shifted image as new baseline
    success = detector.recalibrate(shifted_image)
    assert success == True

    # Verify new baseline - same image should now be VALID
    result2 = detector.process_frame(shifted_image)
    assert result2['status'] == 'VALID'
```

**E2E Test Examples**:
```python
# tests/test_e2e.py
def test_operator_workflow_with_real_images():
    """Test complete operator workflow using sample images"""
    # Use real sample images from sample_images/
    sample_dir = Path('sample_images/of_jerusalem')
    images = sorted(sample_dir.glob('*.jpg'))

    # 1. Setup (simulated config from ROI selection)
    config = create_test_config(roi={'x': 100, 'y': 50, 'width': 400, 'height': 300})

    # 2. Initialize detector
    detector = CameraMovementDetector(config)

    # 3. Capture baseline
    baseline_image = cv2.imread(str(images[0]))
    detector.set_baseline(baseline_image)

    # 4. Process subsequent frames
    results = []
    for img_path in images[1:5]:
        image = cv2.imread(str(img_path))
        result = detector.process_frame(image)
        results.append(result)

    # 5. Verify history
    history = detector.get_history()
    assert len(history) == 4

    # 6. Verify all frames are VALID (static camera setup)
    for result in results:
        assert result['status'] == 'VALID'
        assert result['displacement'] < 2.0
```

**Performance Test Examples**:
```python
# tests/test_performance.py
def test_process_frame_performance(benchmark):
    """Verify process_frame() executes in <500ms (AC from Tech Spec)"""
    detector = setup_detector_with_baseline()
    test_image = load_standard_test_image()

    result = benchmark(detector.process_frame, test_image)

    # Benchmark automatically measures execution time
    # Verify result is valid while benchmarking
    assert 'status' in result
    assert result['status'] in ['VALID', 'INVALID']


def test_set_baseline_performance(benchmark):
    """Verify set_baseline() executes in <2s (AC from Tech Spec)"""
    detector = CameraMovementDetector('tests/fixtures/configs/valid.json')
    baseline_image = load_standard_test_image()

    benchmark(detector.set_baseline, baseline_image)

    # No assertion needed - benchmark fails if >2s threshold exceeded
```

**Coverage Configuration** (`pytest.ini`):
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --cov=src
    --cov-report=html
    --cov-report=term
    --cov-fail-under=80
    -v
```

**Test Execution Commands**:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/test_integration.py

# Run performance benchmarks
pytest tests/test_performance.py --benchmark-only

# Run tests in parallel (faster)
pytest -n auto
```

### Project Structure Notes

**Test Data Storage**:
- `tests/fixtures/configs/` - Sample config.json files (valid, invalid, edge cases)
- `tests/fixtures/images/` - Test images (baseline, shifted_2px, feature_poor, etc.)
- `sample_images/` - Real images from DAF sites (used for E2E tests)

**Coverage Reporting**:
- `htmlcov/` - HTML coverage reports (generated by pytest-cov)
- `.coverage` - Coverage data file (generated by pytest-cov)
- Coverage reports exclude: `tests/`, `tools/`, `__pycache__/`

**CI/CD Integration**:
- Test command: `pytest --cov=src --cov-fail-under=80`
- Exit code 0 = all tests pass + coverage >80%
- Exit code non-zero = test failures or coverage <80%

### Testing Standards

**Test Categories**:

1. **Unit Tests** - Test individual components in isolation
   - Mock external dependencies
   - Fast execution (<10ms per test)
   - Deterministic (no randomness)
   - Target: >80% code coverage

2. **Integration Tests** - Test component interactions
   - Minimal mocking (only external I/O)
   - Medium execution (<100ms per test)
   - Verify contracts between components
   - Target: All public APIs tested

3. **End-to-End Tests** - Test complete workflows
   - No mocking (real components)
   - Slower execution (<1s per test)
   - Use real sample images
   - Target: Critical user workflows covered

4. **Performance Tests** - Verify performance requirements
   - Use pytest-benchmark
   - Compare against Tech Spec targets (<500ms process_frame, <2s set_baseline)
   - Run separately from main test suite

**Test Naming Conventions**:
- Unit tests: `test_<component>_<behavior>()`
- Integration tests: `test_<workflow>_<scenario>()`
- E2E tests: `test_<user_story>_with_<condition>()`
- Performance tests: `test_<operation>_performance()`

**Assertion Patterns**:
```python
# Verify exact values
assert result == expected_value

# Verify ranges
assert 0.0 <= confidence <= 1.0

# Verify structure
assert 'status' in result
assert result['status'] in ['VALID', 'INVALID']

# Verify exceptions
with pytest.raises(ValueError) as exc_info:
    detector.process_frame(invalid_image)
assert "invalid format" in str(exc_info.value)
```

### References

- [Source: tech-spec-epic-MVP-001.md#Test Strategy Summary] - Testing framework, levels, coverage targets
- [Source: tech-spec-epic-MVP-001.md#Acceptance Criteria] - AC-001 to AC-011 must all be testable
- [Source: tech-spec-epic-MVP-001.md#NFR → Performance] - Performance targets (<500ms, <2s, <10ms)
- [Source: tech-spec-epic-MVP-001.md#NFR → Reliability] - Error handling requirements and failure modes
- [Source: existing test files] - 191 existing tests provide foundation and patterns to follow

## Dev Agent Record

### Context Reference

- docs/stories/story-context-1.1.8.xml (Generated: 2025-10-23)

### Agent Model Used

claude-sonnet-4-5-20250929

### Debug Log References

### Completion Notes List

### File List
