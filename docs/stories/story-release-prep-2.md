# Story: Integration Guide & Black-Box Documentation

Status: Complete

**PRIORITY:** Required for stakeholder meeting on 2025-10-27

## Story

As a **system integrator**,
I want **comprehensive integration documentation with clear API examples and installation instructions**,
so that **I can understand how to integrate the camera shift detection module into the larger DAF system and discuss integration approaches with confidence**.

## Acceptance Criteria

**AC1: Integration Quick Start Guide Created**
- [x] Document created: `docs/integration-guide.md`
- [x] Black-box API overview (CameraMovementDetector class) documented
- [x] 4 integration flow diagrams included (setup, runtime, recalibration, error handling)
- [x] Code examples for each integration scenario provided
- [x] Common integration patterns explained clearly

**AC2: Installation Instructions Documented**
- [x] Document created: `docs/installation.md`
- [x] Step-by-step installation from source explained
- [x] Development installation (`pip install -e .`) documented
- [x] Dependencies and environment setup detailed
- [x] Quick verification test included
- [x] Troubleshooting section for common issues

**AC3: Integration Scenarios Documented**
- [x] Scenario 1: Simple periodic check (every 5-10 min)
- [x] Scenario 2: Continuous monitoring integration
- [x] Scenario 3: Error handling and recovery flows
- [x] Scenario 4: Manual recalibration workflow
- [x] Each scenario includes complete code example
- [x] Best practices and gotchas documented

**AC4: Meeting Cheat Sheet Created**
- [x] 1-2 page quick reference created: `docs/integration-cheat-sheet.md`
- [x] API method signatures with brief descriptions
- [x] Result format schema
- [x] Installation one-liner
- [x] Most common integration pattern (copy-paste ready)
- [x] Key points for discussion highlighted

**AC5: Optional Stub Implementation Available**
- [x] Stub module created: `src/camera_movement_detector_stub.py` (optional)
- [x] All public API methods with correct signatures
- [x] Placeholder implementations returning mock data
- [x] Documentation on how to swap stub with real implementation
- [x] Mock response examples matching real result schema

## Tasks / Subtasks

**Phase 1: API Documentation Extraction (AC: #1)**
- [x] Review `docs/tech-spec-epic-MVP-001.md` Section "APIs and Interfaces"
- [x] Extract `CameraMovementDetector` public API specification
- [x] Document all 5 public methods:
  - [x] `__init__(config_path)` - Initialization
  - [x] `set_baseline(image_array)` - Baseline capture
  - [x] `process_frame(image_array, frame_id)` - Detection
  - [x] `recalibrate(image_array)` - Manual recalibration
  - [x] `get_history(frame_id, limit)` - Query history
- [x] Document result dict schema with all fields
- [x] Document configuration file structure (`config.json`)

**Phase 2: Integration Flow Diagrams (AC: #1)**
- [x] Create Setup Flow diagram (initialization + baseline capture)
- [x] Create Runtime Detection Flow diagram (periodic checks)
- [x] Create Error Handling Flow diagram (invalid status handling)
- [x] Create Recalibration Flow diagram (manual baseline reset)
- [x] Use Mermaid or text-based diagrams for simplicity

**Phase 3: Code Examples Development (AC: #1, #3)**
- [x] Scenario 1 example: Simple periodic check
  ```python
  # Basic integration - check every 5 minutes
  detector = CameraMovementDetector('config.json')
  detector.set_baseline(initial_frame)

  result = detector.process_frame(current_frame)
  if result['status'] == 'INVALID':
      # Handle shift
  ```
- [x] Scenario 2 example: Continuous monitoring
- [x] Scenario 3 example: Error handling with try/except
- [x] Scenario 4 example: Recalibration workflow
- [x] Add inline comments explaining each step

**Phase 4: Installation Documentation (AC: #2)**
- [x] Document system requirements (Python 3.11+, OS)
- [x] Write step-by-step installation from source:
  ```bash
  git clone <repo>
  cd cam-shift-detector
  pip install -e .
  ```
- [x] Document dependency installation verification
- [x] Create quick verification test script
- [x] Document common installation issues and fixes

**Phase 5: Integration Guide Assembly (AC: #1)**
- [x] Create `docs/integration-guide.md` structure:
  - [x] Introduction and purpose
  - [x] Black-box API overview
  - [x] Integration patterns section
  - [x] Flow diagrams section
  - [x] Code examples section
  - [x] Best practices section
  - [x] Troubleshooting section
- [x] Write each section with clear, concise language
- [x] Add table of contents for navigation

**Phase 6: Meeting Preparation Materials (AC: #4)**
- [x] Create `docs/integration-cheat-sheet.md`:
  - [x] API Quick Reference table
  - [x] Installation one-liner
  - [x] Most common integration code (copy-paste)
  - [x] Key discussion points for meeting
- [x] Keep to 1-2 pages max for quick reference
- [x] Highlight critical information for meeting

**Phase 7: Optional Stub Implementation (AC: #5)**
- [x] Create `src/camera_movement_detector_stub.py`
- [x] Implement stub class with correct API:
  ```python
  class CameraMovementDetector:
      def __init__(self, config_path: str):
          # Mock initialization

      def process_frame(self, image_array, frame_id=None):
          # Return mock result
          return {
              "status": "VALID",
              "displacement": 0.5,
              "confidence": 0.95,
              "frame_id": frame_id or "stub_frame",
              "timestamp": datetime.utcnow().isoformat()
          }
  ```
- [x] Add README section explaining stub vs real module
- [x] Document how to swap implementations

**Phase 8: Review and Polish**
- [x] Proofread all documentation
- [x] Verify all code examples are syntactically correct
- [x] Check diagrams render correctly
- [x] Ensure consistent terminology throughout
- [x] Get ready for meeting presentation

## Dev Notes

### Technical Summary

**Objective:** Create comprehensive integration documentation that enables external system integrators to understand and implement the camera shift detection module, with special focus on preparing for stakeholder meeting on 2025-10-27.

**Key Technical Decisions:**
- **Documentation Format:** Markdown for portability and GitHub compatibility
- **Code Examples:** Python with inline comments, copy-paste ready
- **Diagrams:** Text-based (Mermaid) or simple ASCII for maintainability
- **Stub Module:** Optional but recommended for immediate integration testing
- **Meeting Focus:** 1-page cheat sheet for quick reference during discussion

**Critical Path Items:**
- Meeting cheat sheet is highest priority (AC4)
- Integration guide (AC1) provides comprehensive reference
- Installation guide (AC2) enables immediate setup
- Stub implementation (AC5) allows parallel development

**Integration Points:**
- References `docs/tech-spec-epic-MVP-001.md` for authoritative API specification
- Uses existing `pyproject.toml` for installation instructions
- May reference validation results from Story 1 (if available)

### Meeting Preparation Strategy

**Materials Needed for 2025-10-27 Meeting:**

1. **Integration Cheat Sheet** (1 page) - Print or share screen
2. **Installation Guide** (reference) - If needed for setup discussion
3. **Integration Guide** (comprehensive) - Backup detailed reference
4. **Optional:** Stub module demo - Live code integration example

**Key Talking Points:**
- Black-box API is simple: 5 methods, 1 result dict format
- Installation is standard Python: `pip install -e .`
- Integration pattern: init → set_baseline → process_frame loop
- Error handling: Check `status` field in result
- Recalibration: Call `recalibrate()` when lighting changes

### Project Structure Notes

**Files to Create:**
- `docs/integration-guide.md` (comprehensive integration documentation)
- `docs/installation.md` (step-by-step installation instructions)
- `docs/integration-cheat-sheet.md` (1-page quick reference for meeting)
- `src/camera_movement_detector_stub.py` (optional stub implementation)

**Expected Deliverable Sizes:**
- Integration Guide: 10-15 pages (comprehensive with examples)
- Installation Guide: 3-5 pages (step-by-step with troubleshooting)
- Cheat Sheet: 1-2 pages (quick reference)
- Stub Module: ~100 lines (minimal working implementation)

**Estimated Effort:** 2 story points (2-3 hours: 1 hour API extraction, 1 hour examples, 1 hour documentation assembly)

### References

- **Tech Spec (Authoritative):** `docs/tech-spec-epic-MVP-001.md` Section "APIs and Interfaces"
- **Package Config:** `pyproject.toml` for dependencies
- **README:** Current project documentation for context
- **Existing Code:** `src/camera_movement_detector.py` (if exists) for API verification

### Integration Examples Source

All code examples should be based on the authoritative API specification in `tech-spec-epic-MVP-001.md`:

**CameraMovementDetector API:**
```python
class CameraMovementDetector:
    def __init__(self, config_path: str = 'config.json') -> None
    def process_frame(self, image_array: np.ndarray, frame_id: str = None) -> Dict
    def get_history(self, frame_id: str = None, limit: int = None) -> List[Dict]
    def recalibrate(self, image_array: np.ndarray = None) -> bool
    def set_baseline(self, image_array: np.ndarray) -> None
```

**Result Schema:**
```python
{
    "status": str,          # "VALID" | "INVALID"
    "displacement": float,  # pixels
    "confidence": float,    # [0.0, 1.0]
    "frame_id": str,
    "timestamp": str        # ISO 8601 UTC
}
```

## Dev Agent Record

### Context Reference

- **Story Context XML:** Not generated (proceeded with tech-spec reference)

### Agent Model Used

- **Model**: Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)
- **Date**: 2025-10-26
- **Workflow**: dev-story (BMAD Phase 4 Implementation)

### Debug Log References

No errors encountered during execution.

### Completion Notes List

**Files Created:**
1. `docs/integration-guide.md` (~600 lines) - Comprehensive integration documentation with API overview, 4 flow diagrams, 4 code examples, best practices, and troubleshooting
2. `docs/installation.md` (~400 lines) - Complete installation guide with 3 installation methods, system requirements, verification tests, and troubleshooting
3. `docs/integration-cheat-sheet.md` (~250 lines) - 1-2 page quick reference for 2025-10-27 stakeholder meeting with API table, copy-paste code, and talking points
4. `src/camera_movement_detector_stub.py` (~300 lines) - Full stub implementation with identical API, input validation, and realistic mock responses
5. `README.md` (updated) - Added Integration section documenting stub usage and quick start guide

**Implementation Highlights:**
- All 5 acceptance criteria met
- Meeting-ready cheat sheet prepared for 2025-10-27 stakeholder discussion
- Stub enables parallel integration development (zero code changes when swapping to real module)
- Three-tier documentation strategy: comprehensive guide, installation guide, quick reference
- 4 integration flow diagrams: Setup, Runtime, Error Handling, Recalibration
- 4 complete code example scenarios with inline comments
- Extracted authoritative API from tech-spec-epic-MVP-001.md
- MockCameraMovementDetector extension for configurable test responses

**Story Completion:**
- All 8 phases completed successfully
- All tasks and subtasks verified
- Ready for stakeholder meeting on 2025-10-27
