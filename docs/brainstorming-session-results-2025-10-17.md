# Brainstorming Session Results

**Session Date:** 2025-10-17
**Facilitator:** AI Brainstorming Facilitator Claude
**Participant:** Tomer

## Executive Summary

**Topic:** MVP PRD Optimization - Reducing Density & Preventing Overengineering

**Session Goals:**
- Restructure existing 2000+ line PRD into lightweight, implementation-focused version
- Strip down to absolute minimal viable features for cam-shift detection in DAF environment
- Create digestible format that engineers can actually build from without scope creep
- Prevent document abandonment by reducing overwhelming detail
- Maintain technical accuracy where needed, freedom to simplify elsewhere
- Primary audience: Engineers (implementation team)

**Techniques Used:**
1. First Principles Thinking (15 min)
2. Resource Constraints (15 min)
3. Assumption Reversal (15 min)

**Total Ideas Generated:** 30+ ideas/decisions

### Key Themes Identified:

**Theme 1: YAGNI Philosophy**
- Ship simple working version first, add features based on real performance data
- Avoid theoretical concerns and overengineering
- Accept limitations for MVP, iterate based on actual problems

**Theme 2: Black-Box Module Design**
- Clean API boundaries between cam-shift-detector and DAF system
- Direct return values instead of side effects (flag files)
- History buffer for queryability without complexity

**Theme 3: Manual Over Automatic**
- Manual recalibration only (defer automatic drift detection)
- Manual static region definition (defer ML-based auto-detection)
- Manual triggers instead of complex autonomous systems

**Theme 4: Ruthless Scope Reduction**
- From 2000+ lines → ~400 lines (80% reduction)
- From 5 weeks → 2 weeks timeline
- Eliminated: UI, logging, RANSAC (initially), auto-recalibration, real-time monitoring

## Technique Sessions

### Session 1: First Principles Thinking (15 min)

**Goal:** Strip away assumptions and rebuild from fundamental truths

**Core Problem Statement:**
Camera moves → ROI misalignment → Neural net measures wrong region → Bad water flow/turbidity data

**The Solution (Minimum):**
Detect movement > threshold → Alert → Manual fix (reposition OR recalibrate)

**4 Fundamental Difficulty Elements Identified:**

1. **Static Region Identification** - Avoid water/moving elements, handle site-specific layouts
2. **Static Region Tracking** - Distinguish camera movement from scene movement
3. **Feature Robustness** - Stable across lighting/environmental changes
4. **Periodic Recalibration** - Handle outdoor lighting drift without undetected camera drift

**BREAKTHROUGH: MVP Scope Decision**

| Element | MVP Status | Rationale |
|---------|-----------|-----------|
| 1. Static Region ID | ✅ MUST SOLVE | Operator draws static region manually during setup (no generalization) |
| 2. Static Region Tracking | ✅ MUST SOLVE | Core detection problem - RANSAC/homography |
| 3. Feature Robustness | ⚠️ SIMPLIFY | ORB features, accept lighting sensitivity limitations |
| 4. Periodic Recalibration | 🚫 DEFER | Manual recalibration only. No automatic drift handling. |

**Impact:** Eliminated ~30-40% of PRD complexity by removing automatic recalibration system

**1-Week MVP Validated:**
```
SETUP: Camera → Live feed → Operator clicks static region → Extract ORB baseline
RUNTIME: Every 1s → Extract features → RANSAC → Displacement > 2px? → Alert + Flag
RECALIBRATION: Manual "Recapture Baseline" button when needed
VALIDATION: Simulated transforms + Real shifts + Live site video
```

**Key Decision:** YAGNI philosophy - Ship simple working version, add features based on real performance data

---

### Session 2: Resource Constraints (15 min)

**Goal:** Force essential priorities with extreme limitations (2-week / 1-engineer constraint)

**Constraint Scenario:** Ship working detection in 2 weeks or lose pilot site

**Feature Cuts & Simplifications:**

| Feature | Original Scope | 2-Week MVP |
|---------|---------------|------------|
| Static Region UI | Interactive drawing tool | ❌ Manual hardcoded coordinates |
| Visual Alarm | Qt5/Flask UI | ⚠️ Terminal print → REST API hook |
| Data Validity Flag | File-based system | ✅ Single source of truth with alarm |
| Recalibration | 4-step wizard + auto-recal | ❌ **Option C: Manual only** (operator script/button) |
| Event Logging | SQLite + diagnostics | ❌ Minimal/none, add ad-hoc |
| RANSAC Detection | Full implementation | ✅ KEEP (cheap & reliable) |

**Testing Scope Reductions:**

- ❌ **CUT:** Bubbles/water dynamics testing (PRD error, not actual requirement)
- ❌ **CUT:** Varying water conditions tests (defer to post-MVP)
- ❌ **CUT:** Lab validation setup (nice-to-have, not blocking)
- ✅ **KEEP:** Simulated transforms + Real shifts + Live video validation

**Recalibration Decision: Option C (Manual Only)**
- Rationale: Options A & B feel "hacky and flaky" - would waste time on something that might not work
- Impact: Simplest path, eliminates Element 4 complexity entirely
- Trade-off: Limits outdoor deployment for MVP (acceptable)

**Documentation Cuts:**
- ❌ User Guide, Developer Guide, RANSAC Explanation, Testing Report, Deployment Guide
- ✅ Single README with setup instructions only

**Code Structure Decision:**
- ✅ **Option C: Modular OOP from Day 1** (detection_engine, camera_manager, ransac_detection, etc.)
- Rationale: Team works faster with structure; avoids tech debt refactoring later
- Trade-off: Slightly more upfront design, but natural working style

---

### Session 3: Assumption Reversal (15 min)

**Goal:** Challenge hidden assumptions to reveal what's NOT actually required

**Assumptions Challenged:**

**1. "We need RANSAC for robust detection"**
- ✅ **REVERSED:** Start with simple homography estimation first
- Only add RANSAC if simple matching shows insufficient results in testing
- Impact: Faster initial implementation, add complexity only if proven necessary

**2. "We need real-time (1 Hz) continuous monitoring"**
- ✅ **CHALLENGED:** Periodic checks are sufficient, max 10-minute intervals
- Could be every 5-10 minutes instead of every second
- Impact: Reduced resource usage, simpler deployment

**3. "We need REST API integration for alerts"**
- ✅ **REVERSED:** REST API handled by other side, NOT part of MVP
- This MVP only writes status to flag/file that other system reads
- Impact: Zero integration work needed for MVP

**4. "Static region definition without UI"**
- ⚠️ **PROBLEM IDENTIFIED:** No UI on site, can't do manual coordinate definition
- ✅ **SOLUTION:** Two-path approach leveraging existing infrastructure

**Static Region Setup - Two Options:**

**Option 1: Local Laptop Setup**
- Run ROI selection tool on laptop with GUI
- Click/draw static region on live camera feed or sample image
- Save coordinates to config.json
- Deploy config to site

**Option 2: Remote Setup via Cloud Images**
- Site already uploads images to cloud every 10 minutes (existing mechanism)
- Download recent cloud image
- Run ROI selection tool offline on downloaded image
- Publish ROI coordinates back to site config
- No need for on-site UI or SSH access

**Impact:** Reuses existing cloud infrastructure, zero new deployment mechanisms needed

---

{{technique_sessions}}

## Idea Categorization

### ✅ MUST HAVE - Core MVP Features

**Absolute essentials that prove "camera movement detection works":**

1. ~~Camera frame capture~~ **ALREADY EXISTS** - Existing interface provides image arrays
2. **Static region definition tool** - Simple GUI for ROI selection (local laptop or cloud-based workflow)
3. **ORB feature extraction** - From static region only
4. **Simple homography estimation** - Start without RANSAC (add only if needed)
5. **Movement detection logic** - Threshold check (> 2 pixels = movement detected)
6. **Flag file output** - Write VALID/INVALID status for external system to poll
7. **Manual recalibration** - Simple script/button to recapture baseline
8. **Basic validation** - Test with simulated transforms + real shifts + live video

**Total Scope:** 7 core components (camera interface already exists)


### 🔮 FUTURE INNOVATIONS - Defer to Post-MVP

**Consider AFTER simple version proves itself:**

1. **RANSAC homography** - Add only if simple matching shows too many false positives
2. **Automatic recalibration** - Time-based or multi-baseline matching (solve drift paradox)
3. **Optimized monitoring frequency** - Move from 5-10 min to 1-second intervals if needed
4. **REST API integration** - Direct API calls instead of flag file polling
5. **Scene quality monitoring** - Inlier ratio tracking, diagnostics, warnings
6. **Multi-site generalization** - Auto-detect static regions across different DAF layouts
7. **Comprehensive test suite** - Dynamic scene tests, lab validation, operator surveys
8. **Full documentation suite** - User guides, developer guides, RANSAC explanations, deployment guides

### ❌ NOT NEEDED - Cut Entirely

**Recognized as out of scope or unnecessary:**

1. **Automatic periodic recalibration with drift detection** - Too complex, too risky for MVP
2. **Bubble/water dynamics testing** - PRD error, not actual requirement
3. **4-step recalibration wizard UI** - Overkill, simple button/script sufficient
4. **Multi-camera support** - Single camera only for MVP
5. **Interactive static region masking UI on-site** - Solved via cloud image workflow
6. **Real-time 1Hz continuous monitoring** - Periodic checks (5-10 min) sufficient
7. **Rich UI (Qt5/Flask)** - Out of scope entirely, no visual interface needed
8. **SQLite event logging** - Out of scope entirely, minimal/no logging for MVP

### Insights and Learnings

_Key realizations from the session_

**Insight 1: Dense PRDs Create Scope Creep**
The original 2000+ line PRD wasn't just documentation bloat - it was actively creating scope creep by including features that seemed "required" but weren't essential for proving the concept works.

**Insight 2: Camera Infrastructure Already Exists**
Discovering that camera frame capture already exists eliminated an entire component from the MVP. Always verify existing infrastructure before designing from scratch.

**Insight 3: The Drift Paradox is Real**
Element 4 (periodic recalibration vs. drift detection) is genuinely hard. Manual-only recalibration for MVP is the right call - solve the simpler problem first, tackle drift later with real data.

**Insight 4: Interfaces Matter More Than Implementation**
Spending time on the black-box interface contract was more valuable than detailing internal algorithms. Clear boundaries enable independent development and testing.

**Insight 5: Flag Files are Side Effects**
Direct return values are cleaner than side effects (flag files). The API redesign from flag file to `process_frame()` return makes integration simpler and more predictable.

**Insight 6: Start Simple, Add RANSAC Only If Needed**
Assumption reversal revealed that RANSAC might not be necessary if static regions are truly static. Test with simple homography first - add complexity only when proven necessary.

## Action Planning

### Top 3 Priority Ideas

#### #1 Priority: Implement Simplified MVP PRD

- **Rationale:** New PRD is 80% smaller (400 lines vs 2000+), 2-week timeline instead of 5 weeks, focuses on proving core detection works
- **Next steps:**
  1. Review and approve simplified PRD (docs/MVP_Camera_Movement_Detection_SIMPLIFIED.md)
  2. Set up project structure with modular OOP design
  3. Implement ROI selection tool (leverage cloud image workflow)
  4. Build CameraMovementDetector class with process_frame() API
- **Resources needed:** 1 engineer, 2 weeks, access to cloud images for ROI setup
- **Timeline:** Week 1: Core detection + ROI tool | Week 2: Integration + validation testing

#### #2 Priority: Define Static Region for Initial Site

- **Rationale:** Static region definition is critical blocker - must be done before any detection can work. Using cloud image workflow makes this possible remotely.
- **Next steps:**
  1. Download recent image from cloud (site already uploads every 10 minutes)
  2. Run ROI selection tool on downloaded image (OpenCV GUI)
  3. Identify static elements: tank walls, pipes, equipment (NOT water surface)
  4. Validate ≥50 features detected in static region
  5. Save coordinates to config.json
  6. Deploy config to site
- **Resources needed:** Recent cloud image, laptop with GUI, 30 minutes
- **Timeline:** Can be done immediately once ROI tool is built

#### #3 Priority: Validation Testing Strategy

- **Rationale:** Three-stage validation (simulated + real + live) is lightweight but comprehensive. Proves detection works without extensive test infrastructure.
- **Next steps:**
  1. **Stage 1:** Create 20-30 test images with known camera shifts (2px, 5px, 10px)
  2. **Stage 2:** Use existing recordings where camera actually moved (if available)
  3. **Stage 3:** Deploy to 1 site, monitor for 1 week, manually verify all alerts
  4. Measure: detection accuracy, false positive rate, false negative rate
  5. Go/No-Go decision based on: >95% accuracy, <5% false positives, 0% missed movements
- **Resources needed:** Test images, 1 pilot site, 1 week monitoring period
- **Timeline:** Stage 1-2 during development (week 1-2) | Stage 3 after deployment (week 3)

## Reflection and Follow-up

### What Worked Well

**First Principles Thinking:**
- Immediately cut through 2000 lines to reveal the core problem: camera moves → ROI misalignment → bad data
- Breaking down into 4 fundamental difficulty elements was extremely clarifying
- Enabled ruthless prioritization (Element 4 deferred entirely)

**Resource Constraints:**
- The "2-week / 1-engineer" constraint forced honest assessment of what's truly essential
- Made it obvious which features were theoretical concerns vs. actual requirements
- Revealed existing infrastructure (camera interface already exists)

**Assumption Reversal:**
- Challenged RANSAC necessity - exposed opportunity to start simpler
- Questioned flag file approach - led to cleaner API design
- Revealed cloud image workflow as elegant solution for remote ROI setup

**Session Flow:**
- Progressive techniques built on each other naturally
- Each technique revealed different angles on the same problem
- Concrete decisions made at each step (not just ideas generated)

### Areas for Further Exploration

**1. RANSAC Performance Testing**
Once simple homography is implemented, systematically test false positive rates with actual DAF footage to determine if/when RANSAC is needed.

**2. Static Region Optimization**
Explore how to identify optimal static regions across different DAF site configurations. Could patterns emerge for "good" vs. "bad" ROI placement?

**3. Lighting Drift Handling**
After MVP proves itself, revisit Element 4 (drift detection). Collect real data on how quickly lighting changes affect feature matching. Design automatic recalibration strategy based on actual drift patterns, not theoretical concerns.

**4. Multi-Site Generalization**
Once system works at 1 site, explore what's needed to generalize across sites with different cameras, angles, DAF configurations.

### Recommended Follow-up Techniques

**For Implementation Planning:**
- **Mind Mapping** - Break down each component (ROI tool, feature extractor, detector) into implementation tasks
- **SCAMPER Method** - Systematically improve the API design before locking it in

**For Testing Strategy:**
- **Five Whys** - If false positives occur, drill into root causes
- **Morphological Analysis** - Systematically explore all combinations of test parameters (lighting, movement magnitude, ROI size)

**For Future Features:**
- **First Principles Thinking** (again) - Before adding RANSAC or auto-recalibration, rebuild from fundamentals with real data

### Questions That Emerged

**Technical Questions:**
1. What's the actual false positive rate with simple homography in real DAF environments? (Need real data)
2. How many features are typically detectable in static regions across different sites? (≥50 assumption needs validation)
3. How quickly does lighting drift affect feature matching? (Element 4 deferred but not forgotten)
4. What's the optimal history buffer size? (100 is arbitrary - may need tuning)

**Integration Questions:**
1. How will DAF system call `process_frame()` - synchronous or async?
2. What happens if detector raises exception? (Error handling strategy needed)
3. Should history buffer persist across restarts or start fresh? (Deferred for MVP)

**Operational Questions:**
1. How will operators know when to manually recalibrate? (Lighting changes, maintenance, etc.)
2. What's the failure mode if static region is poorly defined? (How obvious will it be?)
3. Should there be a "confidence score" in addition to VALID/INVALID? (Deferred for MVP)

### Next Session Planning

**Suggested Topics:**

1. **Implementation Task Breakdown** (1 hour)
   - Use Mind Mapping to break down each component into specific coding tasks
   - Identify dependencies between components
   - Create 2-week sprint plan with daily milestones

2. **API Contract Finalization** (30 minutes)
   - Review black-box interface with team
   - Discuss error handling strategies
   - Mock integration with DAF system to validate design

3. **Test Data Preparation Strategy** (45 minutes)
   - How to create realistic simulated camera transforms
   - Identify existing recordings with known camera movements
   - Select pilot site for Stage 3 live testing

**Recommended Timeframe:**
- Implementation planning: Within 1-2 days (before coding starts)
- API finalization: Before week 1 of development
- Test data prep: During week 1 (parallel with development)

**Preparation Needed:**
- Download sample images from cloud (multiple sites, times of day)
- Review existing DAF system integration points
- Confirm availability of 1 engineer for 2-week focused work
- Identify pilot site willing to participate in 1-week monitoring trial

---

_Session facilitated using the BMAD CIS brainstorming framework_
