# cam-shift-detector Project

## Sample Images for Testing

### Overview
A diverse sample of 50 real images from DAF (Digital Agriculture Framework) sites has been added to the project for testing camera shift detection features. These images were systematically sampled from the greenpipe datasets to ensure representative coverage.

### Dataset Distribution

**Total Images**: 50 (systematically sampled from 30,205 available images)

#### 1. OF_JERUSALEM Dataset
- **Location**: `sample_images/of_jerusalem/`
- **Count**: 23 images
- **Source**: 9bc4603f-0d21-4f60-afea-b6343d372034 (14,039 total images)
- **Sampling**: Every 610th image (systematic sampling)

#### 2. CARMIT Dataset
- **Location**: `sample_images/carmit/`
- **Count**: 17 images
- **Source**: e2336087-0143-4895-bc12-59b8b8f97790 (10,430 total images)
- **Sampling**: Every 613th image (systematic sampling)

#### 3. GAD Dataset
- **Location**: `sample_images/gad/`
- **Count**: 10 images
- **Source**: f10f17d4-ac26-4c28-b601-06c64b8a22a4 (5,736 total images)
- **Sampling**: Every 574th image (systematic sampling)

### Purpose
These sample images provide real-world data for:
- Testing camera shift detection algorithms
- Validating feature extraction and matching
- Benchmarking performance on actual DAF site imagery
- Ensuring algorithms work on diverse agricultural scenes

### Source Data
All images sourced from: `/home/thh3/data/greenpipe/`

Dataset mappings defined in: `/home/thh3/data/greenpipe/datasets_uuid_legend.md`

---

## Project Scope & YAGNI Enforcement

**Backlog Location**: [documentation/backlog.md](documentation/backlog.md)

### Scope Discipline Guidelines

When working on features or receiving requests for enhancements:

1. **Stay Focused on Current Objectives**: Build only what is explicitly required for the current task or sprint
2. **Defer Sophisticated Solutions**: If a proposed solution seems oversophisticated, complex, or beyond immediate needs → **Add to backlog instead**
3. **YAGNI Principle**: "You Aren't Gonna Need It" - Resist the urge to add features "just in case" or for future convenience
4. **Backlog for Future Work**: Use the backlog to capture ideas, enhancements, and sophisticated features for proper prioritization

### When to Add to Backlog vs. Implement Now

**Add to Backlog** ✓
- Feature is sophisticated but not immediately needed
- Enhancement would be nice-to-have but isn't blocking current work
- Optimization that doesn't address a measured performance issue
- Architectural refactoring without clear current benefit
- Additional abstraction layers beyond current requirements

**Implement Now** ✓
- Directly solves the current problem or requirement
- Fixes a blocking bug or issue
- Required for current sprint/stage objectives
- Measured performance bottleneck with clear impact
- Simplifies existing complexity (not adds new layers)

### Example Scenarios

**Scenario**: "While implementing ROI selection, should we add ML-based auto-detection?"
- **Decision**: Add ML approach to backlog, implement simple geometric ROI for now

**Scenario**: "Should we build a full configuration management system?"
- **Decision**: Add to backlog, use simple config file for current needs

**Scenario**: "Add real-time video processing alongside single image detection?"
- **Decision**: Add to backlog, focus on perfecting single image detection first

### Backlog Review

- Review backlog weekly or after completing major milestones
- Prioritize based on measured needs and user feedback
- Promote items from backlog to current sprint only when justified by actual requirements
