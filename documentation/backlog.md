# Project Backlog

## Current Sprint / In Progress

### Stage 3: [Current Work]
- [ ] TBD - Define next objectives

---

## High Priority

### Camera Shift Detection
- [ ] Set ROI (Region of Interest) automatically
- [ ] Support for excluding moving objects in frame (using RANSAC or similar outlier rejection)
- [ ] Handle abrupt environmental changes (lighting on/off, sudden illumination shifts)
- [ ] Improve detection robustness across diverse lighting conditions
- [ ] Optimize threshold parameters for different agricultural scenes
- [ ] Add confidence scoring for detection results
- [ ] Handle edge cases (occlusions, extreme weather)

### Testing & Validation
- [ ] Launch cam-shift-detector alongside Charuco/Aruco detector and compare results for live testing mode
- [ ] Add tools to analyze quality of feature tracking and matching
- [ ] Create feature matching visualization tool (side-by-side images with detected features, matches, and homography info)
- [ ] Expand test coverage beyond current sample images
- [ ] Add unit tests for core detection algorithms
- [ ] Create benchmark suite for performance metrics
- [ ] Validate across all three DAF sites (OF_JERUSALEM, CARMIT, GAD)

### Documentation
- [ ] Document detection algorithm details
- [ ] Add usage examples and tutorials
- [ ] Create troubleshooting guide
- [ ] Document calibration procedures

---

## Medium Priority

### Performance Optimization
- [ ] Profile detection pipeline for bottlenecks
- [ ] Optimize feature extraction for speed
- [ ] Implement batch processing for multiple images
- [ ] Add GPU acceleration support

### Features
- [ ] Add calibration report generation during calibration process
- [ ] Re-establish baseline using slow moving average or periodic reset when natural drift is small
- [ ] Add support for video stream processing
- [ ] Implement real-time shift detection
- [ ] Create visualization tools for detection results
- [ ] Add export functionality for detection reports

### Integration
- [ ] Create API for shift detection service
- [ ] Add support for additional image formats
- [ ] Integrate with DAF pipeline workflows
- [ ] Create command-line interface

---

## Low Priority / Future Enhancements

### Advanced Features
- [ ] Machine learning-based detection enhancement
- [ ] Multi-camera synchronization support
- [ ] Historical shift pattern analysis
- [ ] Automated camera recalibration recommendations

### Maintenance
- [ ] Set up continuous integration
- [ ] Add automated deployment pipeline
- [ ] Create monitoring and alerting system
- [ ] Implement logging and diagnostics

### Research
- [ ] Explore alternative feature detection methods
- [ ] Investigate deep learning approaches
- [ ] Benchmark against commercial solutions
- [ ] Study temporal shift patterns across seasons

---

## Completed

### Stage 2 ✓
- [x] Achieve 100% detection rate with affine model
- [x] Optimize threshold parameters
- [x] Implement ChArUco board calibration
- [x] Add sample image dataset (50 images from DAF sites)

### Stage 1 ✓
- [x] Initial project setup
- [x] Basic camera shift detection implementation
- [x] ArUco marker detection tools

---

## Backlog Management Notes

**Last Updated**: 2025-10-27

**Priority Definitions**:
- **High**: Critical for core functionality or blocking issues
- **Medium**: Important improvements and features
- **Low**: Nice-to-have enhancements and future research

**Status Indicators**:
- `[ ]` Not started
- `[~]` In progress
- `[x]` Completed
- `[!]` Blocked

**Review Schedule**: Update weekly or after major milestones
