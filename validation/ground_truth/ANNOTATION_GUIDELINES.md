# Ground Truth Annotation Guidelines

## Objective
Manually review each of the 50 sample images and annotate whether camera movement/shift is present.

## Camera Shift Detection Criteria

### Positive Cases (has_camera_shift = true)
Camera movement is considered present if ANY of the following are observed:
1. **Translation**: Camera moved horizontally or vertically
2. **Rotation**: Camera rotated around its optical axis
3. **Pan/Tilt**: Camera view direction changed
4. **Zoom**: Camera zoom level changed between frames

### Negative Cases (has_camera_shift = false)
No camera movement when:
1. Static scene with no camera position changes
2. Only subject movement (people, vehicles, animals) with fixed camera
3. Environmental changes (wind, lighting) with stationary camera mount

## Annotation Process

### Step 1: Visual Inspection
For each image:
1. Open image in viewer
2. Assess overall scene composition
3. Look for indicators of camera stability:
   - Fixed horizon lines
   - Stable background reference points
   - Consistent field of view

### Step 2: Confidence Assessment
Assign confidence level:
- **high**: Clear indicators, no ambiguity
- **medium**: Some indicators present, minor uncertainty
- **low**: Difficult to determine, requires additional context

### Step 3: Notes (Optional)
Document any observations:
- Specific movement patterns noticed
- Ambiguous cases requiring discussion
- Image quality issues

## Expected Distribution

Based on Stage 2 validation results (97.58% detection rate):
- **Expect**: Most images will NOT have camera shift (stable monitoring)
- **Expect**: ~2-5% may show camera shift patterns
- **Reality**: DAF imagery is typically from fixed security cameras

## Annotation Format

For each image, create entry in ground_truth.json:
```json
{
  "image_path": "sample_images/of_jerusalem/[uuid].jpg",
  "site_id": "of_jerusalem",
  "has_camera_shift": true|false,
  "confidence": "high|medium|low",
  "notes": "Optional: reason for classification"
}
```

## Quality Assurance

After annotation:
1. Verify all 50 images annotated
2. Check for missing entries
3. Review low-confidence cases
4. Ensure consistent criteria application

## Time Estimate
- **Expected**: 4-6 hours for 50 images
- **Rate**: ~5-7 minutes per image (including viewing and documentation)
