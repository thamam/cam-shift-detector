# Stage 1 Validation Results - Synthetic Transformations

**Test Date**: 2025-10-25T00:10:17.279776
**Total Samples**: 1225
**Accuracy Threshold**: >95%
**Result**: ✅ PASSED

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 99.92% |
| Precision | 100.00% |
| Recall | 99.91% |
| F1-Score | 99.96% |

## Confusion Matrix

| | Predicted VALID | Predicted INVALID |
|---|---|---|
| **Actual VALID** | 49 (TN) | 0 (FP) |
| **Actual INVALID** | 1 (FN) | 1175 (TP) |

## Performance by Shift Magnitude

### 10PX

- **Samples**: 392
- **Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1-Score**: 100.00%
- **Confusion**: TP=392, TN=0, FP=0, FN=0

### 2PX

- **Samples**: 392
- **Accuracy**: 99.74%
- **Precision**: 100.00%
- **Recall**: 99.74%
- **F1-Score**: 99.87%
- **Confusion**: TP=391, TN=0, FP=0, FN=1

### 5PX

- **Samples**: 392
- **Accuracy**: 100.00%
- **Precision**: 100.00%
- **Recall**: 100.00%
- **F1-Score**: 100.00%
- **Confusion**: TP=392, TN=0, FP=0, FN=0

## Failure Analysis

- **False Negatives**: 1 cases where camera movement was NOT detected

## Recommendations

✅ **GO**: Detection accuracy exceeds 95% threshold. System is ready for Stage 2 validation.