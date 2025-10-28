#!/bin/bash
# Online Comparison Tool - Live Camera Test
#
# Usage: ./run_online_comparison.sh [camera_id]
# Default camera_id: 0

CAMERA_ID=${1:-0}
OUTPUT_DIR="online_test_results_$(date +%Y%m%d_%H%M%S)"

echo "🎬 Starting Online Comparison Tool"
echo "📹 Camera ID: $CAMERA_ID"
echo "📁 Output: $OUTPUT_DIR"
echo ""
echo "Instructions:"
echo "  1. Position ChArUco board in camera view"
echo "  2. Press 's' to set baseline"
echo "  3. Move the camera or board"
echo "  4. Press 'q' to quit and generate reports"
echo ""

python tools/validation/comparison_tool.py \
    --mode online \
    --camera-id "$CAMERA_ID" \
    --camera-yaml camera.yaml \
    --charuco-config comparison_config.json \
    --camshift-config config_session_001.json \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "✅ Session complete! Results saved to: $OUTPUT_DIR"
