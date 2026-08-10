#!/bin/bash
# Run G1 Deploy in simulation mode
# Usage: ./run_sim.sh [extra_args...]

cd "$(dirname "$0")"

# Configuration
TARGET="lo"
CHECKPOINT_DECODER="policy/release/model_decoder.onnx"
CHECKPOINT_ENCODER="policy/release/model_encoder.onnx"
MOTION_DATA="reference/example/"
OBS_CONFIG="policy/release/observation_config.yaml"
PLANNER="planner/target_vel/V2/planner_sonic.onnx"
INPUT_TYPE="keyboard"
OUTPUT_TYPE="zmq"
ZMQ_HOST="localhost"
EXTRA_ARGS="--disable-crc-check $@"

echo "🚀 Starting G1 Deploy in simulation mode..."
echo "   Target: $TARGET"
echo "   Motion Data: $MOTION_DATA"
echo ""

# Source ROS2 and run
source /opt/ros/humble/setup.bash 2>/dev/null || true

./target/release/g1_deploy_onnx_ref "$TARGET" "$CHECKPOINT_DECODER" "$MOTION_DATA" \
    --obs-config "$OBS_CONFIG" \
    --encoder-file "$CHECKPOINT_ENCODER" \
    --planner-file "$PLANNER" \
    --input-type "$INPUT_TYPE" \
    --output-type "$OUTPUT_TYPE" \
    --zmq-host "$ZMQ_HOST" \
    $EXTRA_ARGS