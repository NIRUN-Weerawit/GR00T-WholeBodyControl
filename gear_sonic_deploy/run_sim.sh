#!/bin/bash
# Run G1 Deploy in simulation mode
# Usage: ./run_sim.sh [extra_args...]

cd "$(dirname "$0")"

# The executable was built on the host, so its embedded RUNPATH contains the
# host checkout path. In Docker this repository is mounted at /workspace,
# therefore make the vendored CycloneDDS libraries discoverable explicitly.
DEPLOY_ROOT="$(pwd)"
DDS_LIB_DIR="$DEPLOY_ROOT/thirdparty/unitree_sdk2/thirdparty/lib/$(uname -m)"
if [[ ! -f "$DDS_LIB_DIR/libddsc.so.0" ]]; then
    echo "ERROR: vendored CycloneDDS library is missing: $DDS_LIB_DIR/libddsc.so.0" >&2
    exit 1
fi
export LD_LIBRARY_PATH="$DDS_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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