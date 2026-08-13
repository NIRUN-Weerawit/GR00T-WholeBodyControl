#!/bin/bash
# Run G1 C++ bridge with ZMQ input in Docker
# This receives commands from Quest 3 via quest_to_g1_bridge.py
#
# Architecture:
#   Quest 3 → MQTT → quest_to_g1_bridge.py → ZMQ (5556) → g1_deploy_onnx_ref → MuJoCo
#
# Usage:
#   1. Run this script first
#   2. Run quest_to_g1_bridge.py: python3.11 scripts/quest_to_g1_bridge.py --mqtt_broker "mqtt://sora2.uclab.jp:1883"
#   3. Run MuJoCo simulation: python gear_sonic/scripts/run_sim_loop.py

cd "$(dirname "$0")"

echo "🔄 Starting G1 Deploy with ZMQ input..."
echo "   This bridge receives commands from Quest 3 via MQTT → ZMQ"
echo ""
echo "📋 Architecture:"
echo "   Quest 3 → MQTT → quest_to_g1_bridge.py → ZMQ (5556) → g1_deploy_onnx_ref → MuJoCo"
echo ""
echo "⚠️  Make sure to run these in separate terminals:"
echo "   1. This script (C++ bridge)"
echo "   2. python3.11 scripts/quest_to_g1_bridge.py --mqtt_broker \"mqtt://sora2.uclab.jp:1883\""
echo "   3. python gear_sonic/scripts/run_sim_loop.py (MuJoCo simulation)"
echo ""

# Stop any existing container
docker stop g1-zmq 2>/dev/null || true

# Run with proper terminal access
# Use --verbose and --zmq-verbose to see ZMQ message logs
docker run -it --rm \
    --gpus all \
    --network host \
    -v $(pwd):/workspace/g1_deploy \
    -e LD_LIBRARY_PATH=/workspace/g1_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64 \
    --name g1-zmq \
    g1-deploy-tensorrt10:latest \
    bash -c "source /opt/ros/humble/setup.bash && cd /workspace/g1_deploy && ./target/release/g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \
        --obs-config policy/release/observation_config.yaml \
        --encoder-file policy/release/model_encoder.onnx \
        --planner-file planner/target_vel/V2/planner_sonic.onnx \
        --input-type zmq_manager \
        --zmq-port 5556 \
        --output-type zmq \
        --verbose \
        --zmq-verbose \
        --disable-crc-check"