#!/bin/bash
# Run the G1 SONIC deploy with ZMQ-manager input in Docker for MuJoCo.
# It accepts protocol-compatible command, pose, and planner messages on ZMQ :5556,
# e.g. from pico_manager_thread_server.py or run_swing_episode_replayer.py.
#
# Architecture:
#   PICO manager or recorded-episode replayer → ZMQ (:5556) → g1_deploy_onnx_ref
#       → ZMQ joint actions → run_sim_loop.py → MuJoCo G1
#
# Usage:
#   1. Run this script first and wait for the deploy to initialize.
#   2. Run run_sim_loop.py in a second terminal.
#   3. Run either pico_manager_thread_server.py or run_swing_episode_replayer.py
#      in a third terminal to publish teleoperation/replay messages.

cd "$(dirname "$0")"

echo "🔄 Starting G1 SONIC deploy with ZMQ-manager input..."
echo "   Receives PICO-manager or recorded-episode messages on ZMQ :5556"
echo ""
echo "📋 Simulation architecture:"
echo "   PICO manager / episode replayer → ZMQ :5556 → SONIC deploy → ZMQ actions → MuJoCo"
echo ""
echo "⚠️  Run these in separate terminals:"
echo "   1. This script (SONIC deploy; wait for initialization)"
echo "   2. python gear_sonic/scripts/run_sim_loop.py (MuJoCo simulation)"
echo "   3. pico_manager_thread_server.py or run_swing_episode_replayer.py (ZMQ publisher)"
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
    bash -c "source /opt/ros/humble/setup.bash && cd /workspace/g1_deploy && ./target/release/g1_deploy_onnx_ref lo policy/low_latency/model_decoder.onnx reference/example/ \
        --obs-config policy/low_latency/observation_config.yaml \
        --encoder-file policy/low_latency/model_encoder.onnx \
        --planner-file planner/target_vel/V2/planner_sonic.onnx \
        --input-type zmq_manager \
        --zmq-port 5556 \
        --output-type zmq \
        --verbose \
        --zmq-verbose \
        --disable-crc-check"