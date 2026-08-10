#!/bin/bash
# Run G1 simulation with keyboard control in Docker
# This script provides proper terminal access for keyboard input

cd "$(dirname "$0")"

echo "🎮 Starting G1 Deploy with keyboard control..."
echo "   Make sure to run this in an interactive terminal!"
echo ""

# Stop any existing container
docker stop g1-keyboard 2>/dev/null || true

# Run with proper terminal access for keyboard input
docker run -it --rm \
    --gpus all \
    --network host \
    -v $(pwd):/workspace/g1_deploy \
    --name g1-keyboard \
    g1-deploy-tensorrt10:latest \
    bash -c "source /opt/ros/humble/setup.bash && cd /workspace/g1_deploy && ./run_sim.sh"