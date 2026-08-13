#!/bin/bash
# Run G1 simulation in Docker container
cd "$(dirname "$0")"

docker run -it --rm --gpus all --network host \
  -v $(pwd):/workspace/g1_deploy \
  -e LD_LIBRARY_PATH=/workspace/g1_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64 \
  --name g1-sim \
  g1-deploy-tensorrt10:latest \
  bash -c "source /opt/ros/humble/setup.bash && cd /workspace/g1_deploy && ./run_sim.sh"