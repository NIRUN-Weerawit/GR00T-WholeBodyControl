# Real G1 SONIC Deployment with Recorded Motion Playback

> **Scope:** deploy SONIC on a physical Unitree G1 and replay a recorded swing episode through `run_swing_episode_replayer.py`.
>
> **Safety:** This procedure can command a physical robot. Perform the first run only with an E-stop accessible, a clear workspace, a spotter, and the robot in a known safe standing state. First validate every new model / episode in MuJoCo.

---

## 1. Overview

### 1.1 Architecture and launch roles

```text
Recorded .npz episode
  -> run_swing_episode_replayer.py (ZMQ publisher :5556)
  -> g1_deploy_onnx_ref (SONIC encoder + decoder)
  -> ROS2 / DDS / Unitree SDK2
  -> physical G1 low-level motor commands
```
### 1.2 Available Playback Movements

1. **Walking forward about 2 meters**

<img width="300" height="300" alt="walk_forward" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/walk_forward.gif" />

2. **Walking backward about 2 meters**

|  |  |
|---|---|
| <img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/walk_backward.gif" /> | <img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/walk_backward_1.gif" /> |

3. **Waving left hand**
<img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/waving_hand.gif" />

4. **Swinging tennis racket**

|  |  |
|---|---|
| <img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/swing_1.gif" /> | <img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/swing_2.gif" /> |
| <img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/swing_3.gif" /> | <img width="300" height="300" alt="walk_backward_1" src="https://github.com/NIRUN-Weerawit/GR00T-WholeBodyControl/blob/feature/g1-camera-teleop-sim/swing_episodes/videos/swing_4.gif" /> |

Before using the recorded movement on real G1, please verify the movement in MuJoCo first. 

---

## 2. Requirements

### 2.1 Required environments

Run commands from the repository root unless a step says otherwise:

```bash
cd ~/GR00T-WholeBodyControl
```

| Component | Environment | Package |
|---|---|---|
| Real deploy binary | Docker image `g1-deploy-tensorrt10:latest` | TensorRT, ROS2, Unitree SDK2, GPU runtime |
| Recorded-episode replayer | `.venv_data_collection` | Python, NumPy, ZMQ, Tyro |
| Live PICO manager, if used instead of replay | `.venv_teleop` | XRoboToolkit / PICO dependencies |
| MuJoCo only | `.venv_sim` | Not needed for a physical G1 run |

* If you don't have `.venv_sim` yet, install MuJoCo environment using this command
```bash
bash install_scripts/install_mujoco_sim.sh 
```

* If you don't have `.venv_data_collection` yet, create with the following command:
```bash
bash install_scripts/install_data_collection.sh
```

### 2.2 Required scripts, recorded movement files, and models

1. `gear_sonic/scripts/run_swing_episode_replayer.py` - main script for sending a stream of recorded SMPL movement to `SONIC` model 

2. `gear_sonic_deploy/run_docker_real.sh` - Docker-version deployment script (substitute of `./deploy.sh`) 

3. `gear_sonic_deploy/run_docker_zmq.sh` (Optional, for sim only deployment)

4. `swing_episodes/*` - recorded movements to be used with `run_swing_episode_replayer.py` If it does not exist, download from here: 

5. `gear_sonic_deploy/policy/low_latency` low-latency model - see [6.Verify required model and configuration assets](#6-verify-required-model-and-configuration-assets)

6. `gear_sonic_deploy/Dockerfile.tensorrt10` Docker file
---

## 3. Mandatory hardware/network checks

### 3.1 Connect the correct Ethernet adapter

Connect the host's **wired Ethernet** adapter to the G1 control network. The host adapter must have an IPv4 address on:

```text
192.168.123.x
```

Check it:

```bash
ip -4 -o addr show
ip -4 -o addr show | grep '192\.168\.123\.'
```

If the expected wired interface is not auto-detected, identify it with:

```bash
ip -br link
ip -4 addr show <interface>
```

Then use it explicitly later:

```bash
./gear_sonic_deploy/run_docker_real.sh --interface <interface>
```

### 3.2 Verify G1 reachability

Obtain the robot controller IP from the G1 network setup; With the wired interface configured, verify reachability:

```bash
ping -c 3 <G1_CONTROLLER_IP>
```


## 4. Docker and GPU prerequisites

Check Docker, GPU visibility, and the deployment image:

```bash
docker --version
nvidia-smi
docker image inspect g1-deploy-tensorrt10:latest \
  --format '{{.Id}} {{.Created}}'
```

The Docker runtime must support NVIDIA GPUs. A quick check is:

```bash
docker info --format 'default={{.DefaultRuntime}}'
```

If the image is absent, build it from the deployment directory:

```bash
cd gear_sonic_deploy
docker build -t g1-deploy-tensorrt10:latest -f Dockerfile.tensorrt10 .
cd ..
```

---

## 5. Critical ROS2/DDS build gate

### 5.1 Check the prior build output

A valid real-robot build must **not** report either of these:

```text
ROS2 disabled (HAS_ROS2 unset or 0)
ROS2 not found
```

### 5.2 Build an ROS2-enabled binary

Run from the repository root. The Docker command builds into the bind-mounted repository, so the resulting binary is used by `run_docker_real.sh`.

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HAS_ROS2=1 \
  -v "$PWD/gear_sonic_deploy:/workspace/g1_deploy" \
  -w /workspace/g1_deploy \
  g1-deploy-tensorrt10:latest \
  bash -lc 'source /opt/ros/humble/setup.bash && just build'
```

Inspect the build output. Continue only if ROS2 was found/enabled. If it reports `ROS2 not found`, do not launch real hardware; fix the image/build dependencies first.

Verify the new executable exists:

```bash
stat gear_sonic_deploy/target/release/g1_deploy_onnx_ref
```

---

## 6. Verify required model and configuration assets

The current `run_docker_real.sh` is configured for the local **low-latency** model variant. Use this command to install:
```bash
python download_from_hf.py --low-latency
```
Verify all files before launch:

```bash
for f in \
  gear_sonic_deploy/policy/low_latency/model_decoder.onnx \
  gear_sonic_deploy/policy/low_latency/model_encoder.onnx \
  gear_sonic_deploy/policy/low_latency/observation_config.yaml \
  gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx \
  gear_sonic_deploy/target/release/g1_deploy_onnx_ref
 do
  test -f "$f" && stat -c 'OK %n (%s bytes)' "$f" || echo "MISSING $f"
done

test -d gear_sonic_deploy/reference/example/ && echo 'OK reference/example/'
```
If there are missing, download from this Google Drive:

Expected categories:

| Asset | Purpose |
|---|---|
| `policy/low_latency/model_decoder.onnx` | SONIC action decoder |
| `policy/low_latency/model_encoder.onnx` | SONIC motion/teleop encoder |
| `policy/low_latency/observation_config.yaml` | Encoder observation schema / modes |
| `planner/target_vel/V2/planner_sonic.onnx` | Standing/locomotion planner |
| `reference/example/` | Required deploy motion-reference directory |
| `target/release/g1_deploy_onnx_ref` | ROS2-enabled deployment executable |

---

## 8. Launch order

### 8.1 Test the desired behavior in MuJoCo before deploying on real G1 (Optional)

### Terminal A — MuJoCo sim loop
```bash
source .venv_sim/bin/activate
python3 gear_sonic/scripts/run_sim_loop.py   --enable-onscreen --enable-offscreen
```

### Terminal B — sim G1 deploy 
```bash
./gear_sonic_deploy/run_docker_zmq.sh
```
`run_docker_zmq.sh` uses loopback (`lo`) and ZMQ-only action output for MuJoCo.

### Terminal C — recorded-episode replayer

Wait until Terminal B finishes initialization and is listening on ZMQ port `5556`.

```bash
source .venv_data_collection/bin/activate
unset PYTHONPATH

python gear_sonic/scripts/run_swing_episode_replayer.py \
  --data-dir swing_episodes/forehand_right \
```

The replayer binds ZMQ `:5556`; deploy subscribers connect to it.


### 8.2 recorded full-body motion playback in real G1

Use three terminals.

### Terminal A — real G1 deploy

```bash
./gear_sonic_deploy/run_docker_real.sh --interface <G1_WIRED_INTERFACE>
```

The container is named:

```text
g1-real
```

It uses host networking so DDS and ZMQ use the host network namespace.

### Terminal B — recorded-episode replayer

Wait until Terminal A finishes initialization and is listening on ZMQ port `5556`.

```bash
source .venv_data_collection/bin/activate
unset PYTHONPATH

python gear_sonic/scripts/run_swing_episode_replayer.py \
  --data-dir swing_episodes/forehand_right
# change `swing_episodes/forehand_right` to your desired movement 
```

The replayer binds ZMQ `:5556`; deploy subscribers connect to it.

### Terminal C — optional monitoring

Useful read-only checks:

```bash
docker ps --filter name=g1-real
docker logs -f g1-real
```

---

## 9. Interactive replay sequence

The replayer starts in full-body mode. 
Read this 
1. This script will auto-detect all episodes in the specified `--data-dir`, so once you start the script, you can select/change the episode you'd like directly from the CLI. You don't need to re-run the script for changing episode of the same movement (but you still need to re-run new command for changing the movement)
2. Before pressing `r`, confirm the deploy reports its planner/control transition. 
3. When changing to another episode while the current one is running, it will stop the current episode.  
4. Each episode has varying frequency of data. Match the frequency of data streaming to the recorded frequency by using `UP` and `DOWN` arrows for increasing or decreasing the frequency.
   - Higher = Faster movement
   - You can keep it at 30 Hz for all movement for safety.
   - You can increase it higher than the recorded for faster movement, but it might not be stable.

### Full-body SMPL playback

Order of operation (Quick start)
```text
p → m  → r 
(robot starts to stand still on its own) → (change mode to receive playback command) → (start to replay at 30 Hz)
```

| Key | Meaning |
|---|---|
| `p` | Enter PLANNER and start policy; verify the deploy enters `CONTROL` |
| `m` | Switch to streamed full-body SMPL motion |
| `r` | Start the selected episode |
| `s` | Stop replay |
| `o` | Emergency stop command |
| `q` | Quit replayer |
| `UP` arrow | Increase frequency |
| `DOWN` arrow | Decrease frequency |
| `RIGHT` arrow | Next episode |
| `LEFT` arrow | Previous episode |



## 10. Stop and emergency procedures

### Planned stop

1. Press `s` in the replayer.
2. Press `o` in the replayer to send the emergency stop command if required.
3. Stop the deploy container with `Ctrl+C` in Terminal A.

---
