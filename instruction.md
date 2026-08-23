# Real G1 SONIC Deployment with Recorded Motion Playback

> **Scope:** deploy SONIC on a physical Unitree G1 and replay a recorded swing episode through `run_swing_episode_replayer.py`.
>
> **Safety:** This procedure can command a physical robot. Perform the first run only with an E-stop accessible, a clear workspace, a spotter, and the robot in a known safe standing state. First validate every new model / episode in MuJoCo.

---

## 1. Architecture and launch roles

```text
Recorded .npz episode
  -> run_swing_episode_replayer.py (ZMQ publisher :5556)
  -> g1_deploy_onnx_ref (SONIC encoder + decoder)
  -> ROS2 / DDS / Unitree SDK2
  -> physical G1 low-level motor commands
```

Use the **real-robot launcher**:

```bash
./gear_sonic_deploy/run_docker_real.sh
```

For MuJoCo deployment use 
```bash
./gear_sonic_deploy/run_docker_zmq.sh
```
`run_docker_zmq.sh` uses loopback (`lo`) and ZMQ-only action output for MuJoCo.

---

## 2. Required environments

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

*If you don't have `.venv_sim` yet, install MuJoCo environment using this command
```bash
cd ~/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh 
```
### Terminal A — MuJoCo sim loop
```bash
cd ~/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python3 gear_sonic/scripts/run_sim_loop.py   --enable-onscreen --enable-offscreen
```

### Terminal B — sim G1 deploy 
```bash
cd ~/GR00T-WholeBodyControl
./gear_sonic_deploy/run_docker_zmq.sh
```
`run_docker_zmq.sh` uses loopback (`lo`) and ZMQ-only action output for MuJoCo.

### Terminal C — recorded-episode replayer


Wait until Terminal A finishes initialization and is listening on ZMQ port `5556`.

```bash
cd ~/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
unset PYTHONPATH

python gear_sonic/scripts/run_swing_episode_replayer.py \
  --data-dir swing_episodes/forehand_right \
```

The replayer binds ZMQ `:5556`; deploy subscribers connect to it.

* If you don't have `.venv_data_collection` yet, create with the following command:
```bash
bash install_scripts/install_data_collection.sh
```


### 8.2 recorded full-body motion playback in real G1

Use three terminals.

### Terminal A — real G1 deploy

```bash
cd ~/GR00T-WholeBodyControl
./gear_sonic_deploy/run_docker_real.sh --interface <G1_WIRED_INTERFACE>
```

Do not use `--no-prompt` initially. Read the printed model paths and interface carefully, then answer `y` only after the safety gate is complete.

The container is named:

```text
g1-real
```

It uses host networking so DDS and ZMQ use the host network namespace.

### Terminal B — recorded-episode replayer

Wait until Terminal A finishes initialization and is listening on ZMQ port `5556`.

```bash
cd ~/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
unset PYTHONPATH

python gear_sonic/scripts/run_swing_episode_replayer.py \
  --data-dir swing_episodes/forehand_right \
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

### Full-body SMPL playback

Order of replay 
```text
p → m  → r 
(robot starts to stand still on its own) → (change mode to receive playback command) → (start to replay)
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

Before pressing `r`, confirm the deploy reports its planner/control transition. 



## 10. Stop and emergency procedures

### Planned stop

1. Press `s` in the replayer.
2. Press `o` in the replayer to send the emergency stop command if required.
3. Stop the deploy container with `Ctrl+C` in Terminal A.

---
