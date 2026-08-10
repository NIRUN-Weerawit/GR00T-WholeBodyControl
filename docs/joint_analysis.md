# GR00T Whole-Body Control: Joint Space & Perception Analysis

## Overview

This document analyzes the joint control pipeline for NVIDIA's GR00T Whole-Body Control system deployed on the Unitree G1 humanoid robot, using PICO 4 Pro VR headset + ankle IMU sensors for perceived whole-body teleoperation.

All information is derived from actual source code in `gear_sonic_deploy/` and `gear_sonic/scripts/pico_manager_thread_server.py`.

## Hardware Platform: Unitree G1 Humanoid (29 DOF)

The Unitree G1 humanoid has hardware motor indices 0–28, totaling **29 actuated joints**:

| Index | Joint Name | Body Region |
|-------|------------|-------------|
| 0 | LeftHipPitch | Left Leg |
| 1 | LeftHipRoll | Left Leg |
| 2 | LeftHipYaw | Left Leg |
| 3 | LeftKnee | Left Leg |
| 4 | LeftAnklePitch (also: LeftAnkleB) | Left Ankle |
| 5 | LeftAnkleRoll (also: LeftAnkleA) | Left Ankle |
| 6 | RightHipPitch | Right Leg |
| 7 | RightHipRoll | Right Leg |
| 8 | RightHipYaw | Right Leg |
| 9 | RightKnee | Right Leg |
| 10 | RightAnklePitch (also: RightAnkleB) | Right Ankle |
| 11 | RightAnkleRoll (also: RightAnkleA) | Right Ankle |
| 12 | WaistYaw | Waist |
| 13 | WaistRoll (also: WaistA) *[1]* | Waist |
| 14 | WaistPitch (also: WaistB) *[1]* | Waist |
| 15 | LeftShoulderPitch | Left Arm |
| 16 | LeftShoulderRoll | Left Arm |
| 17 | LeftShoulderYaw | Left Arm |
| 18 | LeftElbow | Left Arm |
| 19 | LeftWristRoll | Left Arm |
| 20 | LeftWristPitch *[2]* | Left Wrist |
| 21 | LeftWristYaw *[2]* | Left Wrist |
| 22 | RightShoulderPitch | Right Arm |
| 23 | RightShoulderRoll | Right Arm |
| 24 | RightShoulderYaw | Right Arm |
| 25 | RightElbow | Right Arm |
| 26 | RightWristRoll | Right Arm |
| 27 | RightWristPitch *[2]* | Right Wrist |
| 28 | RightWristYaw *[2]* | Right Wrist |

*[1] Waist Roll/Pitch joints are INVALID for G1 23-DOF and waist-locked variants*  
*[2] Left/Right Wrist Pitch/Yaw joints are INVALID for G1 23-DOF variant*  

## Motor Command Structure

Each control command sent to the low-level controller publishes a `LowCmd_` message via DDS with this structure:

```cpp
struct MotorCommand {
    std::array<float, 29> q_target;   // Target position (radians)
    std::array<float, 29> dq_target;  // Target velocity (rad/s)  
    std::array<float, 29> kp;         // Position gain (Nm/rad)
    std::array<float, 29> kd;         // Velocity gain (Nm·s/rad)
    std::array<float, 29> tau_ff;     // Feed-forward torque (Nm)
};
```

### Control Loop Timing

| Thread | Rate | Responsibility |
|--------|------|----------------|
| Input thread | 100 Hz | Poll perception inputs, handle commands |
| Control thread | 50 Hz | Gather observations, run policy, compute motor targets |
| Planner thread | 10 Hz | Re-plan locomotion trajectory (streamed modes) |
| Command Writer | 500 Hz | Publish motor commands via DDS to robot hardware |

## Perception Stream from PICO 4 Pro + Ankle IMUs

The perception pipeline sources data from:
- **PICO 4 Pro VR headset** → Head pose, controller poses (left/right wrists)
- **2× Ankle IMU sensors** → Left/right ankle orientation in base frame
- **Derived** → Body angular velocity averaged from both ankles

### ZMQ Message Format for POSE Mode

When publishing to control the robot via POSE mode (full body teleoperation), the ZMQ message contains:

```python
numpy_data = {
    "smpl_pose": np.ndarray(N, 63),        # 21 SMPL body joint rotations (axis-angle)
    "smpl_joints": np.ndarray(N, 24, 3),   # 24 SMPL joint positions in local frame 
    "body_quat_w": np.ndarray(N, 24, 4),   # Body quaternion orientations in world frame
    "joint_pos": np.ndarray(N, 29),        # G1 joint positions (the actual control signal!)
    "joint_vel": np.ndarray(N, 29),        # Joint velocities (zeros in streamed mode)
    "vr_position": np.float32[9],          # VR 3-point positions (L-wrist, R-wrist, Neck)
    "vr_orientation": np.float32[12],      # VR 3-point orientations  
    "frame_index": int64,                  # Motion frame index
    "left_trigger": float32,               # Left controller trigger (gripper control)
    "right_trigger": float32,              # Right controller trigger (gripper control)
    "left_grip": float32,                  # Left grip button state
    "right_grip": float32,                 # Right grip button state  
    "pico_dt": float32,                    # Time delta between frames
    "timestamp_realtime": float64,         # System real-time timestamp
    "timestamp_monotonic": float64,        # Monotonic clock timestamp
    "left_hand_joints": np.float32[7],     # Left hand joint angles (Dex3 gripper)
    "right_hand_joints": np.float32[7],    # Right hand joint angles (Dex3 gripper)
    "heading_increment": float32,          # Yaw heading adjustment from joystick
    
    # Optional fields:
    "toggle_data_collection": bool,        # Start/stop data recording
    "toggle_data_abort": bool,             # Abort current operation
}
```

## Key Control Path: SMPL → G1 Joint Mapping

The system converts human body tracking data to robot joint commands through this pipeline:

```bash
PICO 4 Pro Body Tracking (24 SMPL joints)
    ↓ Coordinate transform (Unity→Robot frame)
Raw SMPL pose parameters (63 dim) + Joint positions (24×3)
    ↓ 
compute_smpl_joints() function
    ↓ Process SMPL pose: body_orientation, global_frame transformations  
Body poses in robot coordinate system
    ↓ Three-point pose extraction: L-wrist, R-wrist, Neck
SMPL → G1 joint mapping via policy parameters
    ↓ Final control signal sent to C++ deploy process
G1 Joint Positions[29] published over ZMQ
```

### What Actually Gets Commanded

The critical output that drives the robot is `joint_pos` — a 29-element array containing:
- Full lower body joint positions (legs, waist) derived from SMPL pose estimation
- Upper body joints directly tracked from VR controller wrist positions via IK solvers
- Wrist roll/pitch/yaw computed through hand IK solver

### Exact Joint Ranges (from `policy_parameters.hpp`)

The C++ control loop computes motor targets as: `target = action * action_scale + default_angle`
Where:
- **action** is the policy/stream input, clipped to [-1, +1]
- **action_scale** = 0.25 × effort_limit / stiffness (motor armature-derived)
- **default_angles** are the standing-pose offset for each joint

Each joint's practical range is ±scale around its default:

| Idx | Joint | Default (rad/°) | ±Scale (°) | Range (°) |
|-----|-------|----------------|------------|-----------|
| 0 | LeftHipPitch | -0.312 / -17.9° | ±20.1° | [-38.0°, +2.2°] |
| 1 | LeftHipRoll | 0.000 / 0.0° | ±20.1° | [-20.1°, +20.1°] |
| 2 | LeftHipYaw | 0.000 / 0.0° | ±31.4° | [-31.4°, +31.4°] |
| 3 | LeftKnee | 0.669 / +38.3° | ±20.1° | [+18.2°, +58.4°] |
| 4 | LeftAnklePitch | -0.363 / -20.8° | ±25.1° | [-45.9°, +4.3°] |
| 5 | LeftAnkleRoll | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 6 | RightHipPitch | -0.312 / -17.9° | ±20.1° | [-38.0°, +2.2°] |
| 7 | RightHipRoll | 0.000 / 0.0° | ±20.1° | [-20.1°, +20.1°] |
| 8 | RightHipYaw | 0.000 / 0.0° | ±31.4° | [-31.4°, +31.4°] |
| 9 | RightKnee | 0.669 / +38.3° | ±20.1° | [+18.2°, +58.4°] |
| 10 | RightAnklePitch | -0.363 / -20.8° | ±25.1° | [-45.9°, +4.3°] |
| 11 | RightAnkleRoll | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 12 | WaistYaw | 0.000 / 0.0° | ±31.4° | [-31.4°, +31.4°] |
| 13 | WaistRoll *[1]* | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 14 | WaistPitch *[1]* | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 15 | LeftShoulderPitch | 0.200 / +11.5° | ±25.1° | [-13.7°, +36.6°] |
| 16 | LeftShoulderRoll | 0.200 / +11.5° | ±25.1° | [-13.7°, +36.6°] |
| 17 | LeftShoulderYaw | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 18 | LeftElbow | 0.600 / +34.4° | ±25.1° | [+9.2°, +59.5°] |
| 19 | LeftWristRoll | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 20 | LeftWristPitch *[2]* | 0.000 / 0.0° | ±4.3° | [-4.3°, +4.3°] |
| 21 | LeftWristYaw *[2]* | 0.000 / 0.0° | ±4.3° | [-4.3°, +4.3°] |
| 22 | RightShoulderPitch | 0.200 / +11.5° | ±25.1° | [-13.7°, +36.6°] |
| 23 | RightShoulderRoll | -0.200 / -11.5° | ±25.1° | [-36.6°, +13.7°] |
| 24 | RightShoulderYaw | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 25 | RightElbow | 0.600 / +34.4° | ±25.1° | [+9.2°, +59.5°] |
| 26 | RightWristRoll | 0.000 / 0.0° | ±25.1° | [-25.1°, +25.1°] |
| 27 | RightWristPitch *[2]* | 0.000 / 0.0° | ±4.3° | [-4.3°, +4.3°] |
| 28 | RightWristYaw *[2]* | 0.000 / 0.0° | ±4.3° | [-4.3°, +4.3°] |

*[1] Waist Roll/Pitch INVALID on G1 23-DOF and waist-locked variants — send 0.0*
*[2] Wrist Pitch/Yaw INVALID on G1 23-DOF variant — send 0.0*

**Warning for video-based control**: If your IK solver outputs angles outside these ranges, the policy clips them to ±scale from default. Clamp your extracted joint angles to these bounds before publishing!

## Ankle IMU Integration

The ankle sensors don't directly send joint angles to the control loop. Instead they:
1. Provide base frame orientation reference for lower body tracking
2. Help compute `body_quat_w` orientations in world coordinates  
3. Enable locomotion mode awareness (walking, running, idle poses)
4. Feed into the planner module's trajectory generation

The actual joint angle extraction happens via forward kinematics from perceived body positions, not raw IMU readings.

## For Video-Based Human Extraction

To control G1 from video-extracted pose data instead of PICO VR:

### Required Data Structure for ZMQ Publishing

```python
import zmq
import numpy as np
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

# Your 29 G1 joint angles extracted from video pose estimation
your_joint_pos = np.array([29], dtype=np.float32)  # q_target values in radians

construct_control_message = {
    "smpl_pose": smpl_parameters,      # 63-dim body pose params  
    "smpl_joints": body_joint_positions, # 24×3 local positions
    "body_quat_w": body_orients_world,   # 24×4 quaternions in world frame
    "joint_pos": your_joint_pos,         # ← THE ACTUAL CONTROL VALUES (29 elems)
    "vr_position": np.zeros(9),         # VR 3pt positions (optional for video mode)
    "vr_orientation": np.zeros(12),     # VR 3pt orientations  
    "left_joints": left_hand_if_present,   # Left Dex3 gripper joint angles
    "right_joints": right_hand_if_present, # Right Dex3 gripper angles
}

zmq_msg = pack_pose_message(message_data)
```

### Critical Mapping: Video Pose → 29 G1 Joints

The video-to-G1 mapping should output exactly these motor indices with proper ranges (radians):
- Lower body derived from video pose estimation hip/knee/ankle positions
- Waist joints controlled by torso orientation relative to legs
- Upper body joints computed via kinematic chain from shoulder/elbow/wrist tracking data
