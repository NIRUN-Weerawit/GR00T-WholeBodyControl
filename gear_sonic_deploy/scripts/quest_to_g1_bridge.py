#!/usr/bin/env python3
"""
MQTT -> ZMQ Bridge: Quest 3 Controller to G1 Robot Control

This script bridges Meta Quest 3 controller data from MQTT to ZMQ for G1 robot control.

MQTT Input (from Quest 3):
    - WebSocket connection to broker at sora2.uclab.jp
    - Topic: control/{device_id}
    - Message format: JSON with left/right controller pose, head pose, buttons

ZMQ Output (to G1):
    - PUB socket on port 5556
    - Message format: msgpack serialized dict with planner/pose data

Usage:
    python quest_to_g1_bridge.py --device_id piper-wee --zmq_port 5556

"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import msgpack
import numpy as np
import paho.mqtt.client as mqtt
import zmq
from scipy.spatial.transform import Rotation as R

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("QuestG1Bridge")

# =============================================================================
# Constants and Enums
# =============================================================================

# Locomotion modes matching pico_manager_thread_server.py
class LocomotionMode(IntEnum):
    """Locomotion mode enum for robot movement."""
    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19


# Joystick deadzone threshold
JOYSTICK_DEADZONE = 0.15

# Coordinate transformation matrix: Unity (Y-up, left-handed) -> Robot (Z-up, right-handed)
# Unity: X-right, Y-up, Z-forward
# Robot: X-forward, Y-left, Z-up
#
# Mapping (matching pico_manager_thread_server.py):
#   Robot X = -Unity X (negated right = left, but used as forward in robot frame)
#   Robot Y = Unity Z (forward in Unity becomes left in robot)
#   Robot Z = Unity Y (up stays up)
#
# Note: This matches the Q matrix in pico_manager_thread_server.py:
#   Q = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
#   Unity [x, y, z] -> Robot [-x, z, y]
#
# Matrix form: [robot_x, robot_y, robot_z]^T = M @ [unity_x, unity_y, unity_z]^T
UNITY_TO_ROBOT = np.array([
    [-1, 0, 0],  # Robot X = -Unity X
    [0, 0, 1],   # Robot Y = Unity Z
    [0, 1, 0]    # Robot Z = Unity Y
], dtype=np.float32)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ControllerState:
    """Represents a single controller's state."""
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # quaternion [w, x, y, z] scalar-first
    trigger: bool
    grip: bool
    thumbstick: np.ndarray  # [x, y]


@dataclass
class QuestInput:
    """Parsed input from Quest 3 MQTT message."""
    time: float
    left_controller: ControllerState
    right_controller: ControllerState
    head_position: np.ndarray
    head_rotation: np.ndarray
    button_a: bool
    button_b: bool
    button_x: bool
    button_y: bool


@dataclass
class G1PlannerMessage:
    """G1 planner message format for ZMQ output."""
    topic: str = "planner"
    mode: int = LocomotionMode.IDLE
    movement: np.ndarray = None  # [x, y, z]
    facing: np.ndarray = None    # [x, y, z]
    speed: float = -1.0
    height: float = -1.0
    vr_3pt_position: np.ndarray = None   # 9 floats: L-wrist, R-wrist, Neck positions
    vr_3pt_orientation: np.ndarray = None  # 12 floats: L-wrist, R-wrist, Neck quaternions

    def __post_init__(self):
        if self.movement is None:
            self.movement = np.zeros(3, dtype=np.float32)
        if self.facing is None:
            self.facing = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if self.vr_3pt_position is None:
            self.vr_3pt_position = np.zeros(9, dtype=np.float32)
        if self.vr_3pt_orientation is None:
            self.vr_3pt_orientation = np.zeros(12, dtype=np.float32)


@dataclass
class G1HandMessage:
    """G1 hand command for Dex3 hands.
    
    Hand positions:
    - 0.0 = fully open
    - 1.0 = fully closed
    """
    left_hand: float = 0.0   # 0.0=open, 1.0=closed
    right_hand: float = 0.0  # 0.0=open, 1.0=closed


# Dex3 hand joint limits (from Unitree example)
# q=0 is fully open, limits define the closed position
# For some joints, closed is positive (max), for others it's negative (min)
DEX3_MAX_LIMITS_LEFT = np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEX3_MIN_LIMITS_LEFT = np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75], dtype=np.float32)
DEX3_MAX_LIMITS_RIGHT = np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75], dtype=np.float32)
DEX3_MIN_LIMITS_RIGHT = np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def normalized_to_joint_positions(normalized: float, is_left: bool) -> np.ndarray:
    """
    Convert normalized hand value (0=open, 1=closed) to 7 joint positions in radians.
    
    The Dex3 hand has different joint limits:
    - q=0 is fully open for all joints
    - Closed position varies: some joints close to max_limit, others to min_limit
    
    Args:
        normalized: 0.0 (open) to 1.0 (closed)
        is_left: True for left hand, False for right
    
    Returns:
        7-element array of joint positions in radians
    """
    if is_left:
        max_limits = DEX3_MAX_LIMITS_LEFT
        min_limits = DEX3_MIN_LIMITS_LEFT
    else:
        max_limits = DEX3_MAX_LIMITS_RIGHT
        min_limits = DEX3_MIN_LIMITS_RIGHT
    
    # For each joint, determine the "closed" position
    # If max_limit > 0 and |max_limit| > |min_limit|, closed is at max_limit
    # If |min_limit| > |max_limit|, closed is at min_limit
    joints = np.zeros(7, dtype=np.float32)
    for i in range(7):
        max_lim = max_limits[i]
        min_lim = min_limits[i]
        
        # Determine which limit represents "closed"
        if abs(max_lim) >= abs(min_lim):
            # Closed is at max_limit (positive direction)
            joints[i] = normalized * max_lim
        else:
            # Closed is at min_limit (negative direction)
            joints[i] = normalized * min_lim
    
    return joints


# =============================================================================
# Coordinate Transformation Functions
# =============================================================================

def unity_position_to_robot(pos: np.ndarray) -> np.ndarray:
    """
    Transform position from Unity coordinate frame to robot frame.
    
    Unity: X-right, Y-up, Z-forward (left-handed)
    Robot: X-forward, Y-left, Z-up (right-handed)
    
    Transformation: [x', y', z'] = [-x, z, y]
    """
    return UNITY_TO_ROBOT @ pos


def unity_quat_to_robot(quat: np.ndarray, scalar_first: bool = True) -> np.ndarray:
    """
    Transform quaternion from Unity frame to robot frame.
    
    Args:
        quat: Quaternion [w, x, y, z] if scalar_first, else [x, y, z, w]
        scalar_first: Whether quaternion is scalar-first (w, x, y, z)
    
    Returns:
        Quaternion [w, x, y, z] in robot frame (scalar-first)
    """
    if not scalar_first:
        # Convert from [x, y, z, w] to [w, x, y, z]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])
    
    # Create rotation from quaternion
    rot = R.from_quat(quat, scalar_first=True)
    
    # Transform rotation matrix: R_robot = Q @ R_unity @ Q.T
    rot_matrix = rot.as_matrix()
    rot_robot = UNITY_TO_ROBOT @ rot_matrix @ UNITY_TO_ROBOT.T
    
    # Convert back to quaternion
    return R.from_matrix(rot_robot).as_quat(scalar_first=True)


def compute_relative_transform(
    pose: np.ndarray,
    reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pose relative to a reference frame.
    
    Args:
        pose: [x, y, z, qw, qx, qy, qz] in robot frame
        reference: [x, y, z, qw, qx, qy, qz] reference frame in robot frame
    
    Returns:
        (rel_pos, rel_quat): Position and quaternion relative to reference
    """
    ref_pos = reference[:3]
    ref_quat = reference[3:]
    
    ref_rot = R.from_quat(ref_quat, scalar_first=True)
    
    # Relative position: rotate delta by inverse of reference rotation
    rel_pos = ref_rot.inv().apply(pose[:3] - ref_pos)
    
    # Relative orientation: ref_inv * pose_rot
    pose_rot = R.from_quat(pose[3:], scalar_first=True)
    rel_rot = ref_rot.inv() * pose_rot
    
    return rel_pos, rel_rot.as_quat(scalar_first=True)


# =============================================================================
# Quest Input Parsing
# =============================================================================

def parse_quest_message(payload: bytes) -> Optional[QuestInput]:
    """
    Parse MQTT message payload into QuestInput.
    
    Expected JSON format:
    {
        "time": number,
        "left_controller": {
            "position": {"x": ..., "y": ..., "z": ...},
            "rotation": {"w": ..., "x": ..., "y": ..., "z": ...},
            "trigger": boolean,
            "grip": boolean,
            "thumbstick": {"x": ..., "y": ...}
        },
        "right_controller": {...},
        "head": {
            "position": {"x": ..., "y": ..., "z": ...},
            "rotation": {"w": ..., "x": ..., "y": ..., "z": ...}
        },
        "buttonA": boolean,
        "buttonB": boolean,
        "buttonX": boolean,
        "buttonY": boolean
    }
    """
    try:
        data = json.loads(payload.decode('utf-8'))
        
        # Debug: log the received message structure
        logger.debug(f"Received message keys: {list(data.keys())}")
        if len(data) > 0:
            logger.debug(f"First few keys and values: {dict(list(data.items())[:3])}")
        
        def parse_controller(ctrl_data: dict) -> ControllerState:
            pos = np.array([
                ctrl_data["position"]["x"],
                ctrl_data["position"]["y"],
                ctrl_data["position"]["z"]
            ], dtype=np.float32)
            
            # Quaternion is scalar-first (w, x, y, z) from Quest
            quat = np.array([
                ctrl_data["rotation"]["w"],
                ctrl_data["rotation"]["x"],
                ctrl_data["rotation"]["y"],
                ctrl_data["rotation"]["z"]
            ], dtype=np.float32)
            
            # Handle None thumbstick or non-dict values (sometimes int 0 is sent)
            thumbstick_data = ctrl_data.get("thumbstick")
            if thumbstick_data is None or not isinstance(thumbstick_data, dict):
                thumbstick = np.array([0.0, 0.0], dtype=np.float32)
            else:
                thumbstick = np.array([
                    float(thumbstick_data.get("x", 0.0)),
                    float(thumbstick_data.get("y", 0.0))
                ], dtype=np.float32)
            
            return ControllerState(
                position=pos,
                rotation=quat,
                trigger=ctrl_data.get("trigger", False),
                grip=ctrl_data.get("grip", False),
                thumbstick=thumbstick
            )
        
        left_ctrl = parse_controller(data["left_controller"])
        right_ctrl = parse_controller(data["right_controller"])
        
        head_pos = np.array([
            data["head"]["position"]["x"],
            data["head"]["position"]["y"],
            data["head"]["position"]["z"]
        ], dtype=np.float32)
        
        head_quat = np.array([
            data["head"]["rotation"]["w"],
            data["head"]["rotation"]["x"],
            data["head"]["rotation"]["y"],
            data["head"]["rotation"]["z"]
        ], dtype=np.float32)
        
        return QuestInput(
            time=data["time"],
            left_controller=left_ctrl,
            right_controller=right_ctrl,
            head_position=head_pos,
            head_rotation=head_quat,
            button_a=data.get("buttonA", False),
            button_b=data.get("buttonB", False),
            button_x=data.get("buttonX", False),
            button_y=data.get("buttonY", False)
        )
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse Quest message: {e}")
        return None


# =============================================================================
# Quest to G1 Conversion
# =============================================================================

class YawAccumulator:
    """Accumulates yaw heading angle based on joystick input."""
    
    def __init__(self, yaw_gain: float = 1.5, deadzone: float = JOYSTICK_DEADZONE):
        self.yaw_gain = yaw_gain
        self.deadzone = deadzone
        self.reset()
    
    def reset(self):
        """Reset facing direction to default (1, 0, 0)."""
        self.heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.yaw_angle_rad = 0.0
        self.dyaw = 0.0
        logger.info("YawAccumulator: reset yaw angle to 0.0")
    
    def yaw_angle(self) -> float:
        """Get current yaw angle in radians."""
        return self.yaw_angle_rad
    
    def yaw_angle_change(self) -> float:
        """Get current yaw angle change in radians."""
        return self.dyaw
    
    def update(self, rx: float, dt: float) -> np.ndarray:
        """
        Update facing direction based on right stick x-axis input.
        
        Args:
            rx: Right stick x-axis value (-1 to 1)
            dt: Time delta in seconds
        
        Returns:
            Facing direction as [x, y, 0.0]
        """
        self.dyaw = self.yaw_gain * (-rx) * dt
        if abs(rx) >= self.deadzone:
            self.yaw_angle_rad += self.dyaw
            self.heading = np.array([
                np.cos(self.yaw_angle_rad),
                np.sin(self.yaw_angle_rad),
                0.0
            ], dtype=np.float32)
        return self.heading


class QuestToG1Converter:
    """Converts Quest 3 controller data to G1 robot control format."""
    
    def __init__(self):
        self.yaw_accumulator = YawAccumulator()
        self.mode = LocomotionMode.IDLE
        self.prev_ab = False
        self.prev_xy = False
        self.last_time = time.time()
        
        # Hand state for Dex3 gripper control
        # 0.0 = fully open, 1.0 = fully closed
        self.left_hand_state = 0.0
        self.right_hand_state = 0.0
        
        # Hand smoothing: how fast to transition (0.0-1.0 per second)
        self.hand_close_speed = 3.0   # Close speed (fast)
        self.hand_open_speed = 4.0     # Open speed (faster)
        
        # Rotation offsets for wrist alignment (from pico_manager_thread_server.py)
        # These align the controller frames with robot wrist frames
        self.l_wrist_offset = R.from_euler("xyz", [90, 0, 0], degrees=True)
        self.r_wrist_offset = R.from_euler("xyz", [-90, 0, 180], degrees=True)
        self.neck_offset_rot = R.from_euler("xyz", [0, 0, -90], degrees=True)
        
        # Global yaw offset: 180° around Y (from pico_manager_thread_server.py)
        # This accounts for the VR controller coordinate system facing opposite to robot forward
        self.global_yaw_offset = R.from_euler("y", 180, degrees=True)
        
        # Wrist position offsets in robot frame (matching C++ VR_3POINT_OFFSETS)
        # These compensate for the offset from controller position to actual wrist joint
        # Left wrist: 18cm forward, 2.5cm toward body center (negative Y in robot frame)
        # Right wrist: 18cm forward, 2.5cm toward body center (positive Y in robot frame)
        # NOTE: These must be SYMMETRIC for left/right arms to behave identically
        self.l_wrist_pos_offset = np.array([0.18, -0.025, 0.0], dtype=np.float32)
        self.r_wrist_pos_offset = np.array([0.00, 0.00, 0.0], dtype=np.float32)
    
    def reset_yaw(self):
        """Reset yaw accumulator."""
        self.yaw_accumulator.reset()
    
    def convert(self, quest_input: QuestInput) -> tuple[G1PlannerMessage, G1HandMessage]:
        """
        Convert Quest input to G1 planner message and hand message.
        
        Args:
            quest_input: Parsed Quest 3 controller data
        
        Returns:
            Tuple of (G1PlannerMessage, G1HandMessage) ready for ZMQ publishing
        """
        dt = time.time() - self.last_time
        self.last_time = time.time()
        
        # Update hand states based on grip buttons
        # Grip button pressed = close hand, released = open hand
        left_grip = quest_input.left_controller.grip
        right_grip = quest_input.right_controller.grip
        
        # Debug: Log grip button states
        if left_grip or right_grip:
            logger.debug(f"Grip buttons: L={left_grip}, R={right_grip}")
        
        # Smooth hand transitions
        if left_grip:
            self.left_hand_state = min(1.0, self.left_hand_state + self.hand_close_speed * dt)
        else:
            self.left_hand_state = max(0.0, self.left_hand_state - self.hand_open_speed * dt)
        
        if right_grip:
            self.right_hand_state = min(1.0, self.right_hand_state + self.hand_close_speed * dt)
        else:
            self.right_hand_state = max(0.0, self.right_hand_state - self.hand_open_speed * dt)
        
        # Handle mode switching with A+B (next) and X+Y (previous)
        a_pressed = quest_input.button_a
        b_pressed = quest_input.button_b
        x_pressed = quest_input.button_x
        y_pressed = quest_input.button_y
        
        ab_now = a_pressed and b_pressed
        xy_now = x_pressed and y_pressed
        
        if ab_now and not self.prev_ab:
            self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
            logger.info(f"Mode -> {self.mode.value}: {self.mode.name}")
        if xy_now and not self.prev_xy:
            self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
            logger.info(f"Mode -> {self.mode.value}: {self.mode.name}")
        
        self.prev_ab = ab_now
        self.prev_xy = xy_now
        
        # Get thumbstick inputs
        lx = quest_input.left_controller.thumbstick[0]
        ly = quest_input.left_controller.thumbstick[1]
        rx = quest_input.right_controller.thumbstick[0]
        
        # Update facing direction from right stick
        facing = self.yaw_accumulator.update(rx, dt)
        
        # Compute movement from left stick
        raw_mag = np.hypot(lx, ly)
        raw_mag = np.clip(raw_mag, 0.0, 1.0)
        
        if np.abs(raw_mag) < JOYSTICK_DEADZONE:
            mag = 0.0
            speed = -1.0
            mode_to_send = LocomotionMode.IDLE
        else:
            mag = (raw_mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
            mag = min(mag, 1.0)
            mode_to_send = self.mode
            
            if self.mode == LocomotionMode.SLOW_WALK:
                speed = 0.1 + 0.5 * mag  # 0.1 .. 0.6
            elif self.mode == LocomotionMode.WALK:
                speed = -1.0
            elif self.mode == LocomotionMode.RUN:
                speed = 1.5 + 3 * mag  # 1.5 .. 4.5
            else:
                speed = mag  # default 0 .. 1.0
        
        # Compute movement in global frame
        # WebXR thumbstick: x=left(-1)/right(+1), y=up(-1)/down(+1)
        # Robot frame: X=forward, Y=left
        # Mapping: forward=-y, left=-x
        denom = raw_mag if raw_mag > 0.0 else 1.0
        scale = mag / denom
        movement_local = np.array([-lx, -ly]) * scale
        
        # Rotate movement by facing direction
        perp_x, perp_y = -facing[1], facing[0]
        rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
        movement_global = rotation_facing @ movement_local
        movement = np.array([movement_global[0], movement_global[1], 0.0], dtype=np.float32)
        
        # Transform controller positions and rotations to robot frame
        # Left wrist
        l_pos_robot = unity_position_to_robot(quest_input.left_controller.position)
        l_quat_robot = unity_quat_to_robot(quest_input.left_controller.rotation)
        
        # Right wrist
        r_pos_robot = unity_position_to_robot(quest_input.right_controller.position)
        r_quat_robot = unity_quat_to_robot(quest_input.right_controller.rotation)
        
        # NOTE: Wrist position offsets are currently DISABLED for testing
        # The C++ code applies offsets rotated by body orientation for motion data,
        # but for VR data, the offset application is unclear.
        # TODO: Re-enable after determining correct offset application method.
        # l_pos_robot = l_pos_robot + self.l_wrist_pos_offset
        # r_pos_robot = r_pos_robot + self.r_wrist_pos_offset
        
        # Head/Neck (use head as neck approximation)
        head_pos_robot = unity_position_to_robot(quest_input.head_position)
        head_quat_robot = unity_quat_to_robot(quest_input.head_rotation)
        
        # Estimate pelvis position from head
        # In robot frame: head is ~40cm above pelvis (z=0.40 in default pose)
        # Pelvis is estimated by subtracting this offset from head position
        PELVIS_FROM_HEAD_OFFSET = np.array([0.0, 0.0, 0.40], dtype=np.float32)
        pelvis_pos_robot = head_pos_robot - PELVIS_FROM_HEAD_OFFSET
        
        # Make positions relative to pelvis (root frame)
        # This matches what the policy expects: positions relative to root
        l_pos_relative = l_pos_robot - pelvis_pos_robot
        r_pos_relative = r_pos_robot - pelvis_pos_robot
        head_pos_relative = head_pos_robot - pelvis_pos_robot  # Should be ~[0, 0, 0.40]
        
        # Debug: Log left vs right positions to find asymmetry
        logger.info(f"L pos (Unity): {quest_input.left_controller.position}")
        logger.info(f"R pos (Unity): {quest_input.right_controller.position}")
        logger.info(f"L pos (Robot): {l_pos_robot}")
        logger.info(f"R pos (Robot): {r_pos_robot}")
        logger.info(f"L pos (Rel): {l_pos_relative}")
        logger.info(f"R pos (Rel): {r_pos_relative}")
        logger.info(f"L-R diff (Unity): {quest_input.left_controller.position - quest_input.right_controller.position}")
        logger.info(f"L-R diff (Robot): {l_pos_robot - r_pos_robot}")
        logger.info(f"L-R diff (Rel): {l_pos_relative - r_pos_relative}")

        # Apply rotation offsets for wrist/neck alignment
        l_rot = R.from_quat(l_quat_robot, scalar_first=True) * self.l_wrist_offset
        r_rot = R.from_quat(r_quat_robot, scalar_first=True) * self.r_wrist_offset
        head_rot = R.from_quat(head_quat_robot, scalar_first=True) * self.neck_offset_rot
        
        # Build VR 3-point pose arrays
        # Position: 9 floats (L-wrist, R-wrist, Head) - RELATIVE TO PELVIS
        # These positions are in the robot's local frame, relative to the pelvis/root
        vr_3pt_position = np.concatenate([
            l_pos_relative,
            r_pos_relative,
            head_pos_relative
        ]).astype(np.float32)
        
        # Orientation: 12 floats (L-wrist, R-wrist, Head quaternions)
        vr_3pt_orientation = np.concatenate([
            l_rot.as_quat(scalar_first=True),
            r_rot.as_quat(scalar_first=True),
            head_rot.as_quat(scalar_first=True)
        ]).astype(np.float32)
        
        planner_msg = G1PlannerMessage(
            topic="planner",
            mode=mode_to_send.value,
            movement=movement,
            facing=facing,
            speed=speed,
            height=-1.0,
            vr_3pt_position=vr_3pt_position,
            vr_3pt_orientation=vr_3pt_orientation
        )
        
        hand_msg = G1HandMessage(
            left_hand=self.left_hand_state,
            right_hand=self.right_hand_state
        )
        
        return planner_msg, hand_msg


# =============================================================================
# ZMQ Publisher
# =============================================================================

def build_packed_message(topic: str, fields: list) -> bytes:
    """
    Build a packed binary message for ZMQPackedMessageSubscriber.
    
    Format:
    [topic_prefix] [1280-byte JSON header] [concatenated binary fields]
    
    Args:
        topic: Topic name prefix (e.g., "command", "planner")
        fields: List of (name, dtype, shape, data) tuples
            - name: Field name (string)
            - dtype: Data type ('f32', 'f64', 'i32', 'i64', 'bool', 'u8')
            - shape: Shape list (e.g., [3] for 3-element vector)
            - data: Binary data as bytes or numpy array
    
    Returns:
        Packed message bytes
    """
    # Build header JSON
    header_dict = {
        "v": 1,
        "endian": "le" if sys.byteorder == 'little' else "be",
        "count": 1,
        "fields": [
            {"name": f[0], "dtype": f[1], "shape": f[2]}
            for f in fields
        ]
    }
    header_json = json.dumps(header_dict)
    
    # Pad header to 1280 bytes
    header_bytes = header_json.encode('utf-8')
    if len(header_bytes) > 1280:
        raise ValueError(f"Header too large: {len(header_bytes)} > 1280")
    header_bytes = header_bytes + b'\x00' * (1280 - len(header_bytes))
    
    # Concatenate binary fields
    data_bytes = b''
    for name, dtype, shape, field_data in fields:
        if isinstance(field_data, np.ndarray):
            # Convert numpy array to bytes
            if dtype == 'f32':
                data_bytes += field_data.astype('<f4').tobytes()
            elif dtype == 'f64':
                data_bytes += field_data.astype('<f8').tobytes()
            elif dtype == 'i32':
                data_bytes += field_data.astype('<i4').tobytes()
            elif dtype == 'i64':
                data_bytes += field_data.astype('<i8').tobytes()
            elif dtype == 'bool' or dtype == 'u8':
                data_bytes += field_data.astype('uint8').tobytes()
            else:
                data_bytes += field_data.tobytes()
        else:
            data_bytes += field_data
    
    # Prepend topic prefix
    topic_bytes = topic.encode('utf-8')
    return topic_bytes + header_bytes + data_bytes


def build_planner_message(msg: G1PlannerMessage, left_hand: float = 0.0, right_hand: float = 0.0) -> bytes:
    """
    Build packed binary planner message for ZMQPackedMessageSubscriber.
    
    Format:
    [topic: "planner"] [1280-byte JSON header] [mode:i32] [movement:3xf32] [facing:3xf32] [speed:f32] [height:f32]
                        [vr_position:9xf32] [vr_orientation:12xf32] [left_hand_joints:7xf32] [right_hand_joints:7xf32]
    
    Note: Field names must match C++ ZMQManager expectations:
          - vr_position (not vr_3pt_position)
          - vr_orientation (not vr_3pt_orientation)
    
    Hand joints: 7 values per hand in radians (0=open, converted to proper joint limits)
    """
    # Convert normalized hand values to actual joint positions
    left_hand_joints = normalized_to_joint_positions(left_hand, is_left=True)
    right_hand_joints = normalized_to_joint_positions(right_hand, is_left=False)
    
    fields = [
        ("mode", "i32", [1], np.array([int(msg.mode)], dtype=np.int32)),
        ("movement", "f32", [3], msg.movement.astype(np.float32)),
        ("facing", "f32", [3], msg.facing.astype(np.float32)),
        ("speed", "f32", [1], np.array([float(msg.speed)], dtype=np.float32)),
        ("height", "f32", [1], np.array([float(msg.height)], dtype=np.float32)),
        ("vr_position", "f32", [9], msg.vr_3pt_position.astype(np.float32)),
        ("vr_orientation", "f32", [12], msg.vr_3pt_orientation.astype(np.float32)),
        ("left_hand_joints", "f32", [7], left_hand_joints),
        ("right_hand_joints", "f32", [7], right_hand_joints),
    ]
    return build_packed_message("planner", fields)


def build_command_message(start: bool = True, stop: bool = False, planner: bool = True) -> bytes:
    """
    Build packed binary command message for ZMQPackedMessageSubscriber.
    
    Format:
    [topic: "command"] [1280-byte JSON header] [start:bool] [stop:bool] [planner:bool]
    """
    fields = [
        ("start", "bool", [1], np.array([1 if start else 0], dtype=np.uint8)),
        ("stop", "bool", [1], np.array([1 if stop else 0], dtype=np.uint8)),
        ("planner", "bool", [1], np.array([1 if planner else 0], dtype=np.uint8)),
    ]
    return build_packed_message("command", fields)


# =============================================================================
# MQTT Bridge Class
# =============================================================================

class QuestG1Bridge:
    """
    MQTT to ZMQ bridge for Quest 3 to G1 robot control.
    
    Subscribes to MQTT topic for Quest controller data,
    converts to G1 format, and publishes via ZMQ.
    """
    
    def __init__(
        self,
        device_id: str,
        mqtt_client_id: str,
        mqtt_broker: str = "sora2.uclab.jp",
        zmq_port: int = 5556,
    ):
        self.device_id = device_id
        self.mqtt_broker = mqtt_broker
        self.zmq_port = zmq_port
        self.mqtt_client_id = mqtt_client_id or f"quest_g1_bridge_{device_id}"
        
        # Components
        self.converter = QuestToG1Converter()
        
        # State
        self.running = False
        self.mqtt_connected = False
        self.last_message_time = 0.0
        self.message_count = 0
        
        # ZMQ
        self.zmq_context = None
        self.zmq_socket = None
        
        # MQTT
        self.mqtt_client = None
        
        # Statistics
        self.stats_interval = 5.0
        self.stats_last_time = time.time()
        self.stats_message_count = 0
    
    def setup_zmq(self):
        """Initialize ZMQ publisher socket."""
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(f"tcp://*:{self.zmq_port}")
        logger.info(f"ZMQ PUB socket bound to port {self.zmq_port}")

        # Give subscribers time to connect
        time.sleep(0.5)

        # Send startup command to start the robot in planner mode
        startup_cmd = build_command_message(start=True, stop=False, planner=True)
        self.zmq_socket.send(startup_cmd)
        logger.info("Sent startup command: start=True, planner=True")

        # Log message format details
        logger.info("Message format: packed binary with 1280-byte JSON header")
        logger.info("  - Command topic: 'command' with fields: start, stop, planner")
        logger.info("  - Planner topic: 'planner' with fields: mode, movement, facing, speed, height, vr_position, vr_orientation")
    
    def setup_mqtt(self):
        """Initialize MQTT client for WebSocket connection."""
        self.mqtt_client = mqtt.Client(
            client_id=self.mqtt_client_id,
            transport="websockets"
        )
        
        # Set callbacks
        self.mqtt_client.on_connect = self._on_mqtt_connect
        self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
        self.mqtt_client.on_message = self._on_mqtt_message
        self.mqtt_client.on_subscribe = self._on_mqtt_subscribe
        
        # Configure WebSocket connection
        # Parse WebSocket URL
        if self.mqtt_broker.startswith("wss://"):
            host = self.mqtt_broker[6:].split("/")[0]
            if ":" in host:
                host, port = host.split(":")
                port = int(port)
            else:
                port = 443
            self.mqtt_client.tls_set()
            self.mqtt_client.ws_set_options(path="/mqws")
        elif self.mqtt_broker.startswith("ws://"):
            host = self.mqtt_broker[5:].split("/")[0]
            if ":" in host:
                host, port = host.split(":")
                port = int(port)
            else:
                port = 1883
            self.mqtt_client.ws_set_options(path="/mqws")
        elif self.mqtt_broker.startswith("mqtt://"):
            host = self.mqtt_broker[7:].split("/")[0]
            if ":" in host:
                host, port = host.split(":")
                port = int(port)
            else:
                port = 1883
            # Raw MQTT - recreate client without WebSocket transport
            self.mqtt_client = mqtt.Client(client_id=self.mqtt_client_id)
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_subscribe = self._on_mqtt_subscribe
        else:
            raise ValueError(f"Invalid MQTT broker URL: {self.mqtt_broker}")
        
        self.mqtt_client.host = host
        self.mqtt_client.port = port
        
        logger.info(f"MQTT client configured for {host}:{port}")
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback when MQTT connects."""
        if rc == 0:
            self.mqtt_connected = True
            topic = f"control/{self.device_id}"
            # topic = f"robot/{self.device_id}"
            client.subscribe(topic)
            logger.info(f"Connected to MQTT broker, subscribed to '{topic}'")
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback when MQTT disconnects."""
        self.mqtt_connected = False
        if rc != 0:
            logger.warning(f"MQTT unexpected disconnect (rc={rc}), will reconnect...")
    
    def _on_mqtt_subscribe(self, client, userdata, mid, granted_qos):
        """Callback when MQTT subscription is confirmed."""
        logger.info(f"MQTT subscription confirmed (mid={mid})")
    
    def send_command(self, start: bool = True, stop: bool = False, planner: bool = True):
        """
        Send a command message to the C++ bridge.
        
        Args:
            start: Start the robot
            stop: Stop the robot
            planner: Use planner mode (True) or streamed motion mode (False)
        """
        if self.zmq_socket:
            cmd = build_command_message(start=start, stop=stop, planner=planner)
            self.zmq_socket.send(cmd)
            logger.info(f"Sent command: start={start}, stop={stop}, planner={planner}")
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Callback when MQTT message is received."""
        try:
            self.message_count += 1
            self.stats_message_count += 1
            
            # Parse Quest input
            quest_input = parse_quest_message(msg.payload)
            if quest_input is None:
                logger.warning("Failed to parse Quest message, skipping")
                return
            
            # Convert to G1 format
            planner_msg, hand_msg = self.converter.convert(quest_input)
            
            # Build and send ZMQ planner message (includes hand joints)
            zmq_payload = build_planner_message(
                planner_msg, 
                left_hand=hand_msg.left_hand, 
                right_hand=hand_msg.right_hand
            )
            self.zmq_socket.send(zmq_payload)
            
            self.last_message_time = time.time()
            
            # Log statistics periodically
            now = time.time()
            if now - self.stats_last_time >= self.stats_interval:
                fps = self.stats_message_count / (now - self.stats_last_time)
                # Get actual joint values for debug
                left_joints = normalized_to_joint_positions(hand_msg.left_hand, is_left=True)
                right_joints = normalized_to_joint_positions(hand_msg.right_hand, is_left=False)
                logger.info(f"Stats: {fps:.1f} msg/s, mode={planner_msg.mode}, "
                           f"movement=({planner_msg.movement[0]:.2f}, {planner_msg.movement[1]:.2f}), "
                           f"hand=({hand_msg.left_hand:.2f}, {hand_msg.right_hand:.2f}), "
                           f"joints=L[{left_joints[0]:.2f},{left_joints[1]:.2f},{left_joints[2]:.2f}] "
                           f"R[{right_joints[0]:.2f},{right_joints[1]:.2f},{right_joints[2]:.2f}]")
                self.stats_message_count = 0
                self.stats_last_time = now
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}", exc_info=True)
    
    def start(self):
        """Start the bridge."""
        logger.info("Starting Quest to G1 bridge...")
        
        self.running = True
        
        # Setup ZMQ
        self.setup_zmq()
        
        # Setup MQTT
        self.setup_mqtt()
        
        # Connect to MQTT broker
        try:
            self.mqtt_client.connect(self.mqtt_client.host, self.mqtt_client.port, keepalive=60)
            logger.info(f"Connecting to MQTT broker at {self.mqtt_client.host}:{self.mqtt_client.port}")
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise
        
        # Start MQTT loop in background
        self.mqtt_client.loop_start()
        
        logger.info("Bridge started, waiting for Quest controller data...")
        
        # Main loop
        try:
            while self.running:
                time.sleep(0.1)
                
                # Check for MQTT connection issues
                if not self.mqtt_connected:
                    logger.warning("MQTT not connected, waiting...")
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the bridge."""
        logger.info("Stopping bridge...")
        self.running = False
        
        # Send stop command before shutting down
        if self.zmq_socket:
            try:
                stop_cmd = build_command_message(start=False, stop=True, planner=True)
                self.zmq_socket.send(stop_cmd)
                logger.info("Sent stop command: stop=True")
                time.sleep(0.1)  # Give time for message to be sent
            except Exception as e:
                logger.warning(f"Failed to send stop command: {e}")
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            logger.info("MQTT client disconnected")
        
        if self.zmq_socket:
            self.zmq_socket.close()
            logger.info("ZMQ socket closed")
        
        if self.zmq_context:
            self.zmq_context.term()
            logger.info("ZMQ context terminated")
        
        logger.info("Bridge stopped")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MQTT to ZMQ bridge for Quest 3 to G1 robot control"
    )
    parser.add_argument(
        "--device_id",
        type=str,
        required=False,
        default="piper-wee",
        help="Device ID for MQTT topic subscription (e.g., 'piper-wee')"
    )
    parser.add_argument(
        "--zmq_port",
        type=int,
        default=5556,
        help="ZMQ publisher port (default: 5556)"
    )
    parser.add_argument(
        "--mqtt_broker",
        type=str,
        default="mqtt://sora2.uclab.jp:1883",
        help="MQTT broker WebSocket URL (default: sora2.uclab.jp)"
    )
    parser.add_argument(
        "--client_id",
        type=str,
        default='PiPER-control-wee',
        help="MQTT client ID (default: auto-generated from device_id)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start bridge
    bridge = QuestG1Bridge(
        device_id=args.device_id,
        mqtt_broker=args.mqtt_broker,
        zmq_port=args.zmq_port,
        mqtt_client_id=args.client_id
    )
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        bridge.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start bridge
    bridge.start()


if __name__ == "__main__":
    main()
