#!/usr/bin/env python3
"""
Replay a recorded swing-episode npz as a ZMQ ``pose`` stream (protocol v3).

Purpose: verify tennis-primitive data collection in SONIC mode 2 ("smpl").
The recorded episode contains exactly the SMPL fields the deployed ZMQ endpoint
needs; republishing them as protocol v3 auto-selects encoder mode 2 (no motion
folder, no retargeting):

  recorded episode npz (smpl_pose, smpl_joints, body_quat_w, wrist joints)
    -> protocol v3 pose stream on :5556
    -> SONIC encoder (mode 2) -> 64-D token
    -> SONIC decoder -> 29-D G1 action
    -> MuJoCo G1 (run_sim.sh + deploy.sh --input-type zmq_manager sim)

Merged with run_sim_loop.py (which no longer needs to be the pose source) and
deploy.sh, this lets the deploy self-start in streamed-motion mode and, once it
first sees smpl_joints/smpl_pose, set encode_mode 2 automatically
(zmq_endpoint_interface.hpp lines 1706-1711).

Usage (from repo root, collection venv):
    source .venv_data_collection/bin/activate
    unset PYTHONPATH
    python gear_sonic/scripts/run_swing_episode_replayer.py \
        --npz swing_episodes/forehand_right_000000.npz --fps 50
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import tyro
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

# G1 wrist joints in joint_pos[29] (IsaacLab ordering used by the streamer).
G1_L_WRIST_ROLL_IDX = 23
G1_L_WRIST_PITCH_IDX = 25
G1_L_WRIST_YAW_IDX = 27
G1_R_WRIST_ROLL_IDX = 24
G1_R_WRIST_PITCH_IDX = 26
G1_R_WRIST_YAW_IDX = 28


@dataclass
class ReplayerConfig:
    """CLI config for the swing-episode replayer."""

    npz: str
    """Path to the recorded episode npz (from run_swing_episode_recorder.py)."""

    zmq_host: str = "localhost"
    """ZMQ host to publish the pose stream to."""

    zmq_port: int = 5556
    """ZMQ port for the pose topic (matches pico_manager / deploy)."""

    topic: str = "pose"
    """Pose ZMQ topic."""

    fps: int = 50
    """Replay rate (Hz). Match the deploy control loop (50 Hz default)."""

    loop: bool = False
    """Repeat the episode indefinitely (for repeated state-machine replay)."""


def load_episode(npz: str) -> dict:
    """Load a swing episode npz into per-frame numpy arrays."""
    with np.load(npz, allow_pickle=True) as data:
        out = {}
        # smpl_pose (T,21,3) axis-angle, smpl_joints (T,24,3) root-local,
        # body_quat_w (T,4), wrist joints (T,3 each).
        for key in (
            "smpl_pose",
            "smpl_joints",
            "body_quat_w",
            "left_wrist_joints",
            "right_wrist_joints",
            "frame_index",
        ):
            if key not in data:
                raise ValueError(f"npz missing '{key}'; not a swing-episode file")
            arr = data[key]
            # frame_index is (T,1) or (T,); flatten to scalar per frame.
            if key == "frame_index":
                out[key] = np.asarray(arr).reshape(-1, 1).astype(np.int64)
            else:
                out[key] = np.asarray(arr)
        out["T"] = len(out["smpl_pose"])
        if out["T"] == 0:
            raise ValueError("episode has no frames")
    return out


def build_pose_message(ep: dict, t: int) -> dict:
    """Assemble the protocol v3 pose message for frame *t*.

    joint_pos[29] is zeros with the 6 G1 wrist joints filled (matching
    pico_manager_thread_server's live output); joint_vel zeros.
    """
    joint_pos = np.zeros((1, 29), dtype=np.float32)
    lw = ep["left_wrist_joints"][t].astype(np.float32)
    rw = ep["right_wrist_joints"][t].astype(np.float32)
    joint_pos[0, G1_L_WRIST_ROLL_IDX] = lw[0]
    joint_pos[0, G1_L_WRIST_PITCH_IDX] = lw[1]
    joint_pos[0, G1_L_WRIST_YAW_IDX] = lw[2]
    joint_pos[0, G1_R_WRIST_ROLL_IDX] = rw[0]
    joint_pos[0, G1_R_WRIST_PITCH_IDX] = rw[1]
    joint_pos[0, G1_R_WRIST_YAW_IDX] = rw[2]

    return {
        "smpl_pose": ep["smpl_pose"][t : t + 1].astype(np.float32),
        "smpl_joints": ep["smpl_joints"][t : t + 1].astype(np.float32),
        "body_quat_w": ep["body_quat_w"][t : t + 1].astype(np.float32),
        "joint_pos": joint_pos,
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "frame_index": ep["frame_index"][t].reshape(1).astype(np.int64),
    }


class SwingEpisodeReplayer:
    """Publishes a recorded episode as protocol v3 pose messages on ZMQ."""

    def __init__(self, host: str, port: int, topic: str, fps: int, loop: bool) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        self.topic = topic
        self.frame_period = 1.0 / fps
        self.loop = loop
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(self.endpoint)
        print(f"[Replayer] Publishing 'pose' (protocol v3) to {self.endpoint} @ {fps} Hz")
        time.sleep(0.3)  # subscriber handshake

    def close(self) -> None:
        self.sock.close(0)
        self.ctx.term()

    def replay(self, npz: str) -> None:
        ep = load_episode(npz)
        T = ep["T"]
        print(f"[Replayer] Loaded {npz}: {T} frames")
        epoch = 0
        try:
            while True:
                if self.loop:
                    print(f"[Replayer] Repetition {epoch + 1}")
                frame_start = time.monotonic()
                for t in range(T):
                    fields = build_pose_message(ep, t)
                    packed = pack_pose_message(fields, topic=self.topic, version=3)
                    self.sock.send(packed)
                    # Pace to fps.
                    elapsed = time.monotonic() - frame_start
                    if elapsed < self.frame_period:
                        time.sleep(self.frame_period - elapsed)
                    frame_start = time.monotonic()
                epoch += 1
                if not self.loop:
                    print(f"[Replayer] Episode finished ({T} frames sent)")
                    return
        except KeyboardInterrupt:
            print("\n[Replayer] Shutting down...")
        finally:
            self.close()


def main() -> None:
    config = tyro.cli(ReplayerConfig)
    replayer = SwingEpisodeReplayer(
        host=config.zmq_host,
        port=config.zmq_port,
        topic=config.topic,
        fps=config.fps,
        loop=config.loop,
    )
    replayer.replay(config.npz)


if __name__ == "__main__":
    main()
