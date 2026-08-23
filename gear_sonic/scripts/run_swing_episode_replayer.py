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

WHY THE ROBOT USED TO FALL (root cause, confirmed in zmq_manager.hpp):
  The deploy's policy only runs when program_state_ == CONTROL, which is entered
  only when operator_state.start == true. The ZMQ command(start=True) flag is
  consumed in PLANNER mode (handlePlannerInput, line 527) but DROPPED in
  STREAMED_MOTION mode (pose_interface has its own start_control member that is
  only set by a local ']' keypress). So sending command(start=True,
  planner=False) directly switches the deploy into streamed-motion mode where it
  decodes/merges the pose stream but NEVER enters CONTROL -> the policy is never
  inferred -> the robot stands uncontrolled and falls.

  The PICO manager avoids this by doing it in two steps:
    1. command(start=True,  planner=True)  -> PLANNER mode -> deploy enters CONTROL
    2. command(start=True,  planner=False) -> STREAMED_MOTION -> already in CONTROL,
                                              keeps running policy on the pose stream

  This replayer replicates that sequence. In interactive mode you drive each
  step with a key so you can watch the deploy log confirm each transition.

INTERACTIVE MODE (default) — keyboard controls (single char, non-blocking):
  [ / b   Select previous episode (stops replay before switching)
  ] / n   Select next episode (stops replay before switching)
  i       Print selected episode status
  f       Toggle selected full-body SMPL / upper-body VR-3-point replay type
  p       Enter PLANNER mode + start policy
  m       Activate selected mode (STREAMED_MOTION for full body; PLANNER for upper body)
  r       Start selected episode from frame 0
  s   Stop replay
  l   Toggle looping for the selected episode
  ↑/↓ Increase/decrease replay rate by 5 Hz (minimum 5 Hz)
  o   Emergency stop                      (command start=False, stop=True, planner=True)
  ?   Show help
  q   Quit
  Recommended sequence:  p  ->  m  ->  r

AUTO MODE (--auto) — non-interactive, sends the full correct sequence then
replays (optionally looping):
  p (PLANNER+start) -> 0.5 s -> m (STREAMED_MOTION) -> 0.3 s -> replay

Usage (from repo root, collection venv):
    source .venv_data_collection/bin/activate
    unset PYTHONPATH
    python gear_sonic/scripts/run_swing_episode_replayer.py \
        --data-dir swing_episodes --fps 30

``--data-dir`` is searched recursively, so pass either a single recorder run
(e.g. ``swing_episodes/run_2``) or the parent containing multiple ``run_N``
directories. ``--npz`` remains supported for one-off backward-compatible runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import numpy as np
import tyro
import zmq

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

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

    data_dir: str | None = None
    """Directory containing episode ``.npz`` files, searched recursively (recommended)."""

    npz: str | None = None
    """Deprecated single-episode input. Use ``--data-dir`` for interactive selection."""

    episode_index: int = 0
    """Initially selected episode index within the discovered directory (0-based)."""

    zmq_host: str = "localhost"
    """ZMQ host to publish the pose stream to."""

    zmq_port: int = 5556
    """ZMQ port for the pose topic (matches pico_manager / deploy)."""

    topic: str = "pose"
    """Pose ZMQ topic."""

    fps: int = 30
    """Replay rate (Hz). Episodes were captured at 28-37 Hz; 30 is the
    recommended replay rate (50 would play back faster than real-time)."""

    loop: bool = False
    """Repeat the selected episode indefinitely; toggle with ``l`` in interactive mode."""

    auto: bool = False
    """Non-interactive: send the full correct sequence (PLANNER+start ->
    STREAMED_MOTION -> replay) with no keyboard. Default is interactive."""


_NATURAL_NUMBER = re.compile(r"(\d+)")


def _natural_path_key(path: Path) -> list[object]:
    """Sort ``run_2`` before ``run_10`` and episode 2 before episode 10."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in _NATURAL_NUMBER.split(str(path))
    ]


def discover_episode_paths(data_dir: str) -> list[str]:
    """Return all recursively discovered episode archives in natural path order."""
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise ValueError(f"data directory does not exist or is not a directory: {root}")
    paths = sorted((path for path in root.rglob("*.npz") if path.is_file()), key=_natural_path_key)
    if not paths:
        raise ValueError(f"no .npz episodes found under: {root}")
    return [str(path) for path in paths]


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
        out["upper_body_available"] = False
        if "vr_3pt_position" in data and "vr_3pt_orientation" in data:
            vr_position = np.asarray(data["vr_3pt_position"], dtype=np.float32)
            vr_orientation = np.asarray(data["vr_3pt_orientation"], dtype=np.float32)
            if (
                vr_position.shape == (out["T"], 9)
                and vr_orientation.shape == (out["T"], 12)
                and np.isfinite(vr_position).all()
                and np.isfinite(vr_orientation).all()
            ):
                out["vr_3pt_position"] = vr_position
                out["vr_3pt_orientation"] = vr_orientation
                out["upper_body_available"] = True
        out["recorded_fps"] = None
        if "timestamp_realtime" in data:
            timestamps = np.asarray(data["timestamp_realtime"], dtype=np.float64).reshape(-1)
            timestamps = timestamps[np.isfinite(timestamps)]
            if timestamps.size >= 2:
                span_seconds = float(timestamps[-1] - timestamps[0])
                if span_seconds > 0.0:
                    out["recorded_fps"] = (timestamps.size - 1) / span_seconds
    return out


def build_upper_body_planner_message(ep: dict, t: int) -> bytes:
    """Encode recorded VR head/wrist targets as an idle planner command.

    Planner/VR-3-point mode keeps locomotion at ``IDLE`` while the deploy maps
    the recorded left wrist, right wrist, and head targets into upper-body
    control.  It is intentionally distinct from protocol-v3 SMPL pose replay.
    """
    return build_planner_message(
        mode=0,  # LocomotionMode.IDLE
        movement=(0.0, 0.0, 0.0),
        facing=(1.0, 0.0, 0.0),
        speed=-1.0,
        height=-1.0,
        vr_3pt_position=ep["vr_3pt_position"][t].astype(np.float32),
        vr_3pt_orientation=ep["vr_3pt_orientation"][t].astype(np.float32),
    )


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
    """Publishes a recorded episode as protocol v3 pose messages on ZMQ.

    Supports interactive (keyboard-driven step-by-step) and auto modes.
    """

    def __init__(self, host: str, port: int, topic: str, fps: int, loop: bool) -> None:
        if fps < 5:
            raise ValueError("fps must be at least 5")
        self.endpoint = f"tcp://{host}:{port}"
        self.topic = topic
        self.fps = fps
        self.frame_period = 1.0 / self.fps
        self.loop = loop
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(self.endpoint)
        print(f"[Replayer] Publishing episode control messages (protocol v3) to {self.endpoint} @ {fps} Hz")
        time.sleep(0.3)  # subscriber handshake
        self.control_mode = "full_body"
        self.replaying = False
        self.frame_idx = 0
        self.ep: dict | None = None
        self.episode_paths: list[str] = []
        self.episode_cursor = 0
        self._old_termios = None
        self._stdin_fd = sys.stdin.fileno()

    # ------------------------------------------------------------------
    # Raw (non-blocking) single-char stdin, matching the C++ deploy pattern
    # ------------------------------------------------------------------
    def _enable_raw_stdin(self) -> None:
        self._old_termios = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)

    def _disable_raw_stdin(self) -> None:
        if self._old_termios is not None:
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
            self._old_termios = None

    def _read_key(self) -> str | None:
        """Read one keypress without blocking, decoding terminal arrow keys."""
        ready, _, _ = select.select([self._stdin_fd], [], [], 0)
        if not ready:
            return None
        ch = os.read(self._stdin_fd, 1)
        if ch != b"\x1b":
            return ch.decode("utf-8", errors="replace")

        # Arrow keys arrive as ESC [ A/B/C/D. Wait briefly so ESC alone still
        # remains harmless while a complete terminal escape sequence is decoded.
        ready, _, _ = select.select([self._stdin_fd], [], [], 0.01)
        if not ready:
            return "ESC"
        suffix = os.read(self._stdin_fd, 2)
        return {b"[A": "UP", b"[B": "DOWN", b"[C": "RIGHT", b"[D": "LEFT"}.get(suffix, "ESC")

    def adjust_fps(self, delta_hz: int) -> None:
        """Update live replay pacing, enforcing a safe 5 Hz lower bound."""
        self.fps = max(5, self.fps + delta_hz)
        self.frame_period = 1.0 / self.fps
        print(f"[Replayer] Replay rate: {self.fps} Hz")

    # ------------------------------------------------------------------
    # Episode library and selection
    # ------------------------------------------------------------------
    def set_episode_library(self, episode_paths: list[str], initial_index: int = 0) -> None:
        if not episode_paths:
            raise ValueError("episode library is empty")
        self.episode_paths = episode_paths
        self.episode_cursor = initial_index % len(episode_paths)
        self._load_selected_episode()

    def _load_selected_episode(self) -> None:
        path = self.episode_paths[self.episode_cursor]
        self.ep = load_episode(path)
        self.frame_idx = 0
        recorded_rate = self.ep["recorded_fps"]
        recorded_rate_text = (
            f", recorded {recorded_rate:.1f} Hz"
            if recorded_rate is not None
            else ", recorded rate unavailable"
        )
        print(
            f"[Replayer] Selected {self.episode_cursor + 1}/{len(self.episode_paths)}: "
            f"{path} ({self.ep['T']} frames{recorded_rate_text})"
        )

    def select_episode(self, step: int) -> None:
        """Select the adjacent episode, always stopping the current stream first."""
        if not self.episode_paths:
            raise RuntimeError("episode library has not been configured")
        if self.replaying:
            self.stop_replay()
        self.episode_cursor = (self.episode_cursor + step) % len(self.episode_paths)
        self._load_selected_episode()

    def print_selected_episode(self) -> None:
        assert self.ep is not None
        recorded_rate = self.ep["recorded_fps"]
        recorded_text = f"{recorded_rate:.1f} Hz" if recorded_rate is not None else "unavailable"
        print(
            f"[Replayer] Episode {self.episode_cursor + 1}/{len(self.episode_paths)} | "
            f"{self.episode_paths[self.episode_cursor]} | {self.ep['T']} frames | "
            f"recorded={recorded_text}"
        )
        print(
            f"           replay={1.0 / self.frame_period:.0f} Hz | mode={self.control_mode} | "
            f"loop={'on' if self.loop else 'off'}"
        )

    # ------------------------------------------------------------------
    # Command senders (the three deploy state-machine transitions)
    # ------------------------------------------------------------------
    def _send_command(self, start: bool, stop: bool, planner: bool) -> None:
        self.sock.send(build_command_message(start=start, stop=stop, planner=planner))

    def enter_planner_start(self) -> None:
        """Step 1: PLANNER mode + start policy (like A+B+X+Y, then ']').

        This is the step that actually gets the deploy into CONTROL state,
        because handlePlannerInput consumes command(start=True).
        """
        self._send_command(start=True, stop=False, planner=True)
        print("[Replayer] -> PLANNER mode + START policy  (command start=True, planner=True)")
        print("           [watch deploy log for: 'transitioning to CONTROL state']")

    def enter_streamed_motion(self) -> None:
        """Step 2: switch to STREAMED_MOTION / POSE mode (like A+X).

        The deploy is already in CONTROL from step 1, so it keeps running the
        policy and now consumes the pose stream (use_zmq_stream enabled by the
        mode-switch safety reset + TriggerZMQToggle).
        """
        self._send_command(start=True, stop=False, planner=False)
        print("[Replayer] -> STREAMED_MOTION (POSE) mode  (command start=True, planner=False)")
        print("           [watch deploy log for: 'Switched to: STREAMED MOTION mode']")

    def toggle_control_mode(self) -> None:
        """Toggle the selected replay type without sending any deploy command."""
        self.control_mode = (
            "upper_body" if self.control_mode == "full_body" else "full_body"
        )
        print(f"[Replayer] Selected control mode: {self.control_mode}")

    def _send_upper_body_frame(self, t: int) -> None:
        assert self.ep is not None
        self.sock.send(build_upper_body_planner_message(self.ep, t))

    def activate_selected_control_mode(self) -> None:
        """Apply the selected mode after ``p`` has started policy/planner control."""
        self.stop_replay()
        if self.control_mode == "full_body":
            # Clear any VR-3-point capability latched by a prior upper-body run
            # before handing control to protocol-v3 SMPL streamed motion.
            self.sock.send(
                build_planner_message(
                    mode=0,
                    movement=(0.0, 0.0, 0.0),
                    facing=(1.0, 0.0, 0.0),
                    speed=-1.0,
                    height=-1.0,
                )
            )
            self.enter_streamed_motion()
            print("[Replayer] FULL-BODY mode active: replay sends protocol-v3 SMPL pose frames")
            return
        if self.ep is None or not self.ep["upper_body_available"]:
            print("[Replayer] UPPER-BODY mode unavailable: episode lacks valid VR 3-point targets")
            return
        # VR-3-point is a planner input, not a streamed-motion input. Preload its
        # first target before switching, matching the live PICO manager ordering.
        # Teleop encoder mode also needs a planner-generated lower-body reference,
        # so upper-body replay must remain in PLANNER rather than STREAMED_MOTION.
        self._send_upper_body_frame(0)
        self.enter_planner_start()
        print("[Replayer] UPPER-BODY mode active: PLANNER provides lower-body reference and VR 3-point targets")

    def emergency_stop(self) -> None:
        """Step 3: emergency stop (like 'O')."""
        self._send_command(start=False, stop=True, planner=True)
        print("[Replayer] -> EMERGENCY STOP  (command start=False, stop=True, planner=True)")
        self.replaying = False

    # ------------------------------------------------------------------
    # Replay control
    # ------------------------------------------------------------------
    def start_replay(self) -> None:
        if self.ep is None:
            print("[Replayer] No episode loaded")
            return
        self.replaying = True
        self.frame_idx = 0
        print(
            f"[Replayer] REPLAY START  ({self.ep['T']} frames @ {1.0 / self.frame_period:.0f} Hz, "
            f"mode={self.control_mode})"
        )

    def stop_replay(self) -> None:
        self.replaying = False
        print("[Replayer] REPLAY STOP")

    def _send_one_frame(self) -> None:
        assert self.ep is not None
        T = self.ep["T"]
        if self.frame_idx >= T:
            if self.loop:
                self.frame_idx = 0
            else:
                self.replaying = False
                print("[Replayer] Episode finished")
                return
        if self.control_mode == "upper_body":
            self._send_upper_body_frame(self.frame_idx)
        else:
            fields = build_pose_message(self.ep, self.frame_idx)
            packed = pack_pose_message(fields, topic=self.topic, version=3)
            self.sock.send(packed)
        self.frame_idx += 1

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    def interactive(self, episode_paths: list[str], initial_index: int = 0) -> None:
        self.set_episode_library(episode_paths, initial_index)
        print()
        print("=== Interactive swing replayer ===")
        print("  <- / b   Previous episode (stops replay before switching)")
        print("  -> / n   Next episode (stops replay before switching)")
        print("  i       Print selected episode status")
        print("  f       Toggle selected full-body SMPL / upper-body VR-3-point replay type")
        print("  p       Enter PLANNER mode + start policy")
        print("  m       Activate selected mode (STREAMED_MOTION for full body; PLANNER for upper body)")
        print("  r       Start selected episode from frame 0")
        print("  s       Stop replay")
        print("  l       Toggle loop for the selected episode")
        print("  ↑ / ↓   Increase / decrease replay rate by 5 Hz (minimum 5 Hz)")
        print("  o       Emergency stop")
        print("  ?       Show this help")
        print("  q       Quit")
        print()
        print("  Sequence: p -> f (if upper body) -> m -> r. Default selection is full body.")
        print()
        self._enable_raw_stdin()
        next_frame_time = 0.0
        try:
            while True:
                now = time.monotonic()
                key = self._read_key()
                if key is not None:
                    if key == "UP":
                        self.adjust_fps(5)
                    elif key == "DOWN":
                        self.adjust_fps(-5)
                    elif key in ("q", "Q"):
                        break
                    elif key in ("LEFT", "b", "B"):
                        self.select_episode(-1)
                    elif key in ("RIGHT", "n", "N"):
                        self.select_episode(1)
                    elif key in ("i", "I"):
                        self.print_selected_episode()
                    elif key in ("f", "F"):
                        self.toggle_control_mode()
                    elif key in ("p", "P"):
                        self.enter_planner_start()
                    elif key in ("m", "M"):
                        self.activate_selected_control_mode()
                    elif key in ("r", "R"):
                        self.start_replay()
                        next_frame_time = now
                    elif key in ("s", "S"):
                        self.stop_replay()
                    elif key in ("l", "L"):
                        self.loop = not self.loop
                        print(f"[Replayer] Loop {'ON' if self.loop else 'OFF'}")
                    elif key in ("o", "O"):
                        self.emergency_stop()
                    elif key in ("?", "h", "H"):
                        print("  [/b=previous  ]/n=next  i=status  f=toggle full/upper body  "
                              "p=PLANNER+start  m=activate selection  ↑/↓=rate±5Hz  "
                              "r=replay  s=stop  l=loop  o=estop  q=quit")
                # Send a frame if replaying and the frame period has elapsed.
                if self.replaying and now >= next_frame_time:
                    self._send_one_frame()
                    next_frame_time = now + self.frame_period
                # Sleep to avoid busy-waiting (loop runs at ~200 Hz).
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n[Replayer] Shutting down...")
        finally:
            self._disable_raw_stdin()
            self.close()

    # ------------------------------------------------------------------
    # Auto (non-interactive) loop
    # ------------------------------------------------------------------
    def auto(self, episode_paths: list[str], initial_index: int = 0) -> None:
        self.set_episode_library(episode_paths, initial_index)
        assert self.ep is not None
        T = self.ep["T"]
        print("[Replayer] AUTO: step 1/3  PLANNER mode + start policy")
        self.enter_planner_start()
        time.sleep(0.5)
        print("[Replayer] AUTO: step 2/3  switch to STREAMED_MOTION")
        self.enter_streamed_motion()
        time.sleep(0.3)
        print("[Replayer] AUTO: step 3/3  starting replay")
        self.start_replay()
        frame_start = time.monotonic()
        epoch = 0
        try:
            while True:
                if self.frame_idx >= T:
                    if not self.loop:
                        break
                    self.frame_idx = 0
                    epoch += 1
                    print(f"[Replayer] Repetition {epoch + 1}")
                    # Re-send the streamed-motion command on each loop so the
                    # deploy re-enters the pose-consumption path cleanly.
                    self._send_command(start=True, stop=False, planner=False)
                    time.sleep(0.2)
                fields = build_pose_message(self.ep, self.frame_idx)
                packed = pack_pose_message(fields, topic=self.topic, version=3)
                self.sock.send(packed)
                self.frame_idx += 1
                elapsed = time.monotonic() - frame_start
                if elapsed < self.frame_period:
                    time.sleep(self.frame_period - elapsed)
                frame_start = time.monotonic()
        except KeyboardInterrupt:
            print("\n[Replayer] Shutting down...")
        finally:
            self.close()

    def close(self) -> None:
        self.sock.close(0)
        self.ctx.term()


def main() -> None:
    config = tyro.cli(ReplayerConfig)
    if config.data_dir is not None:
        if config.npz is not None:
            raise ValueError("provide either --data-dir or deprecated --npz, not both")
        episode_paths = discover_episode_paths(config.data_dir)
    elif config.npz is not None:
        episode_paths = [config.npz]
    else:
        raise ValueError("provide --data-dir containing recorded episodes")

    print(f"[Replayer] Found {len(episode_paths)} episode(s)")
    replayer = SwingEpisodeReplayer(
        host=config.zmq_host,
        port=config.zmq_port,
        topic=config.topic,
        fps=config.fps,
        loop=config.loop,
    )
    if config.auto:
        replayer.auto(episode_paths, config.episode_index)
    else:
        replayer.interactive(episode_paths, config.episode_index)


if __name__ == "__main__":
    main()
