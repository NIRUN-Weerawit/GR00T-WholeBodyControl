#!/usr/bin/env python3
"""
Standalone episode recorder for tennis swing primitives — NO deploy / sim / camera.

Records the essential replay fields for the primitive state machine, reusing the
protocol of ``run_data_exporter.py``:

  - ZMQ SUB on ``pose``, ``planner``, ``manager_state`` at ``:5556``
  - frames unpacked with ``unpack_pose_message`` (1280-byte JSON header +
    concatenated little-endian fields)
  - a two-state recorder: IDLE -> RECORDING -> IDLE; Left-Grip+A starts,
    and the next Left-Grip+A stops and saves immediately
  - recording controls from ``manager_state``: ``toggle_data_collection``
    (Left-Grip+A) toggle start/save, ``toggle_data_abort`` (Left-Grip+B) discard.
    An optional keyboard fallback (``c``/``x``, reuse ``run_recording_keyboard.py``
    on port 5580) is also supported.

Output difference vs ``run_data_exporter.py``: writes ONE labeled ``.npz`` per
episode instead of the lerobot parquet + MP4 layout. Every invocation creates a
fresh ``run_N`` directory under ``--output-dir`` (default: ``swing_episodes``).

Runs standalone: no MuJoCo, no deployment binary, no camera server, no Remote
Vision. PICO body tracking (``pico_manager_thread_server.py --manager``) is the
only required source, publishing ``pose`` / ``manager_state`` on ``:5556``.

IMPORTANT: the manager must be in POSE mode for pose frames to be published.
It starts in StreamMode.OFF (manager_state only). Flow:
  1. A+B+X+Y (start combo) -> PLANNER (calibrates VR 3pt in zero-ref pose)
  2. A+X                   -> POSE (pose frames now stream)
Recording while not in POSE mode captures zero frames and the episode is
discarded with a warning.

Usage (from repo root):
    source .venv_data_collection/bin/activate
    unset PYTHONPATH
    python gear_sonic/scripts/run_swing_episode_recorder.py --primitive forehand_right
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import threading
import time

import numpy as np
import tyro
import zmq

from gear_sonic.utils.data_collection.episode_state import EpisodeState
from gear_sonic.utils.data_collection.keyboard_subscriber import ZMQKeyboardSubscriber


_RUN_DIR_PATTERN = re.compile(r"run_(\d+)$")


def create_run_output_dir(output_root: str) -> str:
    """Create and return the next numbered ``run_N`` directory under a root.

    The mkdir loop makes an invocation collision-safe: if another recorder creates
    the candidate first, this recorder simply retries with the next number.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    existing_indices = [
        int(match.group(1))
        for child in root.iterdir()
        if child.is_dir() and (match := _RUN_DIR_PATTERN.fullmatch(child.name))
    ]
    next_index = max(existing_indices, default=0) + 1

    while True:
        run_dir = root / f"run_{next_index}"
        try:
            run_dir.mkdir()
            return str(run_dir)
        except FileExistsError:
            next_index += 1


# ---------------------------------------------------------------------------
# Wire protocol (mirrors run_data_exporter.py's unpack_pose_message)
# ---------------------------------------------------------------------------

POSE_HEADER_SIZE = 1280

# StreamMode values from pico_manager_thread_server.StreamMode.
STREAM_MODE_NAMES = {
    0: "OFF",
    1: "POSE",
    2: "PLANNER",
    3: "PLANNER_FROZEN_UPPER_BODY",
    4: "POSE_PAUSE",
    5: "PLANNER_VR_3PT",
}


def unpack_pose_message(packed_data: bytes, topic: str = "pose") -> dict:
    """Unpack a single-frame packed message from pico_manager_thread_server.

    Wire format: [topic_prefix][1280-byte JSON header][concatenated binary fields]
    """
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + POSE_HEADER_SIZE:
        raise ValueError(
            f"Packed data too small: {len(packed_data)} < {offset + POSE_HEADER_SIZE}"
        )

    header_bytes = packed_data[offset : offset + POSE_HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]

    header = json.loads(header_bytes.decode("utf-8"))
    fields = header.get("fields", [])

    result = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + POSE_HEADER_SIZE
    dtype_map = {
        "f32": np.float32,
        "f64": np.float64,
        "i32": np.int32,
        "i64": np.int64,
        "bool": bool,
    }

    for field in fields:
        dtype = dtype_map.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(
                packed_data[current_offset : current_offset + n_bytes], dtype=dtype
            )
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes

    return result


# G1 wrist joints in joint_pos[29] (IsaacLab ordering used by the streamer).
G1_L_WRIST_ROLL_IDX = 23
G1_L_WRIST_PITCH_IDX = 25
G1_L_WRIST_YAW_IDX = 27
G1_R_WRIST_ROLL_IDX = 24
G1_R_WRIST_PITCH_IDX = 26
G1_R_WRIST_YAW_IDX = 28


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SwingRecorderConfig:
    """CLI config for the standalone swing-primitive episode recorder."""

    primitive: str = "forehand_right"
    """Primitive label written into each episode file (default forehand_right)."""

    output_dir: str = "swing_episodes"
    """Parent directory; each invocation creates a fresh numbered ``run_N`` inside it."""

    zmq_host: str = "localhost"
    """ZMQ host for Sonic SMPL pose / manager_state messages."""

    zmq_port: int = 5556
    """ZMQ port for Sonic SMPL pose / manager_state messages."""

    keyboard_fallback: bool = False
    """Also listen for 'c'/'x' recording controls from run_recording_keyboard.py."""

    keyboard_port: int = 5580
    """ZMQ port for the optional keyboard recording-control publisher."""

    episode_index_start: int = 0
    """First episode index to use in the output filename."""


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

# Replay-essential fields captured per pose message, in §4.4 order.
_REPLAY_KEYS = (
    "smpl_pose",       # (21, 3) local axis-angle
    "smpl_joints",     # (24, 3) root-local Z-up joint positions
    "body_quat_w",     # (4,)    root orientation, SONIC convention
    "left_wrist_joints",  # (3,)  G1 wrist roll/pitch/yaw (racked hand)
    "right_wrist_joints",  # (3,)
    "vr_3pt_position",    # (9,)
    "vr_3pt_orientation",  # (12,)
    "frame_index",       # ()
    "timestamp_realtime",  # () original PICO timestamp (seconds)
    "timestamp_monotonic",  # ()
)


class SwingEpisodeRecorder:
    """Subscribes to Sonic pose/manager_state and writes per-episode npz files."""

    def __init__(
        self,
        zmq_host: str,
        zmq_port: int,
        output_dir: str,
        primitive: str,
        keyboard_fallback: bool = False,
        keyboard_port: int = 5580,
        episode_index_start: int = 0,
    ) -> None:
        self.host = zmq_host
        self.port = zmq_port
        self.output_dir = output_dir
        self.primitive = primitive
        self.episode_index = episode_index_start

        self._episode_state = EpisodeState()
        self._keyboard_listener = None
        if keyboard_fallback:
            self._keyboard_listener = ZMQKeyboardSubscriber(port=keyboard_port, host=zmq_host)

        self._manager_toggle_dc = False
        self._manager_toggle_da = False

        # Track the manager's StreamMode so we can warn before the user
        # records an empty episode (pose frames only stream in POSE mode).
        self._stream_mode = None
        self._warned_not_pose = False

        # Per-episode buffers.
        self._buffers = {key: [] for key in _REPLAY_KEYS}
        self._episode_start_ns = None

        self._socket = None
        self._ctx = None
        self._stop = threading.Event()
        self._init_socket()

    def _init_socket(self) -> None:
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.CONFLATE, 0)
        self._socket.setsockopt(zmq.RCVHWM, 20)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
        self._socket.connect(f"tcp://{self.host}:{self.port}")
        time.sleep(0.5)
        print(
            f"[Recorder] Connected to ZMQ at {self.host}:{self.port} "
            "(pose, planner, manager_state)"
        )
        print(f"[Recorder] Writing to {self.output_dir} (primitive={self.primitive})")

    def stop(self) -> None:
        """Signal the run loop to exit. Safe to call from another thread."""
        self._stop.set()

    def close(self) -> None:
        """Release ZMQ resources. Idempotent; run() calls it from its own thread."""
        self._stop.set()
        if self._socket is not None:
            try:
                self._socket.close()
            except zmq.ZMQError:
                pass
            self._socket = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass
            self._ctx = None
        if self._keyboard_listener is not None:
            self._keyboard_listener.close()

    def filename(self) -> str:
        return f"{self.primitive}_{self.episode_index:06d}.npz"

    # -- control handling (mirrors run_data_exporter.py) --

    def _check_recording_commands(self) -> None:
        key = None
        if self._keyboard_listener is not None:
            key = self._keyboard_listener.read_msg()

        if self._manager_toggle_da:
            key = "x"
            self._manager_toggle_da = False
        elif self._manager_toggle_dc:
            key = "c"
            self._manager_toggle_dc = False

        if key == "c":
            state = self._episode_state.get_state()
            if state == self._episode_state.IDLE:
                self._episode_state.change_state()
                self._episode_start_ns = time.monotonic()
                print(f"[Recorder] Started episode {self.episode_index}")
            elif state == self._episode_state.RECORDING:
                # This standalone recorder has no review/annotation stage, so
                # one second toggle deliberately means stop *and* save.
                if self.save_episode():
                    self.episode_index += 1
                    print("[Recorder] Saved episode and back to idle")
                self._episode_state.reset_state()
        elif key == "x":
            if self._episode_state.get_state() == self._episode_state.RECORDING:
                self.discard_episode()
                self._episode_state.reset_state()
                self._episode_start_ns = None
                print("[Recorder] Discarded episode")

    def _handle_manager_state(self, raw: bytes) -> None:
        try:
            data = unpack_pose_message(raw, topic="manager_state")
        except Exception:
            return

        if "stream_mode" in data:
            self._stream_mode = int(data["stream_mode"].flat[0])
            if self._stream_mode != 1 and not self._warned_not_pose:
                self._warned_not_pose = True
                mode_name = STREAM_MODE_NAMES.get(self._stream_mode, str(self._stream_mode))
                print(
                    f"[Recorder] WARNING: manager is in StreamMode.{mode_name} - pose "
                    "frames are only published in POSE mode. Press A+B+X+Y then A+X on "
                    "the PICO to enter POSE mode, or episodes will be empty."
                )
            elif self._stream_mode == 1:
                self._warned_not_pose = False

        if self._extract_bool(data, "toggle_data_collection"):
            self._manager_toggle_dc = True
        if self._extract_bool(data, "toggle_data_abort"):
            self._manager_toggle_da = True

    @staticmethod
    def _extract_bool(data: dict, key: str) -> bool:
        if key not in data:
            return False
        val = data[key]
        if val.size:
            return bool(val.flat[0])
        return False

    # -- frame capture --

    def _handle_pose_message(self, raw: bytes) -> None:
        if self._episode_state.get_state() != self._episode_state.RECORDING:
            return

        try:
            pose_data = unpack_pose_message(raw, topic="pose")
        except Exception as e:
            print(f"[Recorder] Error unpacking pose message: {e}")
            return

        if "smpl_joints" not in pose_data or len(pose_data["smpl_joints"].shape) != 3:
            return

        b = self._buffers

        # smpl_joints is stacked with a leading batch dim; take first frame (as
        # run_data_exporter does).
        b["smpl_joints"].append(pose_data["smpl_joints"][0].copy())

        smpl_pose = np.zeros((21, 3), dtype=np.float32)
        if "smpl_pose" in pose_data:
            raw_pose = pose_data["smpl_pose"]
            if raw_pose.ndim == 3:
                smpl_pose = raw_pose[0].reshape(21, 3).astype(np.float32)
            elif raw_pose.ndim == 2:
                smpl_pose = raw_pose.reshape(21, 3).astype(np.float32)
        b["smpl_pose"].append(smpl_pose)

        body_quat_w = np.zeros(4, dtype=np.float32)
        if "body_quat_w" in pose_data:
            body_quat_w = pose_data["body_quat_w"][0].astype(np.float32)
        b["body_quat_w"].append(body_quat_w)

        # G1 wrist joints from joint_pos[29].
        left_wrist = np.zeros(3, dtype=np.float32)
        right_wrist = np.zeros(3, dtype=np.float32)
        if "joint_pos" in pose_data and len(pose_data["joint_pos"].shape) == 2:
            jp = pose_data["joint_pos"][0]
            left_wrist = np.array(
                [
                    jp[G1_L_WRIST_ROLL_IDX],
                    jp[G1_L_WRIST_PITCH_IDX],
                    jp[G1_L_WRIST_YAW_IDX],
                ],
                dtype=np.float32,
            )
            right_wrist = np.array(
                [
                    jp[G1_R_WRIST_ROLL_IDX],
                    jp[G1_R_WRIST_PITCH_IDX],
                    jp[G1_R_WRIST_YAW_IDX],
                ],
                dtype=np.float32,
            )
        b["left_wrist_joints"].append(left_wrist)
        b["right_wrist_joints"].append(right_wrist)

        # VR 3-point pose (flat).
        b["vr_3pt_position"].append(
            pose_data["vr_position"].flatten().astype(np.float32)
            if "vr_position" in pose_data and pose_data["vr_position"].size == 9
            else np.zeros(9, dtype=np.float32)
        )
        b["vr_3pt_orientation"].append(
            pose_data["vr_orientation"].flatten().astype(np.float32)
            if "vr_orientation" in pose_data and pose_data["vr_orientation"].size == 12
            else np.zeros(12, dtype=np.float32)
        )

        b["frame_index"].append(
            np.array([pose_data["frame_index"].flat[0]], dtype=np.int64)
            if "frame_index" in pose_data
            else np.array([0], dtype=np.int64)
        )
        b["timestamp_realtime"].append(
            np.array([pose_data["timestamp_realtime"].flat[0]], dtype=np.float64)
            if "timestamp_realtime" in pose_data
            else np.array([time.time()], dtype=np.float64)
        )
        b["timestamp_monotonic"].append(
            np.array([pose_data["timestamp_monotonic"].flat[0]], dtype=np.float64)
            if "timestamp_monotonic" in pose_data
            else np.array([time.monotonic()], dtype=np.float64)
        )

    # -- episode save / discard --

    def save_episode(self) -> bool:
        """Assemble and write one labeled npz for the just-finished episode.

        Returns True if a file was written, False if the episode had no pose
        frames (typically because the manager was not in POSE mode).
        """
        import os

        n_frames = len(self._buffers["smpl_joints"])
        if n_frames == 0:
            mode_name = STREAM_MODE_NAMES.get(self._stream_mode or 0, "unknown")
            print(
                f"[Recorder] WARNING: no pose frames captured (manager StreamMode="
                f"{mode_name}) - discarding empty episode. Enter POSE mode (A+X) and "
                "record again."
            )
            self._clear_buffers()
            self._episode_start_ns = None
            return False

        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, self.filename())

        arrays: dict[str, np.ndarray] = {}
        for key in _REPLAY_KEYS:
            stacked = np.stack(self._buffers[key])
            arrays[key] = stacked

        arrays["primitive_label"] = np.array([self.primitive.encode("utf-8")])
        arrays["episode_index"] = np.array([self.episode_index], dtype=np.int32)
        arrays["recording_seconds"] = np.array(
            [time.monotonic() - self._episode_start_ns]
            if self._episode_start_ns is not None
            else [0.0],
            dtype=np.float64,
        )

        np.savez_compressed(path, **arrays)

        print(f"[Recorder] Saved {path} ({n_frames} frames)")
        self._clear_buffers()
        self._episode_start_ns = None
        return True

    def discard_episode(self) -> None:
        """Discard current episode buffers without writing."""
        self._clear_buffers()
        self._episode_start_ns = None

    def _clear_buffers(self) -> None:
        self._buffers = {key: [] for key in _REPLAY_KEYS}

    # -- run loop --

    def run(self) -> None:
        print("[Recorder] Waiting for episodes. Left-Grip+A = start/save, "
              "Left-Grip+B = discard.")
        try:
            while not self._stop.is_set():
                self._check_recording_commands()
                max_polls = 20
                for _ in range(max_polls):
                    try:
                        raw = self._socket.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    except zmq.ZMQError:
                        return  # socket closed by another thread
                    if raw.startswith(b"manager_state"):
                        self._handle_manager_state(raw)
                    elif raw.startswith(b"pose"):
                        self._handle_pose_message(raw)
                    # planner is subscribed for protocol fidelity but not needed
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\n[Recorder] Shutting down...")
        finally:
            self.close()


def main() -> None:
    config = tyro.cli(SwingRecorderConfig)
    run_output_dir = create_run_output_dir(config.output_dir)
    print(f"[Recorder] Created run directory: {run_output_dir}")
    recorder = SwingEpisodeRecorder(
        zmq_host=config.zmq_host,
        zmq_port=config.zmq_port,
        output_dir=run_output_dir,
        primitive=config.primitive,
        keyboard_fallback=config.keyboard_fallback,
        keyboard_port=config.keyboard_port,
        episode_index_start=config.episode_index_start,
    )
    recorder.run()


if __name__ == "__main__":
    main()
