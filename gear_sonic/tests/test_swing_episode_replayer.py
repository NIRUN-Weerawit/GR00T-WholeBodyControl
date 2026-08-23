"""Tests for the swing-episode replayer (protocol v3 pose stream)."""

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import zmq

from gear_sonic.scripts.run_swing_episode_replayer import (
    G1_L_WRIST_YAW_IDX,
    G1_R_WRIST_YAW_IDX,
    SwingEpisodeReplayer,
    build_pose_message,
    build_upper_body_planner_message,
    discover_episode_paths,
    load_episode,
)
from gear_sonic.scripts.run_swing_episode_recorder import unpack_pose_message

HEADER_SIZE = 1280
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def make_episode_npz(save_path: str, T: int = 6, recorded_fps: float | None = None) -> None:
    """Write a synthetic swing-episode npz matching the recorder schema."""
    arrays = {
        "smpl_pose": np.zeros((T, 21, 3), dtype=np.float32),
        "smpl_joints": np.random.rand(T, 24, 3).astype(np.float32),
        "body_quat_w": np.random.rand(T, 4).astype(np.float32),
        "left_wrist_joints": np.random.rand(T, 3).astype(np.float32),
        "right_wrist_joints": np.random.rand(T, 3).astype(np.float32),
        "frame_index": np.arange(T, dtype=np.int64).reshape(T, 1),
        "primitive_label": np.array([b"forehand_right"]),
        "episode_index": np.array([0], dtype=np.int32),
        "recording_seconds": np.array([2.0], dtype=np.float64),
    }
    if recorded_fps is not None:
        arrays["timestamp_realtime"] = (
            1_000.0 + np.arange(T, dtype=np.float64) / recorded_fps
        ).reshape(T, 1)
    np.savez_compressed(save_path, **arrays)


class EpisodeDiscoveryTest(unittest.TestCase):
    def test_recursively_discovers_episodes_in_natural_run_order(self):
        with tempfile.TemporaryDirectory() as data_dir:
            run_2 = os.path.join(data_dir, "run_2")
            run_10 = os.path.join(data_dir, "run_10")
            os.mkdir(run_2)
            os.mkdir(run_10)
            first = os.path.join(run_2, "forehand_right_000000.npz")
            second = os.path.join(run_2, "forehand_right_000002.npz")
            third = os.path.join(run_10, "backhand_left_000000.npz")
            for path in (first, second, third):
                make_episode_npz(path)
            with open(os.path.join(data_dir, "notes.txt"), "w", encoding="utf-8") as handle:
                handle.write("not an episode")

            paths = discover_episode_paths(data_dir)

            self.assertEqual(paths, [first, second, third])

    def test_episode_selection_wraps_and_stops_active_replay(self):
        with tempfile.TemporaryDirectory() as data_dir:
            first = os.path.join(data_dir, "episode_0.npz")
            second = os.path.join(data_dir, "episode_1.npz")
            make_episode_npz(first, T=3)
            make_episode_npz(second, T=5)

            # Avoid opening a real ZMQ socket: selection is independent of transport.
            player = object.__new__(SwingEpisodeReplayer)
            player.replaying = True
            player.frame_idx = 99
            player.episode_paths = []
            player.episode_cursor = 0
            player.set_episode_library([first, second])
            player.select_episode(-1)

            self.assertFalse(player.replaying)
            self.assertEqual(player.episode_cursor, 1)
            self.assertEqual(player.ep["T"], 5)
            self.assertEqual(player.frame_idx, 0)

    def test_control_mode_toggle_changes_selection_without_sending_commands(self):
        player = object.__new__(SwingEpisodeReplayer)
        player.control_mode = "full_body"

        player.toggle_control_mode()
        self.assertEqual(player.control_mode, "upper_body")

        player.toggle_control_mode()
        self.assertEqual(player.control_mode, "full_body")

    def test_upper_body_activation_preloads_target_then_enters_planner(self):
        class Socket:
            def __init__(self):
                self.messages = []

            def send(self, message):
                self.messages.append(message)

        player = object.__new__(SwingEpisodeReplayer)
        player.sock = Socket()
        player.replaying = False
        player.control_mode = "upper_body"
        player.ep = {
            "upper_body_available": True,
            "vr_3pt_position": np.zeros((1, 9), dtype=np.float32),
            "vr_3pt_orientation": np.zeros((1, 12), dtype=np.float32),
        }

        player.activate_selected_control_mode()

        topics = [message[: message.index(b"{")] for message in player.sock.messages]
        self.assertEqual(topics, [b"planner", b"command"])
        self.assertEqual(list(_command_payload(player.sock.messages[1])), [1, 0, 1])

    def test_replay_fps_adjustment_uses_five_hz_steps_and_floor(self):
        player = object.__new__(SwingEpisodeReplayer)
        player.fps = 30
        player.frame_period = 1.0 / player.fps

        player.adjust_fps(5)
        self.assertEqual(player.fps, 35)
        self.assertAlmostEqual(player.frame_period, 1.0 / 35)

        player.adjust_fps(-100)
        self.assertEqual(player.fps, 5)
        self.assertAlmostEqual(player.frame_period, 0.2)


class LoadEpisodeTest(unittest.TestCase):
    def test_load_and_shapes(self):
        _, path = tempfile.mkstemp(suffix=".npz")
        os.remove(path)
        try:
            make_episode_npz(path, T=5)
            ep = load_episode(path)
            self.assertEqual(ep["T"], 5)
            self.assertEqual(ep["smpl_pose"].shape, (5, 21, 3))
            self.assertEqual(ep["smpl_joints"].shape, (5, 24, 3))
            self.assertEqual(ep["body_quat_w"].shape, (5, 4))
            self.assertEqual(ep["frame_index"].shape, (5, 1))
            self.assertIsNone(ep["recorded_fps"])
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_loads_timestamp_derived_recording_frequency(self):
        _, path = tempfile.mkstemp(suffix=".npz")
        os.remove(path)
        try:
            make_episode_npz(path, T=6, recorded_fps=75.0)

            ep = load_episode(path)

            self.assertAlmostEqual(ep["recorded_fps"], 75.0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_missing_key_raises(self):
        # npz missing smpl_joints -> ValueError
        _, path = tempfile.mkstemp(suffix=".npz")
        os.remove(path)
        try:
            np.savez(path, smpl_pose=np.zeros((2, 21, 3)))
            with self.assertRaises(ValueError):
                load_episode(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


class BuildUpperBodyPlannerMessageTest(unittest.TestCase):
    def test_encodes_recorded_vr_targets_with_idle_locomotion(self):
        ep = {
            "vr_3pt_position": np.arange(18, dtype=np.float32).reshape(2, 9),
            "vr_3pt_orientation": np.arange(24, dtype=np.float32).reshape(2, 12),
        }

        packed = build_upper_body_planner_message(ep, t=1)
        data = unpack_pose_message(packed, topic="planner")

        self.assertEqual(data["mode"].flat[0], 0)  # LocomotionMode.IDLE
        np.testing.assert_array_equal(data["movement"], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(data["facing"], np.array([1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(data["vr_position"], ep["vr_3pt_position"][1])
        np.testing.assert_array_equal(data["vr_orientation"], ep["vr_3pt_orientation"][1])


class BuildPoseMessageTest(unittest.TestCase):
    def setUp(self):
        _, self.path = tempfile.mkstemp(suffix=".npz")
        os.remove(self.path)
        make_episode_npz(self.path, T=8)
        self.ep = load_episode(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_message_fields_protocol_v3(self):
        fields = build_pose_message(self.ep, t=3)
        # Required protocol v3 fields present.
        for key in ("smpl_pose", "smpl_joints", "body_quat_w", "joint_pos",
                    "joint_vel", "frame_index"):
            self.assertIn(key, fields)
        # joint_pos is the 29-D vector with wrist slots filled.
        jp = fields["joint_pos"][0]
        self.assertEqual(jp.shape, (29,))
        # Left wrist yaw and right wrist yaw indices carry the recorded values.
        self.assertAlmostEqual(
            jp[G1_L_WRIST_YAW_IDX], self.ep["left_wrist_joints"][3][2]
        )
        self.assertAlmostEqual(
            jp[G1_R_WRIST_YAW_IDX], self.ep["right_wrist_joints"][3][2]
        )
        # We only filled the 6 wrist indices; the other 23 are exactly zero.
        self.assertEqual(np.count_nonzero(fields["joint_pos"][0]), 6)

    def test_wire_roundtrip_headers(self):
        fields = build_pose_message(self.ep, t=0)
        packed = pack_pose_message_from_local(fields)
        data = unpack_pose_message(packed, topic="pose")
        self.assertEqual(data["smpl_joints"].shape[0], 1)
        self.assertAlmostEqual(
            data["smpl_joints"][0, 0, 0], fields["smpl_joints"][0, 0, 0], places=5
        )


def pack_pose_message_from_local(fields: dict) -> bytes:
    """Same topic+header+payload layout as pack_pose_message (protocol v3)."""
    import json

    header = {
        "v": 3,
        "endian": "le",
        "count": 1,
        "fields": [
            {
                "name": k,
                "dtype": _dtype_str(v.dtype),
                "shape": list(v.shape),
            }
            for k, v in fields.items()
        ],
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    assert len(header_json) <= HEADER_SIZE
    payload = b"".join(
        np.ascontiguousarray(v).astype(_astype(v.dtype)).tobytes()
        for v in fields.values()
    )
    return b"pose" + header_json.ljust(HEADER_SIZE, b"\x00") + payload


def _dtype_str(dt):
    dt = np.dtype(dt)
    if dt == np.float32:
        return "f32"
    if dt == np.float64:
        return "f64"
    if dt == np.int32:
        return "i32"
    if dt == np.int64:
        return "i64"
    if dt == np.bool_:
        return "bool"
    return "f32"


def _astype(dt):
    dt = np.dtype(dt)
    return np.float32 if dt == np.float64 else dt


def _command_payload(msg: bytes) -> bytes:
    """Extract the 3-byte [start, stop, planner] payload from a command message.

    Layout: topic (7 bytes, 'command') + 1280-byte header + 3-byte payload.
    """
    topic_end = msg.index(b"{")
    return msg[topic_end + HEADER_SIZE : topic_end + HEADER_SIZE + 3]


class ReplayerCommandFirstTest(unittest.TestCase):
    """Regression: the replayer must send the TWO-STEP command sequence BEFORE
    the pose stream, so the C++ deploy actually enters CONTROL state.

    Root cause (zmq_manager.hpp): command(start=True) is only consumed in
    PLANNER mode (handlePlannerInput). In STREAMED_MOTION mode the start flag is
    dropped. So the deploy must first be told PLANNER+start (enters CONTROL),
    then switched to STREAMED_MOTION (keeps running policy on the pose stream).
    Sending command(start=True, planner=False) directly leaves the deploy in
    WAIT_FOR_CONTROL -> the robot falls.

    Expected sequence:
      1. command [start=1, stop=0, planner=1]  (PLANNER + start policy)
      2. command [start=1, stop=0, planner=0]  (switch to STREAMED_MOTION)
      3. pose frames...
    """

    def _run_replayer_auto(self, npz_path: str, port: int, duration: float = 2.0):
        received = []
        stop_evt = threading.Event()
        sub_ctx = zmq.Context()
        sub = sub_ctx.socket(zmq.SUB)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.setsockopt(zmq.RCVTIMEO, 100)
        sub.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.3)

        def recv_loop():
            while not stop_evt.is_set():
                try:
                    msg = sub.recv()
                except zmq.Again:
                    continue
                received.append(msg)

        t = threading.Thread(target=recv_loop, daemon=True)
        t.start()

        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "gear_sonic/scripts/run_swing_episode_replayer.py"),
                "--npz", npz_path,
                "--zmq-host", "127.0.0.1",
                "--zmq-port", str(port),
                "--fps", "30",
                "--auto",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            time.sleep(duration)
        finally:
            proc.terminate()
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
            stop_evt.set()
            t.join(timeout=1.0)
            sub.close(0)
            sub_ctx.term()
            os.remove(npz_path)
        return received

    def _free_port(self) -> int:
        probe = zmq.Context()
        probe_sock = probe.socket(zmq.PUB)
        port = probe_sock.bind_to_random_port("tcp://127.0.0.1")
        probe_sock.close(0)
        probe.term()
        return port

    def test_command_precedes_pose(self):
        port = self._free_port()
        _, npz_path = tempfile.mkstemp(suffix=".npz")
        os.remove(npz_path)
        make_episode_npz(npz_path, T=6)

        received = self._run_replayer_auto(npz_path, port)

        from collections import Counter

        topics = [m[: m.index(b"{")] for m in received]
        counts = Counter(topics)
        self.assertGreaterEqual(counts.get(b"command", 0), 1, "no command message sent")
        self.assertGreaterEqual(counts.get(b"pose", 0), 1, "no pose messages sent")
        # The very first message must be the command (mode switch), not a pose frame.
        self.assertEqual(topics[0], b"command", "command must precede the pose stream")

    def test_two_step_command_sequence(self):
        """The fix: PLANNER+start command must come BEFORE the STREAMED_MOTION
        command, and both must precede the pose stream."""
        port = self._free_port()
        _, npz_path = tempfile.mkstemp(suffix=".npz")
        os.remove(npz_path)
        make_episode_npz(npz_path, T=6)

        received = self._run_replayer_auto(npz_path, port)

        commands = [m for m in received if m[: m.index(b"{")] == b"command"]
        poses = [m for m in received if m[: m.index(b"{")] == b"pose"]
        self.assertGreaterEqual(len(commands), 2, "expected at least two command messages")

        # First command: PLANNER + start  -> [start=1, stop=0, planner=1]
        self.assertEqual(
            list(_command_payload(commands[0])),
            [1, 0, 1],
            "first command must be PLANNER+start to enter CONTROL state",
        )
        # Second command: STREAMED_MOTION  -> [start=1, stop=0, planner=0]
        self.assertEqual(
            list(_command_payload(commands[1])),
            [1, 0, 0],
            "second command must switch to STREAMED_MOTION",
        )
        # Both commands must precede any pose frame.
        first_pose_idx = received.index(poses[0])
        last_command_idx = max(i for i, m in enumerate(received) if m[: m.index(b"{")] == b"command")
        self.assertLess(
            last_command_idx,
            first_pose_idx,
            "both command messages must precede the pose stream",
        )


if __name__ == "__main__":
    unittest.main()
