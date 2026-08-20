"""Synthetic end-to-end tests for the swing episode recorder.

Publishes ``manager_state``/``pose`` messages (mirroring pico_manager_thread_server)
on a ZMQ PUB socket, drives the recorder through an episode, and verifies the
per-episode npz file.
"""

import json
import os
import time
import unittest

import numpy as np
import zmq

from gear_sonic.scripts.run_swing_episode_recorder import (
    SwingEpisodeRecorder,
    create_run_output_dir,
    unpack_pose_message,
)

HEADER_SIZE = 1280


def pack_pose_message(fields: dict, topic: str = "pose") -> bytes:
    """Wire-compatible packer (same layout as pico_manager's pack_pose_message).

    Layout: [topic_bytes][1280-byte JSON header][concatenated LE binary fields].
    """
    dtype_map = {
        np.dtype(np.float32): "f32",
        np.dtype(np.float64): "f64",
        np.dtype(np.int32): "i32",
        np.dtype(np.int64): "i64",
        np.dtype(bool): "bool",
    }
    header_fields = []
    binary = []
    for key, value in fields.items():
        if not isinstance(value, np.ndarray):
            continue
        dtype_str = dtype_map.get(value.dtype, "f32")
        if dtype_str == "f32" and value.dtype != np.float32:
            value = value.astype(np.float32)
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        header_fields.append(
            {"name": key, "dtype": dtype_str, "shape": list(value.shape)}
        )
        binary.append(value.tobytes())
    header = {"v": 3, "endian": "le", "count": 1, "fields": header_fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    assert len(header_json) <= HEADER_SIZE
    return (
        topic.encode("utf-8")
        + header_json.ljust(HEADER_SIZE, b"\x00")
        + b"".join(binary)
    )


def make_pose_frame(idx: int) -> dict:
    """A realistic pose frame matching the fields the exporter consumes."""
    return {
        "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
        "smpl_joints": np.random.rand(1, 24, 3).astype(np.float32),
        "body_quat_w": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        "joint_pos": np.random.rand(1, 29).astype(np.float32),
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "vr_position": np.zeros(9, dtype=np.float32),
        "vr_orientation": np.zeros(12, dtype=np.float32),
        "frame_index": np.array([idx], dtype=np.int64),
        "timestamp_realtime": np.array([float(idx) / 50.0 + 1e6], dtype=np.float64),
        "timestamp_monotonic": np.array([float(idx) / 50.0 + 2e6], dtype=np.float64),
    }


def _send_toggle(pub, collection=True, abort=False):
    """Send one rising-edge manager_state toggle (press)."""
    pub.send(
        pack_pose_message(
            {
                "stream_mode": np.array([2], dtype=np.int32),
                "toggle_data_collection": np.array([collection], dtype=bool),
                "toggle_data_abort": np.array([abort], dtype=bool),
            },
            topic="manager_state",
        )
    )
    time.sleep(0.15)


class RunOutputDirectoryTest(unittest.TestCase):
    def test_creates_next_numbered_run_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as base_dir:
            os.mkdir(os.path.join(base_dir, "run_1"))
            os.mkdir(os.path.join(base_dir, "run_3"))
            os.mkdir(os.path.join(base_dir, "notes"))

            run_dir = create_run_output_dir(base_dir)

            self.assertEqual(run_dir, os.path.join(base_dir, "run_4"))
            self.assertTrue(os.path.isdir(run_dir))


class _ZMQHarness(unittest.TestCase):
    def setUp(self):
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.port = self.pub.bind_to_random_port("tcp://127.0.0.1")
        self.out_dir = self._temp_dir()
        os.makedirs(self.out_dir, exist_ok=True)

    def tearDown(self):
        self.pub.close(0)
        self.ctx.term()

    def _temp_dir(self):
        import tempfile

        return tempfile.mkdtemp(prefix="swing_episodes_")

    def _start_recorder(self, primitive="forehand_right"):
        import threading

        rec = SwingEpisodeRecorder(
            zmq_host="127.0.0.1",
            zmq_port=self.port,
            output_dir=self.out_dir,
            primitive=primitive,
            keyboard_fallback=False,
            episode_index_start=0,
        )
        thread = threading.Thread(target=rec.run, daemon=True)
        thread.start()
        time.sleep(0.3)  # allow SUB connect + PUB handshake
        return rec, thread

    def _drive_episode(self, primitive="forehand_right", n_send=10):
        """Press start, feed frames, then press once more to stop and save."""
        rec, thread = self._start_recorder(primitive)
        try:
            # Press 1: start recording (IDLE -> RECORDING).
            _send_toggle(self.pub)

            # Feed pose frames during recording.
            for i in range(n_send):
                self.pub.send(pack_pose_message(make_pose_frame(i), topic="pose"))
                time.sleep(0.05)

            # Press 2: stop and save immediately (RECORDING -> IDLE + save).
            _send_toggle(self.pub)

            deadline = time.monotonic() + 3.0
            npz_path = os.path.join(self.out_dir, f"{primitive}_000000.npz")
            while time.monotonic() < deadline:
                if os.path.exists(npz_path):
                    break
                time.sleep(0.05)
            return npz_path
        finally:
            rec.stop()
            thread.join(timeout=1.0)
            rec.close()

    def test_save_episode_npz(self):
        primitive = "forehand_right"
        npz_path = self._drive_episode(primitive, n_send=10)

        self.assertTrue(os.path.exists(npz_path), f"Expected {npz_path} after save")
        with np.load(npz_path, allow_pickle=True) as data:
            self.assertEqual(
                np.array(data["primitive_label"]).item().decode("utf-8"), primitive
            )
            self.assertEqual(data["episode_index"].flat[0], 0)
            n = len(data["smpl_joints"])
            self.assertGreaterEqual(n, 1)
            for key, shape in [
                ("smpl_pose", (n, 21, 3)),
                ("smpl_joints", (n, 24, 3)),
                ("body_quat_w", (n, 4)),
                ("left_wrist_joints", (n, 3)),
                ("right_wrist_joints", (n, 3)),
                ("vr_3pt_position", (n, 9)),
                ("vr_3pt_orientation", (n, 12)),
                ("frame_index", (n, 1)),
                ("timestamp_realtime", (n, 1)),
                ("timestamp_monotonic", (n, 1)),
            ]:
                self.assertEqual(
                    data[key].shape, shape, f"{key}: got {data[key].shape}, want {shape}"
                )
            self.assertTrue(np.isfinite(data["smpl_joints"]).all())
            self.assertTrue(np.isfinite(data["smpl_pose"]).all())

    def test_discard_episode_writes_nothing(self):
        import threading

        recorder = SwingEpisodeRecorder(
            zmq_host="127.0.0.1",
            zmq_port=self.port,
            output_dir=self.out_dir,
            primitive="discard_check",
            keyboard_fallback=False,
            episode_index_start=0,
        )
        thread = threading.Thread(target=recorder.run, daemon=True)
        thread.start()
        time.sleep(0.3)
        try:
            # Press 1: start recording.
            _send_toggle(self.pub)
            for i in range(5):
                self.pub.send(pack_pose_message(make_pose_frame(i), topic="pose"))
                time.sleep(0.05)
            # Left-Grip+B rising edge: abort/discard.
            _send_toggle(self.pub, collection=False, abort=True)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and recorder._episode_state.get_state() != "idle":
                time.sleep(0.05)
        finally:
            recorder.stop()
            thread.join(timeout=1.0)
            recorder.close()

        self.assertEqual(os.listdir(self.out_dir), [], "abort must not write an episode")
        self.assertEqual(recorder._buffers["smpl_joints"], [])

    def test_unpack_roundtrip(self):
        fields = {
            "smpl_pose": np.zeros((1, 21, 3), dtype=np.float32),
            "smpl_joints": np.random.rand(1, 24, 3).astype(np.float32),
            "frame_index": np.array([7], dtype=np.int64),
        }
        packed = pack_pose_message(fields, topic="pose")
        data = unpack_pose_message(packed, topic="pose")
        self.assertEqual(data["smpl_pose"].shape, (1, 21, 3))
        self.assertAlmostEqual(
            data["smpl_joints"][0, 0, 0], fields["smpl_joints"][0, 0, 0]
        )
        self.assertEqual(data["frame_index"].flat[0], 7)


if __name__ == "__main__":
    unittest.main()
