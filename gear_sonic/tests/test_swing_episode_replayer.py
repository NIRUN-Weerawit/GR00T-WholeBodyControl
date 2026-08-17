"""Tests for the swing-episode replayer (protocol v3 pose stream)."""

import os
import tempfile
import time
import unittest

import numpy as np
import zmq

from gear_sonic.scripts.run_swing_episode_replayer import (
    G1_L_WRIST_YAW_IDX,
    G1_R_WRIST_YAW_IDX,
    build_pose_message,
    load_episode,
)
from gear_sonic.scripts.run_swing_episode_recorder import unpack_pose_message

HEADER_SIZE = 1280


def make_episode_npz(save_path: str, T: int = 6) -> None:
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
    np.savez_compressed(save_path, **arrays)


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


if __name__ == "__main__":
    unittest.main()
