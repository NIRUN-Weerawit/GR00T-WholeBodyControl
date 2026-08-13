"""Behavior tests for the XRoboToolkit Remote Vision bridge."""

import importlib.util
from pathlib import Path
import socket
import struct
import sys
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_xrobotoolkit_camera_bridge.py"
spec = importlib.util.spec_from_file_location("run_xrobotoolkit_camera_bridge", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


def make_open_camera_payload(
    width=2160,
    height=810,
    fps=60,
    bitrate=4_000_000,
    camera="VR",
    ip="192.168.197.34",
    port=12345,
):
    return (
        b"\xca\xfe\x01"
        + struct.pack("<7i", width, height, fps, bitrate, 0, 0, port)
        + bytes([len(camera)])
        + camera.encode()
        + bytes([len(ip)])
        + ip.encode()
    )


def make_control_body(command, payload=b""):
    command_bytes = command.encode()
    return struct.pack("<i", len(command_bytes)) + command_bytes + struct.pack("<i", len(payload)) + payload


def make_control_record(command, payload=b""):
    body = make_control_body(command, payload)
    return struct.pack(">I", len(body)) + body


class ControlProtocolTests(unittest.TestCase):
    def test_open_camera_payload_is_deserialized(self):
        request = bridge.deserialize_camera_request(make_open_camera_payload())
        self.assertEqual(request.width, 2160)
        self.assertEqual(request.height, 810)
        self.assertEqual(request.fps, 60)
        self.assertEqual(request.bitrate, 4_000_000)
        self.assertEqual(request.camera, "VR")
        self.assertEqual(request.ip, "192.168.197.34")
        self.assertEqual(request.port, 12345)

    def test_control_record_handles_fragmented_tcp_reads(self):
        server, client = socket.socketpair()
        self.addCleanup(server.close)
        self.addCleanup(client.close)
        record = make_control_record("OPEN_CAMERA", make_open_camera_payload())
        for chunk in (record[:3], record[3:11], record[11:37], record[37:]):
            client.sendall(chunk)
        command, payload = bridge.recv_control_record(server)
        self.assertEqual(command, "OPEN_CAMERA")
        self.assertEqual(bridge.deserialize_camera_request(payload).port, 12345)

    def test_control_response_has_big_endian_outer_length_and_little_endian_body(self):
        body = make_control_body("PONG", b"token")
        packet = bridge.serialize_control_record("PONG", b"token")
        self.assertEqual(packet[:4], struct.pack(">I", len(body)))
        self.assertEqual(packet[4:], body)
    def test_audio_config_disables_unimplemented_audio_and_declares_mono_video(self):
        payload = bridge.make_audio_config_payload("0123456789abcdef")
        import json
        config = json.loads(payload)
        self.assertEqual(config["schema"], "g1_wuji_audio_ports_v2")
        self.assertEqual(config["audio_request_id"], "0123456789abcdef")
        self.assertEqual(config["audio_stream_port"], 0)
        self.assertEqual(config["microphone_upload_port"], 0)
        self.assertEqual(config["video_projection"], "flat")
        self.assertEqual(config["video_stereo_layout"], "mono")


class CanvasTests(unittest.TestCase):
    def test_compose_canvas_creates_requested_operational_multiview(self):
        front = np.zeros((120, 160, 3), dtype=np.uint8)
        front[..., 0] = 220
        side = np.zeros((120, 160, 3), dtype=np.uint8)
        side[..., 1] = 180
        canvas = bridge.compose_canvas(
            {"teleop_front": front, "teleop_side": side},
            width=640,
            height=240,
            camera_order=("teleop_front", "teleop_side"),
        )
        self.assertEqual(canvas.shape, (240, 640, 3))
        self.assertGreater(float(canvas[:, :320, 0].mean()), 100.0)
        self.assertGreater(float(canvas[:, 320:, 1].mean()), 80.0)


class H264ProtocolTests(unittest.TestCase):
    def test_send_video_sample_uses_big_endian_length_prefix(self):
        server, client = socket.socketpair()
        self.addCleanup(server.close)
        self.addCleanup(client.close)
        payload = b"\x00\x00\x00\x01\x65encoded"
        bridge.send_video_sample(client, payload)
        received = server.recv(4 + len(payload))
        self.assertEqual(received[:4], struct.pack(">I", len(payload)))
        self.assertEqual(received[4:], payload)

    def test_annex_b_parser_groups_access_units_at_aud_nals(self):
        aud = b"\x00\x00\x00\x01\x09\xf0"
        sps = b"\x00\x00\x00\x01\x67abc"
        idr = b"\x00\x00\x00\x01\x65frame1"
        pframe = b"\x00\x00\x00\x01\x41frame2"
        parser = bridge.AnnexBAccessUnitParser()
        self.assertEqual(parser.feed(aud + sps[:3]), [])
        units = parser.feed(sps[3:] + idr + aud + pframe + aud)
        self.assertEqual(units, [aud + sps + idr, aud + pframe])


if __name__ == "__main__":
    unittest.main()
