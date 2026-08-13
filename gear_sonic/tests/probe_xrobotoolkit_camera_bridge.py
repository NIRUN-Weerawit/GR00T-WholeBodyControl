"""Local end-to-end probe for the XRoboToolkit camera bridge.

Publishes synthetic MuJoCo camera packets, emulates the PICO control/video TCP
endpoints, and verifies that length-prefixed Annex-B H.264 reaches the receiver.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from gear_sonic.utils.mujoco_sim.sensor_server import ImageMessageSchema, SensorServer


ROOT = Path(__file__).parents[2]
BRIDGE_PATH = ROOT / "gear_sonic/scripts/run_xrobotoolkit_camera_bridge.py"
spec = importlib.util.spec_from_file_location("xrobot_bridge_probe", BRIDGE_PATH)
assert spec is not None and spec.loader is not None
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)

CAMERA_PORT = 15565
CONTROL_PORT = 13589
VIDEO_PORT = 12359


def control_record(command: str, payload: bytes) -> bytes:
    command_bytes = command.encode()
    body = (
        struct.pack("<i", len(command_bytes))
        + command_bytes
        + struct.pack("<i", len(payload))
        + payload
    )
    return struct.pack(">I", len(body)) + body


def open_payload() -> bytes:
    camera = b"VR"
    ip = b"127.0.0.1"
    return (
        b"\xca\xfe\x01"
        + struct.pack("<7i", 320, 120, 15, 500_000, 0, 0, VIDEO_PORT)
        + bytes([len(camera)]) + camera
        + bytes([len(ip)]) + ip
    )


def recv_exact(sock: socket.socket, size: int) -> bytes:
    output = bytearray()
    while len(output) < size:
        chunk = sock.recv(size - len(output))
        if not chunk:
            raise ConnectionError("video connection closed early")
        output.extend(chunk)
    return bytes(output)


def main() -> None:
    publisher = SensorServer()
    publisher.start_server(CAMERA_PORT)
    stop_publish = threading.Event()

    def publish() -> None:
        front = np.zeros((120, 160, 3), dtype=np.uint8)
        side = np.zeros((120, 160, 3), dtype=np.uint8)
        count = 0
        while not stop_publish.is_set():
            front[..., 0] = (count * 7) % 255
            front[..., 2] = 180
            side[..., 1] = (count * 11) % 255
            message = ImageMessageSchema(
                {"teleop_front": time.time(), "teleop_side": time.time()},
                {"teleop_front": front, "teleop_side": side},
            ).serialize()
            publisher.send_message(message)
            count += 1
            time.sleep(1 / 30)

    publish_thread = threading.Thread(target=publish, daemon=True)
    publish_thread.start()

    process = subprocess.Popen(
        [
            sys.executable,
            str(BRIDGE_PATH),
            "--camera-port", str(CAMERA_PORT),
            "--control-host", "127.0.0.1",
            "--control-port", str(CONTROL_PORT),
            "--first-frame-timeout", "5",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
    )

    video_listener = socket.socket()
    video_listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    video_listener.bind(("127.0.0.1", VIDEO_PORT))
    video_listener.listen(1)
    video_listener.settimeout(10)

    control = None
    video = None
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                control = socket.create_connection(("127.0.0.1", CONTROL_PORT), timeout=1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)
        control.sendall(control_record("OPEN_CAMERA", open_payload()))
        video, _ = video_listener.accept()
        video.settimeout(10)
        samples = []
        while len(samples) < 3:
            size = struct.unpack(">I", recv_exact(video, 4))[0]
            if not 0 < size < 4_000_000:
                raise AssertionError(f"invalid framed sample size {size}")
            payload = recv_exact(video, size)
            if not payload.startswith(b"\x00\x00\x00\x01\x09"):
                raise AssertionError("H.264 access unit does not begin with Annex-B AUD")
            samples.append(payload)
        control.sendall(control_record("CLOSE_CAMERA", b""))
        print(f"PASS samples={len(samples)} sizes={[len(sample) for sample in samples]}")
        print("PASS framing=uint32_be codec=Annex-B-H264 AUD=yes")
    finally:
        if control is not None:
            control.close()
        if video is not None:
            video.close()
        video_listener.close()
        process.terminate()
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        print("BRIDGE LOG:\n" + output)
        stop_publish.set()
        publish_thread.join(timeout=2)
        publisher.stop_server()


if __name__ == "__main__":
    main()
