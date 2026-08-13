#!/usr/bin/env python3
"""Bridge MuJoCo ZMQ cameras to XRoboToolkit Remote Vision on PICO.

XRoboToolkit connects to this process on TCP port 13579 and sends OPEN_CAMERA.
The bridge then composes the latest simulator views, encodes Annex-B H.264 with
FFmpeg, and connects back to the headset's requested video port (normally 12345).
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import socket
import struct
import subprocess
import threading
import time
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import tyro

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


MAX_COMMAND_BYTES = 256
MAX_PAYLOAD_BYTES = 4096


@dataclass(frozen=True)
class CameraRequest:
    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str


@dataclass
class BridgeConfig:
    """MuJoCo-to-XRoboToolkit bridge settings."""

    camera_host: str = "127.0.0.1"
    camera_port: int = 5555
    control_host: str = "0.0.0.0"
    control_port: int = 13579
    cameras: tuple[str, ...] = ("teleop_front", "teleop_side")
    encoder: str = "h264_nvenc"
    keyframe_interval: int = 15
    connect_timeout: float = 5.0
    first_frame_timeout: float = 10.0
    max_width: int = 3840
    max_height: int = 2160
    max_fps: int = 60
    allow_destination_ip_mismatch: bool = False


def _read_i32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError("camera request ended before an int32 field")
    return struct.unpack_from("<i", data, offset)[0]


def _read_compact_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset >= len(data):
        raise ValueError("camera request ended before a string length")
    size = data[offset]
    offset += 1
    if offset + size > len(data):
        raise ValueError("camera request ended inside a string")
    try:
        value = data[offset : offset + size].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("camera request contains invalid UTF-8") from exc
    return value, offset + size


def deserialize_camera_request(data: bytes) -> CameraRequest:
    """Decode XRoboToolkit's OPEN_CAMERA payload."""
    if len(data) < 33 or data[:2] != b"\xca\xfe":
        raise ValueError("invalid camera request magic or truncated payload")
    if data[2] != 1:
        raise ValueError(f"unsupported camera request version {data[2]}")
    values = struct.unpack_from("<7i", data, 3)
    offset = 31
    camera, offset = _read_compact_string(data, offset)
    ip, offset = _read_compact_string(data, offset)
    if offset != len(data):
        raise ValueError("unexpected trailing bytes in camera request")
    return CameraRequest(
        width=values[0],
        height=values[1],
        fps=values[2],
        bitrate=values[3],
        enable_mv_hevc=values[4],
        render_mode=values[5],
        port=values[6],
        camera=camera,
        ip=ip,
    )


def recv_exact(sock: socket.socket, size: int) -> bytes | None:
    """Read exactly size bytes; return None only on clean EOF before a record."""
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            if remaining == size:
                return None
            raise ConnectionError("TCP connection closed inside a control record")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def deserialize_control_body(body: bytes) -> tuple[str, bytes]:
    """Decode the little-endian NetworkDataProtocol body."""
    if len(body) < 8:
        raise ValueError("control body is too short")
    command_size = struct.unpack_from("<i", body, 0)[0]
    if not 0 < command_size <= MAX_COMMAND_BYTES:
        raise ValueError(f"invalid command length {command_size}")
    payload_size_offset = 4 + command_size
    if payload_size_offset + 4 > len(body):
        raise ValueError("control body ended before payload length")
    payload_size = struct.unpack_from("<i", body, payload_size_offset)[0]
    if not 0 <= payload_size <= MAX_PAYLOAD_BYTES:
        raise ValueError(f"invalid payload length {payload_size}")
    expected_size = payload_size_offset + 4 + payload_size
    if expected_size != len(body):
        raise ValueError(
            f"control body size mismatch: expected {expected_size}, received {len(body)}"
        )
    try:
        command = body[4:payload_size_offset].rstrip(b"\0").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("control command contains invalid UTF-8") from exc
    return command, body[payload_size_offset + 4 :]


def serialize_control_record(command: str, payload: bytes = b"") -> bytes:
    """Encode a control response with big-endian outer framing."""
    command_bytes = command.encode("utf-8")
    if not 0 < len(command_bytes) <= MAX_COMMAND_BYTES:
        raise ValueError("invalid control command length")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("control payload is too large")
    body = (
        struct.pack("<i", len(command_bytes))
        + command_bytes
        + struct.pack("<i", len(payload))
        + payload
    )
    return struct.pack(">I", len(body)) + body


def send_control_record(sock: socket.socket, command: str, payload: bytes = b"") -> None:
    sock.sendall(serialize_control_record(command, payload))


def make_audio_config_payload(audio_request_id: str) -> bytes:
    """Tell the current Unity client that this video-only bridge disables audio."""
    payload = {
        "schema": "g1_wuji_audio_ports_v2",
        "audio_request_id": audio_request_id,
        "audio_stream_port": 0,
        "microphone_upload_port": 0,
        "sample_rate": 16000,
        "channels": 1,
        "sample_format": "s16le",
        "video_projection": "flat",
        "video_stereo_layout": "mono",
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def recv_control_record(sock: socket.socket) -> tuple[str, bytes] | None:
    """Read one big-endian-length-framed XRoboToolkit control record."""
    header = recv_exact(sock, 4)
    if header is None:
        return None
    body_size = struct.unpack(">I", header)[0]
    max_body_size = 8 + MAX_COMMAND_BYTES + MAX_PAYLOAD_BYTES
    if not 8 <= body_size <= max_body_size:
        raise ValueError(f"invalid control body length {body_size}")
    body = recv_exact(sock, body_size)
    assert body is not None
    return deserialize_control_body(body)


def send_video_sample(sock: socket.socket, payload: bytes) -> None:
    """Send one H.264 access unit using XRoboToolkit's framing."""
    if not payload:
        return
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def _fit_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 frame, got {frame.shape}")
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized_width = max(1, round(frame.shape[1] * scale))
    resized_height = max(1, round(frame.shape[0] * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    output[y : y + resized_height, x : x + resized_width] = resized
    return output


def compose_canvas(
    images: Mapping[str, np.ndarray],
    width: int,
    height: int,
    camera_order: Sequence[str],
) -> np.ndarray:
    """Create a mono operational multi-view canvas at the requested dimensions."""
    selected = [(name, images[name]) for name in camera_order if name in images]
    if not selected:
        selected = sorted(images.items())
    if not selected:
        raise ValueError("no camera images available")
    panel_count = len(selected)
    panel_widths = [width // panel_count] * panel_count
    panel_widths[-1] += width - sum(panel_widths)
    panels = [_fit_rgb(frame, panel_widths[index], height) for index, (_, frame) in enumerate(selected)]
    return np.ascontiguousarray(np.concatenate(panels, axis=1))


class AnnexBAccessUnitParser:
    """Incrementally split Annex-B H.264 at Access Unit Delimiter NALs."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    @staticmethod
    def _aud_offsets(data: bytes | bytearray) -> list[int]:
        offsets: list[int] = []
        index = 0
        while True:
            index = data.find(b"\x00\x00\x00\x01\x09", index)
            if index < 0:
                return offsets
            offsets.append(index)
            index += 5

    def feed(self, data: bytes) -> list[bytes]:
        self._buffer.extend(data)
        offsets = self._aud_offsets(self._buffer)
        if len(offsets) < 2:
            return []
        units = [bytes(self._buffer[offsets[i] : offsets[i + 1]]) for i in range(len(offsets) - 1)]
        del self._buffer[: offsets[-1]]
        return units

    def flush(self) -> bytes | None:
        if not self._buffer:
            return None
        result = bytes(self._buffer)
        self._buffer.clear()
        return result


def ffmpeg_command(request: CameraRequest, config: BridgeConfig) -> list[str]:
    """Build a low-latency H.264 Annex-B encoder command."""
    common = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{request.width}x{request.height}",
        "-r", str(request.fps), "-i", "pipe:0", "-an", "-c:v", config.encoder,
    ]
    if config.encoder == "libx264":
        common += [
            "-preset", "ultrafast", "-tune", "zerolatency", "-bf", "0",
            "-x264-params", f"aud=1:repeat-headers=1:keyint={config.keyframe_interval}:min-keyint={config.keyframe_interval}:scenecut=0",
        ]
    elif config.encoder == "h264_nvenc":
        common += [
            "-preset", "p1", "-tune", "ull", "-delay", "0", "-bf", "0",
            "-g", str(config.keyframe_interval), "-forced-idr", "1",
            "-bsf:v", "h264_metadata=aud=insert",
        ]
    else:
        common += ["-g", str(config.keyframe_interval), "-bf", "0", "-bsf:v", "h264_metadata=aud=insert"]
    common += [
        "-b:v", str(request.bitrate), "-maxrate", str(request.bitrate),
        "-bufsize", str(max(request.bitrate // 2, 1)), "-pix_fmt", "yuv420p",
        "-f", "h264", "pipe:1",
    ]
    return common


def validate_request(request: CameraRequest, config: BridgeConfig, peer_ip: str) -> None:
    if not 16 <= request.width <= config.max_width or request.width % 2:
        raise ValueError(f"unsupported width {request.width}")
    if not 16 <= request.height <= config.max_height or request.height % 2:
        raise ValueError(f"unsupported height {request.height}")
    if not 1 <= request.fps <= config.max_fps:
        raise ValueError(f"unsupported FPS {request.fps}")
    if not 100_000 <= request.bitrate <= 100_000_000:
        raise ValueError(f"unsupported bitrate {request.bitrate}")
    if not 1 <= request.port <= 65535:
        raise ValueError(f"invalid destination port {request.port}")
    ipaddress.ip_address(request.ip)
    if not config.allow_destination_ip_mismatch and request.ip != peer_ip:
        raise ValueError(f"destination IP {request.ip} does not match control peer {peer_ip}")


class StreamingSession:
    def __init__(self, request: CameraRequest, config: BridgeConfig) -> None:
        self.request = request
        self.config = config
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="remote-vision-stream")
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _read_encoder(self, process: subprocess.Popen[bytes], video_socket: socket.socket) -> None:
        parser = AnnexBAccessUnitParser()
        assert process.stdout is not None
        try:
            while not self.stop_event.is_set():
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                for access_unit in parser.feed(chunk):
                    send_video_sample(video_socket, access_unit)
            tail = parser.flush()
            if tail and not self.stop_event.is_set():
                send_video_sample(video_socket, tail)
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            self.error = exc
            self.stop_event.set()

    def _run(self) -> None:
        camera = ComposedCameraClientSensor(self.config.camera_host, self.config.camera_port)
        process: subprocess.Popen[bytes] | None = None
        video_socket: socket.socket | None = None
        output_thread: threading.Thread | None = None
        try:
            print(f"Connecting H.264 stream to PICO {self.request.ip}:{self.request.port}")
            video_socket = socket.create_connection(
                (self.request.ip, self.request.port), timeout=self.config.connect_timeout
            )
            video_socket.settimeout(None)
            process = subprocess.Popen(
                ffmpeg_command(self.request, self.config),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                bufsize=0,
            )
            output_thread = threading.Thread(
                target=self._read_encoder, args=(process, video_socket), daemon=True
            )
            output_thread.start()
            deadline = time.monotonic() + self.config.first_frame_timeout
            period = 1.0 / self.request.fps
            sent_frames = 0
            while not self.stop_event.is_set():
                started = time.monotonic()
                packet = camera.read(blocking=False)
                images = packet.get("images") if packet else None
                if images:
                    canvas = compose_canvas(images, self.request.width, self.request.height, self.config.cameras)
                    assert process.stdin is not None
                    process.stdin.write(canvas.tobytes())
                    sent_frames += 1
                    if sent_frames == 1:
                        print(
                            f"Remote Vision streaming {self.request.width}x{self.request.height} "
                            f"@ {self.request.fps} FPS from {list(images)}"
                        )
                elif sent_frames == 0 and time.monotonic() >= deadline:
                    raise TimeoutError("no MuJoCo camera frames received on the configured ZMQ port")
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    self.stop_event.wait(remaining)
                if process.poll() is not None:
                    raise RuntimeError(f"FFmpeg encoder exited with status {process.returncode}")
                if self.error is not None:
                    raise RuntimeError("PICO video connection failed") from self.error
        except BaseException as exc:
            self.error = exc
            if not self.stop_event.is_set():
                print(f"Remote Vision stream error: {exc}")
        finally:
            self.stop_event.set()
            camera.close()
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if output_thread is not None:
                output_thread.join(timeout=2)
            if video_socket is not None:
                try:
                    video_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                video_socket.close()


def serve_client(client: socket.socket, peer_ip: str, config: BridgeConfig) -> None:
    session: StreamingSession | None = None
    try:
        while True:
            record = recv_control_record(client)
            if record is None:
                return
            command, payload = record
            print(f"Received protocol command: {command!r}")
            if command == "AUDIO_SESSION":
                try:
                    audio_request_id = payload.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ValueError("AUDIO_SESSION request ID is not ASCII") from exc
                send_control_record(
                    client,
                    "AUDIO_CONFIG",
                    make_audio_config_payload(audio_request_id),
                )
                print("Audio disabled for this video-only Remote Vision session")
            elif command == "PING":
                send_control_record(client, "PONG", payload)
            elif command == "PONG":
                pass
            elif command == "OPEN_CAMERA":
                request = deserialize_camera_request(payload)
                validate_request(request, config, peer_ip)
                print(
                    "Camera config - "
                    f"{request.width}x{request.height}@{request.fps}, bitrate={request.bitrate}, "
                    f"type={request.camera}, destination={request.ip}:{request.port}"
                )
                if session is not None:
                    session.stop()
                session = StreamingSession(request, config)
                session.start()
            elif command == "CLOSE_CAMERA":
                if session is not None:
                    session.stop()
                    session = None
                print("Remote Vision camera closed")
            else:
                print(f"Ignoring unsupported control command {command!r}")
    finally:
        if session is not None:
            session.stop()


def main(config: BridgeConfig) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((config.control_host, config.control_port))
        server.listen(1)
        print(f"XRoboToolkit control server listening on {config.control_host}:{config.control_port}")
        print(f"MuJoCo camera source: tcp://{config.camera_host}:{config.camera_port}")
        print("In PICO Remote Vision, enter this PC's IP address (without :5555).")
        try:
            while True:
                client, address = server.accept()
                print(f"PICO control client connected from {address[0]}:{address[1]}")
                with client:
                    try:
                        serve_client(client, address[0], config)
                    except (ConnectionError, OSError, ValueError) as exc:
                        print(f"Control connection error: {exc}")
                print("PICO control client disconnected")
        except KeyboardInterrupt:
            print("\nStopping XRoboToolkit camera bridge.")


if __name__ == "__main__":
    main(tyro.cli(BridgeConfig))
