#!/usr/bin/env python3
"""Expose MuJoCo camera streams as a headset-accessible browser view.

The simulator publishes RGB frames over ZMQ (normally port 5555). This process
subscribes, tiles all available views, JPEG-encodes one combined frame, and
serves it as MJPEG over HTTP. Open the printed URL in the PICO4 browser.

Example:
    python gear_sonic/scripts/run_camera_web_viewer.py \
      --camera-host 127.0.0.1 --camera-port 5555 \
      --http-host 0.0.0.0 --http-port 8080
"""

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from typing import Optional

import cv2
import numpy as np
import tyro

from gear_sonic.camera.composed_camera import ComposedCameraClientSensor


INDEX_HTML = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>G1 MuJoCo Cameras</title>
<style>
html,body{margin:0;width:100%;height:100%;background:#080b10;color:#e7eef8;
font-family:system-ui,sans-serif;overflow:hidden}main{height:100%;display:flex;
flex-direction:column}.bar{display:flex;align-items:center;gap:1rem;padding:.55rem 1rem;
background:#121926}.bar strong{color:#65d9ff}.bar span{color:#a9b8ca;font-size:.9rem}
.view{min-height:0;flex:1;display:flex;align-items:center;justify-content:center}
img{display:block;max-width:100%;max-height:100%;object-fit:contain}button{margin-left:auto;
border:1px solid #506278;background:#1d2938;color:white;border-radius:.4rem;padding:.4rem .8rem}
</style></head><body><main><div class="bar"><strong>G1 MuJoCo</strong>
<span id="status">connecting...</span><button onclick="document.documentElement.requestFullscreen()">Fullscreen</button>
</div><div class="view"><img src="/stream.mjpg" alt="Live MuJoCo camera views"></div></main>
<script>setInterval(async()=>{try{let r=await fetch('/health',{cache:'no-store'}),j=await r.json();
document.getElementById('status').textContent=j.camera_names.join(' + ')+' | '+j.frame_age_seconds.toFixed(1)+'s ago';
}catch(e){document.getElementById('status').textContent='disconnected'}},1000)</script></body></html>"""


@dataclass
class CameraWebViewerConfig:
    camera_host: str = "127.0.0.1"
    """MuJoCo ZMQ camera publisher host."""

    camera_port: int = 5555
    """MuJoCo ZMQ camera publisher port."""

    http_host: str = "0.0.0.0"
    """HTTP bind address. Keep 0.0.0.0 so the headset can connect."""

    http_port: int = 8080
    """Browser viewer port."""

    fps: float = 20.0
    """Maximum browser stream frame rate."""

    jpeg_quality: int = 80
    """JPEG quality from 1 to 100."""

    max_tile_width: int = 640
    """Maximum width of each camera tile before JPEG encoding."""

    first_frame_timeout: float = 15.0
    """Seconds to wait for the MuJoCo camera publisher."""


def build_camera_canvas(
    images_rgb: dict[str, np.ndarray], max_tile_width: int
) -> np.ndarray:
    """Tile labelled RGB camera frames and return one BGR canvas."""
    if not images_rgb:
        raise ValueError("at least one camera image is required")
    if max_tile_width <= 0:
        raise ValueError("max_tile_width must be positive")

    tiles: list[np.ndarray] = []
    for name in sorted(images_rgb):
        image = images_rgb[name]
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"camera {name!r} is not an HxWx3 image")
        tile = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width = tile.shape[:2]
        if width > max_tile_width:
            scale = max_tile_width / width
            tile = cv2.resize(tile, (max_tile_width, max(1, round(height * scale))))
        cv2.putText(
            tile,
            name,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        tiles.append(tile)

    max_height = max(tile.shape[0] for tile in tiles)
    padded: list[np.ndarray] = []
    for tile in tiles:
        if tile.shape[0] < max_height:
            tile = cv2.copyMakeBorder(
                tile,
                0,
                max_height - tile.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        padded.append(tile)
    return np.hstack(padded)


class LatestFrame:
    """Thread-safe latest-JPEG slot shared by ZMQ and HTTP threads."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._sequence = 0
        self._camera_names: list[str] = []
        self._updated_at = 0.0

    def update(self, jpeg: bytes, camera_names: list[str]) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._camera_names = list(camera_names)
            self._sequence += 1
            self._updated_at = time.monotonic()
            self._condition.notify_all()

    def wait_after(self, sequence: int, timeout: float = 2.0):
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > sequence, timeout=timeout)
            return self._sequence, self._jpeg

    def snapshot(self):
        with self._condition:
            return self._jpeg

    def health(self) -> dict:
        with self._condition:
            age = time.monotonic() - self._updated_at if self._updated_at else -1.0
            return {
                "ready": self._jpeg is not None,
                "camera_count": len(self._camera_names),
                "camera_names": list(self._camera_names),
                "frame_age_seconds": age,
            }


def create_http_server(host: str, port: int, state: LatestFrame) -> ThreadingHTTPServer:
    """Create the browser-view server without starting its serving loop."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", INDEX_HTML)
            elif path == "/health":
                body = json.dumps(state.health()).encode()
                self.send_bytes(HTTPStatus.OK, "application/json", body)
            elif path == "/snapshot.jpg":
                jpeg = state.snapshot()
                if jpeg is None:
                    self.send_bytes(HTTPStatus.SERVICE_UNAVAILABLE, "text/plain", b"No frame yet\n")
                else:
                    self.send_bytes(HTTPStatus.OK, "image/jpeg", jpeg)
            elif path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                sequence = 0
                try:
                    while True:
                        sequence, jpeg = state.wait_after(sequence)
                        if jpeg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_bytes(HTTPStatus.NOT_FOUND, "text/plain", b"Not found\n")

    return ThreadingHTTPServer((host, port), Handler)


def discover_lan_ip() -> str:
    """Best-effort LAN address for the URL printed to the headset user."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def main(config: CameraWebViewerConfig) -> None:
    if config.fps <= 0:
        raise ValueError("fps must be positive")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    state = LatestFrame()
    server = create_http_server(config.http_host, config.http_port, state)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    display_host = discover_lan_ip() if config.http_host in ("0.0.0.0", "::") else config.http_host
    print(f"PICO4 viewer: http://{display_host}:{server.server_port}")
    print(f"Waiting for MuJoCo cameras at tcp://{config.camera_host}:{config.camera_port} ...")

    client = ComposedCameraClientSensor(
        server_ip=config.camera_host, port=config.camera_port
    )
    deadline = time.monotonic() + config.first_frame_timeout
    period = 1.0 / config.fps
    received_first_frame = False

    try:
        while True:
            started = time.monotonic()
            packet = client.read(blocking=False)
            images = packet.get("images") if packet else None
            if images:
                canvas = build_camera_canvas(images, config.max_tile_width)
                ok, encoded = cv2.imencode(
                    ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
                )
                if not ok:
                    raise RuntimeError("failed to JPEG-encode the camera canvas")
                camera_names = sorted(images)
                state.update(encoded.tobytes(), camera_names)
                if not received_first_frame:
                    received_first_frame = True
                    print(f"Streaming cameras: {camera_names} at up to {config.fps:g} FPS")
            elif not received_first_frame and time.monotonic() >= deadline:
                raise TimeoutError(
                    "no camera frames received; start run_sim_loop.py with "
                    "--enable-image-publish --enable-offscreen"
                )

            remaining = period - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopping browser camera viewer.")
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


if __name__ == "__main__":
    main(tyro.cli(CameraWebViewerConfig))
