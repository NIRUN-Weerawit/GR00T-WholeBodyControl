"""Behavior tests for the headset-accessible MuJoCo camera web viewer."""

import importlib.util
from pathlib import Path
import sys
import threading
import unittest
import urllib.request

import numpy as np


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_camera_web_viewer.py"
spec = importlib.util.spec_from_file_location("run_camera_web_viewer", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
viewer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = viewer
spec.loader.exec_module(viewer)


class CameraCanvasTests(unittest.TestCase):
    def test_build_camera_canvas_tiles_and_labels_rgb_frames(self):
        red = np.zeros((40, 80, 3), dtype=np.uint8)
        red[..., 0] = 255
        green = np.zeros((60, 40, 3), dtype=np.uint8)
        green[..., 1] = 255

        canvas = viewer.build_camera_canvas(
            {"teleop_front": red, "teleop_side": green}, max_tile_width=80
        )

        self.assertEqual(canvas.shape, (60, 120, 3))
        # Returned canvas is browser/OpenCV BGR: RGB red becomes BGR (0, 0, 255).
        self.assertEqual(canvas[35, 20].tolist(), [0, 0, 255])
        self.assertEqual(canvas[35, 100].tolist(), [0, 255, 0])


class CameraHttpTests(unittest.TestCase):
    def test_health_and_snapshot_endpoints_serve_latest_frame(self):
        state = viewer.LatestFrame()
        state.update(b"jpeg-test-payload", ["teleop_front", "teleop_side"])
        server = viewer.create_http_server("127.0.0.1", 0, state)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as response:
                body = response.read().decode()
                self.assertEqual(response.status, 200)
                self.assertIn('"camera_count": 2', body)
            with urllib.request.urlopen(base + "/snapshot.jpg", timeout=2) as response:
                self.assertEqual(response.headers.get_content_type(), "image/jpeg")
                self.assertEqual(response.read(), b"jpeg-test-payload")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
