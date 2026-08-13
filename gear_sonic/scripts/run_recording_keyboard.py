#!/usr/bin/env python3
"""Interactive ZMQ publisher for GR00T data-recording controls.

This intentionally uses a dedicated terminal and port 5580 rather than the
G1 deployment terminal: the deployment's ``c`` / ``x`` keys have robot-hand
semantics, whereas the data exporter interprets those bytes as recording
commands.

Run after starting ``run_data_exporter.py``:
    python gear_sonic/scripts/run_recording_keyboard.py

Controls:
    c  toggle recording / save the current episode
    x  discard the current episode
    q  quit this publisher
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

import zmq


class RecordingKeyboardPublisher:
    """Owns the dedicated PUB socket used only for exporter controls."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5580) -> None:
        self.endpoint = f"tcp://{host}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(self.endpoint)

    def send(self, command: str) -> None:
        if command not in {"c", "x"}:
            raise ValueError(f"Unsupported recording command: {command!r}")
        self.socket.send_string(command)

    def close(self) -> None:
        self.socket.close()
        self.context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5580, help="ZMQ PUB port (default: 5580)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not sys.stdin.isatty():
        print("ERROR: run this script in an interactive terminal.", file=sys.stderr)
        return 2

    try:
        publisher = RecordingKeyboardPublisher(args.host, args.port)
    except zmq.ZMQError as exc:
        print(f"ERROR: cannot bind recording-control publisher at tcp://{args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    print(f"[Recording controls] Publishing to tcp://{args.host}:{args.port}")
    print("  c: toggle recording/save | x: discard episode | q: quit")
    print("  Keep this terminal separate from the G1 deployment terminal.")
    # A PUB/SUB subscription needs a short handshake before the first key.
    time.sleep(0.3)

    old_settings = termios.tcgetattr(sys.stdin.fileno())
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.25)
            if not ready:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                return 0
            if key in {"c", "x"}:
                publisher.send(key)
                print("[Recording controls] sent " + ("toggle" if key == "c" else "discard"))
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        publisher.close()


if __name__ == "__main__":
    raise SystemExit(main())
