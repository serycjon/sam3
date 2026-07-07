# Copyright (c) 2026 Jonas Serych
"""
Server for visualizing streaming-tracker debug dumps.

Usage:
    python scripts/memory_viz/server.py DUMP_DIR [--host 127.0.0.1] [--port 8123]

DUMP_DIR is a directory produced by `sam3.debug_dump.DebugDumpWriter`
(see scripts/memory_viz/PLAN.md for the format). Uses only the Python standard
library, so it has no third-party dependencies (avoids fastapi/pydantic version
conflicts with HPC module environments).
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_IMG_NAME = re.compile(r"^\d{6}\.(jpg|png)$")
_STATIC_NAME = re.compile(r"^[\w.-]+$")
_MM_PNG = re.compile(r"^/multimask/(\d+)/(\d+|token0)\.png$")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".jpg": "image/jpeg",
    ".png": "image/png",
}


def make_handler(dump_dir: Path, static_dir: Path, meta: dict, log: list, mm_meta: dict):
    """Build a request handler bound to one loaded dump."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the console quiet
            pass

        # ---- response helpers ----
        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, obj) -> None:
            self._send_bytes(
                200, _CONTENT_TYPES[".json"], json.dumps(obj).encode("utf-8")
            )

        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_bytes(404, "text/plain", b"not found")
                return
            ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            self._send_bytes(200, ctype, path.read_bytes())

        # ---- routing ----
        def do_GET(self) -> None:
            path = urlparse(self.path).path

            if path == "/":
                self._send_file(static_dir / "index.html")
            elif path == "/api/meta":
                self._send_json(meta)
            elif path == "/api/log":
                self._send_json({"log": log})
            elif path.startswith("/api/multimask/"):
                self._route_multimask_meta(path)
            elif path.startswith("/static/"):
                self._route_dir(static_dir, path[len("/static/"):], _STATIC_NAME)
            elif path.startswith("/thumbs/"):
                self._route_dir(dump_dir / "thumbs", path[len("/thumbs/"):], _IMG_NAME)
            elif path.startswith("/masks/"):
                self._route_dir(dump_dir / "masks", path[len("/masks/"):], _IMG_NAME)
            elif path.startswith("/multimask/"):
                self._route_multimask_png(path)
            else:
                self._send_bytes(404, "text/plain", b"not found")

        do_HEAD = do_GET

        def _route_dir(self, base: Path, name: str, pattern: "re.Pattern") -> None:
            # Validate the name (no traversal) before touching the filesystem.
            if not pattern.match(name):
                self._send_bytes(404, "text/plain", b"not found")
                return
            self._send_file(base / name)

        def _route_multimask_meta(self, path: str) -> None:
            frame = path[len("/api/multimask/"):]
            m = mm_meta.get(frame)
            if m is None:
                self._send_json({"available": False})
                return
            out = {
                "available": True,
                "ious": m["ious"],
                "n_candidates": m["n_candidates"],
            }
            if "token0_iou" in m:
                out["token0_iou"] = m["token0_iou"]
                out["token0_stability"] = m["token0_stability"]
            self._send_json(out)

        def _route_multimask_png(self, path: str) -> None:
            match = _MM_PNG.match(path)
            if not match:
                self._send_bytes(404, "text/plain", b"not found")
                return
            frame_idx, key = int(match.group(1)), match.group(2)
            self._send_file(dump_dir / "multimask" / f"{frame_idx:06d}_{key}.png")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="dump written by DebugDumpWriter")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    dump_dir = args.dump_dir
    debug_path = dump_dir / "debug.json"
    if not debug_path.is_file():
        raise FileNotFoundError(f"no debug.json in {dump_dir}")
    data = json.loads(debug_path.read_text())
    log = data.get("log", [])
    meta = {"config": data.get("config", {}), "num_records": len(log)}
    mm_meta_path = dump_dir / "multimask.json"
    mm_meta = json.loads(mm_meta_path.read_text()) if mm_meta_path.is_file() else {}
    static_dir = Path(__file__).resolve().parent / "static"

    handler = make_handler(dump_dir, static_dir, meta, log, mm_meta)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"serving {dump_dir} ({len(log)} records) at "
        f"http://{args.host}:{args.port} — Ctrl-C to stop"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        server.shutdown()


if __name__ == "__main__":
    main()
