#!/usr/bin/env python3
"""Web wrapper for yolo_botsort_interpolated_pipeline.py."""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PIPELINE = PROJECT_DIR / "yolo_botsort_interpolated_pipeline.py"
SUBSAMPLE_SCRIPT = PROJECT_DIR / "wednesday-yolov4" / "tracker_evaluation" / "subsample_detections.py"
DEFAULT_CFG = PROJECT_DIR / "wednesday-yolov4" / "model" / "cfg" / "yolov4-tiny-wednesday-v1_2.cfg"
DEFAULT_WEIGHTS = PROJECT_DIR / "wednesday-yolov4" / "model" / "yolov4-tiny-wednesday-v1_2_best.weights"
DEFAULT_NAMES = PROJECT_DIR / "wednesday-yolov4" / "model" / "cfg" / "obj.names"
# Inputs and outputs are transient server-side staging files. They are kept out
# of the workspace and removed when the wrapper shuts down.
RUNS_DIR = Path(tempfile.mkdtemp(prefix="tern_pipeline_web_"))
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024

jobs: dict[str, dict[str, object]] = {}
jobs_lock = threading.Lock()
http_server: ThreadingHTTPServer | None = None


def safe_filename(name: str) -> str:
    name = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9_.() -]+", "_", name).strip(" .")
    return cleaned or "upload"


def parse_upload(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = (safe_filename(filename), payload)
        else:
            fields[name] = payload.decode("utf-8", errors="replace")
    return fields, files


def stream_job(job_id: str, command: list[str], output_dir: Path, fields: dict[str, str], has_detections: bool) -> None:
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "running"
            jobs[job_id]["output"] = "Starting pipeline run\n"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=os.name != "nt",
        )
        with jobs_lock:
            jobs[job_id]["process"] = process
        assert process.stdout is not None
        for line in process.stdout:
            with jobs_lock:
                jobs[job_id]["output"] = str(jobs[job_id]["output"]) + line
        return_code = process.wait()
        files = []
        if return_code == 0 and output_dir.is_dir():
            files = prepare_outputs(output_dir, fields, has_detections)
        with jobs_lock:
            cancelled = bool(jobs[job_id].get("cancel_requested"))
            jobs[job_id].update(
                status="cancelled" if cancelled else ("complete" if return_code == 0 else "failed"),
                return_code=return_code,
                files=files,
            )
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="failed", return_code=1, output=f"{exc}\n", files=[])


def terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


class Handler(BaseHTTPRequestHandler):
    server_version = "PipelineWeb/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.serve_file(BASE_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/style.css":
            self.serve_file(BASE_DIR / "style.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self.serve_file(BASE_DIR / "app.js", "text/javascript; charset=utf-8")
            return
        match = re.fullmatch(r"/download/([a-f0-9]+)/(.+)", parsed.path)
        if match:
            self.download(match.group(1), unquote(match.group(2)))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/cancel":
            self.cancel_requested_job()
            return
        if path == "/shutdown":
            self.shutdown_server()
            return
        if path != "/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_UPLOAD_BYTES:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Uploads are too large")
            return
        body = self.rfile.read(length)
        try:
            fields, files = parse_upload(body, self.headers.get("Content-Type", ""))
            job_id, output_dir = self.create_job(fields, files)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        command = self.build_command(job_id, fields, output_dir)
        with jobs_lock:
            jobs[job_id] = {
                "status": "queued",
                "output": "",
                "files": [],
                "return_code": None,
                "process": None,
                "cancel_requested": False,
            }
        threading.Thread(
            target=stream_job,
            args=(job_id, command, output_dir, fields, files.get("detections") is not None),
            daemon=True,
        ).start()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write((json.dumps({"job_id": job_id, "event": "started"}) + "\n").encode())
        self.wfile.flush()
        last_output = 0
        while True:
            with jobs_lock:
                job = dict(jobs[job_id])
            output = str(job["output"])
            if len(output) > last_output:
                event = {"event": "output", "text": output[last_output:]}
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
                last_output = len(output)
            if job["status"] in {"complete", "failed", "cancelled"}:
                finished_event = {
                    "event": "finished",
                    "job_id": job_id,
                    "status": job["status"],
                    "output": job["output"],
                    "files": job["files"],
                    "return_code": job["return_code"],
                }
                self.wfile.write((json.dumps(finished_event) + "\n").encode())
                self.wfile.flush()
                break
            threading.Event().wait(0.15)

    def cancel_requested_job(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        job_id = payload.get("job_id")
        if not isinstance(job_id, str):
            self.send_error(HTTPStatus.BAD_REQUEST, "A job_id is required")
            return
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Job not found")
                return
            if job["status"] in {"complete", "failed", "cancelled"}:
                self.send_json({"status": job["status"]})
                return
            job["cancel_requested"] = True
            process = job.get("process")
        terminate_process(process)
        self.send_json({"status": "cancelling", "job_id": job_id})

    def shutdown_server(self) -> None:
        with jobs_lock:
            active_ids = [
                job_id for job_id, job in jobs.items()
                if job["status"] in {"queued", "running"}
            ]
        for job_id in active_ids:
            self.cancel_job_by_id(job_id)
        self.send_json({"status": "shutting_down"})
        if http_server is not None:
            threading.Thread(target=http_server.shutdown, daemon=True).start()

    def cancel_job_by_id(self, job_id: str) -> None:
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                return
            job["cancel_requested"] = True
            process = job.get("process")
        terminate_process(process)

    def send_json(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def create_job(self, fields: dict[str, str], files: dict[str, tuple[str, bytes]]) -> tuple[str, Path]:
        video = files.get("video")
        detections = files.get("detections")
        truth = files.get("truth")
        if not video and not detections:
            raise ValueError("Upload a video or a detection CSV")
        if fields.get("run_metrics") == "yes" and not truth:
            raise ValueError("Ground truth XML is required when metrics are enabled")
        if fields.get("output_filtered") == "yes" and not fields.get("class_name", "").strip():
            raise ValueError("Class name is required for filtered tracks")
        run_id = uuid.uuid4().hex
        output_dir = RUNS_DIR / run_id / "output"
        upload_dir = RUNS_DIR / run_id / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        for key, uploaded in (("video", video), ("detections", detections), ("truth", truth)):
            if uploaded:
                filename, payload = uploaded
                (upload_dir / filename).write_bytes(payload)
        return run_id, output_dir

    def build_command(self, job_id: str, fields: dict[str, str], output_dir: Path) -> list[str]:
        upload_dir = RUNS_DIR / job_id / "uploads"
        uploaded = {path.name: path for path in upload_dir.iterdir()}
        video_path = next((p for p in uploaded.values() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}), None)
        detections_path = next((p for p in uploaded.values() if p.suffix.lower() == ".csv"), None)
        truth_path = next((p for p in uploaded.values() if p.suffix.lower() == ".xml"), None)
        command = [sys.executable, "-u", str(PIPELINE), "--out-dir", str(output_dir)]
        if video_path:
            command.extend(["--video", str(video_path)])
        if detections_path:
            command.extend(["--detection-csv", str(detections_path)])
        if fields.get("visualise_detections") == "yes":
            command.append("--visualise-detections")
        if fields.get("run_metrics") == "yes":
            command.extend(["--run-metrics", "--gt-xml", str(truth_path)])
            class_name = fields.get("class_name", "").strip()
            if class_name:
                command.extend(["--class-name", class_name])
            command.extend(["--iou", fields.get("iou", "0.3")])
        if not detections_path:
            def model_path(field: str, default: Path) -> str:
                value = fields.get(field, "").strip()
                path = Path(value) if value else default
                if not path.is_absolute():
                    path = PROJECT_DIR / path
                return str(path.resolve())

            command.extend([
                "--conf", fields.get("conf", "0.25"),
                "--nms", fields.get("nms", "0.45"),
                "--vid-stride", fields.get("vid_stride", "1"),
                "--max-frames", fields.get("max_frames", "0"),
                "--cfg", model_path("cfg", DEFAULT_CFG),
                "--weights", model_path("weights", DEFAULT_WEIGHTS),
                "--names", model_path("names", DEFAULT_NAMES),
            ])
        command.extend([
            "--tracker-params", fields.get("tracker_params", "optuna_thresholds_updated"),
            "--tracker-stride", fields.get("tracker_stride", "1"),
            "--interpolation-max-gap", fields.get("interpolation_max_gap", "150"),
        ])
        if fields.get("visualise_tracks") != "yes":
            command.append("--no-visualise-tracks")
        return command

    def serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def download(self, job_id: str, relative_path: str) -> None:
        root = (RUNS_DIR / job_id / "output").resolve()
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{html.escape(path.name)}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[web] {format % args}")


def csv_path(output_dir: Path, suffix: str) -> Path | None:
    matches = sorted(output_dir.rglob(suffix))
    return matches[0] if matches else None


def prepare_outputs(output_dir: Path, fields: dict[str, str], has_detections: bool) -> list[str]:
    """Create requested derived files and return only the files exposed to the user."""
    interpolated = csv_path(output_dir, "*_tracked_detections_interpolated.csv")
    if interpolated is None:
        return []

    if fields.get("output_stride4") == "yes":
        stride4 = output_dir / f"{interpolated.stem}_stride4_detections.csv"
        subprocess.run(
            [sys.executable, str(SUBSAMPLE_SCRIPT), "--in", str(interpolated), "--out", str(stride4), "--stride", "4"],
            check=True,
        )

    if fields.get("output_filtered") == "yes":
        import pandas as pd

        filtered = output_dir / f"{interpolated.stem}_filtered_{fields['class_name'].strip()}.csv"
        frame = pd.read_csv(interpolated)
        frame = frame[frame["class_name"] == fields["class_name"].strip()].copy()
        frame.to_csv(filtered, index=False)

    selected: list[Path] = [interpolated]
    if fields.get("output_tracked") == "yes":
        raw = csv_path(output_dir, "*_tracked_detections.csv")
        if raw is not None and raw != interpolated:
            selected.append(raw)
    if fields.get("output_filtered") == "yes":
        selected.append(output_dir / f"{interpolated.stem}_filtered_{fields['class_name'].strip()}.csv")
    if fields.get("output_stride4") == "yes":
        selected.append(output_dir / f"{interpolated.stem}_stride4_detections.csv")
    if not has_detections:
        selected.extend(path for path in output_dir.rglob("*_detections.csv") if "_tracked_detections" not in path.name)
    if fields.get("run_metrics") == "yes":
        selected.extend(output_dir.rglob("*_eval_summary.csv"))
        selected.extend(output_dir.rglob("metrics_summary.csv"))
    if fields.get("visualise_detections") == "yes" or fields.get("visualise_tracks") == "yes":
        selected.extend(output_dir.rglob("*.mp4"))
    return sorted({str(path.relative_to(output_dir)) for path in selected if path.is_file()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Web wrapper for the YOLOv4 + BoT-SORT pipeline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    global http_server
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    http_server = server
    print(f"Pipeline web wrapper: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    finally:
        shutil.rmtree(RUNS_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
