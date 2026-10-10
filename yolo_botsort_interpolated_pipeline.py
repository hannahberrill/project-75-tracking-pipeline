#!/usr/bin/env python3
r"""Run a YOLOv4 detection -> BoT-SORT tracking -> linear interpolation of tracks pipeline on one video.

Inputs:
  --video /path/to/video.mp4
      Single-video mode. Process one source clip.
  --detection-csv /path/to/detections.csv
      Single-video mode. Skip YOLO inference and track an existing detection CSV.
  --out-dir /path/to/output_root
      Output directory for generated CSVs and optional videos.
  --gt-xml /path/to/ground_truth.xml
      Optional GT XML for single-video evaluation when --run-metrics is used.
  --visualise-detections
      Save an annotated detection-only preview video (
      <video_stem>_annotated_detection_only.mp4).
  --visualise-tracks / --no-visualise-tracks
      Enable or disable the annotated tracked output video; enabled by default.
  --run-metrics
      Evaluate tracking/detection accuracy against GT after tracking.
      Also writes a combined metrics_summary.csv in the output root.
  --tracker-params NAME
      BoT-SORT parameter set name from botsort.py; default is optuna_thresholds_updated.
  --tracker-stride N
      Tracking stride applied to the detection CSV; default 1.
  --interpolation-max-gap N
      Maximum missing-frame gap to fill after tracking; default 150.
  --conf FLOAT
      YOLO confidence threshold for detections; default 0.25.
  --nms FLOAT
      NMS threshold; default 0.45.
  --vid-stride N
      Process every Nth frame during YOLO detection.
  --max-frames N
      Maximum number of frames to process. 0 = all frames.
  --cfg /path/to/model.cfg
      YOLOv4 config; defaults to the 17-class v1_2 model.
  --weights /path/to/model.weights
      YOLOv4 weights; defaults to the 17-class v1_2 model.
  --names /path/to/classes.names
      YOLO class-name file; defaults to obj.names, including STESTR_FLY.
  --class-name NAME
      Optional class filter for metrics evaluation, for example STESTR_FLY.
  --iou FLOAT
      IoU threshold for metrics evaluation; default 0.3 and overrideable.

Outputs:
  <out-dir>/<video_stem>/
      - <video_stem>_detections.csv
      - <video_stem>_tracked_detections.csv
      - <video_stem>_tracked_detections_interpolated.csv
      - optional <video_stem>_annotated_detection_only.mp4
      - optional <video_stem>_annotated.mp4
      - optional <video_stem>_stride4_detections.csv
      - optional <video_stem>_eval_summary.csv
    <out-dir>/metrics_summary.csv
            Combined MOTA/HOTA summary when --run-metrics is enabled.

Example:
  python yolo_botsort_interpolated_pipeline.py \
      --video path/to/video.mp4 \
      --out-dir outputs/test_run \
      --visualise-detections \
      --run-metrics \
      --gt-xml path/to/ground_truth.xml
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
WEDNESDAY_DIR = PROJECT_DIR / "wednesday-yolov4"
TRACKER_DIR = PROJECT_DIR / "tracker"

DEFAULT_CFG = WEDNESDAY_DIR / "model" / "cfg" / "yolov4-tiny-wednesday-v1_2.cfg"
DEFAULT_WEIGHTS = WEDNESDAY_DIR / "model" / "yolov4-tiny-wednesday-v1_2_best.weights"
DEFAULT_NAMES = WEDNESDAY_DIR / "model" / "cfg" / "obj.names"

TRACKER_SCRIPT = TRACKER_DIR / "botsort.py"
INTERPOLATION_SCRIPT = TRACKER_DIR / "interpolate_tracks.py"
VISUALISER_SCRIPT = TRACKER_DIR / "visualise_tracked_detections.py"
SUBSAMPLE_SCRIPT = TRACKER_DIR / "subsample_detections.py"
EVAL_SCRIPT = PROJECT_DIR / "evaluate_mota_xml.py"
SUMMARY_SCRIPT = TRACKER_DIR / "summarize_eval_summaries.py"


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def parse_net_size(cfg_path: Path) -> tuple[int, int]:
    width = height = None
    for line in cfg_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("width=") and width is None:
            width = int(line.split("=", 1)[1].strip())
        elif line.startswith("height=") and height is None:
            height = int(line.split("=", 1)[1].strip())
        if width and height:
            return width, height
    return 416, 416


def load_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def detect_frame(
    net: Any,
    output_names: list[str],
    frame: np.ndarray,
    inp_size: tuple[int, int],
    classes: list[str],
    conf_thr: float,
    nms_thr: float,
) -> list[dict[str, Any]]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, inp_size, swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_names)

    boxes: list[list[int]] = []
    confidences: list[float] = []
    class_ids: list[int] = []

    for out in outs:
        for det in out:
            if len(det) < 6:
                continue
            obj_score = float(det[4])
            scores = det[5:]
            class_id = int(np.argmax(scores))
            class_score = float(scores[class_id])
            conf = obj_score * class_score
            if conf < conf_thr:
                continue
            cx, cy, bw, bh = det[:4]
            x = int((float(cx) - float(bw) / 2) * w)
            y = int((float(cy) - float(bh) / 2) * h)
            ww = int(float(bw) * w)
            hh = int(float(bh) * h)
            boxes.append([x, y, x + ww, y + hh])
            confidences.append(conf)
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).reshape(-1).tolist()

    dets: list[dict[str, Any]] = []
    for idx in idxs:
        x1, y1, x2, y2 = boxes[idx]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)
        class_id = class_ids[idx]
        class_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
        dets.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidences[idx]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
            }
        )
    return dets


def draw_detections(frame: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    out = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        colour = (50 + (det["class_id"] * 37) % 206, 80 + (det["class_id"] * 53) % 176, 120 + (det["class_id"] * 97) % 136)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, max(th + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def run_yolo_detection(
    video_path: Path,
    output_dir: Path,
    cfg: Path,
    weights: Path,
    names: Path,
    conf: float,
    nms: float,
    vid_stride: int,
    max_frames: int,
    visualise_detections: bool,
) -> Path:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    classes = load_names(names)
    inp_width, inp_height = parse_net_size(cfg)
    inp_size = (inp_width, inp_height)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    annotated_path = output_dir / f"{safe_name(video_path)}_annotated_detection_only.mp4"
    if visualise_detections:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_path), fourcc, max(1.0, fps / max(1, vid_stride)), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create detection-only video writer at {annotated_path}")

    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_names = net.getUnconnectedOutLayersNames()

    rows: list[dict[str, Any]] = []
    det_counts: list[int] = []
    infer_times: list[float] = []
    processed = 0
    frame_idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % vid_stride != 0:
                continue
            if max_frames and processed >= max_frames:
                break

            start = time.time()
            dets = detect_frame(net, output_names, frame, inp_size, classes, conf, nms)
            infer_times.append(time.time() - start)
            det_counts.append(len(dets))

            for det_index, det in enumerate(dets):
                rows.append(
                    {
                        "processed_frame": processed,
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "det_index": det_index,
                        "tracker_id": None,
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "x1": det["x1"],
                        "y1": det["y1"],
                        "x2": det["x2"],
                        "y2": det["y2"],
                        "cx": det["cx"],
                        "cy": det["cy"],
                        "width": det["width"],
                        "height": det["height"],
                    }
                )

            annotated = draw_detections(frame, dets)
            if writer is not None:
                writer.write(annotated)

            processed += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    csv_path = output_dir / f"{safe_name(video_path)}_detections.csv"
    fieldnames = [
        "processed_frame",
        "frame",
        "time_s",
        "det_index",
        "tracker_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "width",
        "height",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    summary = {
        "source": str(video_path),
        "cfg": str(cfg),
        "weights": str(weights),
        "names": str(names),
        "classes": classes,
        "input_size": inp_size,
        "conf": conf,
        "nms": nms,
        "stride": vid_stride,
        "frame_count": frame_count,
        "processed_frames": processed,
        "det_counts": det_counts,
        "infer_times": infer_times,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved detection CSV: {csv_path}")
    if visualise_detections:
        print(f"Saved detection visualisation: {annotated_path}")
    return csv_path


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_stage_time(stage_name: str, start_time: float) -> None:
    print(f"{stage_name} runtime: {time.perf_counter() - start_time:.2f} s")


def run_tracker_on_detections(
    det_csv: Path,
    tracker_params: str,
    tracker_stride: int,
    output_dir: Path,
) -> Path:
    botsort_mod = load_module_from_path("botsort_mod", TRACKER_SCRIPT)
    botsort_mod.process_tracking_for_csv(
        det_csv,
        tracker_stride,
        tracker_params,
        out_dir_override=output_dir,
    )

    base_stem = det_csv.stem.removesuffix("_detections")
    tracked_csv = output_dir / f"{base_stem}_tracked_detections.csv"
    if not tracked_csv.is_file():
        raise FileNotFoundError(f"Expected tracked CSV was not created: {tracked_csv}")
    return tracked_csv


def run_interpolation(tracked_csv: Path, output_dir: Path, max_gap: int) -> Path:
    interpolation_mod = load_module_from_path("interpolate_tracks_mod", INTERPOLATION_SCRIPT)
    interpolated_csv = interpolation_mod.interpolate_tracked_detections(
        input_csv=tracked_csv,
        output_dir=output_dir,
        max_gap=max_gap,
    )
    interpolated_csv = Path(interpolated_csv)
    if not interpolated_csv.is_file():
        raise FileNotFoundError(f"Expected interpolated CSV was not created: {interpolated_csv}")
    return interpolated_csv


def run_visualisation(tracked_csv: Path, source_video: Path, output_dir: Path) -> Path:
    if not VISUALISER_SCRIPT.is_file():
        raise FileNotFoundError(f"Visualiser script not found: {VISUALISER_SCRIPT}")
    cmd = [
        sys.executable,
        str(VISUALISER_SCRIPT),
        "--out-dir",
        str(output_dir),
        "--detections",
        str(tracked_csv),
        "--video",
        str(source_video),
    ]
    subprocess.run(cmd, check=True)
    output_path = output_dir / f"{safe_name(source_video)}_annotated.mp4"
    if not output_path.is_file():
        raise FileNotFoundError(f"Expected annotated tracking video was not created: {output_path}")
    return output_path


def run_metrics_for_video(gt_xml: Path, tracked_csv: Path, output_dir: Path, class_name: str | None, iou: float) -> Path:
    if not gt_xml.is_file():
        raise FileNotFoundError(f"Ground truth XML does not exist: {gt_xml}")
    if not tracked_csv.is_file():
        raise FileNotFoundError(f"Tracked CSV does not exist: {tracked_csv}")

    stride4_out = output_dir / f"{tracked_csv.stem}_stride4_detections.csv"
    subprocess.run(
        [
            sys.executable,
            str(SUBSAMPLE_SCRIPT),
            "--in",
            str(tracked_csv),
            "--out",
            str(stride4_out),
            "--stride",
            "4",
        ],
        check=True,
    )

    if class_name:
        filtered_csv = output_dir / f"{tracked_csv.stem}_filtered_{class_name}.csv"
        import pandas as pd
        df = pd.read_csv(stride4_out)
        if "class_name" in df.columns:
            df = df[df["class_name"] == class_name].copy()
        else:
            raise ValueError(f"No class_name column found in {stride4_out}; cannot filter by class {class_name}")
        df.to_csv(filtered_csv, index=False)
        det_csv_for_eval = filtered_csv
    else:
        det_csv_for_eval = stride4_out

    eval_args = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--gt-xml",
        str(gt_xml),
        "--det-csv",
        str(det_csv_for_eval),
        "--iou",
        str(iou),
    ]
    if class_name:
        eval_args.extend(["--class-name", class_name])
    summary_csv = output_dir / f"{tracked_csv.stem}_eval_summary.csv"
    eval_args.extend(["--out-csv", str(summary_csv)])
    subprocess.run(eval_args, check=True)
    return summary_csv


def run_metrics_summary(input_root: Path) -> Path:
    if not SUMMARY_SCRIPT.is_file():
        raise FileNotFoundError(f"Metrics summarizer not found: {SUMMARY_SCRIPT}")
    output_path = input_root / "metrics_summary.csv"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            str(input_root),
            "--out",
            str(output_path),
        ],
        check=True,
    )
    if not output_path.is_file():
        raise FileNotFoundError(f"Expected metrics summary was not created: {output_path}")
    return output_path


def process_single_video(args: argparse.Namespace) -> None:
    video_path = args.video.resolve() if args.video is not None else None
    if args.out_dir is None:
        raise ValueError("--out-dir is required when processing a single video")
    output_root = args.out_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_label = safe_name(video_path) if video_path is not None else "outputs"
    video_dir = output_root / source_label
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.detection_csv is not None:
        det_csv = args.detection_csv.resolve()
        if not det_csv.is_file():
            raise FileNotFoundError(f"Detection CSV does not exist: {det_csv}")
        print(f"Using existing detection CSV: {det_csv}")
    else:
        if video_path is None:
            raise ValueError("Either --video or --detection-csv must be supplied for single-video mode")
        stage_start = time.perf_counter()
        det_csv = run_yolo_detection(
            video_path=video_path,
            output_dir=video_dir,
            cfg=args.cfg,
            weights=args.weights,
            names=args.names,
            conf=args.conf,
            nms=args.nms,
            vid_stride=args.vid_stride,
            max_frames=args.max_frames,
            visualise_detections=args.visualise_detections,
        )
        log_stage_time("Detection stage", stage_start)

    stage_start = time.perf_counter()
    tracked_csv = run_tracker_on_detections(
        det_csv=det_csv,
        tracker_params=args.tracker_params,
        tracker_stride=args.tracker_stride,
        output_dir=video_dir,
    )
    log_stage_time("Tracking stage", stage_start)

    stage_start = time.perf_counter()
    tracked_csv = run_interpolation(
        tracked_csv=tracked_csv,
        output_dir=video_dir,
        max_gap=args.interpolation_max_gap,
    )
    log_stage_time("Interpolation stage", stage_start)

    if args.visualise_tracks:
        if video_path is None:
            print("Skipping tracked-video visualisation because no source video was supplied.")
        else:
            stage_start = time.perf_counter()
            run_visualisation(tracked_csv=tracked_csv, source_video=video_path, output_dir=video_dir)
            log_stage_time("Visualisation stage", stage_start)

    if args.run_metrics:
        if args.gt_xml is None:
            raise ValueError("--gt-xml is required when --run-metrics is used in single-video mode")
        stage_start = time.perf_counter()
        run_metrics_for_video(
            gt_xml=args.gt_xml.resolve(),
            tracked_csv=tracked_csv,
            output_dir=video_dir,
            class_name=args.class_name,
            iou=args.iou,
        )
        log_stage_time("Metrics stage", stage_start)
        stage_start = time.perf_counter()
        run_metrics_summary(output_root)
        log_stage_time("Metrics summary stage", stage_start)

    if video_path is not None:
        print(f"Completed pipeline for {video_path}")
    else:
        print(f"Completed tracking pipeline for {det_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a YOLOv4 + BoT-SORT pipeline for one video.")
    parser.add_argument("--video", type=Path, help="Single source video path.")
    parser.add_argument("--detection-csv", type=Path, help="Optional existing detection CSV to skip straight to tracking.")
    parser.add_argument("--out-dir", type=Path, help="Output root directory for single-video runs.")
    parser.add_argument("--gt-xml", type=Path, help="Ground-truth XML for single-video metrics.")
    parser.add_argument("--visualise-detections", action="store_true", help="Save annotated detection-only preview videos.")
    parser.add_argument("--visualise-tracks", dest="visualise_tracks", action="store_true", default=True, help="Save annotated tracked output videos (default: enabled).")
    parser.add_argument("--no-visualise-tracks", dest="visualise_tracks", action="store_false", help="Disable the tracked annotated output video.")
    parser.add_argument("--run-metrics", action="store_true", help="Downsample tracked detections to stride 4 and evaluate against GT XML.")
    parser.add_argument("--tracker-params", default="optuna_thresholds_updated", help="BoT-SORT parameter set name.")
    parser.add_argument("--tracker-stride", type=int, default=1, help="Tracking stride applied to the detection CSV.")
    parser.add_argument("--interpolation-max-gap", type=int, default=150, help="Maximum missing-frame gap to interpolate after tracking.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.45, help="Non-maximum suppression threshold.")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame during detection.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process up to N frames. 0 means all frames.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="YOLOv4 config path (defaults to the 17-class v1_2 model).")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLOv4 weights path (defaults to the 17-class v1_2 model).")
    parser.add_argument("--names", type=Path, default=DEFAULT_NAMES, help="YOLO class-name file (defaults to obj.names, including STESTR_FLY).")
    parser.add_argument("--class-name", default=None, help="Optional class filter for metrics evaluation; use the exact GT class label, e.g. STESTR_FLY.")
    parser.add_argument("--iou", type=float, default=0.3, help="Intersection-over-union threshold for metrics evaluation; override with --iou VALUE.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_start = time.perf_counter()

    try:
        if args.video is None and args.detection_csv is None:
            parser.error("Either --video or --detection-csv must be supplied")
        process_single_video(args)
    finally:
        print(f"Total pipeline runtime: {time.perf_counter() - run_start:.2f} s")


if __name__ == "__main__":
    main()
