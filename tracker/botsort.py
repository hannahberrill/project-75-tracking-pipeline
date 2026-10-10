#!/usr/bin/env python3
"""Run Ultralytics BoT-SORT on pre-computed detection CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from ultralytics.trackers.bot_sort import BOTSORT


# Add or edit named BoT-SORT configurations here.
# Every value is passed to BOTSORT via SimpleNamespace.
TRACKER_PARAMETER_SETS: dict[str, dict[str, Any]] = {
    "default": {
        "track_high_thresh": 0.1,
        "track_low_thresh": 0.01,
        "match_thresh": 0.99,
        "new_track_thresh": 0.1,
        "track_buffer": 1000,
        "frame_rate": 30,
        "with_reid": False,
        "proximity_thresh": 0.9,
        "appearance_thresh": 0.1,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "cmc_method": "sparseOptFlow",
        "model": None,
        "device": "cpu",
        "fp16": False},
    "optuna_best": {
        "track_high_thresh": 0.2,
        "track_low_thresh": 0.24000000000000002,
        "new_track_thresh": 0.15000000000000002,
        "match_thresh": 0.95,
        "track_buffer": 150,
        "frame_rate": 30,
        "with_reid": False,
        "proximity_thresh": 0.95,
        "appearance_thresh": 0.56,
        "fuse_score": False,
        "gmc_method": "sparseOptFlow",
        "cmc_method": "sparseOptFlow",
        "model": None,
        "device": "cpu",
        "fp16": False
    },

     "optuna_thresholds_updated": {
        "track_high_thresh": 0.2,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.2,
        "match_thresh": 0.95,
        "track_buffer": 150,
        "frame_rate": 30,
        "with_reid": False,
        "proximity_thresh": 0.95,
        "appearance_thresh": 0.56,
        "fuse_score": False,
        "gmc_method": "sparseOptFlow",
        "cmc_method": "sparseOptFlow",
        "model": None,
        "device": "cpu",
        "fp16": False
        }   
}


class DetectionResults:
    """Minimal results wrapper compatible with Ultralytics BoT-SORT."""

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.int32)
        if len(self.xywh):
            self.xyxy = np.column_stack(
                (
                    self.xywh[:, 0] - self.xywh[:, 2] / 2,
                    self.xywh[:, 1] - self.xywh[:, 3] / 2,
                    self.xywh[:, 0] + self.xywh[:, 2] / 2,
                    self.xywh[:, 1] + self.xywh[:, 3] / 2,
                )
            ).astype(np.float32)
        else:
            self.xyxy = np.empty((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: Any) -> "DetectionResults":
        if isinstance(index, np.ndarray) and index.dtype == bool:
            index = np.flatnonzero(index)
        return DetectionResults(self.xywh[index], self.conf[index], self.cls[index])

    @classmethod
    def from_detections(cls, detections: list[dict[str, Any]]) -> "DetectionResults":
        xywh: list[list[float]] = []
        conf: list[float] = []
        class_ids: list[int] = []
        for detection in detections:
            x1, y1 = float(detection["x1"]), float(detection["y1"])
            x2, y2 = float(detection["x2"]), float(detection["y2"])
            xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, max(1e-3, x2 - x1), max(1e-3, y2 - y1)])
            conf.append(float(detection.get("confidence", detection.get("conf", 0.0))))
            class_ids.append(int(detection.get("class_id", detection.get("cls", 0))))
        return cls(np.asarray(xywh, dtype=np.float32).reshape(-1, 4), np.asarray(conf), np.asarray(class_ids))


def load_detection_rows(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def process_tracking_for_csv(
    csv_path: Path,
    vid_stride: int = 1,
    tracker_params: str | dict[str, Any] | SimpleNamespace = "default",
    max_frames: int = 0,
    conf_thresh: float = 0.0,
    out_dir_override: Path | None = None,
) -> None:
    if isinstance(tracker_params, str):
        if tracker_params not in TRACKER_PARAMETER_SETS:
            raise ValueError(f"parameter set does not exist: {tracker_params}")
        tracker_config = dict(TRACKER_PARAMETER_SETS[tracker_params])
    elif isinstance(tracker_params, SimpleNamespace):
        tracker_config = dict(vars(tracker_params))
    elif isinstance(tracker_params, dict):
        tracker_config = dict(TRACKER_PARAMETER_SETS["default"])
        tracker_config.update(tracker_params)
    else:
        raise TypeError("tracker_params must be str, dict, or SimpleNamespace")

    rows = load_detection_rows(csv_path)
    if not rows:
        print(f"Skipping empty detection CSV: {csv_path}")
        return

    frame_map: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        frame_key = "processed_frame" if "processed_frame" in row else "frame"
        frame_map.setdefault(int(float(row[frame_key])), []).append(row)

    min_frame, max_frame = min(frame_map), max(frame_map)
    if max_frames > 0:
        max_frame = min(max_frame, min_frame + max_frames - 1)

    output_dir = out_dir_override or csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_stem = csv_path.stem.removesuffix("_detections")
    output_csv = output_dir / f"{base_stem}_tracked_detections.csv"

    if "frame_rate" not in tracker_config:
        tracker_config["frame_rate"] = max(1, int(round(30 / max(1, vid_stride))))
    tracker = BOTSORT(SimpleNamespace(**tracker_config))
    tracked_rows: list[dict[str, Any]] = []

    for frame_idx in range(min_frame, max_frame + 1):
        if frame_idx % max(1, vid_stride) != 0:
            continue
        detections = frame_map.get(frame_idx, [])
        if conf_thresh > 0:
            detections = [d for d in detections if float(d.get("confidence", d.get("conf", 0))) >= conf_thresh]
        tracker.frame_id = frame_idx
        tracked = tracker.update(DetectionResults.from_detections(detections), img=None)
        for item in tracked:
            x1, y1, x2, y2, track_id, score, class_id, _ = item.tolist()
            class_id = int(class_id)
            class_name = next(
                (str(d.get("class_name", f"class_{class_id}")) for d in detections if int(float(d.get("class_id", class_id))) == class_id),
                f"class_{class_id}",
            )
            tracked_rows.append({
                "processed_frame": frame_idx // max(1, vid_stride),
                "frame": frame_idx,
                "time_s": frame_idx / 25.0,
                "det_index": len(tracked_rows),
                "tracker_id": int(track_id),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(score),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2),
                "width": float(x2 - x1), "height": float(y2 - y1),
            })

    fieldnames = ["processed_frame", "frame", "time_s", "det_index", "tracker_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height"]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tracked_rows)
    print(f"Saved tracked CSV: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply BoT-SORT to pre-computed detection CSV files.")
    parser.add_argument("--det-csv", "--source", dest="det_csv", required=True, type=Path)
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--tracker_params", default="default")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.det_csv.is_file():
        raise FileNotFoundError(f"Detection CSV does not exist: {args.det_csv}")
    process_tracking_for_csv(args.det_csv.resolve(), args.vid_stride, args.tracker_params, args.max_frames, args.conf, args.out)


if __name__ == "__main__":
    main()
