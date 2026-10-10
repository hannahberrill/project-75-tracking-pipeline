#!/usr/bin/env python3
"""Render tracked-detection CSV files over their source videos."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]


def safe_name(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def parse_frame(row: dict[str, str]) -> int:
    """Use the source-video frame column, with compatibility fallbacks."""
    for key in ("frame", "source_frame", "processed_frame"):
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    raise ValueError("CSV row has no frame, source_frame, or processed_frame column")


def optional_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def tracker_id(row: dict[str, str]) -> int | None:
    value = row.get("tracker_id")
    if value in (None, "", "nan", "NaN"):
        return None
    return int(float(value))


def load_rows(csv_path: Path) -> dict[int, list[dict[str, str]]]:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            by_frame[parse_frame(row)].append(row)
    return by_frame


def track_colour(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (0, 255, 255)
    return (50 + (track_id * 37) % 206, 80 + (track_id * 53) % 176, 120 + (track_id * 97) % 136)


def draw_detections(
    frame: np.ndarray,
    rows: list[dict[str, str]],
    histories: dict[int, deque[tuple[float, float]]],
    trail_length: int,
) -> np.ndarray:
    output = frame.copy()
    for row in rows:
        x1 = int(round(optional_float(row, "x1")))
        y1 = int(round(optional_float(row, "y1")))
        x2 = int(round(optional_float(row, "x2")))
        y2 = int(round(optional_float(row, "y2")))
        track = tracker_id(row)
        colour = track_colour(track)
        class_name = row.get("class_name") or f"class_{row.get('class_id', '0')}"
        confidence = optional_float(row, "confidence")
        label = f"{class_name} {confidence:.2f}"
        if track is not None:
            label += f" | T{track}"
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            history = histories.setdefault(track, deque(maxlen=trail_length))
            history.append(center)
            if len(history) > 1:
                points = np.asarray(history, dtype=np.int32)
                cv2.polylines(output, [points], False, colour, 2)

        cv2.rectangle(output, (x1, y1), (x2, y2), colour, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        label_y = max(0, y1 - text_height - 6)
        cv2.rectangle(
            output,
            (x1, label_y),
            (x1 + text_width + 4, y1),
            colour,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x1 + 2, max(text_height + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return output


def render_video(
    detections_csv: Path,
    source_video: Path,
    output_dir: Path,
    trail_length: int,
    max_frames: int,
) -> Path:
    if not detections_csv.is_file():
        raise FileNotFoundError(f"Tracked detections CSV does not exist: {detections_csv}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source_video}")
    if trail_length < 1:
        raise ValueError("trail length must be at least 1")

    detections_by_frame = load_rows(detections_csv)
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_name(source_video)}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, source_fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    histories: dict[int, deque[tuple[float, float]]] = {}
    processed = 0
    annotated_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rows = detections_by_frame.get(processed, [])
            writer.write(draw_detections(frame, rows, histories, trail_length))
            if rows:
                annotated_frames += 1
            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        capture.release()
        writer.release()

    print(
        f"Saved {output_path} ({processed}/{frame_count or '?'} frames, "
        f"{annotated_frames} with detections)"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise tracked detections over source videos."
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory in which the annotated video will be saved.")
    parser.add_argument("--detections", type=Path, required=True, help="Tracked detections CSV.")
    parser.add_argument("--video", type=Path, required=True, help="Source video.")
    parser.add_argument("--trail-length", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process the complete video.")
    args = parser.parse_args()

    render_video(
        args.detections,
        args.video,
        args.out_dir,
        args.trail_length,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
