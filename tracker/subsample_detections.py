#!/usr/bin/env python3
"""
Subsample a tracked detections CSV to every Nth original video frame,
renumbering processed_frame as the original frame number divided by N so it
lines up with how CVAT numbers frames in a stride-N annotation export.

This keeps `frame` as the ORIGINAL video frame number (so you can always trace
a row back to the source footage) and rewrites `processed_frame` to be the
sequential index within the subsampled set.

Usage:
    python subsample_detections.py --in tracked_detections.csv --stride 4 --out stride4_detections.csv

    python evaluate_mota_xml.py \
        --gt-xml annotations.xml \
        --det-csv stride4_detections.csv \
        --class-name STESTR_FLY \
        --iou 0.3 \
        --out-csv detection_only_eval.csv

The target stride is relative to the original video, regardless of the stride
used when the tracker produced the input file. Therefore use --stride 4 for
inputs tracked at stride 1, 2, or 4.

NOTE: if you're evaluating a *tracker* (BoT-SORT/ByteTrack) at this stride, run
the tracker on the OUTPUT of this script, not the other way around -- tracking
association behaves differently with frame gaps, so subsampling an already-
tracked full-rate run does not correctly simulate running the tracker at that
stride. Detection-only metrics (precision/recall/MODA) are fine to subsample
after the fact either way.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--stride",
        required=True,
        type=int,
        help="Target stride in original video frames (use 4 for stride-4 output).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    if "frame" not in df.columns:
        raise SystemExit("Expected a 'frame' column (original video frame number) in the input CSV.")

    kept = df[df["frame"] % args.stride == 0].copy()
    kept["processed_frame"] = kept["frame"] // args.stride
    unique_frames = sorted(kept["frame"].unique())

    kept = kept.sort_values(["processed_frame", "det_index"]).reset_index(drop=True)
    kept.to_csv(args.out, index=False)

    print(f"Input rows:  {len(df)}  (original frame range {df['frame'].min()}-{df['frame'].max()})")
    print(f"Output rows: {len(kept)}  (stride {args.stride}, {len(unique_frames)} sampled frames, "
          f"processed_frame range {kept['processed_frame'].min()}-{kept['processed_frame'].max()})")
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
