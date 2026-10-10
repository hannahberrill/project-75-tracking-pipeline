#!/usr/bin/env python3
"""
Evaluate detection/tracking accuracy (MOTA, HOTA, and friends) against a CVAT XML export,
and export summary metrics to a CSV file in the same directory as the detection CSV.

Inputs:
  --gt-xml:
      Path to the CVAT XML ground-truth export.
  --det-csv:
      Path to the detection/tracking CSV.
  --class-name:
      Optional class filter (e.g. STESTR_FLY).
  --iou:
      Intersection-over-union threshold (default: 0.5).
  --out-csv:
      Optional path or filename for summary CSV export.
      If omitted or given as a relative path/filename, it will be saved in the same directory as --det-csv.
    Defaults to '<video_name>_eval_summary.csv' in the --det-csv directory.

Usage:
    python evaluate_mota_xml.py \
        --gt-xml annotations.xml \
        --det-csv /path/to/detections.csv \
        --class-name STESTR_FLY \
        --iou 0.3
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def load_gt_xml(xml_path: Path, label_filter: str | None) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for track in root.findall("track"):
        if label_filter and track.get("label") != label_filter:
            continue
        gt_id = int(track.get("id"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            rows.append({
                "frame": int(box.get("frame")),
                "gt_id": gt_id,
                "x1": float(box.get("xtl")),
                "y1": float(box.get("ytl")),
                "x2": float(box.get("xbr")),
                "y2": float(box.get("ybr")),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["frame", "gt_id"]).reset_index(drop=True)
    return df


def load_det_csv(csv_path: Path, class_name: str | None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "processed_frame" in df.columns:
        df = df.drop(columns=[c for c in ["frame"] if c in df.columns])
        df = df.rename(columns={"processed_frame": "frame"})
    if class_name and "class_name" in df.columns:
        df = df[df["class_name"] == class_name].reset_index(drop=True)
    return df


def iou(a, b) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def run_detection_only(gt: pd.DataFrame, det: pd.DataFrame, iou_thresh: float) -> dict:
    max_common = min(gt["frame"].max(), det["frame"].max())
    min_common = max(gt["frame"].min(), det["frame"].min())
    if max_common < gt["frame"].max() or min_common > gt["frame"].min():
        print(f"NOTE: restricting to overlapping frame range [{min_common}, {max_common}] "
              f"(GT spans [{gt['frame'].min()}, {gt['frame'].max()}], "
              f"detections span [{det['frame'].min()}, {det['frame'].max()}])")
    gt_c = gt[gt["frame"].between(min_common, max_common)]
    det_c = det[det["frame"].between(min_common, max_common)]
    frames = sorted(set(gt_c["frame"]) | set(det_c["frame"]))

    tp = fp = fn = gt_total = 0
    for f in frames:
        g = gt_c[gt_c["frame"] == f][["x1", "y1", "x2", "y2"]].values
        d = det_c[det_c["frame"] == f][["x1", "y1", "x2", "y2"]].values
        gt_total += len(g)
        if len(g) == 0:
            fp += len(d)
            continue
        if len(d) == 0:
            fn += len(g)
            continue
        m = np.zeros((len(g), len(d)))
        for i in range(len(g)):
            for j in range(len(d)):
                m[i, j] = iou(g[i], d[j])
        r, c = linear_sum_assignment(1 - m)
        matched_g, matched_d = set(), set()
        for ri, ci in zip(r, c):
            if m[ri, ci] >= iou_thresh:
                matched_g.add(ri)
                matched_d.add(ci)
        tp += len(matched_g)
        fn += len(g) - len(matched_g)
        fp += len(d) - len(matched_d)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    moda = 1 - (fn + fp) / gt_total if gt_total else float("nan")

    print()
    print("=== Detection-only metrics (no tracker_id present -- no IDSW computed) ===")
    print(f"IoU threshold: {iou_thresh}")
    print(f"Frames evaluated: {len(frames)}")
    print(f"GT boxes: {gt_total}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"MODA (MOTA without identity switches): {moda:.3f}")

    return {
        "eval_type": "Detection-only",
        "num_frames": len(frames),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "moda": moda,
    }


def compute_hota(gt: pd.DataFrame, det: pd.DataFrame, iou_thresh: float) -> dict:
    from trackeval.metrics import HOTA as TrackEvalHOTA

    gt_id_to_idx = {gt_id: i for i, gt_id in enumerate(sorted(gt["gt_id"].unique()))}
    tr_id_to_idx = {tr_id: i for i, tr_id in enumerate(sorted(det["tracker_id"].astype(int).unique()))}

    frames = sorted(set(gt["frame"]).union(det["frame"]))
    min_common = gt["frame"].min()
    max_common = gt["frame"].max()
    frames = [f for f in frames if min_common <= f <= max_common]

    gt_ids_by_t = []
    tracker_ids_by_t = []
    similarity_scores = []

    for f in frames:
        g = gt[gt["frame"] == f]
        d = det[det["frame"] == f]
        gt_ids = np.array([gt_id_to_idx[gid] for gid in g["gt_id"].tolist()], dtype=int)
        tr_ids = np.array([tr_id_to_idx[int(tid)] for tid in d["tracker_id"].tolist()], dtype=int)

        if len(g) == 0 or len(d) == 0:
            sim = np.zeros((len(g), len(d)), dtype=float)
        else:
            g_boxes = g[["x1", "y1", "x2", "y2"]].values
            d_boxes = d[["x1", "y1", "x2", "y2"]].values
            sim = np.zeros((len(g_boxes), len(d_boxes)), dtype=float)
            for i in range(len(g_boxes)):
                for j in range(len(d_boxes)):
                    val = iou(g_boxes[i], d_boxes[j])
                    sim[i, j] = val if val >= iou_thresh else 0.0

        gt_ids_by_t.append(gt_ids)
        tracker_ids_by_t.append(tr_ids)
        similarity_scores.append(sim)

    metrics = TrackEvalHOTA()
    data = {
        "gt_ids": gt_ids_by_t,
        "tracker_ids": tracker_ids_by_t,
        "similarity_scores": similarity_scores,
        "num_gt_ids": len(gt_id_to_idx),
        "num_tracker_ids": len(tr_id_to_idx),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(det),
    }
    res = metrics.eval_sequence(data)
    return {
        "HOTA": float(np.mean(res["HOTA"])),
        "DetA": float(np.mean(res["DetA"])),
        "AssA": float(np.mean(res["AssA"])),
        "DetRe": float(np.mean(res["DetRe"])),
        "DetPr": float(np.mean(res["DetPr"])),
        "LocA": float(np.mean(res["LocA"])),
    }


def run_full_mota(gt: pd.DataFrame, det: pd.DataFrame, iou_thresh: float) -> dict:
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    max_common = min(gt["frame"].max(), det["frame"].max())
    min_common = max(gt["frame"].min(), det["frame"].min())
    frames = sorted(set(gt["frame"]).union(det["frame"]))
    frames = [f for f in frames if min_common <= f <= max_common]

    for f in frames:
        g = gt[gt["frame"] == f]
        d = det[det["frame"] == f]
        g_ids = g["gt_id"].tolist()
        d_ids = d["tracker_id"].astype(int).tolist()
        g_boxes = g[["x1", "y1", "x2", "y2"]].values
        d_boxes = d[["x1", "y1", "x2", "y2"]].values

        if len(g_boxes) and len(d_boxes):
            dist = np.full((len(g_boxes), len(d_boxes)), np.nan)
            for i in range(len(g_boxes)):
                for j in range(len(d_boxes)):
                    v = iou(g_boxes[i], d_boxes[j])
                    dist[i, j] = 1 - v if v >= iou_thresh else np.nan
        else:
            dist = np.full((len(g_boxes), len(d_boxes)), np.nan)

        acc.update(g_ids, d_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["num_frames", "mota", "motp", "num_switches",
                                        "num_false_positives", "num_misses",
                                        "num_matches", "precision", "recall"], name="acc")
    hota = compute_hota(gt, det, iou_thresh)

    print()
    print("=== Full MOTA and HOTA (tracker_id present) ===")
    print(f"IoU threshold: {iou_thresh}")
    print(summary.to_string())
    print(f"HOTA: {hota['HOTA']:.3f}  DetA: {hota['DetA']:.3f}  AssA: {hota['AssA']:.3f}  DetRe: {hota['DetRe']:.3f}  DetPr: {hota['DetPr']:.3f}  LocA: {hota['LocA']:.3f}")

    results = {
        "eval_type": "Full MOTA and HOTA",
        "num_frames": int(summary["num_frames"].iloc[0]),
        "mota": float(summary["mota"].iloc[0]),
        "motp": float(summary["motp"].iloc[0]),
        "num_switches": int(summary["num_switches"].iloc[0]),
        "num_false_positives": int(summary["num_false_positives"].iloc[0]),
        "num_misses": int(summary["num_misses"].iloc[0]),
        "num_matches": int(summary["num_matches"].iloc[0]),
        "precision": float(summary["precision"].iloc[0]),
        "recall": float(summary["recall"].iloc[0]),
    }
    results.update(hota)
    return results


def export_csv(
    output_path: Path,
    gt_xml_path: Path,
    det_csv_path: Path,
    class_name: str | None,
    iou_thresh: float,
    gt: pd.DataFrame,
    det: pd.DataFrame,
    eval_results: dict,
) -> None:
    gt_min_f = gt["frame"].min() if len(gt) else 0
    gt_max_f = gt["frame"].max() if len(gt) else 0
    det_min_f = det["frame"].min() if len(det) else 0
    det_max_f = det["frame"].max() if len(det) else 0

    rows = [
        ["Category", "Metric / Field", "Value"],
        # Input Sources & Config
        ["Input Source", "GT XML File", str(gt_xml_path.resolve())],
        ["Input Source", "Detection CSV File", str(det_csv_path.resolve())],
        ["Configuration", "Class Name Filter", class_name if class_name else "ALL"],
        ["Configuration", "IoU Threshold", iou_thresh],
        # Dataset Summaries
        ["GT Summary", "GT Total Boxes", len(gt)],
        ["GT Summary", "GT Total Tracks", gt["gt_id"].nunique() if len(gt) else 0],
        ["GT Summary", "GT Frame Range", f"{gt_min_f}-{gt_max_f}"],
        ["Detection Summary", "Detections Total Boxes", len(det)],
        ["Detection Summary", "Detections Frame Range", f"{det_min_f}-{det_max_f}"],
    ]

    # Add calculated evaluation metrics
    for metric_name, value in eval_results.items():
        val_str = f"{value:.6f}" if isinstance(value, float) else str(value)
        rows.append(["Evaluation Metrics", metric_name, val_str])

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nSaved evaluation summary CSV to: {output_path.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-xml", required=True, type=Path)
    ap.add_argument("--det-csv", required=True, type=Path)
    ap.add_argument("--class-name", default=None, help="e.g. STESTR_FLY. Applied to both GT track label and detections class_name.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Path or custom filename for summary CSV export. "
             "If omitted or given as a relative path, it is saved in the same directory as --det-csv."
    )
    args = ap.parse_args()

    gt = load_gt_xml(args.gt_xml, args.class_name)
    det = load_det_csv(args.det_csv, args.class_name)

    print(f"GT: {len(gt)} boxes, {gt['gt_id'].nunique() if len(gt) else 0} tracks, frames {gt['frame'].min() if len(gt) else 0}-{gt['frame'].max() if len(gt) else 0}")
    print(f"Detections: {len(det)} boxes, frames {det['frame'].min() if len(det) else 0}-{det['frame'].max() if len(det) else 0}")

    has_tracker_ids = "tracker_id" in det.columns and det["tracker_id"].notna().any()
    if has_tracker_ids:
        det_eval = det[det["tracker_id"].notna()].copy()
        results = run_full_mota(gt, det_eval, args.iou)
    else:
        print("\nNOTE: tracker_id column is empty -- these are raw untracked detections.")
        print("Computing detection-only metrics. Run BoT-SORT/ByteTrack first for full MOTA with IDSW.")
        results = run_detection_only(gt, det, args.iou)

    # Resolve output path: relative paths or defaults are placed in the same directory as det-csv
    if args.out_csv is None:
        video_name = args.det_csv.stem.split("_tracked_detections", maxsplit=1)[0]
        out_csv_path = args.det_csv.parent / f"{video_name}_eval_summary.csv"
    elif not args.out_csv.is_absolute():
        out_csv_path = args.det_csv.parent / args.out_csv
    else:
        out_csv_path = args.out_csv

    export_csv(out_csv_path, args.gt_xml, args.det_csv, args.class_name, args.iou, gt, det, results)


if __name__ == "__main__":
    main()