#!/usr/bin/env python3
"""
Combine per-video evaluation summaries into one comparison CSV.

Input:
    A root directory containing one or more parameter-set subdirectories. Each
    subdirectory should contain evaluation files named ``*_eval_summary.csv``.
    Each summary file must include ``mota`` and ``HOTA`` metric rows.

Output:
    ``metrics_summary.csv`` in the input directory by default, or the path
    supplied with ``--out``. The output contains the parameter set, video name,
    MOTA, HOTA, and per-parameter-set averages for MOTA and HOTA.

Example:
    python summarize_eval_summaries.py /path/to/evaluation_outputs \
        --out /path/to/evaluation_outputs/metrics_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SUMMARY_SUFFIX = "_tracked_detections_eval_summary"
OUTPUT_COLUMNS = [
    "param_set",
    "video",
    "mota",
    "hota",
    "avg_mota_per_param_set",
    "avg_hota_per_param_set",
]


def video_name(summary_path: Path) -> str:
    stem = summary_path.stem
    if stem.endswith(SUMMARY_SUFFIX):
        return stem[: -len(SUMMARY_SUFFIX)]
    if stem == "eval_summary":
        return summary_path.parent.name
    return re.sub(r"_eval_summary$", "", stem)


def read_metrics(summary_path: Path) -> dict[str, float]:
    summary = pd.read_csv(summary_path, header=None, names=["category", "metric", "value"])
    metrics: dict[str, float] = {}
    for metric in ("mota", "HOTA"):
        values = summary.loc[summary["metric"] == metric, "value"]
        if values.empty:
            raise ValueError(f"Missing {metric} in {summary_path}")
        metrics[metric.lower()] = float(values.iloc[0])
    return metrics


def build_summary(input_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary_paths = sorted(input_root.glob("**/*_eval_summary.csv"))
    if not summary_paths:
        raise FileNotFoundError(f"No *_eval_summary.csv files found under {input_root}")

    for summary_path in summary_paths:
        metrics = read_metrics(summary_path)
        rows.append(
            {
                "param_set": summary_path.parent.name,
                "video": video_name(summary_path),
                "mota": metrics["mota"],
                "hota": metrics["hota"],
            }
        )

    result = pd.DataFrame(rows)
    averages = result.groupby("param_set")[["mota", "hota"]].transform("mean")
    result["avg_mota_per_param_set"] = averages["mota"]
    result["avg_hota_per_param_set"] = averages["hota"]
    return result[OUTPUT_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize evaluation summary CSV files.")
    parser.add_argument("input_root", type=Path, help="Root folder containing parameter-set subfolders.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: <input_root>/metrics_summary.csv).",
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    output_path = (args.out or input_root / "metrics_summary.csv").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = build_summary(input_root)
    result.to_csv(output_path, index=False)
    print(f"Summarized {len(result)} evaluation files into {output_path}")


if __name__ == "__main__":
    main()
