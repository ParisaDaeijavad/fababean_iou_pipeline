from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .mask_utils import mask_area, mask_iou, polygon_to_mask, points_from_json


def _read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required = {"id", "area", "bbox_x0", "bbox_y0", "bbox_w", "bbox_h", "predicted_iou"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")
    return df


def _find_candidate_mask(
    sam_image_dir: Path,
    metadata: pd.DataFrame,
    center_x: float,
    center_y: float,
) -> Optional[Path]:
    candidates = []
    x = int(round(center_x))
    y = int(round(center_y))

    for _, row in metadata.iterrows():
        mask_path = sam_image_dir / f"{int(row['id'])}.png"
        if not mask_path.exists():
            continue
        mask = _read_mask(mask_path)
        if y < 0 or x < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            continue
        if mask[y, x] > 0:
            candidates.append((float(row.get("predicted_iou", 0.0)), int(row["area"]), mask_path))

    if not candidates:
        return None

    # Prefer the highest predicted_iou; if tied, prefer the smaller mask.
    candidates.sort(key=lambda t: (-t[0], t[1]))
    return candidates[0][2]


def enrich_annotations_with_sam_and_iou(
    annotations_xlsx: str | Path,
    sam_output_dir: str | Path,
    output_xlsx: str | Path,
    summary_xlsx: str | Path | None = None,
) -> pd.DataFrame:
    annotations_xlsx = Path(annotations_xlsx)
    sam_output_dir = Path(sam_output_dir)
    output_xlsx = Path(output_xlsx)
    summary_xlsx = Path(summary_xlsx) if summary_xlsx else None

    df = pd.read_excel(annotations_xlsx)

    sam_mask_files = []
    sam_mask_areas = []
    ious = []
    gt_mask_areas = []
    matched_mask_relpaths = []
    status_list = []

    for _, row in df.iterrows():
        image_stem = str(row["image_stem"])
        center_x = float(row["center_x"])
        center_y = float(row["center_y"])
        img_w = int(row["image_width"])
        img_h = int(row["image_height"])

        sam_image_dir = sam_output_dir / image_stem
        metadata_path = sam_image_dir / "metadata.csv"

        if not metadata_path.exists():
            sam_mask_files.append(None)
            sam_mask_areas.append(None)
            ious.append(None)
            gt_mask_areas.append(None)
            matched_mask_relpaths.append(None)
            status_list.append("missing_sam_folder_or_metadata")
            continue

        metadata = _load_metadata(metadata_path)
        matched_mask_path = _find_candidate_mask(sam_image_dir, metadata, center_x, center_y)

        if matched_mask_path is None:
            sam_mask_files.append(None)
            sam_mask_areas.append(None)
            ious.append(None)
            gt_mask_areas.append(None)
            matched_mask_relpaths.append(None)
            status_list.append("no_mask_contains_center")
            continue

        sam_mask = _read_mask(matched_mask_path)
        sam_area = mask_area(sam_mask)
        sam_mask_areas.append(sam_area)
        matched_mask_relpaths.append(str(matched_mask_path.relative_to(sam_output_dir)))

        pts = points_from_json(str(row["polygon_points_json"]))
        gt_mask = polygon_to_mask(pts, img_w, img_h)
        gt_area = mask_area(gt_mask)
        gt_mask_areas.append(gt_area)

        iou = mask_iou(gt_mask, sam_mask)
        ious.append(iou)
        sam_mask_files.append(matched_mask_path.name)
        status_list.append("ok")

    df["sam_mask_file"] = sam_mask_files
    df["sam_mask_relpath"] = matched_mask_relpaths
    df["sam_mask_area_px"] = sam_mask_areas
    df["gt_mask_area_px"] = gt_mask_areas
    df["iou"] = ious
    df["compare_status"] = status_list

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx, index=False)

    if summary_xlsx is not None:
        summary = (
            df[df["compare_status"] == "ok"]
            .groupby("tag", as_index=False)
            .agg(
                num_objects=("tag", "count"),
                mean_iou=("iou", "mean"),
                median_iou=("iou", "median"),
                mean_manual_area=("manual_area_px", "mean"),
                mean_sam_area=("sam_mask_area_px", "mean"),
            )
            .sort_values("tag")
        )
        summary.to_excel(summary_xlsx, index=False)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Match SAM masks to manual annotations and compute IoU."
    )

    parser.add_argument(
        "--annotations-xlsx",
        required=True,
        help="Path to manual annotations Excel file"
    )

    parser.add_argument(
        "--sam-output-dir",
        required=True,
        help="Directory containing SAM output folders"
    )

    parser.add_argument(
        "--output-xlsx",
        default="work/annotations_with_sam_and_iou.xlsx",
        help="Output Excel file with object-level IoU results"
    )

    parser.add_argument(
        "--summary-xlsx",
        default="work/iou_summary_by_tag.xlsx",
        help="Output Excel summary grouped by tag"
    )

    args = parser.parse_args()

    df = enrich_annotations_with_sam_and_iou(
        annotations_xlsx=args.annotations_xlsx,
        sam_output_dir=args.sam_output_dir,
        output_xlsx=args.output_xlsx,
        summary_xlsx=args.summary_xlsx,
    )

    print("\nIoU comparison completed successfully.")
    print(f"Saved detailed results to: {args.output_xlsx}")
    print(f"Saved summary results to: {args.summary_xlsx}")

    print("\nComparison status counts:")
    print(df["compare_status"].value_counts())

    if "iou" in df.columns:
        valid_iou = df["iou"].dropna()

        if len(valid_iou) > 0:
            print("\nIoU statistics:")
            print(valid_iou.describe())