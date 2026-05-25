from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from .image_sampling import list_images, sample_images
from .polygon_annotator import OpenCVPolygonAnnotator
from .sam_match_iou import enrich_annotations_with_sam_and_iou


def write_sample_list(sampled_images, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "image_path"])
        for p in sampled_images:
            writer.writerow([p.name, str(p)])


def main():
    parser = argparse.ArgumentParser(description="Faba-bean manual annotation + SAM IoU pipeline")
    parser.add_argument("--input-dir", required=True, help="Directory with original images")
    parser.add_argument("--work-dir", default="./work", help="Directory to store outputs")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotate", action="store_true", help="Open the manual annotation GUI")
    parser.add_argument("--sam-output", default=None, help="SAM output root directory")
    parser.add_argument("--export-sampled-dir", default=None, help="Optional directory to copy sampled images into for SAM processing")
    parser.add_argument("--compare", action="store_true", help="Match SAM masks and compute IoU")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_dir)
    if args.sample_size:
        sampled = sample_images(images, sample_size=args.sample_size, seed=args.seed)
        sample_csv = work_dir / "selected_images.csv"
        write_sample_list(sampled, sample_csv)
        print(f"Saved selected image list to {sample_csv}")
        if args.export_sampled_dir:
            sampled_dir = Path(args.export_sampled_dir)
            sampled_dir.mkdir(parents=True, exist_ok=True)
            for src in sampled:
                shutil.copy2(src, sampled_dir / src.name)
            print(f"Copied sampled images to {sampled_dir}")


    annotations_xlsx = work_dir / "manual_annotations.xlsx"
    annotated_dir = work_dir / "annotated_images"

    if args.annotate:
        
        annotator = OpenCVPolygonAnnotator(
            sampled,
            annotations_xlsx,
            annotated_dir=annotated_dir,
        )
        annotator.run()
        print(f"Manual annotations saved to {annotations_xlsx}")
        print(f"Annotated images saved to {annotated_dir}")

    if args.compare:
        if not args.sam_output:
            raise SystemExit("--sam-output is required when using --compare")
        output_xlsx = work_dir / "annotations_with_sam_and_iou.xlsx"
        summary_xlsx = work_dir / "iou_summary_by_tag.xlsx"
        df = enrich_annotations_with_sam_and_iou(
            annotations_xlsx=annotations_xlsx,
            sam_output_dir=args.sam_output,
            output_xlsx=output_xlsx,
            summary_xlsx=summary_xlsx,
        )
        print(f"Comparison results saved to {output_xlsx}")
        print(f"Summary saved to {summary_xlsx}")
        print(df["compare_status"].value_counts())

        if "iou" in df.columns:
            valid_iou = df["iou"].dropna()

            if len(valid_iou) > 0:
                print("\nIoU statistics:")
                print(valid_iou.describe())


if __name__ == "__main__":
    main()
