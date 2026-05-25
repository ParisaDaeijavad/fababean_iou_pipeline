#!/usr/bin/env python
import os
import sys
import shutil
from math import ceil

import cv2
import torch
import pandas as pd
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
CHECKPOINT = "sam2/checkpoints/sam2.1_hiera_large.pt"
EXPECTED_WIDTH = 4000
EXPECTED_HEIGHT = 6000


def get_slurm_task_info():
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        task_id = int(os.environ["SLURM_PROCID"])
        num_tasks = int(os.environ["SLURM_NTASKS"])
        print(f"[SLURM srun mode] Task {task_id + 1}/{num_tasks}", flush=True)
        return task_id, num_tasks

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", os.environ.get("SLURM_ARRAY_TASK_MAX", task_id + 1)))
        print(f"[SLURM array mode] Task {task_id + 1}/{num_tasks}", flush=True)
        return task_id, num_tasks

    print("[Sequential mode] No SLURM environment detected", flush=True)
    return 0, 1


def distribute_files(files, task_id, num_tasks):
    if num_tasks <= 1:
        return files

    total_files = len(files)
    chunk_size = ceil(total_files / num_tasks)
    start_idx = task_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_files)
    assigned_files = files[start_idx:end_idx]
    print(f"Task {task_id}: Processing files {start_idx} to {end_idx-1} ({len(assigned_files)} files)", flush=True)
    return assigned_files


def _build_sam_model(device="cpu"):
    torch.cuda.empty_cache()
    sam_model = build_sam2(MODEL_CFG, CHECKPOINT)
    sam_model.to(device)

    mask_generator = SAM2AutomaticMaskGenerator(sam_model)
    mask_generator188 = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=64,
        pred_iou_thresh=0.6,
        min_mask_region_area=500,
    )
    return mask_generator, mask_generator188


def _process_single_image(image_path, output_dir, mask_generator, mask_generator188):
    image_name = os.path.basename(image_path)

    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️  Warning: Could not read '{image_name}'. Skipping...", flush=True)
        return (False, image_name, "Unreadable image")

    height, width, _ = image.shape
    if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
        error_msg = f"Invalid dimensions (found {width}x{height}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})"
        print(f"❌ Error: Image '{image_name}' - {error_msg}. Skipping...", flush=True)
        return (False, image_name, error_msg)

    image_base_name = os.path.splitext(image_name)[0]
    image_output_dir = os.path.join(output_dir, image_base_name)
    metadata_file = os.path.join(image_output_dir, "metadata.csv")

    if os.path.exists(metadata_file):
        print(f"Skipping {image_name} (already processed - metadata.csv exists)", flush=True)
        return (True, image_name, "Already processed")

    if os.path.exists(image_output_dir):
        print(f"Removing incomplete output directory for {image_name}", flush=True)
        shutil.rmtree(image_output_dir)

    os.makedirs(image_output_dir, exist_ok=True)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        masks = mask_generator188.generate(image_rgb) if "188" in image_name else mask_generator.generate(image_rgb)

    metadata = []
    metadata_header = [
        "id", "area", "bbox_x0", "bbox_y0", "bbox_w", "bbox_h",
        "point_input_x", "point_input_y", "predicted_iou", "stability_score",
        "crop_box_x0", "crop_box_y0", "crop_box_w", "crop_box_h"
    ]

    for i, mask in enumerate(masks):
        mask_image = (mask["segmentation"] * 255).astype("uint8")
        output_path = os.path.join(image_output_dir, f"{i}.png")
        Image.fromarray(mask_image).save(output_path)

        mask_metadata = [
            i,
            mask["area"],
            *mask["bbox"],
            *mask["point_coords"][0],
            mask["predicted_iou"],
            mask["stability_score"],
            *mask["crop_box"],
        ]
        metadata.append(mask_metadata)

    df_metadata = pd.DataFrame(metadata, columns=metadata_header)
    metadata_path = os.path.join(image_output_dir, "metadata.csv")
    df_metadata.to_csv(metadata_path, index=False)

    print(f"✅ Masks and metadata saved for {image_name} ({len(masks)} masks).", flush=True)
    torch.cuda.empty_cache()
    return (True, image_name, None)


def _sam_segment(files, output_dir, device="cpu"):
    if not files:
        print("No files to process in this chunk", flush=True)
        return []

    print(f"Processing {len(files)} files", flush=True)
    print(f"First file: {files[0]}", flush=True)
    if len(files) > 1:
        print(f"Last file: {files[-1]}", flush=True)

    print(f"Building SAM2 model on device: {device}", flush=True)
    mask_generator, mask_generator188 = _build_sam_model(device)
    print("SAM2 model built successfully", flush=True)

    results = []
    invalid_images = []

    for idx, image_path in enumerate(files):
        print(f"Processing image {idx+1}/{len(files)}: {os.path.basename(image_path)}", flush=True)
        result = _process_single_image(image_path, output_dir, mask_generator, mask_generator188)
        results.append(result)
        if not result[0] and result[2] != "Already processed":
            invalid_images.append((result[1], result[2]))

    print(f"Chunk processing complete! Processed {len(files)} images.", flush=True)
    if invalid_images:
        print("The following images were skipped due to errors:", flush=True)
        for name, reason in invalid_images:
            print(f"   - {name}: {reason}", flush=True)

    return results


def main(input_dir, output_dir, device="cpu"):
    task_id, num_tasks = get_slurm_task_info()

    print("========================================", flush=True)
    print(f"SAM2 Segmentation - Task {task_id}/{num_tasks}", flush=True)
    print("========================================", flush=True)
    print(f"Input directory: {input_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png")
    all_image_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(valid_extensions)
    ])

    if not all_image_files:
        raise ValueError(f"The input directory '{input_dir}' does not contain any valid image files.")

    print(f"Found {len(all_image_files)} total images", flush=True)
    my_files = distribute_files(all_image_files, task_id, num_tasks)

    if not my_files:
        print(f"Task {task_id}: No files assigned, exiting.", flush=True)
        return

    files_to_process = []
    for image_path in my_files:
        image_base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(output_dir, image_base_name)
        metadata_file = os.path.join(image_output_dir, "metadata.csv")
        if os.path.exists(metadata_file):
            print(f"Skipping {os.path.basename(image_path)} - already processed", flush=True)
        else:
            files_to_process.append(image_path)

    if not files_to_process:
        print(f"Task {task_id}: All assigned images already processed.", flush=True)
        return

    print(f"Task {task_id}: {len(files_to_process)} images to process (after filtering)", flush=True)
    _sam_segment(files_to_process, output_dir=output_dir, device=device)
    print(f"Task {task_id}: SAM2 segmentation completed!", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python my-sam-code.py <input_directory> <output_directory> [--device <device>]", flush=True)
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    device = "cpu"
    if "--device" in sys.argv:
        idx = sys.argv.index("--device")
        if idx + 1 < len(sys.argv):
            device = sys.argv[idx + 1]
    main(input_dir, output_dir, device=device)
