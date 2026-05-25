# Faba-bean manual annotation + SAM IoU pipeline

This project does three things:

1. Randomly selects 30 images from a folder.
2. Lets you manually draw polygons on each image and saves the annotations to Excel.
3. Matches the corresponding SAM/SAM2 masks by image name and point location, then computes IoU.

## Folder naming expected by your data

Your input images should be in a folder such as:

- `Faba-Seed-CC_VfN-N-N.jpg`
- `Faba-Seed-CC_VfN-N-N.png`

The SAM output should look like this:

```text
sam_output/
  Faba-Seed-CC_VfN-1-1/
    0.png
    1.png
    metadata.csv
  Faba-Seed-CC_VfN-1-2/
    0.png
    1.png
    metadata.csv
```

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fababean_iou_env
```

### SAM2.1 repository

Clone the SAM2 repository and download checkpoints separately:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..
```

Then copy `my-sam-code.py` into the cloned `sam2` folder if you want to use the provided segmentation script.

## Workflow

### 1) Sample 30 images and annotate them

Run:

```bash
python -m src.pipeline   --input-dir /path/to/Faba-Seed-CC_images   --work-dir ./work   --sample-size 30   --seed 42
```
Example:
```bash
python -m src.pipeline --input-dir ../../../../../../data/phenomics_images/faba_images --work-dir ./work   --sample-size 30   --seed 42
```

This creates:

- `work/selected_images.csv`
- `work/manual_annotations.xlsx`

### Annotate resume:
```bash
python -m src.polygon_annotator   --selected-csv work/selected_images.csv   --output-xlsx work/manual_annotations.xlsx
```

### 2) Run SAM2 on the same 30 selected images

Copy the sampled images into a separate folder first, then run SAM2 on that folder. For example:

```bash
python -m src.pipeline   --input-dir /path/to/Faba-Seed-CC_images   --work-dir ./work   --sample-size 30   --seed 42   --export-sampled-dir ./work/sampled_images
```

Then run your SAM2 script on `./work/sampled_images` so it processes exactly the same 30 images.

You can use the provided `sam2/my-sam-code.py` inside the cloned `sam2` repo.

### 3) Match SAM masks and compute IoU

After SAM output exists, run:

```bash
python -m src.pipeline   --input-dir /path/to/Faba-Seed-CC_images   --work-dir ./work   --sample-size 30   --seed 42   --sam-output /path/to/sam_output   --compare
```

This creates:

- `work/annotations_with_sam_and_iou.xlsx`
- `work/iou_summary_by_tag.xlsx`

## What is saved in Excel

Each annotation row contains:

- image file name
- tag: `bean`, `coin`, `ruler`, or `color-card`
- polygon points
- center x/y
- manual area in pixels
- SAM mask area in pixels
- IoU

## Keyboard controls during annotation

- Left click: add polygon point
- `f`: finish polygon and assign a tag
- `c`: clear current polygon
- `n`: next image
- `p`: previous image
- `q`: quit and save

## Notes

- The manual annotation is done with OpenCV so you do not need Label Studio.
- The comparison step uses the stored center point and image filename to find the SAM mask for the same object.
- If multiple SAM masks contain the center point, the script chooses the candidate with the highest predicted IoU.
