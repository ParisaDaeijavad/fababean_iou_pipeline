#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${1:?Usage: ./run_full_pipeline.sh /path/to/images /path/to/sam_output}"
SAM_OUTPUT="${2:?Usage: ./run_full_pipeline.sh /path/to/images /path/to/sam_output}"
WORK_DIR="${3:-./work}"

python -m src.pipeline \
  --input-dir "$INPUT_DIR" \
  --work-dir "$WORK_DIR" \
  --sample-size 30 \
  --seed 42 \
  --export-sampled-dir "$WORK_DIR/sampled_images" \
  --annotate

echo "Run SAM2 on: $WORK_DIR/sampled_images"
echo "After SAM output is ready, run:"
echo "python -m src.pipeline --input-dir "$INPUT_DIR" --work-dir "$WORK_DIR" --sample-size 30 --seed 42 --sam-output "$SAM_OUTPUT" --compare"
