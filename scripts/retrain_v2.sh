#!/usr/bin/env bash
# Asset v2 retraining chain (drift gate failed on 2026-09-17): synthetic dataset -> hardened
# YOLO11n training -> static ONNX + TensorRT engines -> calibrated thresholds.  Runs detached;
# progress and the final "CHAIN DONE"/"CHAIN FAILED" marker go to reports/retrain_v2.log.
set -o pipefail
cd "$(dirname "$0")/.."
LOG=reports/retrain_v2.log
{
  echo "=== chain start $(date -Iseconds) ==="
  echo "=== [A] SDG v2: 2400 frames, 75/25 split ==="
  uv run python src/simulation/sdg_pipeline.py --frames 2400 --val-fraction 0.25 --out data --seed 20260917 2>&1 \
    | grep -vE "FindAppliedAPIPrimDefinition|Could not find UsdPrimDefinition|OMNI_USD|^#|^$" || { echo "CHAIN FAILED: sdg"; exit 1; }
  echo "=== [B] train yolo11n_v2: sweep 32/64/128, 30 epochs ==="
  uv run python src/training/train.py --name yolo11n_v2 --epochs 30 --sweep-batches 32,64,128 2>&1 \
    | grep -vE "^\s+[0-9]+/[0-9]+ .*it/s" || { echo "CHAIN FAILED: train"; exit 1; }
  echo "=== [C] export ==="
  uv run python src/export/export_engine.py --best models/runs/yolo11n_v2/weights/best.pt 2>&1 | tail -80 || { echo "CHAIN FAILED: export"; exit 1; }
  echo "=== [D] calibrate thresholds ==="
  uv run python src/inference/calibrate_thresholds.py 2>&1 | tail -40 || { echo "CHAIN FAILED: calibrate"; exit 1; }
  echo "=== CHAIN DONE $(date -Iseconds) ==="
} > "$LOG" 2>&1
