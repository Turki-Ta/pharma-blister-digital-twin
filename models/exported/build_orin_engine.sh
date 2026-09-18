#!/usr/bin/env bash
# Build and benchmark the Jetson Orin TensorRT engine ON THE TARGET.
# TensorRT engines are bound to the GPU architecture and TensorRT version: the RTX 5090 engine in
# this folder (sm_120, TensorRT 11.3) will not deserialize on Orin (sm_87, TensorRT 10.x).
# Portable artifact: yolo11n_blister.onnx (opset 17, static [1,3,640,640], FP32 weights, raw
# [1,8,8400] output). JetPack 6.2 = TensorRT 10.3 / CUDA 12.6; JetPack 7.2 = TensorRT 10.16 / CUDA 13.2.
set -euo pipefail
cd "$(dirname "$0")"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX=yolo11n_blister.onnx

sha256sum -c "${ONNX}.sha256"                       # refuse to build from a modified graph
sudo nvpmodel -m 0 2>/dev/null || true               # MAXN (MAXN_SUPER on NX / Nano Super)
sudo jetson_clocks 2>/dev/null || true

# --- FP16 engine (the deployment default) --------------------------------------------------
"$TRTEXEC" --onnx="$ONNX" \
           --saveEngine=yolo11n_orin_fp16.engine \
           --fp16 \
           --memPoolSize=workspace:2048 \
           --timingCacheFile=orin_timing.cache \
           --skipInference

# --- benchmark: 500 warm-up + 500 timed iterations, CUDA graph, p99 -----------------------
"$TRTEXEC" --loadEngine=yolo11n_orin_fp16.engine --useCudaGraph --warmUp=500 --iterations=500 --percentile=99 \
           | tee orin_fp16_benchmark.txt

# --- optional INT8 (calibrated on synthetic validation frames) ---------------------------
# 1. Copy the validation frames listed in data/val.txt to this folder (e.g. ./calib_frames/).
# 2. Produce the calibration cache ON THE JETSON with the entropy calibrator (needs the
#    JetPack TensorRT python bindings and torch):
#       python3 int8_calibrate_orin.py --onnx "$ONNX" --images-dir ./calib_frames --n 500 --cache orin_int8.cache
# 3. Build the mixed INT8/FP16 engine from the SAME ONNX with that cache:
#       "$TRTEXEC" --onnx="$ONNX" --saveEngine=yolo11n_orin_int8.engine --int8 --fp16 \
#                  --calib=orin_int8.cache --memPoolSize=workspace:2048 --timingCacheFile=orin_timing.cache --skipInference
# 4. Accept INT8 only if per-class recall on the frozen validation set is unchanged at the
#    deployed thresholds (expect < 1 ms gain on a nano model; INT8 is worth it for s-models).
# Note: TensorRT 11.x (not on JetPack) removed the --fp16/--int8 builder flags and the calibrator
# API; on TensorRT 11 hosts precision comes from typed ONNX graphs (see export_engine.py).
