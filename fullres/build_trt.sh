#!/usr/bin/env bash
# Build TensorRT engines for the Spiral Blue SE-2 (Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3).
# UNTESTED ON JETSON: written for JetPack 6.2 but never run on an Orin. Run ON the Jetson; engines are not portable.
# Full-resolution model (fr_neg, threshold 0.5): static input reflectance 1x8x200x200 (10 m, 2 km chip).
set -euo pipefail
ONNX=${ONNX:-methane_se2_fullres_unet_mobilenet_v2.onnx}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

# Optional: pick a power mode first, e.g. sudo nvpmodel -m <id>  (see /etc/nvpmodel.conf for the 10/15/25 W ids)

# 1) FP16 engine (the safe default)
$TRTEXEC --onnx="$ONNX" --saveEngine=methane_fullres_fp16.engine --fp16 \
  --shapes=reflectance:1x8x200x200 --memPoolSize=workspace:1024

# 2) INT8 engine with post-training calibration (pip install polygraphy), using calibration_train256.npy (train split, float16)
cat > calib_loader.py <<'PY'
import numpy as np
def load_data():
    a = np.load("calibration_train256.npy").astype(np.float32)
    for i in range(len(a)):
        yield {"reflectance": a[i:i+1]}
PY
polygraphy convert "$ONNX" --int8 --fp16 \
  --data-loader-script calib_loader.py \
  --calibration-cache methane_fullres_int8.cache \
  -o methane_fullres_int8.engine

# Calibration notes:
# - The first ops compute log(B12/B11) target-vs-reference ratios in-graph; that signal is ~1 % of reflectance and
#   INT8 can erase it. If INT8 accuracy drops, keep the head in FP16:
#     --precisionConstraints=obey --layerPrecisions="/Log*:fp16,/Div*:fp16,/Sub*:fp16,/norm/*:fp16"
#   (inspect layer names with: polygraphy inspect model methane_fullres_int8.engine --show layers)
# - Re-check IoU/F1 of each engine against the ONNX on held-out chips before trusting it:
#     polygraphy run "$ONNX" --trt --onnxrt --int8 --calibration-cache methane_fullres_int8.cache

# 3) Latency at batch 1
$TRTEXEC --loadEngine=methane_fullres_fp16.engine --iterations=500 --warmUp=500
$TRTEXEC --loadEngine=methane_fullres_int8.engine --iterations=500 --warmUp=500
