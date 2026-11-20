#!/usr/bin/env bash
# Build TensorRT engines for the Spiral Blue SE-2 (Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3).
# UNTESTED ON JETSON: written for JetPack 6.2 but not run on an Orin. These commands were not executed
# on the build machine (no GPU). Run them ON the Jetson; engines are not portable between devices.
set -euo pipefail
ONNX=${ONNX:-methane_se2_unet_mbv2.onnx}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}

# Optional: pick a power mode first, e.g. sudo nvpmodel -m <id>  (check /etc/nvpmodel.conf for 10/15/25 W ids)

# 1) FP16 engine (the safe default)
$TRTEXEC --onnx="$ONNX" --saveEngine=methane_fp16.engine --fp16 \
  --shapes=reflectance:1x8x100x100 --memPoolSize=workspace:1024 --verbose=false

# 2) INT8 engine with post-training calibration.
#    trtexec cannot generate a calibration cache from data by itself. Make one with Polygraphy
#    (pip install polygraphy) using the 256 TRAIN-split chips shipped as calibration_train256.npy:
python3 - <<'PY'
import numpy as np
open('calib_loader.py','w').write('''import numpy as np
def load_data():
    a = np.load("calibration_train256.npy").astype(np.float32)
    for i in range(len(a)):
        yield {"reflectance": a[i:i+1]}
''')
PY
polygraphy convert "$ONNX" --int8 --fp16 \
  --data-loader-script calib_loader.py \
  --calibration-cache methane_int8.cache \
  -o methane_int8.engine

#    Or with trtexec once the cache exists (INT8 with FP16 fallback):
$TRTEXEC --onnx="$ONNX" --saveEngine=methane_int8_trtexec.engine --int8 --fp16 \
  --calib=methane_int8.cache --shapes=reflectance:1x8x100x100

# Calibration notes:
# - The first ops compute log(B12/B11) ratios of target vs reference in-graph. That signal is ~1% of
#   reflectance, so INT8 on those ops can erase it. If INT8 accuracy drops, keep the head in FP16:
#     --precisionConstraints=obey --layerPrecisions="/Log*:fp16,/Div*:fp16,/Sub*:fp16,/norm/*:fp16"
#   (inspect layer names with: polygraphy inspect model methane_int8.engine --show layers)
# - Always re-check IoU/F1 of the engine against the ONNX on held-out chips (polygraphy run
#   --trt --onnxrt --load-inputs ...) before trusting INT8.

# 3) Latency at batch 1
$TRTEXEC --loadEngine=methane_fp16.engine --iterations=500 --warmUp=500
$TRTEXEC --loadEngine=methane_int8.engine --iterations=500 --warmUp=500
