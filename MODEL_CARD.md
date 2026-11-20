---
license: cc-by-nc-sa-4.0
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags: [methane, sentinel-2, remote-sensing, edge-ai, jetson, tensorrt, onnx, model-soup]
datasets: [tacofoundation/methaneset]
metrics: [iou, f1, precision, recall]
---

# Methane plume detector for on-orbit edge compute (SE-2 target)

A 4.1 M-parameter U-Net (MobileNetV2 encoder) that segments methane plumes in Sentinel-2
L1C imagery. It takes a target scene and a plume-free reference of the same site. It is sized for
satellite edge computers such as the Spiral Blue SE-2 (NVIDIA Jetson Orin NX 16GB, JetPack 6.2,
TensorRT 10.3, 10–25 W).

**Demo / research model by Simon Loffler ([@sighmon](https://github.com/sighmon)). Not affiliated with,
endorsed by, or tested by Spiral Blue.** It has not been run on Jetson hardware yet.

## What it does

- **Input** `reflectance`: float32 `1×8×100×100`. These are top-of-atmosphere reflectance values
  (L1C DN / 10000) at 20 m, covering a 2 km × 2 km chip.
  - Channels: target B2, B3, B4, B8A, B11, B12, then reference B11, B12.
- **Output** `plume_logits`: `1×1×100×100`. Apply a sigmoid, then threshold at 0.5.
- An MBMP-style SWIR ratio channel, `log(B12/B11)_target − log(B12/B11)_reference` (mean-centred),
  is computed inside the graph. The engine therefore needs no preprocessing beyond reflectance
  scaling. The input is reflect-padded to 128 inside the graph and the output is cropped back to 100.

## Training data

[MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset), `methaneset-s2-finetune` v1.4.0:

- 3,605 Sentinel-2 plume chips (200×200 at 10 m, 13 bands) from MARS-S2L / UNEP IMEO.
- Each chip has a binary plume mask and the MARS reference scene (`bg0`).
- Official emitter-grouped splits: 2,362 train, 193 validation, 1,050 test.
- Chips were 2×2 mean-pooled to 20 m, the native resolution of the SWIR bands. A pixel counts as
  plume where at least 50 % of the 10 m label pixels are plume.
- Plume pixels make up 3.3 % of all pixels.
- **Licence:** the labelled data is CC BY-NC-SA 4.0, so these weights are released under
  **CC BY-NC-SA 4.0**. Non-commercial use only.

## Training

Everything ran on CPU only: 8 vCPU Intel Xeon, no GPU, PyTorch 2.15, segmentation-models-pytorch 0.6.0.
The encoder starts from ImageNet weights. Loss is BCE (pos_weight 3) plus Dice, with AdamW and
cosine learning-rate decay. Augmentation is random rot90 and flips. Runs were tracked with Trackio.

1. `base`: 15 epochs at learning rate 1e-3, about 15 min.
2. Four fine-tunes from `base`, each 8 epochs at learning rate 3e-4. These are the soup ingredients
   and share one initialisation:
   - `ft_s0`: seed 0
   - `ft_s1`: seed 1
   - `ft_aug2`: adds ±10 % brightness jitter
   - `ft_neg`: 30 % of chips replaced with plume-free pairs (a `bg1` reference used as the target
     against `bg0`, with an empty mask)
3. Model soups ([Wortsman et al., 2022](https://arxiv.org/abs/2203.05482)): weight averaging.
   - Uniform: all 4 fine-tunes.
   - Greedy: fine-tunes added in order of val IoU, kept only if val IoU improved. It kept `ft_s1`,
     `ft_neg` and `ft_aug2`.

**The released weights are the greedy soup**, the best checkpoint by val IoU.

## Results

All numbers are measured, at threshold 0.5 and 20 m pixels.

- IoU, F1, precision and recall are pixel-level, pooled over the whole split.
- `det`: the share of plume chips where the prediction has ≥10 plume pixels (0.4 ha) and overlaps
  the label.
- `FA`: the share of plume-free pairs (`bg1` vs `bg0` of the same site) with ≥10 predicted plume
  pixels. These references come from known super-emitter sites and may contain real, unlabelled
  plumes, so FA is an upper bound on true false alarms.

| run | val IoU | val F1 | val P | val R | val det | val FA | test IoU | test F1 | test P | test R | test det | test FA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.493 | 0.660 | 0.601 | 0.732 | 0.969 | 0.969 | 0.494 | 0.662 | 0.689 | 0.636 | 0.911 | 0.923 |
| ft_s0 | 0.482 | 0.651 | 0.601 | 0.709 | 0.964 | 0.938 | 0.493 | 0.660 | 0.699 | 0.626 | 0.911 | 0.906 |
| ft_s1 | 0.495 | 0.662 | 0.605 | 0.730 | 0.974 | 0.964 | 0.502 | 0.669 | 0.683 | 0.655 | 0.924 | 0.945 |
| ft_aug2 | 0.485 | 0.653 | 0.593 | 0.728 | 0.969 | 0.979 | 0.507 | 0.672 | 0.681 | 0.665 | 0.926 | 0.943 |
| ft_neg | 0.493 | 0.661 | 0.637 | 0.686 | 0.907 | **0.570** | 0.483 | 0.651 | 0.745 | 0.578 | 0.814 | **0.562** |
| uniform soup | 0.496 | 0.663 | 0.608 | 0.729 | 0.953 | 0.881 | **0.510** | 0.675 | 0.704 | 0.649 | 0.907 | 0.865 |
| **greedy soup (released)** | **0.501** | 0.667 | 0.616 | 0.728 | 0.953 | 0.834 | 0.509 | 0.675 | 0.710 | 0.643 | 0.899 | 0.825 |

How to read the table:

- The soups beat the best single fine-tune (`ft_s1`) by a small margin.
  - Val IoU: +0.001 for the uniform soup and +0.006 for the greedy soup. The greedy soup was
    chosen on val, so its val gain is optimistic.
  - Test IoU: +0.008 for both soups.
- These are single runs with no confidence intervals. Treat a gain of about 1 IoU point as
  suggestive, not proven.
- Training on plume-free pairs (`ft_neg`) is the change that matters for an onboard alerting use case.
  It cuts false alarms from about 0.94 to 0.56. The cost is lower recall: chip detection falls
  from 0.92 to 0.81 on test.

## Size and speed

| | |
|---|---|
| parameters | 4,133,819 |
| compute | 0.36 GMACs per 100×100 chip (fvcore) |
| ONNX (opset 17, static shape, FP32) | 16.5 MB |
| ONNX vs PyTorch | max abs logit diff 1.0e-4; 100 % mask agreement on 32 test chips |
| CPU latency, onnxruntime, batch 1 | 5.1 ms (1 thread) / 2.8 ms (8 threads), on a shared 8 vCPU Xeon |
| CPU latency, PyTorch eager | 11.0 ms (1 thread) / 10.5 ms (8 threads) |
| Jetson Orin NX FP16 / INT8 | **not measured**; see `build_trt.sh` |

## Files

- `model.safetensors`: weights (`MethaneNet` state dict; see `src/common.py`)
- `methane_se2_unet_mbv2.onnx`: deployable graph
- `infer.py`: runs ONNX on a 200×200 10 m L1C target/reference GeoTIFF pair
- `build_trt.sh`: trtexec and Polygraphy commands for FP16 and INT8 engines on JetPack 6.2. **Untested on Jetson.**
- `calibration_train256.npy`: 256 train-split input chips (float16) for INT8 calibration
- `results.json`, `training_history/`: raw metrics; `training_curves.png`, `example_predictions.png`
- `src/`: full training pipeline (prep, train, soup, export, figures)

## Limitations

- **Chips are centred on known emitters.** The model has only seen 2 km chips around MARS-detected
  sources. Onboard use over full swaths would need tiling, plus far more negative data. Expect many
  more false alarms on arbitrary scenes.
- It needs a plume-free reference image of the same site, co-registered with the target. On orbit,
  that means storing reference tiles onboard or uplinking them.
- Training and evaluation cover onshore oil, gas and coal sites only. Regions are mostly
  Turkmenistan and Algeria. There are no offshore scenes.
- Labels come from MARS-S2L analysts and include label noise. Small and weak plumes are often missed.
- 20 m pooling and the ≥50 % rule shrink thin plumes.
- Emission rates are not estimated.
- INT8 can erase the ~1 % SWIR ratio signal. Validate any engine against the ONNX before trusting it.
- This is not a validated monitoring product. It is a demonstration.

## Intended use

A research and demo model showing that methane plume screening fits a satellite edge-compute
budget. It is not for regulatory, safety or commercial use, which the CC BY-NC-SA 4.0 licence on
the data also rules out.

## Citation (data)

```bibtex
@article{contreras2026methaneset,
  title   = {MethaneSET: Unified Multi-Sensor AI-Ready Datasets for Satellite-Based Methane Plume Detection},
  author  = {Contreras, Julio and Giner, Carlos and Aybar, Cesar and Mateo-Garc{\'i}a, Gonzalo and Gorro{\~n}o, Javier and Montero, David and Mahecha, Miguel D. and Guanter, Luis and G{\'o}mez-Chova, Luis},
  journal = {Scientific Data},
  year    = {2026}
}
```
