---
license: cc-by-nc-sa-4.0
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags: [methane, sentinel-2, remote-sensing, edge-ai, jetson, tensorrt, onnx]
datasets: [tacofoundation/methaneset]
metrics: [iou, f1, precision, recall]
base_model: sighmon/methane-plume-se2
---

# Methane plume detector for on-orbit edge compute, full resolution (10 m, SE-2 target)

This is the full-resolution version of [sighmon/methane-plume-se2](https://huggingface.co/sighmon/methane-plume-se2). It is a U-Net
(mobilenet_v2 encoder, 4,133,819 parameters) that segments methane plumes in Sentinel-2 L1C chips at 10 m. It needs a
plume-free reference image of the same site. It is sized for satellite edge computers such as the Spiral Blue SE-2 (NVIDIA Jetson Orin NX 16GB,
JetPack 6.2, TensorRT 10.3, 10–25 W).

**Demo / research model by Simon Loffler ([@sighmon](https://github.com/sighmon)). Not affiliated with, endorsed by, or tested by
Spiral Blue.** It has not been run on Jetson hardware.

**Released weights: `fr_neg` at threshold 0.5** (fine-tuned with plume-free pairs, chosen for the lowest val false-alarm rate).
The section below explains the change from the two earlier releases (`fr_ft_s0` at 0.7, then `fr_base` at 0.7).

## Input contract

- `reflectance`: float32 `1×8×200×200`, L1C TOA reflectance (DN / 10000) at 10 m (2 km × 2 km). Channels: target B2, B3, B4, B8A,
  B11, B12, then reference B11, B12. The 20 m bands are used as MethaneSET provides them, resampled to 10 m.
- `plume_logits`: float32 `1×1×200×200`. Apply a sigmoid, then threshold at **0.5** (in `config.json`). A chip
  counts as flagged at ≥40 pixels (0.4 ha).
- A SWIR ratio channel `log(B12/B11)_target − log(B12/B11)_reference` (mean-centred) is computed inside the graph in fp32.
  The input is reflect-padded to 224 inside the graph and cropped back to 200.
- The channels and scaling are the same as the 20 m model. Only the spatial size differs. The ONNX file name and tensor names have not changed.

## Which weights are released, and why they changed

The released weights are **`fr_neg` at threshold 0.5**. This is the third release in this repo:

1. `fr_ft_s0` at 0.7: the first Colab run, chosen by highest val IoU.
2. `fr_base` at 0.7: a second Colab run of the same notebook (retrained from scratch), again chosen by highest val IoU.
3. `fr_neg` at 0.5 (this release): the `fr_neg` checkpoint from that second run, chosen by a new rule, lowest val false-alarm rate.

The checkpoint was not retrained for this release. It was exported and re-evaluated on CPU from the second run's saved `final.pt`. The
re-evaluation reproduced the second run's recorded `fr_neg` test numbers (20 m grid, threshold 0.5) to within 0.001, and confirmed on val that
`fr_neg` has the lowest false-alarm rate. Every metric table below comes from that second run, so all models in it share one training run.

- **Original rule (releases 1 and 2):** the notebook picked the model with the highest **val IoU** at its val-picked threshold. That chose `fr_base` (val IoU
  0.507 at 0.7). The val split has only 193 chips, so IoU differences of about 0.01 between models are within noise.
- **What went wrong:** on test, `fr_base` at 0.7 scored IoU 0.497 and FA 0.883 (20 m grid). That
  is *below* the 20 m CPU model (0.509 IoU, FA 0.825). The val IoU ranking did not carry over to test.
- **New rule:** the lowest **val false-alarm rate**, because false alarms are what cost downlink in onboard alerting. `fr_neg` has val FA
  0.585, against 0.782 or more for every other model. That is a much bigger margin than the IoU spread. By val IoU,
  `fr_neg` ranked *lower* (0.495 at 0.65).
- **Threshold:** 0.5, the default, instead of `fr_neg`'s val-F1 threshold of 0.65. On val the two are close (F1 0.660 at 0.5 vs
  0.663 at 0.65; FA 0.591 vs 0.585).
- **Disclosure:** the selection rule and the threshold were changed **after the test numbers had been seen** (from both Colab runs). So the released test numbers are not a
  clean held-out estimate. Treat them as optimistic by an unknown, probably small, amount. Val alone also supports `fr_neg` on false alarms.
- **Released numbers** (test, 20 m-equivalent, threshold 0.5): IoU 0.521, F1 0.685, P 0.745,
  R 0.634, det 0.839, FA 0.550. For comparison, the 20 m CPU greedy soup has IoU 0.509
  and FA 0.825. The trade-off is chip detection: 0.839 here, against 0.899 for the CPU greedy soup.

## Training

The data is [MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset) `methaneset-s2-finetune` v1.4.0. It has 3,605 chips (200×200 at
10 m) in the official MARS-S2L splits of 2,362 train, 193 val and 1,050 test. The split is **temporal** (train 2018–2020 and 2022–2023,
val 2021, test 2024), not site-grouped: 76 % of test chips come from sites seen in training (see "Seen vs unseen sites" below). It is licensed CC BY-NC-SA 4.0. Both Colab runs trained on a Tesla T4 with AMP=True, with the SWIR head in fp32.

- `fr_base`: ImageNet init, 15 epochs, lr 0.001, rot90 and flips, seed 0. This is the CPU `base` recipe at 10 m.
- **`fr_neg` (released)**: a fine-tune of `fr_base` for 8 epochs at lr 0.0003, seed 3. 30 % of chips are replaced by plume-free pairs (bg1 vs bg0).
- `fr_long`: 40 epochs with rot90, flips, ±10 % brightness jitter, 30 % negatives and extra references (bg1–bg3).
- `fr_ft_s*`: 4 fine-tunes of `fr_long` with different seeds, 8 epochs each, averaged as a uniform soup and as a greedy soup on val.
  The greedy soup kept ['fr_ft_s0', 'fr_ft_s3']. Weight averaging did not beat `fr_neg` here.

## Results (test split, measured)

Like-for-like with the CPU model, on the **20 m-equivalent** grid. The 10 m probabilities are 2×2-mean-pooled and scored
against the CPU model's 20 m labels (pooled fraction ≥0.5); det and FA use ≥10 px (0.4 ha).

| model | IoU | F1 | P | R | det | FA |
|---|---|---|---|---|---|---|
| CPU greedy soup (20 m repo release) | 0.509 | 0.675 | 0.710 | 0.643 | 0.899 | 0.825 |
| CPU ft_neg (20 m) | 0.483 | 0.651 | 0.745 | 0.578 | 0.814 | 0.562 |
| fr_base @0.5 | 0.511 | 0.677 | 0.742 | 0.622 | 0.934 | 0.902 |
| **fr_neg @0.5 (released)** | 0.521 | 0.685 | 0.745 | 0.634 | 0.839 | 0.550 |
| fr_long @0.5 | 0.502 | 0.668 | 0.751 | 0.602 | 0.856 | 0.746 |
| fr_ft_s0 @0.5 | 0.506 | 0.672 | 0.758 | 0.603 | 0.876 | 0.733 |
| fr_ft_s1 @0.5 | 0.501 | 0.668 | 0.737 | 0.610 | 0.882 | 0.771 |
| fr_ft_s2 @0.5 | 0.517 | 0.682 | 0.724 | 0.645 | 0.923 | 0.854 |
| fr_ft_s3 @0.5 | 0.514 | 0.679 | 0.732 | 0.633 | 0.894 | 0.796 |
| uniform_soup @0.5 | 0.522 | 0.686 | 0.733 | 0.644 | 0.906 | 0.787 |
| greedy_soup @0.5 | 0.517 | 0.682 | 0.744 | 0.629 | 0.889 | 0.750 |
| fr_base @0.7 (previous release) | 0.497 | 0.664 | 0.772 | 0.582 | 0.930 | 0.883 |

Native 10 m, at each model's val-picked threshold, plus the released model at 0.5:

| model | IoU | F1 | P | R | det | FA |
|---|---|---|---|---|---|---|
| fr_base @0.7 | 0.500 | 0.666 | 0.754 | 0.597 | 0.930 | 0.884 |
| fr_neg @0.65 | 0.513 | 0.678 | 0.749 | 0.619 | 0.839 | 0.546 |
| fr_long @0.75 | 0.482 | 0.650 | 0.774 | 0.560 | 0.852 | 0.738 |
| fr_ft_s0 @0.7 | 0.490 | 0.658 | 0.770 | 0.574 | 0.874 | 0.727 |
| fr_ft_s1 @0.8 | 0.473 | 0.642 | 0.775 | 0.548 | 0.870 | 0.757 |
| fr_ft_s2 @0.8 | 0.494 | 0.661 | 0.768 | 0.580 | 0.917 | 0.837 |
| fr_ft_s3 @0.8 | 0.486 | 0.654 | 0.769 | 0.570 | 0.888 | 0.782 |
| uniform_soup @0.75 | 0.505 | 0.671 | 0.758 | 0.601 | 0.901 | 0.782 |
| greedy_soup @0.75 | 0.498 | 0.664 | 0.767 | 0.586 | 0.887 | 0.742 |
| fr_neg @0.5 (released, 10 m) | 0.519 | 0.683 | 0.728 | 0.643 | 0.839 | 0.550 |

Val split, 10 m, sorted by val false-alarm rate (the new selection rule). The val-picked threshold is the best val F1:

| model | val-picked thr | val IoU | val F1 | **val FA** | val FA @0.5 |
|---|---|---|---|---|---|
| fr_neg ← released | 0.65 | 0.495 | 0.663 | 0.585 | 0.591 |
| fr_ft_s3 | 0.8 | 0.498 | 0.665 | 0.782 | 0.813 |
| greedy_soup | 0.75 | 0.504 | 0.670 | 0.803 | 0.808 |
| fr_long | 0.75 | 0.501 | 0.668 | 0.808 | 0.808 |
| fr_ft_s0 | 0.7 | 0.498 | 0.665 | 0.813 | 0.824 |
| uniform_soup | 0.75 | 0.499 | 0.666 | 0.850 | 0.855 |
| fr_ft_s1 | 0.8 | 0.491 | 0.659 | 0.860 | 0.865 |
| fr_ft_s2 | 0.8 | 0.495 | 0.662 | 0.902 | 0.907 |
| fr_base ← previous release | 0.7 | 0.507 | 0.673 | 0.917 | 0.927 |

Full comparison (`comparison_test.csv`: test split, both grids, at 0.5 and at each model's val-picked threshold; deltas vs the CPU greedy soup):

| model | grid | thr | iou | f1 | precision | recall | det_rate | false_alarm | Δiou_vs_cpu_soup | ΔFA_vs_cpu_soup |
|---|---|---|---|---|---|---|---|---|---|---|
| CPU greedy soup (released) | 20m (CPU) | 0.5 | 0.509 | 0.675 | 0.71 | 0.643 | 0.899 | 0.825 | 0.0 | 0.0 |
| CPU ft_neg | 20m (CPU) | 0.5 | 0.483 | 0.651 | 0.745 | 0.578 | 0.814 | 0.562 | -0.026 | -0.263 |
| fr_base | 20m | 0.5 | 0.511 | 0.677 | 0.742 | 0.622 | 0.934 | 0.902 | 0.002 | 0.077 |
| fr_base | 10m | 0.5 | 0.509 | 0.675 | 0.725 | 0.631 | 0.934 | 0.901 | 0.0 | 0.076 |
| fr_base | 20m | 0.7 | 0.497 | 0.664 | 0.772 | 0.582 | 0.93 | 0.883 | -0.012 | 0.058 |
| fr_base | 10m | 0.7 | 0.5 | 0.666 | 0.754 | 0.597 | 0.93 | 0.884 | -0.009 | 0.059 |
| **fr_neg** (released) | **20m** | **0.5** | **0.521** | **0.685** | **0.745** | **0.634** | **0.839** | **0.55** | **0.012** | **-0.275** |
| **fr_neg** (released) | **10m** | **0.5** | **0.519** | **0.683** | **0.728** | **0.643** | **0.839** | **0.55** | **0.01** | **-0.275** |
| fr_neg | 20m | 0.65 | 0.512 | 0.677 | 0.767 | 0.606 | 0.839 | 0.545 | 0.003 | -0.28 |
| fr_neg | 10m | 0.65 | 0.513 | 0.678 | 0.749 | 0.619 | 0.839 | 0.546 | 0.004 | -0.279 |
| fr_long | 20m | 0.5 | 0.502 | 0.668 | 0.751 | 0.602 | 0.856 | 0.746 | -0.007 | -0.079 |
| fr_long | 10m | 0.5 | 0.5 | 0.667 | 0.735 | 0.611 | 0.857 | 0.745 | -0.009 | -0.08 |
| fr_long | 20m | 0.75 | 0.477 | 0.646 | 0.793 | 0.545 | 0.851 | 0.737 | -0.032 | -0.088 |
| fr_long | 10m | 0.75 | 0.482 | 0.65 | 0.774 | 0.56 | 0.852 | 0.738 | -0.027 | -0.087 |
| fr_ft_s0 | 20m | 0.5 | 0.506 | 0.672 | 0.758 | 0.603 | 0.876 | 0.733 | -0.003 | -0.092 |
| fr_ft_s0 | 10m | 0.5 | 0.504 | 0.67 | 0.742 | 0.611 | 0.878 | 0.733 | -0.005 | -0.092 |
| fr_ft_s0 | 20m | 0.7 | 0.487 | 0.655 | 0.789 | 0.56 | 0.873 | 0.725 | -0.022 | -0.1 |
| fr_ft_s0 | 10m | 0.7 | 0.49 | 0.658 | 0.77 | 0.574 | 0.874 | 0.727 | -0.019 | -0.098 |
| fr_ft_s1 | 20m | 0.5 | 0.501 | 0.668 | 0.737 | 0.61 | 0.882 | 0.771 | -0.008 | -0.054 |
| fr_ft_s1 | 10m | 0.5 | 0.499 | 0.666 | 0.721 | 0.618 | 0.882 | 0.772 | -0.01 | -0.053 |
| fr_ft_s1 | 20m | 0.8 | 0.467 | 0.637 | 0.795 | 0.531 | 0.87 | 0.756 | -0.042 | -0.069 |
| fr_ft_s1 | 10m | 0.8 | 0.473 | 0.642 | 0.775 | 0.548 | 0.87 | 0.757 | -0.036 | -0.068 |
| fr_ft_s2 | 20m | 0.5 | 0.517 | 0.682 | 0.724 | 0.645 | 0.923 | 0.854 | 0.008 | 0.029 |
| fr_ft_s2 | 10m | 0.5 | 0.514 | 0.679 | 0.707 | 0.653 | 0.923 | 0.855 | 0.005 | 0.03 |
| fr_ft_s2 | 20m | 0.8 | 0.489 | 0.657 | 0.787 | 0.563 | 0.917 | 0.835 | -0.02 | 0.01 |
| fr_ft_s2 | 10m | 0.8 | 0.494 | 0.661 | 0.768 | 0.58 | 0.917 | 0.837 | -0.015 | 0.012 |
| fr_ft_s3 | 20m | 0.5 | 0.514 | 0.679 | 0.732 | 0.633 | 0.894 | 0.796 | 0.005 | -0.029 |
| fr_ft_s3 | 10m | 0.5 | 0.511 | 0.677 | 0.716 | 0.642 | 0.895 | 0.796 | 0.002 | -0.029 |
| fr_ft_s3 | 20m | 0.8 | 0.482 | 0.65 | 0.789 | 0.553 | 0.888 | 0.781 | -0.027 | -0.044 |
| fr_ft_s3 | 10m | 0.8 | 0.486 | 0.654 | 0.769 | 0.57 | 0.888 | 0.782 | -0.023 | -0.043 |
| uniform_soup | 20m | 0.5 | 0.522 | 0.686 | 0.733 | 0.644 | 0.906 | 0.787 | 0.013 | -0.038 |
| uniform_soup | 10m | 0.5 | 0.519 | 0.683 | 0.717 | 0.653 | 0.907 | 0.787 | 0.01 | -0.038 |
| uniform_soup | 20m | 0.75 | 0.502 | 0.668 | 0.778 | 0.586 | 0.902 | 0.781 | -0.007 | -0.044 |
| uniform_soup | 10m | 0.75 | 0.505 | 0.671 | 0.758 | 0.601 | 0.901 | 0.782 | -0.004 | -0.043 |
| greedy_soup | 20m | 0.5 | 0.517 | 0.682 | 0.744 | 0.629 | 0.889 | 0.75 | 0.008 | -0.075 |
| greedy_soup | 10m | 0.5 | 0.515 | 0.68 | 0.728 | 0.637 | 0.89 | 0.75 | 0.006 | -0.075 |
| greedy_soup | 20m | 0.75 | 0.495 | 0.662 | 0.787 | 0.571 | 0.886 | 0.741 | -0.014 | -0.084 |
| greedy_soup | 10m | 0.75 | 0.498 | 0.664 | 0.767 | 0.586 | 0.887 | 0.742 | -0.011 | -0.083 |

- `det` is the share of plume chips with ≥0.4 ha predicted that overlaps the label. `FA` is the share of plume-free pairs (bg1 vs bg0 of the same site)
  with ≥0.4 ha predicted. The references come from known super-emitter sites and may hold real, unlabelled plumes, so FA is an
  upper bound on true false alarms.
- These are single runs with no confidence intervals. Treat differences of about 1 IoU point as noise.

## Seen vs unseen sites (test split)

The official MethaneSET `s2-finetune` split is inherited from MARS-S2L and is **temporal, not site-grouped**. Train covers
2018–2020 and 2022–2023 (2,362 chips), validation is 2021 (193) and test is 2024 (1,050). **799 of the 1,050 test chips (76 %)
come from sites that also appear in training.** The MethaneSET dataset card describes the split as "grouped by emitter", but the
per-chip metadata (`site:location_name`, `emission:lat`/`emission:lon`) does not match that. Earlier versions of this card described
the split as grouped by site or emitter. That was wrong.

**Rule:** a test chip is *seen* if at least one training chip has the same `site:location_name` **and** an emission point within
1 km (haversine). In this data every name match also has an identical emission point, and no other test chip has a training
emission point within 1 km, so name-only and distance-only rules give the same 799 / 251 split. The 251 unseen chips come from 92
sites, all oil and gas, led by Turkmenistan (62 chips), the USA (46) and Syria (37). Validation is also mostly seen sites: 192 of
193 val chips share a site name with training.

Released weights, test split, threshold 0.5 (measured; det/FA use ≥0.4 ha):

| model | grid | test subset | chips | IoU | F1 | P | R | det | FA |
|---|---|---|---|---|---|---|---|---|---|
| fr_neg (10 m, released) | 10 m native | all | 1050 | 0.519 | 0.683 | 0.729 | 0.643 | 0.839 | 0.550 |
| fr_neg (10 m, released) | 20 m-equivalent | all | 1050 | 0.521 | 0.685 | 0.745 | 0.634 | 0.839 | 0.550 |
| fr_neg (10 m, released) | 10 m native | seen sites | 799 | 0.541 | 0.702 | 0.739 | 0.668 | 0.872 | 0.581 |
| fr_neg (10 m, released) | 20 m-equivalent | seen sites | 799 | 0.543 | 0.704 | 0.755 | 0.659 | 0.872 | 0.579 |
| fr_neg (10 m, released) | 10 m native | **unseen sites** | 251 | 0.434 | 0.605 | 0.683 | 0.543 | 0.733 | 0.450 |
| fr_neg (10 m, released) | 20 m-equivalent | **unseen sites** | 251 | 0.435 | 0.607 | 0.699 | 0.536 | 0.733 | 0.454 |
| CPU greedy soup (20 m, released) | 20 m native | all | 1050 | 0.509 | 0.675 | 0.710 | 0.643 | 0.899 | 0.825 |
| CPU greedy soup (20 m, released) | 20 m native | seen sites | 799 | 0.526 | 0.689 | 0.724 | 0.658 | 0.926 | 0.835 |
| CPU greedy soup (20 m, released) | 20 m native | **unseen sites** | 251 | 0.447 | 0.618 | 0.654 | 0.586 | 0.813 | 0.793 |

- On unseen sites `fr_neg` loses about 0.11 IoU and 0.14 chip detection compared with seen sites. The CPU soup loses about 0.08 IoU
  and 0.11 detection. The all-test numbers are therefore optimistic for new sites. Expect something nearer the unseen rows.
- FA is lower on the unseen subset (0.45 against 0.58 for `fr_neg`). That reflects a different mix of scenes, not better
  generalisation, and FA is still an upper bound measured on references from known emitter sites.
- 251 chips from 92 sites is a small sample: single runs, no confidence intervals. The subsets were defined after release and were
  not used to pick weights or thresholds.
- Per-row numbers: `test_seen_unseen.csv`. Script: `src/score_seen_unseen.py` in the
  [GitHub repo](https://github.com/grok-sibot/methane-plume-se2).

## Size and speed (released weights)

- 4,133,819 parameters; 1.10 GMACs per 200×200 chip (fvcore); ONNX 16.5 MB (opset 17, static).
- ONNX vs PyTorch: max abs logit diff 7.6e-04, mask agreement 100.00 % on 32 test chips.
- Latency (batch 1, measured on a shared 8 vCPU x86_64 machine with onnxruntime; the earlier `fr_base` export measured 31.7 ms on 1 thread on a Colab CPU and 10.9 ms in PyTorch FP16 on a Tesla T4, same architecture): ort_cpu_ms_1t: 15.9 ms, ort_cpu_ms_8t: 5.9 ms. CPU: x86_64. Jetson Orin NX FP16/INT8 was **not measured**; see `build_trt.sh`.
- The 10 m model does about 3× the compute of the 20 m model per 2 km chip.

## Files

`model.safetensors` and `methane_se2_fullres_unet_mobilenet_v2.onnx` (fr_neg), `config.json` (input contract and threshold 0.5), `infer.py` (default threshold 0.5),
`build_trt.sh` (untested on Jetson), `calibration_train256.npy` (INT8 calibration inputs from the train split, float16; they do not depend on the model),
`results.json` (full threshold sweeps for every model, plus the `selection` record), `comparison_test.csv`, `training_history/`, `training_curves.png`,
`threshold_sweep_val.png` (first run), `example_predictions.png` (fr_neg at 0.5), and `src/methane_fullres.py`.

## Limitations

- 76 % of test chips come from sites seen in training (the official split is by year). On the 251 unseen-site test chips,
  IoU and detection are clearly lower; see "Seen vs unseen sites".
- The chips are centred on known emitters. Onboard use over full swaths needs tiling and far more negative data.
- It needs a co-registered plume-free reference of the same site.
- The data covers onshore oil, gas and coal sites only, mostly Turkmenistan and Algeria.
- The labels come from MARS-S2L analysts and are noisy. Emission rates are not estimated.
- INT8 can erase the ~1 % SWIR ratio signal, so validate any engine against the ONNX.
- This is a demonstration, not a validated monitoring product. It is non-commercial (CC BY-NC-SA 4.0).

## Citation (data)

```bibtex
@article{contreras2026methaneset,
  title   = {MethaneSET: Unified Multi-Sensor AI-Ready Datasets for Satellite-Based Methane Plume Detection},
  author  = {Contreras, Julio and Giner, Carlos and Aybar, Cesar and Mateo-Garc{\'i}a, Gonzalo and Gorro{\~n}o, Javier and Montero, David and Mahecha, Miguel D. and Guanter, Luis and G{\'o}mez-Chova, Luis},
  journal = {Scientific Data},
  year    = {2026}
}
```
