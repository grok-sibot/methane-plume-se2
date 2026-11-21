**Time, disk and RAM (measured 21 Nov 2026 where marked; the T4 column is an estimate for a standard Colab T4 with 2 vCPUs and 12.7 GB RAM, the runtime the last full-res run used).**

| step | size | measured | T4 estimate |
|---|---|---|---|
| fine-tune tacozips + 10 m cache (skipped if an earlier run's cache is in this runtime or on Drive) | 6.2 GB download, 8.9 GB cache | last T4 run: 84 s download, 135 s cache | ~4 min |
| `ch4` leaf cache (loss weights) | 3,605 leaves, 0.6 GB | box: 18 s from local zips | ~1 min |
| plume-free pair caches, HTTP range reads | 24,499 pairs, ~650 KB each: ~15.9 GB download, 15.7 GB on disk | box, 2 CPUs pinned, 128 threads: 25–58 pairs/s | 7–25 min (depends on Colab's network) |
| plume bank | 23,118 rows: 5.0 GB zip download, 1.4 GB canvas cache | | 3–6 min |
| stage 1 gate | 64 scenes vs 160 real plumes | box CPU: 6 s | < 1 min |
| stage 2: `syn` + `syn_mbmp`, 12 epochs each | 295 steps of 16 chips per epoch, plus 2,193 val chips scored | last T4 run: 12 s per 148-step epoch | 15–20 min |
| stage 3: 3 models × (1,050 test plumes + 10,499 real plume-free pairs + 2,193 val) | ~9 GB read per model | box CPU: 3.6 min for the plume part of all 3 models | 5–10 min |
| ONNX export, latency, card | | | 1–2 min |
| **total** | **~38 GB disk** | | **≈ 40–70 min; budget 1.5 h** |

RAM: every cache is memory-mapped, so 12.7 GB is enough (the CPU test kernel peaked at 5.2 GB RSS, mostly reclaimable mapped pages). High-RAM only makes the reads faster because more of the caches stay in the page cache. With `PRETRAIN_N = 6000` and `VAL_FA_N = 1000` the caches shrink by ~4.5 GB and the run by ~5 min.
