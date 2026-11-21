**Time and disk for round 2 on a fresh Colab T4 runtime (estimates).** They are scaled from round 1's measured T4 run on 21 Nov 2026: pair caches 24,499 pairs in 19.3 min (~21 pairs/s, 15.7 GB downloaded); bank 1.4 min; 57 s per training epoch; evaluation 3.9 min for 3 models.

| step | size | T4 estimate |
|---|---|---|
| fine-tune tacozips + 10 m cache + `ch4` leaf (zips deleted afterwards) | 6.2 GB download, 9.5 GB kept | ~5 min |
| plume-free pair caches, HTTP range reads: 24,918 train + 2,000 val + 10,499 test FA | ~24 GB download, 24 GB on disk | ~30 min at round 1's rate (20–45 min, depending on Colab's network) |
| plume bank, ~75k assignments (zips deleted afterwards) | 5 GB zips (transient), ~2–3 GB canvases | ~3 min |
| albedo + sampling weights; `fr_neg` scores 24.9k training pairs (hard negatives) | reads ~20 GB from the caches | ~4 min |
| stage 1 gate | 64 scenes | < 1 min |
| stage 2: `syn2`, `syn2_neg`, `syn2_hn`, 20 epochs each | 295 steps of 16 chips per epoch | ~20 min each, ~60 min total |
| stage 3: 4 models, plus calibration on val and the pool diagnostic | | ~7 min |
| ONNX export, latency, card | | ~2 min |
| **total** | **peak ~41 GB disk (~36 GB at the end)** | **≈ 1 h 50 min (range ~1 h 40 to 2 h 15); budget 2.5 h** |

**Disk.** Every cache build first checks free disk for its size plus a 5 GB reserve. If the plume-free train pool does not fit, the notebook shrinks it, says so and records it in `syn_results.json`. For anything else it stops with a message before downloading. With `DELETE_ZIPS_AFTER_USE = False` the peak is ~47 GB.

**Time.** `TIME_BUDGET_MIN = 150` skips a remaining variant (last first) when its estimated time would not leave `EVAL_RESERVE_MIN` for evaluation. Skips are printed and saved; rerun the training cell later to finish them, since finished runs are kept.

**RAM.** Every cache is memory-mapped. The 24 GB of pair caches do not fit in the page cache of a 12.7 GB runtime, so training reads plume-free pairs from disk (~40 chips/s, a few tens of MB/s). High-RAM helps but is not needed.
