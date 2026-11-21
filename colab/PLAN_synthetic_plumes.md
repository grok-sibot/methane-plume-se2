# Plan: synthetic-plume training run (Colab T4), notebook built, not yet run on a GPU

Goal: raise detection on **unseen sites** and measure false alarms on **real plume-free scenes**, without changing the onboard
input contract (`reflectance` 1×8×200×200, 10 m). Motivation: MARS-S2L (arXiv 2511.21777v3) reports that injecting simulated
plumes into plume-free pairs raised recall on unseen sites from 47 % to 78 %. Our released `fr_neg` scores IoU 0.435 and
detection 0.733 on the 251 unseen-site test chips (20 m-equivalent grid), against 0.543 and 0.872 on seen sites
(`test_seen_unseen.csv`).

## What MethaneSET actually ships (checked 20 Nov 2026 against the HF catalogs)

| piece | count | size | notes |
|---|---:|---:|---|
| `methaneset-s2-pretraining` | 56,344 plume-free pairs | 36.9 GB | leaves `target`, `reference`, `dem`; 13 bands, 200×200 at 10 m. Split by year: train 28,076 (2018–20, 2022–23), val 5,352 (2021), test 22,916 (2024). Has `target:sza`, `target:vza`, `target:raa`, `meteo:wind_u/v`, `site:location_name`. |
| `methaneset-bank` | 238,545 enhancements | 5.0 GB | 279 WRF-LES plumes × 855 sun positions (17 SZA rings, 0–71°). XCH4 enhancement in ppb for **3000 kg/h**, 20 m cells, local frame (wind → +x, source at `array:source_col/row`), no noise/background. Scale linearly by Q/3000. Licence CC BY 4.0. |
| `integrated_transmittances.json` | per sensor S2A/S2B/S2C | 55 KB | `transmittance_b11/b12` tables over 8 AMFs (2.0–3.92) × 9 CH4 mixing ratios (≈1,324–8,278 ppb at AMF 2), background 1,800 ppb, `with_ltoa_correction: true`. |
| `methaneset-s2-finetune` `ch4` leaf | 3,605 | (have it) | MBMP enhancement from MARS. Units not stated in the card; one chip's stats span −3,284 to 203,663, so it must be clipped before use. |

The dataset README's "grouped by emitter" claim does not match the metadata (see the cards), so every split below is checked by
site name and emission point, not trusted.

## Recipe

1. **Leak control first.** Train only on pretraining *train* pairs (2018–2023, not 2021), dropping any pair whose
   `site:location_name` is one of the 92 fine-tune unseen-test sites (46 of them occur in pretraining train). Val = 2021.
   Nothing from 2024 is used for training or selection.
2. **Cache** a sample of ~12k pretraining train pairs as the 8 model channels (target B2, B3, B4, B8A, B11, B12 + reference
   B11, B12) in float16: 12,000 × 8 × 200 × 200 × 2 B ≈ 7.7 GB on disk (mmap), read with range requests via tacoreader,
   not by downloading all 36.9 GB. Keep the existing 10 m fine-tune cache.
3. **Inject on the GPU, per batch** (Gorroño / MARS-S2L style, multiplicative transmittance):
   - pick a bank row by nearest `sun:sza` and `sun:raa` to the scene (`target:sza`, wind-relative RAA from `meteo:wind_u/v`);
   - rotate to the scene wind direction, upsample 20 m → 10 m, random sub-pixel offset and source position in the chip;
   - scale to Q drawn log-uniform from ~500 to the bank's per-plume cut limit (`3000 × r / edge:downwind_ppb`, r = scene noise),
     so the LES downwind cut stays hidden;
   - AMF = 1/cos(SZA) + 1/cos(VZA); interpolate the sensor's B11 and B12 transmittance at 1,800 + ΔXCH4 and at 1,800 ppb;
     multiply target B11/B12 by T(1,800 + Δ) / T(1,800). Reference bands are untouched.
   - label = Δ above a detectability floor (e.g. 2× scene noise), and keep Δ as a per-pixel weight.
   - **Sanity gate before any training:** inject Q = 3000 kg/h into ~50 scenes and check the B12/B11 dip and the MBMP
     enhancement match the distribution of real plumes of similar MARS flux rate. Cross-check the table units and the
     `with_ltoa_correction` handling against the MethaneSET / MARS-S2L reference code. If they disagree, stop.
4. **Batches:** 50 % real fine-tune chips, 35 % plume-free pairs with an injected plume, 15 % plume-free pairs untouched.
5. **Loss:** BCE + Dice, with the BCE positive term weighted by clipped enhancement (real chips: the `ch4` leaf, clipped at a
   high percentile and scaled to [0.5, 2]; synthetic: Δ). Faint plume edges are then down-weighted rather than ignored.
6. **MBMP channel:** add Varon-style MBMP, `(c·B12 − B11)/B11` target minus reference with `c` fitted per chip by least squares,
   as a 10th in-graph channel next to the existing log-ratio. Computed in fp32 inside the model, so the ONNX input is unchanged.
   Run with and without it (ablation).
7. **Runs** (init from `fr_neg`, 10 m, AMP, SWIR head fp32): `syn` (steps 3–5), `syn_mbmp` (+6), and `fr_neg` re-scored as
   the baseline under the new protocol. Select on 2021 val only.

## Evaluation (all new protocol, nothing used for training)

- **Unseen-site plumes:** the 251 test chips from 92 unseen sites, plus the 1,050 all-test chips for continuity.
- **Real plume-free scenes for honest FA:** 2024 pretraining test pairs at sites in no training split (8,740 pairs,
  ≈5.7 GB), reported separately for the 3,066 at fine-tune-unseen sites. "Plume-free" means no MARS detection, so FA is
  still slightly pessimistic, but far better than the current bg1-vs-bg0 upper bound.
- **Scene-level PR:** chip score = max of the 2×2-pooled probability, and area above threshold. Report the PR curve and AP for
  plume chips vs real plume-free pairs, plus the operating point by **downlink budget**: the threshold where real FA = 10 %,
  5 % and 1 %, with recall (detection) at each. MARS-S2L reports about 7 % and 1 % FA at its operating points.
- Pixel IoU/F1 at 0.5 as today, so the cards stay comparable.

## Cost estimate (to confirm on the first run)

- Data: ~7.7 GB train cache + ~5.7 GB FA set + bank (5.0 GB, or only the rows near each scene's sun geometry). Fits a Colab
  disk; use High-RAM if available (the existing fine-tune cache alone is 8.9 GB).
- Caching is the slow part: COG decoding on Colab's 2 vCPUs, guess 30–60 min for ~21k pairs. Run once, save to Drive.
- Training: the earlier T4 runs took ~12–13 s per epoch on 2,362 chips. At ~5k chips per epoch (half real) that is ~25–30 s,
  so 20 epochs × 2 runs ≈ 20 min, plus injection overhead. Evaluation on ~10k pairs: a few minutes.
- Total about 1.5–2.5 h of T4 time.

## Go / no-go for releasing a new model

**Go** only if, versus `fr_neg` under the same protocol:
1. at real FA = 5 %, unseen-site detection improves by **≥ 0.10** absolute;
2. unseen-site IoU at 0.5 does not drop by more than 0.01;
3. 2021 val shows the same direction (selection uses val, never test);
4. the ONNX input contract and parameter count are unchanged and CPU latency is within 10 %.

**No-go** (keep `fr_neg`, write up the result) if the injection sanity gate fails, if gains appear only on synthetic data, or if
real-scene FA gets worse at matched detection.

## Update 21 Nov 2026: notebook built (`synthetic_plumes_colab.ipynb`, generator `make_syn_nb.py`, code `synplume.py`)

Smoke-tested end to end on CPU. Not yet run on a GPU, so there are no results yet. What changed from the plan, and what was checked:

- **FA set counts.** 8,740 is right for 2024 pairs at sites in neither fine-tune train nor pretraining train. Only **1,307** of them are at
  unseen-test sites (`real2024_strict_unseen`). The 3,066 figure above is *all* 2024 pairs at unseen-test sites. 46 of those sites also
  occur in pretraining train, but our training drops every unseen-site pair, so the notebook reports all 3,066 separately (`real2024_unseen_all`).
- **Transmittance units.** B11 is a plain transmittance (~0.99). B12 is tabulated on a ~64 scale: its `_bg` column is the constant 64.2987,
  and the 1,800 ppb interpolation matches it at every AMF for every sensor to within 6e-5. The injection uses T(1800 + Δ) / T(1800) at
  the scene AMF, so the scale cancels. At AMF 2.3, +1,000 ppb gives −2.4 % in B12 and −0.39 % in B11.
- **Sun geometry.** `target:raa` equals (90° − SAA) − wind direction to within 6e-14° in every pair (the bank's convention, MethaneSET v1.2.0+).
  All 279 bank plumes share the same 855 sun positions.
- **Label floor.** The floor is not a fixed 2× noise: that would leave almost no label (median 17 px at 2σ against a real median of
  836 px). It is picked by the gate so that synthetic label areas match real ones (`LABEL_FLOOR_SIGMA = 'auto'`).
- **Gate check on the box** (128 pretraining train scenes at 3 t/h against 160 real fine-tune train plumes of 2–4.5 t/h; see
  `syn_checks/`): it passed. Every ratio below is synthetic/real:
  - ln(B12/B11) enhancement 0.69;
  - MBMP enhancement 0.69;
  - label area 1.31 at a 0.25σ floor;
  - 94 % of injected chips show a B12/B11 dip, against 99 % of real chips.
  Scene noise is a median of 348 ppb per 10 m pixel.
- **MBMP run.** It adds 290 parameters (+0.007 %). Criterion 4 is checked as "within 0.01 %".
- **Consistency.** Under the new scoring code, `fr_neg` reproduces its card numbers exactly: IoU 0.521 on all 1,050 test chips, 0.543 on
  seen and 0.435 on unseen.
- **Cost.** About 38 GB of disk and 40–70 min on a standard T4; see the notebook's first cell. The pair caches use range reads,
  measured at 25–58 pairs/s with 2 CPUs.
