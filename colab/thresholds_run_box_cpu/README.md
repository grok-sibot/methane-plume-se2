# fr_neg operating thresholds: full run on a CPU box (25 Nov 2026)

`fr_neg_thresholds_colab.ipynb` run end to end with papermill on an 8 vCPU x86 box, fp32, no GPU, with the released
`model.safetensors` from `sighmon/methane-plume-se2-fullres` (sha256 `1bd2af369bdfaa45…`, full value in `thresholds.json`).
It reused an existing 10 m fine-tune cache and built the 2,000 val + 10,499 test plume-free pair caches (11.3 min in total).
It reproduced the round-2 `fr_neg` numbers (`reproduces_round2: true`).

Thresholds were chosen on 2021 val only (`val_choice.json`, written before any test chip was scored). Test (2024) results: `thresholds.json`,
`thresholds_table.csv`, `thresholds.png`. Config snippet: `config_thresholds.json`. Nothing here was uploaded to Hugging Face.
