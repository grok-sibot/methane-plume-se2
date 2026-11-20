# Methane plume detector for the Spiral Blue SE-2

Onboard Sentinel-2 methane plume segmentation, sized for the Spiral Blue SE-2 (NVIDIA Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3). Demo project, not affiliated with Spiral Blue.

- 20 m CPU model: https://huggingface.co/sighmon/methane-plume-se2 (dashboard: https://huggingface.co/spaces/sighmon/methane-se2-trackio)
- 10 m full-resolution model (released: `fr_neg`, threshold 0.5): https://huggingface.co/sighmon/methane-plume-se2-fullres (dashboard: https://huggingface.co/spaces/sighmon/methane-se2-fullres-trackio)

Layout: `src/` CPU training pipeline, `colab/` full-resolution Colab notebook, `MODEL_CARD.md` metrics and limitations (20 m model), `fullres/` the full-resolution card, config and results. Weights live on Hugging Face, not here.

**Evaluation note:** the official MethaneSET test split is temporal (test = 2024, inherited from MARS-S2L), not grouped by site. 799 of the 1,050 test chips (76 %) come from sites also seen in training. On the 251 unseen-site chips the released full-res model scores IoU 0.435 and chip detection 0.733 (20 m-equivalent grid), against 0.521 and 0.839 on the whole test split. See `test_seen_unseen.csv`, `src/score_seen_unseen.py` and the "Seen vs unseen sites" section of each card. Next step (planned, not run): [`colab/PLAN_synthetic_plumes.md`](colab/PLAN_synthetic_plumes.md).

Training data: MethaneSET (CC BY-NC-SA 4.0), so the models are non-commercial.
