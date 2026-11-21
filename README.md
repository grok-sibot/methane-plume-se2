# Methane plume detector for the Spiral Blue SE-2

Onboard Sentinel-2 methane plume segmentation, sized for the Spiral Blue SE-2 (NVIDIA Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3). Demo project, not affiliated with Spiral Blue.

- 20 m CPU model: https://huggingface.co/sighmon/methane-plume-se2 (dashboard: https://huggingface.co/spaces/sighmon/methane-se2-trackio)
- 10 m full-resolution model (released: `fr_neg`, threshold 0.5): https://huggingface.co/sighmon/methane-plume-se2-fullres (dashboard: https://huggingface.co/spaces/sighmon/methane-se2-fullres-trackio)

Layout: `src/` CPU training pipeline, `colab/` full-resolution Colab notebook, `MODEL_CARD.md` metrics and limitations (20 m model), `fullres/` the full-resolution card, config and results. Weights live on Hugging Face, not here.

**Evaluation note:** the official MethaneSET test split is temporal (test = 2024, inherited from MARS-S2L), not grouped by site. 799 of the 1,050 test chips (76 %) come from sites also seen in training. On the 251 unseen-site chips the released full-res model scores IoU 0.435 and chip detection 0.733 (20 m-equivalent grid), against 0.521 and 0.839 on the whole test split. See `test_seen_unseen.csv`, `src/score_seen_unseen.py` and the "Seen vs unseen sites" section of each card. Next step: synthetic-plume training, planned in [`colab/PLAN_synthetic_plumes.md`](colab/PLAN_synthetic_plumes.md). The notebook is [`colab/synthetic_plumes_colab.ipynb`](colab/synthetic_plumes_colab.ipynb) (generator `colab/make_syn_nb.py`, code `colab/synplume.py`). Round 1 ran on a Colab T4 on 21 Nov 2026 and was a **NO-GO**. The unseen-site hit rate at 5 % real false alarms rose from 0.402 to 0.438 against the +0.10 required, although AP rose from 0.523 to 0.614. Round 2 is [`colab/synthetic_plumes_r2_colab.ipynb`](colab/synthetic_plumes_r2_colab.ipynb) (generator `colab/make_syn_r2_nb.py`). It adds more data and epochs, extra negatives, hard negatives, unseen-like sampling weights and a val-calibrated threshold, under the same go / no-go rule plus a calibrated-IoU check. It has been smoke-tested on CPU (pandas 3.1 and 2.2) and under the fake-GPU device checker (`colab/tests/`) but not yet run on a GPU.

Training data: MethaneSET (CC BY-NC-SA 4.0), so the models are non-commercial.
