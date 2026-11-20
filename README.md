# Methane plume detector for the Spiral Blue SE-2

Onboard Sentinel-2 methane plume segmentation, sized for the Spiral Blue SE-2 (NVIDIA Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3). Demo project, not affiliated with Spiral Blue.

- 20 m CPU model: https://huggingface.co/sighmon/methane-plume-se2 (dashboard: https://huggingface.co/spaces/sighmon/methane-se2-trackio)
- 10 m full-resolution model (released: `fr_neg`, threshold 0.5): https://huggingface.co/sighmon/methane-plume-se2-fullres (dashboard: https://huggingface.co/spaces/sighmon/methane-se2-fullres-trackio)

Layout: `src/` CPU training pipeline, `colab/` full-resolution Colab notebook, `MODEL_CARD.md` metrics and limitations (20 m model), `fullres/` the full-resolution card, config and results. Weights live on Hugging Face, not here.

Training data: MethaneSET (CC BY-NC-SA 4.0), so the models are non-commercial.
