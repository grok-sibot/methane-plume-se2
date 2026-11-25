# Run the full-resolution methane plume detector on one Sentinel-2 L1C target/reference pair (onnxruntime).

# Input: two co-registered 13-band L1C stacks (DN, scale 10000) at 10 m, 200x200 pixels (2 km x 2 km), e.g. the
# MethaneSET `target` and `bg0` leaves.
#
# Two-stage read-out (default; thresholds from config.json, chosen on 2021 val data only, see the model card):
#   p = sigmoid(plume_logits); pp = 2x2 mean pool of p (20 m grid); scene_score = max(pp)
#   stage 1: plume present if scene_score >= --scene-thr (default config scene_threshold, 0.99658 = 5 % val false alarms;
#            the other operating points are in config operating_points, e.g. --scene-thr 0.99436 for 10 %)
#   stage 2: mask = pp >= --thr (default config mask_threshold, 0.6) on the --grid (default 20m); empty if stage 1 says no.
#            The chip is reported only if the mask has >= 0.4 ha (10 px at 20 m, 40 px at 10 m).
# --no-gate (or --scene-thr 0) turns stage 1 off. The previous single-threshold behaviour is:
#   --no-gate --grid 10m --thr 0.5
# Never use a scene threshold (~0.99) as the mask threshold: the masks shrink to a few pixels.
#
# usage: python infer.py target.tif reference.tif [--onnx methane_se2_fullres_unet_mobilenet_v2.onnx] [--config config.json]
#                        [--scene-thr T | --no-gate] [--thr 0.6] [--grid 20m|10m] [--out plume_mask.npy]
import argparse, json, os, numpy as np, rasterio, onnxruntime as ort
BANDS = [2, 3, 4, 9, 12, 13]  # 1-based in the 13-band L1C stack: B2,B3,B4,B8A,B11,B12
MIN_PX = {'20m': 10, '10m': 40}  # 0.4 ha
def load(path):
    with rasterio.open(path) as d:
        a = d.read(BANDS).astype(np.float32) / 10000.0
    assert a.shape[1:] == (200, 200), 'expects a 200x200 10 m chip; tile larger scenes'
    return a
def load_config(path, onnx):
    for p in ([path] if path else []) + [os.path.join(os.path.dirname(os.path.abspath(onnx)), 'config.json'),
                                         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')]:
        if p and os.path.exists(p):
            with open(p) as f: return json.load(f)
    return {}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('target'); ap.add_argument('reference')
    ap.add_argument('--onnx', default='methane_se2_fullres_unet_mobilenet_v2.onnx')
    ap.add_argument('--config', default=None, help='config.json (default: next to the ONNX, then next to this script)')
    ap.add_argument('--scene-thr', type=float, default=None, help='stage-1 scene threshold on max(2x2-pooled prob); default config scene_threshold; 0 disables')
    ap.add_argument('--no-gate', action='store_true', help='disable stage 1 (same as --scene-thr 0)')
    ap.add_argument('--thr', type=float, default=None, help='mask (pixel) threshold; default config mask_threshold, else 0.5')
    ap.add_argument('--grid', choices=['20m', '10m'], default=None, help='mask grid; default 20m (config mask_grid)')
    ap.add_argument('--out', default='plume_mask.npy'); a = ap.parse_args()
    cfg = load_config(a.config, a.onnx)
    scene_thr = 0.0 if a.no_gate else (a.scene_thr if a.scene_thr is not None else float(cfg.get('scene_threshold', 0.0)))
    thr = a.thr if a.thr is not None else float(cfg.get('mask_threshold', cfg.get('threshold', 0.5)))
    grid = a.grid or ('10m' if str(cfg.get('mask_grid', '20m')).startswith('10m') else '20m')
    t, r = load(a.target), load(a.reference)
    x = np.concatenate([t, r[4:6]])[None].astype(np.float32)  # (1,8,200,200)
    s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    prob = 1 / (1 + np.exp(-s.run(None, {'reflectance': x})[0][0, 0].astype(np.float64)))
    pooled = prob.reshape(100, 2, 100, 2).mean((1, 3))  # 20 m grid
    scene_score = float(pooled.max()); present = scene_score >= scene_thr
    q = pooled if grid == '20m' else prob
    mask = (q >= thr) & present
    flagged = bool(mask.sum() >= MIN_PX[grid]); px_ha = 4 if grid == '20m' else 1
    np.save(a.out, mask)
    gate = f'scene_score {scene_score:.5f} vs scene threshold {scene_thr:.5f} -> plume present: {present}' if scene_thr > 0 else f'scene_score {scene_score:.5f} (scene gate off)'
    print(f'{gate}; mask ({grid}, thr {thr}): {int(mask.sum())} px = {mask.sum() * px_ha / 100:.2f} ha; chip flagged: {flagged}')
if __name__ == '__main__':
    main()
