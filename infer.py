"""Run the methane plume detector on one Sentinel-2 L1C target/reference pair with onnxruntime.

Input: two co-registered 13-band L1C TOA stacks (DN, scale 10000) at 10 m, e.g. MethaneSET
`target` and `bg0` leaves, or your own chips. The model was trained on 20 m chips of 100x100 pixels
(2 km x 2 km). This script 2x2-mean-pools a 200x200 10 m chip to that grid.

usage: python infer.py target.tif reference.tif [--onnx methane_se2_unet_mbv2.onnx] [--thr 0.5]
"""
import argparse, numpy as np, rasterio, onnxruntime as ort

BANDS = [2, 3, 4, 9, 12, 13]  # 1-based in the 13-band L1C stack: B2,B3,B4,B8A,B11,B12

def load(path):
    with rasterio.open(path) as d:
        a = d.read(BANDS).astype(np.float32) / 10000.0
    c, h, w = a.shape
    assert (h, w) == (200, 200), 'expects a 200x200 10 m chip; adapt tiling for other sizes'
    return a.reshape(c, 100, 2, 100, 2).mean((2, 4))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('target'); ap.add_argument('reference')
    ap.add_argument('--onnx', default='methane_se2_unet_mbv2.onnx'); ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--out', default='plume_mask.npy'); a = ap.parse_args()
    t, r = load(a.target), load(a.reference)
    x = np.concatenate([t, r[4:6]])[None].astype(np.float32)  # (1,8,100,100)
    s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    logits = s.run(None, {'reflectance': x})[0][0, 0]
    prob = 1 / (1 + np.exp(-logits)); mask = prob >= a.thr
    np.save(a.out, mask)
    print(f'plume pixels: {int(mask.sum())} (20 m px, {mask.sum() * 400 / 1e4:.1f} ha); '
          f'chip flagged: {bool(mask.sum() >= 10)}; max prob {prob.max():.3f}')

if __name__ == '__main__':
    main()
