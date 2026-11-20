# Run the full-resolution methane plume detector on one Sentinel-2 L1C target/reference pair (onnxruntime).

# Input: two co-registered 13-band L1C stacks (DN, scale 10000) at 10 m, 200x200 pixels (2 km x 2 km), e.g. the
# MethaneSET `target` and `bg0` leaves. No pooling: the model runs at the native 10 m grid.
# usage: python infer.py target.tif reference.tif [--onnx methane_se2_fullres_unet_mobilenet_v2.onnx] [--thr 0.5]
import argparse, numpy as np, rasterio, onnxruntime as ort
BANDS = [2, 3, 4, 9, 12, 13]  # 1-based in the 13-band L1C stack: B2,B3,B4,B8A,B11,B12
def load(path):
    with rasterio.open(path) as d:
        a = d.read(BANDS).astype(np.float32) / 10000.0
    assert a.shape[1:] == (200, 200), 'expects a 200x200 10 m chip; tile larger scenes'
    return a
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('target'); ap.add_argument('reference')
    ap.add_argument('--onnx', default='methane_se2_fullres_unet_mobilenet_v2.onnx'); ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--out', default='plume_mask.npy'); a = ap.parse_args()
    t, r = load(a.target), load(a.reference)
    x = np.concatenate([t, r[4:6]])[None].astype(np.float32)  # (1,8,200,200)
    s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    prob = 1 / (1 + np.exp(-s.run(None, {'reflectance': x})[0][0, 0])); mask = prob >= a.thr
    np.save(a.out, mask)
    print(f'plume pixels: {int(mask.sum())} (10 m px, {mask.sum() * 100 / 1e4:.2f} ha); '
          f'chip flagged: {bool(mask.sum() >= 40)}; max prob {prob.max():.3f}')
if __name__ == '__main__':
    main()
