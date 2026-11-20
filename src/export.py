"""Export chosen weights to safetensors + ONNX (opset 17, static 1x8x100x100), verify with onnxruntime,
measure params / FLOPs / size / CPU latency, save INT8 calibration chips."""
import sys, os, json, time, numpy as np, torch, onnx, onnxruntime as ort
from safetensors.torch import save_file
from common import *
ck, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
X, meta = load()
m = MethaneNet(weights=None); m.load_state_dict(torch.load(ck)); m.eval()
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f'{out}/model.safetensors')
x, _ = make_batch(X, np.where(meta.split == 'test')[0][:1])
onnx_p = f'{out}/methane_se2_unet_mbv2.onnx'
torch.onnx.export(m, x, onnx_p, opset_version=17, input_names=['reflectance'], output_names=['plume_logits'],
                  dynamo=False, do_constant_folding=True)
onnx.checker.check_model(onnx_p)
info = {'params': sum(p.numel() for p in m.parameters()), 'onnx_bytes': os.path.getsize(onnx_p)}
# numerical check on 32 test chips
te = np.where(meta.split == 'test')[0][:32]; xb, _ = make_batch(X, te)
so = ort.SessionOptions(); s = ort.InferenceSession(onnx_p, so, providers=['CPUExecutionProvider'])
d = []; agree = []
with torch.no_grad():
    for i in range(len(xb)):
        pt = m(xb[i:i + 1]).numpy(); ox = s.run(None, {'reflectance': xb[i:i + 1].numpy()})[0]
        d.append(np.abs(pt - ox).max()); agree.append(((pt > 0) == (ox > 0)).mean())
info['onnx_max_abs_diff_logits'] = float(max(d)); info['onnx_mask_agreement'] = float(np.mean(agree))
# FLOPs (fvcore counts one fused multiply-add as one flop -> report as MACs)
try:
    from fvcore.nn import FlopCountAnalysis
    info['gmacs'] = FlopCountAnalysis(m, x).unsupported_ops_warnings(False).uncalled_modules_warnings(False).total() / 1e9
except Exception as e: info['gmacs_error'] = str(e)
def bench(fn, n=50):
    for _ in range(5): fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000
xi = x.numpy()
for th in [1, 8]:
    torch.set_num_threads(th)
    with torch.no_grad(): info[f'torch_cpu_ms_{th}t'] = bench(lambda: m(x))
    so = ort.SessionOptions(); so.intra_op_num_threads = th
    s = ort.InferenceSession(onnx_p, so, providers=['CPUExecutionProvider'])
    info[f'ort_cpu_ms_{th}t'] = bench(lambda: s.run(None, {'reflectance': xi}))
# 256 calibration chips from TRAIN split for INT8 (float32, N x 8 x 100 x 100)
cal = np.random.default_rng(0).choice(np.where(meta.split == 'train')[0], 256, replace=False)
np.save(f'{out}/calibration_train256.npy', make_batch(X, cal)[0].numpy())
info['cpu'] = os.popen("lscpu | grep 'Model name'").read().strip()
json.dump(info, open(f'{out}/export_info.json', 'w'), indent=1); print(json.dumps(info, indent=1))
