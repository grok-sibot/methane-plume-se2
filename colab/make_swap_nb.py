import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s, tags=None):
    c = nbf.v4.new_code_cell(s.strip('\n'))
    if tags: c.metadata['tags'] = tags
    C.append(c)

md(r'''
# Swap the released full-res methane model to `fr_neg` at threshold 0.5

This notebook changes the release of [`sighmon/methane-plume-se2-fullres`](https://huggingface.co/sighmon/methane-plume-se2-fullres)
from `fr_ft_s0` (picked by val IoU, threshold 0.7) to `fr_neg` (lowest val false-alarm rate) at threshold 0.5. It also rewrites the model card
and pushes everything as **one commit**.

**It has two paths and picks one automatically:**

1. **Weights present.** If this runtime still has `/content/methane-fullres/runs/fr_neg/final.pt` from the full run, it uses that file. The
   recorded numbers from the first run stay in the card. The weights are re-evaluated as a check.
2. **Weights gone** (a new runtime). It re-downloads MethaneSET, rebuilds the 10 m cache, and retrains `fr_base` then `fr_neg` with the same settings and
   seeds as the original notebook. It then re-evaluates `fr_neg`. The card uses the **new** numbers and says clearly that the weights were retrained and
   can differ slightly from the first run, because GPU training is not bit-for-bit deterministic.

**How to run:** pick a GPU runtime (T4 is fine), keep the `HF_TOKEN` Colab secret, then *Runtime → Run all*. `UPLOAD = True` is the default because the
swap is approved. Set it to False to build the files without publishing.

| path | T4 (estimate) |
|---|---|
| weights present (export, card, upload) | ≈5–10 min |
| retrain (download, cache, `fr_base` 15 ep, `fr_neg` 8 ep, eval, export, upload) | ≈15–25 min |

These estimates come from the measured first run on a T4: 84 s download, 135 s cache build, 3.5 min for `fr_base` and 1.7 min for `fr_neg`.
''')

md('## 1. Config')
code(r'''
SMOKE_TEST = False          # True: a few chips, 1 epoch, never uploads. Proves the notebook executes; numbers mean nothing.
UPLOAD = True               # approved: push the swap to HF_REPO as one commit
UPLOAD_IF_NOT_LOWEST = False  # safety: if a retrained fr_neg no longer has the lowest val FA, do not upload unless this is True
FORCE_RETRAIN = False       # True: retrain fr_neg even if runs/fr_neg/final.pt exists (the old one is moved to runs/fr_neg_first_run)
HF_REPO = 'sighmon/methane-plume-se2-fullres'
RELEASE, RELEASE_THR = 'fr_neg', 0.5
PREV_RELEASE, PREV_THR = 'fr_ft_s0', 0.7
COMMIT_MSG = 'Release fr_neg (lowest val false-alarm rate) at threshold 0.5'
WORK = None                 # None = /content/methane-fullres on Colab (same as the full run), ./methane-fullres elsewhere
# ---- the original notebook's settings for (a) fr_base and (b) fr_neg; do not change if you want a faithful retrain ----
ENCODER, BATCH_SIZE, AMP = 'mobilenet_v2', 16, True
BASE_EPOCHS, NEG_FT_EPOCHS, LR, FT_LR, NEG_P = 15, 8, 1e-3, 3e-4, 0.3
THRESHOLD_METRIC = 'f1'
CAL_N = 256
WORKERS = None
TRACKIO_PROJECT = 'methane-se2-fullres-swap'   # local only; nothing is synced
''', tags=['parameters'])

md('## 2. Setup')
code(r'''
import os, sys, subprocess, json, time, shutil, glob, platform
T0 = time.time()
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.6.0', 'trackio',
                    'rasterio', 'onnx', 'onnxruntime', 'onnxscript', 'fvcore', 'safetensors', 'psutil',
                    'huggingface_hub>=0.34', 'pyarrow'], check=True)
import numpy as np, pandas as pd, torch
if SMOKE_TEST:
    BASE_EPOCHS = NEG_FT_EPOCHS = 1; CAL_N = 8; BATCH_SIZE = 4
WORK = WORK or ('/content/methane-fullres' if IN_COLAB else os.path.abspath('methane-fullres'))
DATA, RUNS, OUT = f'{WORK}/data', f'{WORK}/runs', f'{WORK}/hf_swap'
shutil.rmtree(OUT, ignore_errors=True)
for d in (DATA, RUNS, OUT): os.makedirs(d, exist_ok=True)
os.environ['TRACKIO_DIR'] = f'{WORK}/trackio'; os.makedirs(os.environ['TRACKIO_DIR'], exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = WORKERS if WORKERS is not None else min(4, os.cpu_count() or 2)
GPU = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'none (CPU)'
print(f'torch {torch.__version__} | device {DEV} | GPU {GPU} | workers {WORKERS} | smoke {SMOKE_TEST} | work {WORK}')
''')

md('## 3. Hugging Face login\nThis uses the Colab secret `HF_TOKEN`. The token is never printed or stored.')
code(r'''
HF_OK = False
try:
    from huggingface_hub import login, whoami
    tok = None
    try:
        from google.colab import userdata; tok = userdata.get('HF_TOKEN')
    except Exception: pass
    if tok:
        login(token=tok, add_to_git_credential=False); HF_OK = True
    elif UPLOAD and not SMOKE_TEST:
        from huggingface_hub import notebook_login; notebook_login(); HF_OK = True
    print('logged in as', whoami()['name']) if HF_OK else print('no HF token (fine for a smoke test)')
    del tok
except Exception as e:
    print('HF login skipped:', type(e).__name__, e)
''')

md('## 4. Fetch the first run\'s results and training code from the repo\nThis reads `results.json`, `comparison_test.csv` and `src/methane_fullres.py`. The local copy of the module from the full run is used if it is there. Either way it is the same file.')
code(r'''
from huggingface_hub import hf_hub_download, HfApi
REPO_FILES = set(HfApi().list_repo_files(HF_REPO))
ORIG_DIR = f'{WORK}/first_run'; os.makedirs(ORIG_DIR, exist_ok=True)
for f in ['results.json', 'comparison_test.csv', 'src/methane_fullres.py']:
    hf_hub_download(HF_REPO, f, local_dir=ORIG_DIR)
if not os.path.exists(f'{WORK}/methane_fullres.py'):
    shutil.copy(f'{ORIG_DIR}/src/methane_fullres.py', f'{WORK}/methane_fullres.py')
import importlib
sys.path.insert(0, WORK); import methane_fullres as mf; importlib.reload(mf)
R0 = json.load(open(f'{ORIG_DIR}/results.json')); ORIG = R0['results']; CMP0 = pd.read_csv(f'{ORIG_DIR}/comparison_test.csv')
assert RELEASE in ORIG and PREV_RELEASE in ORIG, 'unexpected results.json'
print('first run: selected', R0['selected'], '@', R0['selected_threshold'], '| GPU', R0.get('gpu'), '| models', list(ORIG))
''')

md('## 5. Which path?')
code(r'''
CK = f'{RUNS}/{RELEASE}/final.pt'
if FORCE_RETRAIN and os.path.exists(CK):
    shutil.rmtree(f'{RUNS}/{RELEASE}_first_run', ignore_errors=True); shutil.move(f'{RUNS}/{RELEASE}', f'{RUNS}/{RELEASE}_first_run')
RETRAINED = not os.path.exists(CK)
BASE_RETRAINED = RETRAINED and not os.path.exists(f'{RUNS}/fr_base/final.pt')
print('weights present: using', CK) if not RETRAINED else print(
    f'{CK} not found: will retrain', 'fr_base and fr_neg' if BASE_RETRAINED else 'fr_neg (reusing the existing fr_base/final.pt)')
''')

md('## 6. Data (MethaneSET S2 finetune v1.4.0 and the 10 m cache)\nThe cache is reused when it is already there. The export and the re-evaluation need it on both paths.')
code(r'''
REPO, SUB = 'tacofoundation/methaneset', 'methaneset-s2-finetune'
CACHE = f'{DATA}/cache10m.npy'
meta_files = {f: hf_hub_download(REPO, f'{SUB}/.tacocat/{f}', repo_type='dataset', local_dir=DATA)
              for f in ['level0.parquet', 'level1.parquet', 'COLLECTION.json']}
l0 = pd.read_parquet(meta_files['level0.parquet']).reset_index(drop=True)
l1 = pd.read_parquet(meta_files['level1.parquet'])
SMALL = ['Oman', 'Poland', 'Russian_Federation', 'Venezuela', 'Argentina', 'China', 'Jordan']
if SMOKE_TEST:
    l0 = l0[l0['internal:source_file'].isin([f'{SUB}_{c}.tacozip' for c in SMALL])].reset_index(drop=True)
ZIPDIR = f'{DATA}/{SUB}'
t = time.time()
if not os.path.exists(CACHE):
    for z in sorted(l0['internal:source_file'].unique()):
        hf_hub_download(REPO, f'{SUB}/{z}', repo_type='dataset', local_dir=DATA)
    print(f'downloaded tacozips in {time.time() - t:.0f}s')
meta = l0[[c for c in mf.META_COLS if c in l0.columns]].copy()
if SMOKE_TEST:  # same round-robin smoke splits as the original notebook
    meta['split'] = [['train', 'train', 'validation', 'test'][i % 4] for i in range(len(meta))]
if not os.path.exists(CACHE):
    mf.build_cache(l0, l1, ZIPDIR, CACHE, workers=os.cpu_count())
X = mf.load_cache(CACHE)
assert X.shape[1:] == (31, 200, 200) and len(X) == len(meta)
tr = np.where(meta.split == 'train')[0]; va = np.where(meta.split == 'validation')[0]; te = np.where(meta.split == 'test')[0]
print('train/val/test', len(tr), len(va), len(te))
''')

md('## 7. Retrain `fr_base` then `fr_neg` (only when the weights are gone)\nThese calls are the same as cells (a) and (b) of the original notebook: same recipe and seeds (`fr_base` seed 0, `fr_neg` seed 3). An existing `fr_base/final.pt` is reused.')
code(r'''
RUN_TIMES = {}
COMMON = dict(encoder=ENCODER, bs=BATCH_SIZE, amp=AMP, workers=WORKERS, project=TRACKIO_PROJECT,
              skip_done=True, preview_every=1 if SMOKE_TEST else 3)
if RETRAINED:
    t = time.time(); CK_BASE = mf.train_run('fr_base', X, meta, DEV, RUNS, epochs=BASE_EPOCHS, lr=LR, aug=1, neg_p=0.0, **COMMON)
    RUN_TIMES['fr_base'] = time.time() - t
    t = time.time(); CK = mf.train_run(RELEASE, X, meta, DEV, RUNS, init=CK_BASE, epochs=NEG_FT_EPOCHS, lr=FT_LR, seed=3, aug=1,
                                       neg_p=NEG_P, **COMMON)
    RUN_TIMES[RELEASE] = time.time() - t
    print({k: f'{v / 60:.1f} min' for k, v in RUN_TIMES.items()})
else:
    print('weights present: no training')
''')

md('## 8. Re-evaluate `fr_neg` on val and test, and check the selection rule\nThe new rule is the **lowest val false-alarm rate**, taken at each model\'s val-picked threshold on the 10 m grid. The other models\' val numbers come from the first run\'s `results.json`.')
code(r'''
SD = torch.load(CK, map_location='cpu')
vc = mf.evaluate_state(SD, ENCODER, X, va, DEV, WORKERS, AMP); tc = mf.evaluate_state(SD, ENCODER, X, te, DEV, WORKERS, AMP)
thr_v = mf.pick_threshold(vc, THRESHOLD_METRIC)
NEW = dict(threshold=thr_v, ingredients=None,
           **{f'{s}@{lab}/{g}': mf.at(c, t, g) for s, c in (('val', vc), ('test', tc))
              for lab, t in (('0.5', 0.5), ('thr', thr_v)) for g in ('10m', '20m')},
           sweep={s: {g: {k: v.tolist() for k, v in c[g].items()} for g in c} for s, c in (('val', vc), ('test', tc))})
K = ['iou', 'f1', 'precision', 'recall', 'det_rate', 'false_alarm']
diff = max(abs(NEW['test@0.5/20m'][k] - ORIG[RELEASE]['test@0.5/20m'][k]) for k in K)
REEVAL_MATCH = (not RETRAINED) and diff <= 0.005
USE_NEW = RETRAINED or not REEVAL_MATCH
REL = NEW if USE_NEW else ORIG[RELEASE]
REL_KEY = f'{RELEASE}_retrained' if RETRAINED else (f'{RELEASE}_reeval' if USE_NEW else RELEASE)
print(f'fr_neg test@0.5/20m  first run: ' + ' '.join(f'{k} {ORIG[RELEASE]["test@0.5/20m"][k]:.3f}' for k in K))
print(f'fr_neg test@0.5/20m  this eval: ' + ' '.join(f'{k} {NEW["test@0.5/20m"][k]:.3f}' for k in K), f'| max |diff| {diff:.4f}')
if not RETRAINED and not REEVAL_MATCH:
    print('WARNING: the weights on disk do not reproduce the recorded numbers; the card will use the new evaluation and say so.')
VALFA = {n: ORIG[n]['val@thr/10m']['false_alarm'] for n in ORIG}
VALFA[RELEASE] = REL['val@thr/10m']['false_alarm']
RANK = sorted(VALFA, key=VALFA.get)
STILL_LOWEST = RANK[0] == RELEASE
print('val false-alarm rate (10 m, val-picked thr):', {n: round(v, 3) for n, v in sorted(VALFA.items(), key=lambda kv: kv[1])})
print('fr_neg still has the lowest val FA:', STILL_LOWEST)
if not STILL_LOWEST: print('WARNING: after retraining, fr_neg is not the lowest-val-FA model. The card says so, and the upload is held unless UPLOAD_IF_NOT_LOWEST = True.')
''')

md('## 9. Update `results.json` (and `comparison_test.csv` if retrained)')
code(r'''
R = dict(R0)
R['selected'], R['selected_threshold'] = RELEASE, RELEASE_THR
R['selection'] = dict(
    criterion="lowest val false-alarm rate (10 m, each model's val-picked threshold); released at the default threshold 0.5",
    previous=dict(selected=PREV_RELEASE, threshold=PREV_THR, criterion="highest val IoU at the val-picked threshold"),
    changed_after_seeing_test=True, weights_retrained=RETRAINED, fr_base_retrained=BASE_RETRAINED,
    released_metrics_key=REL_KEY, smoke_test=SMOKE_TEST)
if USE_NEW:
    R['results'] = dict(R0['results']); R['results'][REL_KEY] = NEW
    R['retrain_minutes'] = {k: v / 60 for k, v in RUN_TIMES.items()}; R['retrain_gpu'] = GPU
json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1)
CPU_SOUP = CMP0[CMP0.model == 'CPU greedy soup (released)'].iloc[0]
CMP = CMP0.copy()
if USE_NEW:
    add = []
    for lab, t in (('0.5', 0.5), ('thr', thr_v)):
        for g in ('20m', '10m'):
            add.append(dict(model=REL_KEY, grid=g, thr=t, **NEW[f'test@{lab}/{g}']))
    add = pd.DataFrame(add).round(3)
    add['Δiou_vs_cpu_soup'] = (add.iou - CPU_SOUP.iou).round(3); add['ΔFA_vs_cpu_soup'] = (add.false_alarm - CPU_SOUP.false_alarm).round(3)
    CMP = pd.concat([CMP0, add], ignore_index=True); CMP.to_csv(f'{OUT}/comparison_test.csv', index=False)
print('selected ->', R['selected'], R['selected_threshold'], '| metrics key', R['selection']['released_metrics_key'])
''')

md('## 10. Export `fr_neg`: safetensors, ONNX (opset 17, static 1×8×200×200), parity, latency, calibration, TensorRT script, infer.py')
code(r'''
import onnx, onnxruntime as ort
from safetensors.torch import save_file
m = mf.MethaneNet(ENCODER, weights=None); m.load_state_dict(SD); m.eval()
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f'{OUT}/model.safetensors')
ONNX_NAME = f'methane_se2_fullres_unet_{ENCODER.replace("-", "")}.onnx'; onnx_p = f'{OUT}/{ONNX_NAME}'
x1, _, _ = mf.eval_batch(torch.from_numpy(np.array(X[te[:1], :mf.NCH_BASIC])))
torch.onnx.export(m, x1, onnx_p, opset_version=17, input_names=['reflectance'], output_names=['plume_logits'],
                  dynamo=False, do_constant_folding=True)
onnx.checker.check_model(onnx_p)
g_in = onnx.load(onnx_p).graph
assert [d.dim_value for d in g_in.input[0].type.tensor_type.shape.dim] == [1, 8, 200, 200] and g_in.output[0].name == 'plume_logits'
MODEL_ID = f'{RELEASE} (retrained)' if RETRAINED else RELEASE
INFO = dict(model=MODEL_ID, encoder=ENCODER, threshold=RELEASE_THR, params=sum(p.numel() for p in m.parameters()),
            onnx_bytes=os.path.getsize(onnx_p), input='reflectance float32 1x8x200x200', output='plume_logits float32 1x1x200x200')
sess = ort.InferenceSession(onnx_p, providers=['CPUExecutionProvider'])
xb, _, _ = mf.eval_batch(torch.from_numpy(np.array(X[te[:min(32, len(te))], :mf.NCH_BASIC]))); d, agree = [], []
with torch.no_grad():
    for i in range(len(xb)):
        pt = m(xb[i:i + 1]).numpy(); ox = sess.run(None, {'reflectance': xb[i:i + 1].numpy()})[0]
        d.append(float(np.abs(pt - ox).max())); agree.append(float(((pt > 0) == (ox > 0)).mean()))
INFO.update(onnx_max_abs_diff_logits=max(d), onnx_mask_agreement=float(np.mean(agree)), parity_chips=len(xb))
assert max(d) < 1e-2 and np.mean(agree) > 0.999, 'ONNX parity check failed'
try:
    from fvcore.nn import FlopCountAnalysis
    INFO['gmacs'] = FlopCountAnalysis(m, x1).unsupported_ops_warnings(False).uncalled_modules_warnings(False).total() / 1e9
except Exception as e: INFO['gmacs_error'] = str(e)
def bench(fn, n):
    for _ in range(3): fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000
NB = 5 if SMOKE_TEST else 50; xi = x1.numpy()
for th in sorted({1, os.cpu_count() or 1}):
    so = ort.SessionOptions(); so.intra_op_num_threads = th
    s = ort.InferenceSession(onnx_p, so, providers=['CPUExecutionProvider'])
    INFO[f'ort_cpu_ms_{th}t'] = bench(lambda: s.run(None, {'reflectance': xi}), NB)
if DEV.type == 'cuda':
    mg = mf.MethaneNet(ENCODER, weights=None); mg.load_state_dict(SD); mg.eval().to(DEV).half(); xg = x1.to(DEV).half()
    def g():
        with torch.no_grad(): mg(xg)
        torch.cuda.synchronize()
    INFO['torch_gpu_fp16_ms_bs1'] = bench(g, 100); INFO['gpu'] = GPU
INFO['cpu'] = platform.processor() or platform.machine()
# The INT8 calibration inputs are train-split chips and do not depend on the model. Reuse the repo's file when it is there.
CAL_REPO = 'calibration_train256.npy'
if CAL_REPO in REPO_FILES and not SMOKE_TEST:
    INFO['calibration'] = CAL_REPO; print('calibration: keeping', CAL_REPO, 'already in the repo (model-independent, unchanged)')
else:
    cal = np.random.default_rng(0).choice(tr, min(CAL_N, len(tr)), replace=False)
    INFO['calibration'] = f'calibration_train{len(cal)}.npy'
    np.save(f'{OUT}/{INFO["calibration"]}', mf.eval_batch(torch.from_numpy(np.array(X[np.sort(cal), :mf.NCH_BASIC])))[0].numpy().astype(np.float16))
json.dump(INFO, open(f'{OUT}/export_info.json', 'w'), indent=1)
json.dump(dict(input_name='reflectance', input_shape=[1, 8, 200, 200], input_bands=mf.INPUT_BANDS, scale='L1C DN / 10000',
               resolution_m=10, output_name='plume_logits', threshold=RELEASE_THR, min_chip_pixels=40, encoder=ENCODER,
               model=MODEL_ID), open(f'{OUT}/config.json', 'w'), indent=1)
print(json.dumps(INFO, indent=1))
''')
code(r'''
CALF = INFO['calibration']
open(f'{OUT}/build_trt.sh', 'w').write(f"""#!/usr/bin/env bash
# Build TensorRT engines for the Spiral Blue SE-2 (Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3).
# UNTESTED ON JETSON: written for JetPack 6.2 but never run on an Orin. Run ON the Jetson; engines are not portable.
# Full-resolution model ({MODEL_ID}, threshold {RELEASE_THR}): static input reflectance 1x8x200x200 (10 m, 2 km chip).
set -euo pipefail
ONNX=${{ONNX:-{ONNX_NAME}}}
TRTEXEC=${{TRTEXEC:-/usr/src/tensorrt/bin/trtexec}}

# Optional: pick a power mode first, e.g. sudo nvpmodel -m <id>  (see /etc/nvpmodel.conf for the 10/15/25 W ids)

# 1) FP16 engine (the safe default)
$TRTEXEC --onnx="$ONNX" --saveEngine=methane_fullres_fp16.engine --fp16 \\
  --shapes=reflectance:1x8x200x200 --memPoolSize=workspace:1024

# 2) INT8 engine with post-training calibration (pip install polygraphy), using {CALF} (train split, float16)
cat > calib_loader.py <<'PY'
import numpy as np
def load_data():
    a = np.load("{CALF}").astype(np.float32)
    for i in range(len(a)):
        yield {{"reflectance": a[i:i+1]}}
PY
polygraphy convert "$ONNX" --int8 --fp16 \\
  --data-loader-script calib_loader.py \\
  --calibration-cache methane_fullres_int8.cache \\
  -o methane_fullres_int8.engine

# Calibration notes:
# - The first ops compute log(B12/B11) target-vs-reference ratios in-graph; that signal is ~1 % of reflectance and
#   INT8 can erase it. If INT8 accuracy drops, keep the head in FP16:
#     --precisionConstraints=obey --layerPrecisions="/Log*:fp16,/Div*:fp16,/Sub*:fp16,/norm/*:fp16"
#   (inspect layer names with: polygraphy inspect model methane_fullres_int8.engine --show layers)
# - Re-check IoU/F1 of each engine against the ONNX on held-out chips before trusting it:
#     polygraphy run "$ONNX" --trt --onnxrt --int8 --calibration-cache methane_fullres_int8.cache

# 3) Latency at batch 1
$TRTEXEC --loadEngine=methane_fullres_fp16.engine --iterations=500 --warmUp=500
$TRTEXEC --loadEngine=methane_fullres_int8.engine --iterations=500 --warmUp=500
""")
open(f'{OUT}/infer.py', 'w').write(f"""# Run the full-resolution methane plume detector on one Sentinel-2 L1C target/reference pair (onnxruntime).

# Input: two co-registered 13-band L1C stacks (DN, scale 10000) at 10 m, 200x200 pixels (2 km x 2 km), e.g. the
# MethaneSET `target` and `bg0` leaves. No pooling: the model runs at the native 10 m grid.
# usage: python infer.py target.tif reference.tif [--onnx {ONNX_NAME}] [--thr {RELEASE_THR}]
import argparse, numpy as np, rasterio, onnxruntime as ort
BANDS = [2, 3, 4, 9, 12, 13]  # 1-based in the 13-band L1C stack: B2,B3,B4,B8A,B11,B12
def load(path):
    with rasterio.open(path) as d:
        a = d.read(BANDS).astype(np.float32) / 10000.0
    assert a.shape[1:] == (200, 200), 'expects a 200x200 10 m chip; tile larger scenes'
    return a
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('target'); ap.add_argument('reference')
    ap.add_argument('--onnx', default='{ONNX_NAME}'); ap.add_argument('--thr', type=float, default={RELEASE_THR})
    ap.add_argument('--out', default='plume_mask.npy'); a = ap.parse_args()
    t, r = load(a.target), load(a.reference)
    x = np.concatenate([t, r[4:6]])[None].astype(np.float32)  # (1,8,200,200)
    s = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    prob = 1 / (1 + np.exp(-s.run(None, {{'reflectance': x}})[0][0, 0])); mask = prob >= a.thr
    np.save(a.out, mask)
    print(f'plume pixels: {{int(mask.sum())}} (10 m px, {{mask.sum() * 100 / 1e4:.2f}} ha); '
          f'chip flagged: {{bool(mask.sum() >= 40)}}; max prob {{prob.max():.3f}}')
if __name__ == '__main__':
    main()
""")
if RETRAINED:
    os.makedirs(f'{OUT}/training_history', exist_ok=True)
    for n in (['fr_base'] if BASE_RETRAINED else []) + [RELEASE]:
        shutil.copy(f'{RUNS}/{n}/history.json', f'{OUT}/training_history/{n}_retrained.json')
# sanity: infer.py's input path through onnxruntime on a real chip pair, and the default threshold it ships with
xi = mf.eval_batch(torch.from_numpy(np.array(X[te[:1], :mf.NCH_BASIC])))[0].numpy()
print('infer-path logits shape', sess.run(None, {'reflectance': xi})[0].shape)
assert f"default={RELEASE_THR})" in open(f'{OUT}/infer.py').read()
assert json.load(open(f'{OUT}/config.json'))['threshold'] == RELEASE_THR
''')

md('## 11. Example predictions from `fr_neg` at threshold 0.5\nThese are the same 6 test chips as the original figure (rng seed 3).')
code(r'''
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(3); pick = np.sort(rng.choice(te, min(6, len(te)), replace=False))
x, y, _ = mf.eval_batch(torch.from_numpy(np.array(X[pick, :mf.NCH_BASIC])))
with torch.no_grad(): p = torch.sigmoid(m(x)).numpy()
ratio = (np.log((x[:, 5] + 1e-3) / (x[:, 4] + 1e-3)) - np.log((x[:, 7] + 1e-3) / (x[:, 6] + 1e-3))).numpy()
f, ax = plt.subplots(len(pick), 4, figsize=(10, 2.6 * len(pick)), squeeze=False)
for i in range(len(pick)):
    ax[i, 0].imshow(np.clip(x[i, [2, 1, 0]].permute(1, 2, 0).numpy() * 3.5, 0, 1))
    ax[i, 1].imshow(ratio[i] - np.median(ratio[i]), cmap='RdBu', vmin=-0.08, vmax=0.08)
    ax[i, 2].imshow(y[i, 0], cmap='gray'); ax[i, 3].imshow(p[i, 0] >= RELEASE_THR, cmap='gray')
    if y[i, 0].max() > 0: ax[i, 3].contour(y[i, 0], levels=[0.5], colors='r', linewidths=0.6)
    ax[i, 0].set_ylabel(str(meta.loc[pick[i], 'detection:case_study'])[:18], fontsize=8)
for j, t in enumerate(['RGB (B4,B3,B2)', 'SWIR ratio vs ref', 'ground truth', f'{MODEL_ID} @ {RELEASE_THR} (red = GT)']):
    ax[0, j].set_title(t, fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
f.tight_layout(); f.savefig(f'{OUT}/example_predictions.png', dpi=110); plt.close(f)
print(sorted(os.listdir(OUT)))
''')

md('## 12. Model card\nThe card is generated from the measured numbers. It discloses the change of selection rule, that test numbers were seen before the change, and the `fr_ft_s0` regression.')
code(r'''
def r3(v): return f'{v:.3f}'
def row(n, d): return f"| {n} | " + ' | '.join(r3(d[k]) for k in K) + ' |'
H = '| model | IoU | F1 | P | R | det | FA |\n|---|---|---|---|---|---|---|'
CPU = {r['model']: {k: r[k] for k in K} for _, r in CMP0[CMP0.grid == '20m (CPU)'].iterrows()}
REL_NAME = f'{RELEASE} (retrained)' if RETRAINED else (f'{RELEASE} (re-evaluated)' if USE_NEW else RELEASE)
ORIG_NAMES = [n for n in ORIG]
lab = lambda n: f'**{n} @0.5 (released)**' if (n == RELEASE and not USE_NEW) else f'{n} @0.5'
cmp20 = '\n'.join([H] + [row(k.replace('(released)', '(20 m repo release)') if '(released)' in k else k + ' (20 m)', v) for k, v in CPU.items()] +
                  [row(lab(n), ORIG[n]['test@0.5/20m']) for n in ORIG_NAMES] +
                  ([row(f'**{REL_NAME} @0.5 (released)**', REL['test@0.5/20m'])] if USE_NEW else []) +
                  [row(f'{PREV_RELEASE} @{PREV_THR} (previous release)', ORIG[PREV_RELEASE]['test@thr/20m'])])
t10 = '\n'.join([H] + [row(f'{n} @{ORIG[n]["threshold"]}', ORIG[n]['test@thr/10m']) for n in ORIG_NAMES] +
                [row(f'{RELEASE} @0.5 (released, 10 m)', REL['test@0.5/10m'])])
def vrow(n, d, mark=''):
    v = d['val@thr/10m']; return f"| {n}{mark} | {d['threshold']} | {r3(v['iou'])} | {r3(v['f1'])} | {r3(v['false_alarm'])} | {r3(d['val@0.5/10m']['false_alarm'])} |"
v10 = '\n'.join(['| model | val-picked thr | val IoU | val F1 | **val FA** | val FA @0.5 |', '|---|---|---|---|---|---|'] +
                [vrow(n, ORIG[n], ' ← released' if (n == RELEASE and not USE_NEW) else (' ← previous release' if n == PREV_RELEASE else ''))
                 for n in sorted(ORIG_NAMES, key=lambda n: ORIG[n]['val@thr/10m']['false_alarm'])] +
                ([vrow(REL_NAME, REL, ' ← released')] if USE_NEW else []))
def csv_md(df):
    cols = list(df.columns); out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        rel = (r['model'] == REL_KEY) and r['grid'] in ('20m', '10m') and abs(r['thr'] - RELEASE_THR) < 1e-9
        cells = [str(r[c]) for c in cols]
        if rel: cells = [f'**{c}**' for c in cells]; cells[0] += ' (released)'
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)
full_table = csv_md(CMP)
lat = ', '.join(f'{k}: {v:.1f} ms' for k, v in INFO.items() if k.endswith('_ms') or '_ms_' in k)
others_fa = sorted(v for n, v in VALFA.items() if n != RELEASE)
o, p = ORIG[RELEASE], ORIG[PREV_RELEASE]
cpu_soup = CPU['CPU greedy soup (released)']
warn = ('\n> **SMOKE TEST BUILD: these numbers come from a 1-epoch run on a handful of chips and mean nothing. '
        'Do not upload this card.**\n') if SMOKE_TEST else ''
if RETRAINED:
    retrain_note = (f"\n> **The released weights were retrained.** The first run's `fr_neg` checkpoint was not kept, so `fr_neg` "
                    f"{'and `fr_base` were' if BASE_RETRAINED else 'was'} retrained on {GPU} with the same recipe, settings and seeds as the first run. "
                    "GPU training is not bit-for-bit deterministic, so the released weights and their numbers can differ slightly from the first run's "
                    f"`fr_neg` row. Every number marked *{REL_NAME}* below was measured on the released (retrained) weights. Retrained `fr_neg`: test IoU "
                    f"{r3(REL['test@0.5/20m']['iou'])} and FA {r3(REL['test@0.5/20m']['false_alarm'])} (first run: {r3(o['test@0.5/20m']['iou'])} and "
                    f"{r3(o['test@0.5/20m']['false_alarm'])}). Val FA {r3(REL['val@thr/10m']['false_alarm'])}"
                    f"{'' if STILL_LOWEST else ', which is **no longer the lowest** of the first-run models'}.\n")
elif not REEVAL_MATCH:
    retrain_note = (f"\n> **Note:** re-evaluating the `fr_neg` weights did not exactly reproduce the first run's recorded numbers "
                    f"(max difference {diff:.3f}), so the released numbers below come from the re-evaluation.\n")
else:
    retrain_note = ''
sel = f"""## Which weights are released, and why they changed

The released weights are **`{REL_NAME}` at threshold {RELEASE_THR}**. They replace `{PREV_RELEASE}` at {PREV_THR}, which the first version of this repo shipped.
{retrain_note}
- **Original rule:** the notebook picked the model with the highest **val IoU** at its val-picked threshold. That chose `{PREV_RELEASE}` (val IoU
  {r3(p['val@thr/10m']['iou'])} at {p['threshold']}). The val split has only 193 chips, so IoU differences of about 0.01 between models are within noise.
- **What went wrong:** on test, `{PREV_RELEASE}` at {PREV_THR} scored IoU {r3(p['test@thr/20m']['iou'])} and FA {r3(p['test@thr/20m']['false_alarm'])} (20 m grid). That
  is *below* the 20 m CPU model ({r3(cpu_soup['iou'])} IoU, FA {r3(cpu_soup['false_alarm'])}). The val IoU ranking did not carry over to test.
- **New rule:** the lowest **val false-alarm rate**, because false alarms are what cost downlink in onboard alerting. `fr_neg` has val FA
  {r3(o['val@thr/10m']['false_alarm'])}, against {r3(others_fa[0])} or more for every other model. That is a much bigger margin than the IoU spread. By val IoU,
  `fr_neg` ranked *lower* ({r3(o['val@thr/10m']['iou'])} at {o['threshold']}).
- **Threshold:** {RELEASE_THR}, the default, instead of `fr_neg`'s val-F1 threshold of {o['threshold']}. On val the two are close (F1 {r3(o['val@0.5/10m']['f1'])} at 0.5 vs
  {r3(o['val@thr/10m']['f1'])} at {o['threshold']}; FA {r3(o['val@0.5/10m']['false_alarm'])} vs {r3(o['val@thr/10m']['false_alarm'])}).
- **Disclosure:** the selection rule and the threshold were changed **after the test numbers had been seen**. So the released test numbers are not a
  clean held-out estimate. Treat them as optimistic by an unknown, probably small, amount. Val alone also supports `fr_neg` on false alarms.
- **Released numbers** (test, 20 m-equivalent, threshold {RELEASE_THR}): IoU {r3(REL['test@0.5/20m']['iou'])}, F1 {r3(REL['test@0.5/20m']['f1'])}, P {r3(REL['test@0.5/20m']['precision'])},
  R {r3(REL['test@0.5/20m']['recall'])}, det {r3(REL['test@0.5/20m']['det_rate'])}, FA {r3(REL['test@0.5/20m']['false_alarm'])}. For comparison, the 20 m CPU greedy soup has IoU {r3(cpu_soup['iou'])}
  and FA {r3(cpu_soup['false_alarm'])}. The trade-off is chip detection: {r3(REL['test@0.5/20m']['det_rate'])} here, against {r3(cpu_soup['det_rate'])} for the CPU greedy soup.
"""
gs = ORIG['greedy_soup'].get('ingredients') if 'greedy_soup' in ORIG else None
card = f"""---
license: cc-by-nc-sa-4.0
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags: [methane, sentinel-2, remote-sensing, edge-ai, jetson, tensorrt, onnx]
datasets: [tacofoundation/methaneset]
metrics: [iou, f1, precision, recall]
base_model: sighmon/methane-plume-se2
---

# Methane plume detector for on-orbit edge compute, full resolution (10 m, SE-2 target)
{warn}
This is the full-resolution version of [sighmon/methane-plume-se2](https://huggingface.co/sighmon/methane-plume-se2). It is a U-Net
({ENCODER} encoder, {INFO['params']:,} parameters) that segments methane plumes in Sentinel-2 L1C chips at 10 m. It needs a
plume-free reference image of the same site. It is sized for satellite edge computers such as the Spiral Blue SE-2 (NVIDIA Jetson Orin NX 16GB,
JetPack 6.2, TensorRT 10.3, 10–25 W).

**Demo / research model by Simon Loffler ([@sighmon](https://github.com/sighmon)). Not affiliated with, endorsed by, or tested by
Spiral Blue.** It has not been run on Jetson hardware.

**Released weights: `{REL_NAME}` at threshold {RELEASE_THR}** (fine-tuned with plume-free pairs, chosen for the lowest val false-alarm rate).
The section below explains the change from `{PREV_RELEASE}`.

## Input contract

- `reflectance`: float32 `1×8×200×200`, L1C TOA reflectance (DN / 10000) at 10 m (2 km × 2 km). Channels: target B2, B3, B4, B8A,
  B11, B12, then reference B11, B12. The 20 m bands are used as MethaneSET provides them, resampled to 10 m.
- `plume_logits`: float32 `1×1×200×200`. Apply a sigmoid, then threshold at **{RELEASE_THR}** (in `config.json`). A chip
  counts as flagged at ≥40 pixels (0.4 ha).
- A SWIR ratio channel `log(B12/B11)_target − log(B12/B11)_reference` (mean-centred) is computed inside the graph in fp32.
  The input is reflect-padded to 224 inside the graph and cropped back to 200.
- The channels and scaling are the same as the 20 m model. Only the spatial size differs. The ONNX file name and tensor names have not changed.

{sel}
## Training

The data is [MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset) `methaneset-s2-finetune` v1.4.0. It has 3,605 chips (200×200 at
10 m) in the official site-grouped splits of 2,362 train, 193 val and 1,050 test, and is licensed CC BY-NC-SA 4.0. The first run was trained on {R0.get('gpu')} with AMP={AMP}, with the SWIR head in fp32.{f' The released weights were retrained on {GPU}.' if RETRAINED else ''}

- `fr_base`: ImageNet init, {BASE_EPOCHS} epochs, lr {LR}, rot90 and flips, seed 0. This is the CPU `base` recipe at 10 m.
- **`fr_neg` (released)**: a fine-tune of `fr_base` for {NEG_FT_EPOCHS} epochs at lr {FT_LR}, seed 3. {int(NEG_P * 100)} % of chips are replaced by plume-free pairs (bg1 vs bg0).
- `fr_long`: 40 epochs with rot90, flips, ±10 % brightness jitter, 30 % negatives and extra references (bg1–bg3).
- `fr_ft_s*`: 4 fine-tunes of `fr_long` with different seeds, 8 epochs each, averaged as a uniform soup and as a greedy soup on val.
  The greedy soup kept {gs}. Weight averaging did not beat `fr_neg` here.

## Results (test split, measured)

Like-for-like with the CPU model, on the **20 m-equivalent** grid. The 10 m probabilities are 2×2-mean-pooled and scored
against the CPU model's 20 m labels (pooled fraction ≥0.5); det and FA use ≥10 px (0.4 ha).

{cmp20}

Native 10 m, at each model's val-picked threshold, plus the released model at {RELEASE_THR}:

{t10}

Val split, 10 m, sorted by val false-alarm rate (the new selection rule). The val-picked threshold is the best val F1:

{v10}

Full comparison (`comparison_test.csv`: test split, both grids, at 0.5 and at each model's val-picked threshold; deltas vs the CPU greedy soup):

{full_table}

- `det` is the share of plume chips with ≥0.4 ha predicted that overlaps the label. `FA` is the share of plume-free pairs (bg1 vs bg0 of the same site)
  with ≥0.4 ha predicted. The references come from known super-emitter sites and may hold real, unlabelled plumes, so FA is an
  upper bound on true false alarms.
- These are single runs with no confidence intervals. Treat differences of about 1 IoU point as noise.

## Size and speed (released weights)

- {INFO['params']:,} parameters; {INFO.get('gmacs', float('nan')):.2f} GMACs per 200×200 chip (fvcore); ONNX {INFO['onnx_bytes'] / 1e6:.1f} MB (opset 17, static).
- ONNX vs PyTorch: max abs logit diff {INFO['onnx_max_abs_diff_logits']:.1e}, mask agreement {INFO['onnx_mask_agreement'] * 100:.2f} % on {INFO['parity_chips']} test chips.
- Latency (batch 1): {lat}. CPU: {INFO['cpu']}. Jetson Orin NX FP16/INT8 was **not measured**; see `build_trt.sh`.
- The 10 m model does about 3× the compute of the 20 m model per 2 km chip.

## Files

`model.safetensors` and `{ONNX_NAME}` ({REL_NAME}), `config.json` (input contract and threshold {RELEASE_THR}), `infer.py` (default threshold {RELEASE_THR}),
`build_trt.sh` (untested on Jetson), `{INFO['calibration']}` (INT8 calibration inputs from the train split, float16; they do not depend on the model),
`results.json` (full threshold sweeps for every model, plus the `selection` record), `comparison_test.csv`, `training_history/`, `training_curves.png`,
`threshold_sweep_val.png` (first run), `example_predictions.png` ({REL_NAME} at {RELEASE_THR}), and `src/methane_fullres.py`.

## Limitations

- The chips are centred on known emitters. Onboard use over full swaths needs tiling and far more negative data.
- It needs a co-registered plume-free reference of the same site.
- The data covers onshore oil, gas and coal sites only, mostly Turkmenistan and Algeria.
- The labels come from MARS-S2L analysts and are noisy. Emission rates are not estimated.
- INT8 can erase the ~1 % SWIR ratio signal, so validate any engine against the ONNX.
- This is a demonstration, not a validated monitoring product. It is non-commercial (CC BY-NC-SA 4.0).

## Citation (data)

```bibtex
@article{{contreras2026methaneset,
  title   = {{MethaneSET: Unified Multi-Sensor AI-Ready Datasets for Satellite-Based Methane Plume Detection}},
  author  = {{Contreras, Julio and Giner, Carlos and Aybar, Cesar and Mateo-Garc{{\\'i}}a, Gonzalo and Gorro{{\\~n}}o, Javier and Montero, David and Mahecha, Miguel D. and Guanter, Luis and G{{\\'o}}mez-Chova, Luis}},
  journal = {{Scientific Data}},
  year    = {{2026}}
}}
```
"""
open(f'{OUT}/README.md', 'w').write(card)
print(card[:6000])
''')

md('## 13. Upload as one commit\nThis pushes only the changed files in `hf_swap/` to the existing repo as a single commit. Files that are not in the folder (training curves, threshold sweep, `src/`, the calibration set) stay as they are.')
code(r'''
print('files to push:', sorted(os.path.relpath(p, OUT) for p in glob.glob(f'{OUT}/**', recursive=True) if os.path.isfile(p)))
HOLD = not STILL_LOWEST and not UPLOAD_IF_NOT_LOWEST
if HOLD: print('HELD: fr_neg is no longer the lowest-val-FA model, so nothing is uploaded. Review the card, then set UPLOAD_IF_NOT_LOWEST = True to publish anyway.')
if UPLOAD and not SMOKE_TEST and not HOLD:
    assert HF_OK, 'log in first (section 3)'
    ci = HfApi().upload_folder(folder_path=OUT, repo_id=HF_REPO, repo_type='model', commit_message=COMMIT_MSG)
    print('uploaded:', getattr(ci, 'commit_url', ci))
else:
    print('UPLOAD is False (or smoke test): nothing uploaded. Files ready in', OUT)
print(f'total time {(time.time() - T0) / 60:.1f} min | path:', 'retrained' if RETRAINED else 'weights present')
''')

nb.cells = C
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
               'language_info': {'name': 'python'}, 'accelerator': 'GPU',
               'colab': {'provenance': [], 'gpuType': 'T4', 'machine_shape': 'hm'}}
nbf.validate(nb)
nbf.write(nb, '/workspace/methane/colab/swap_to_fr_neg.ipynb'); print('written', len(C), 'cells')
