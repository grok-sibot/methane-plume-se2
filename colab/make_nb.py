import nbformat as nbf, json
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s, tags=None):
    c = nbf.v4.new_code_cell(s.strip('\n'))
    if tags: c.metadata['tags'] = tags
    C.append(c)
MOD = open('/workspace/methane/colab/build/methane_fullres.py').read()

md(r'''
# Methane plume detector for the Spiral Blue SE-2, full-resolution run (10 m, 200×200)

This notebook retrains [`sighmon/methane-plume-se2`](https://huggingface.co/sighmon/methane-plume-se2) at the native 10 m
grid of the MethaneSET chips, on a Colab GPU. The CPU model was trained at 20 m (2×2-pooled, 100×100) because the
build box had no GPU.

**Not affiliated with, endorsed by, or tested by Spiral Blue.** The training labels are CC BY-NC-SA 4.0, so the weights are non-commercial.

**How to run:** choose *Runtime → Change runtime type → GPU* (T4, L4 or A100; **High-RAM** helps), add a Colab secret
`HF_TOKEN` (key icon in the left bar) if you want to upload, then *Runtime → Run all*. Nothing is published unless you set
`SYNC_TRACKIO = True` or `UPLOAD = True` in the config cell.

## Input contract (drop-in for the SE-2 pipeline, static shape)

| tensor | shape | meaning |
|---|---|---|
| `reflectance` (input) | float32 `1×8×200×200` | L1C TOA reflectance (DN / 10000) at **10 m** (2 km × 2 km chip). Channels: target B2, B3, B4, B8A, B11, B12, then reference B11, B12. The 20 m bands are used as MethaneSET provides them, resampled to 10 m. |
| `plume_logits` (output) | float32 `1×1×200×200` | sigmoid, then threshold (the val-picked threshold is written to `config.json`) |

The channel order, scaling, in-graph SWIR ratio (`log(B12/B11)` target − reference, mean-centred) and the ONNX names
are the same as the CPU model. Only the spatial size changes from 100×100 to 200×200. Inside the graph the input is
reflect-padded to 224 (a multiple of 32) and the output is cropped back to 200.

## Experiments

| id | run | recipe |
|---|---|---|
| (a) | `fr_base` | CPU `base` recipe at 10 m: ImageNet init, 15 epochs, lr 1e-3, rot90/flip |
| (b) | `fr_neg` | CPU `ft_neg` recipe: fine-tune (a) 8 epochs, lr 3e-4, 30 % plume-free pairs (bg1 vs bg0) |
| (c) | `fr_long` | longer (40 epochs) + flips/rot90 + ±10 % brightness jitter + negatives + extra references bg1–bg3 |
| (d) | `fr_ft_s*` + soups | 4 fine-tunes of (c) with different seeds, then uniform and greedy soups chosen on val |

Metrics are the same as the CPU model's: pixel IoU/F1/precision/recall, chip detection (≥0.4 ha predicted and overlapping the label) and
false-alarm rate on plume-free pairs (bg1 vs bg0). Each is reported on two grids:
- **10 m**: native (≥40 px = 0.4 ha).
- **20 m-equivalent**: 2×2-mean-pooled probabilities scored against the CPU model's 20 m labels (≥10 px = 0.4 ha).
  This is the like-for-like comparison with the CPU numbers.

A threshold sweep (0.05–0.95) picks the threshold on **val** (best F1), and test is reported both at 0.5 and at that threshold.

## Estimated runtimes (estimates, not measured on Colab)

The numbers are scaled from the CPU run (≈60 s/epoch at 100×100 on 8 vCPU) and the ~3× compute of a 224×224 padded
input (≈1.1 GMACs/chip forward vs 0.36). The T4 figures assume a standard runtime where the 8.9 GB cache is read from disk. A High-RAM
runtime holds it in RAM and is faster.

| section | T4 (estimate) | A100 (estimate) |
|---|---|---|
| setup + pip | 1–3 min | 1–3 min |
| download MethaneSET S2 finetune (6.3 GB) | 3–8 min | 3–6 min |
| build 10 m cache (3,605 × 6 GeoTIFFs) | 10–20 min (2 vCPU) | 3–6 min (12 vCPU) |
| (a) 15 epochs | 8–12 min | 2–4 min |
| (b) 8 epochs | 4–7 min | 1–2 min |
| (c) 40 epochs (31-channel reads) | 25–40 min | 7–12 min |
| (d) 4 × 8 epochs + soups | 20–30 min | 6–10 min |
| final val/test sweep of ~9 models | 5–10 min | 2–4 min |
| export, ONNX parity, latency, figures, card | 2–4 min | 1–3 min |
| **total** | **≈1.5–2.5 h** | **≈30–50 min** |

With `USE_DRIVE = True`, a later session copies the cache and the finished runs back from Drive and skips anything already trained.
''')

md('## 1. Config\nThis is the papermill `parameters` cell. `SMOKE_TEST = True` runs a tiny end-to-end check (a few chips, 1 epoch) and its numbers mean nothing.')
code(r'''
# ---- run mode ----
SMOKE_TEST = False          # True: ~14 chips, 1 epoch per run, CPU OK. Proves the notebook executes; numbers are meaningless.
USE_DRIVE = False           # True: mount Google Drive and cache the prepared data, runs and Trackio DB there
SKIP_DONE = True            # skip runs whose final.pt already exists (resume after a disconnect)
UPLOAD = False              # True: create + upload to the NEW model repo below (needs a write token)
SYNC_TRACKIO = False        # True: publish the Trackio dashboard as a static Space (public)

# ---- model / training knobs ----
ENCODER = 'mobilenet_v2'    # 'mobilenet_v2' (default, matches the CPU model) | 'efficientnet-b0' | 'resnet18'
BATCH_SIZE = 16             # CPU recipe used 16; 32 is fine on any of these GPUs (consider lr x1.4)
AMP = True                  # bf16 on A100/L4, fp16 + GradScaler on T4; the SWIR-ratio head always runs in fp32
BASE_EPOCHS = 15            # (a)
NEG_FT_EPOCHS = 8           # (b)
LONG_EPOCHS = 40            # (c)
SOUP_EPOCHS = 8             # (d) per fine-tune
LR, FT_LR = 1e-3, 3e-4
NEG_P = 0.3                 # fraction of training chips replaced with plume-free pairs in (b), (c), (d)
EXTRA_REFS = True           # (c)/(d): use bg1-bg3 as alternative references and as negative "targets"
EXTRA_REF_P = 0.5           # prob. of swapping bg0 for a random bg1-bg3 reference on a plume chip
NEG_MAX_METHANE = 0.02      # only use a reference as a negative target if <=2 % of its pixels show its own enhancement (bg:methaneN); None = no filter
SOUP_SEEDS = [0, 1, 2, 3]   # (d) one fine-tune per seed
THRESHOLD_METRIC = 'f1'     # pick the operating threshold on val by this metric (10 m grid)
SELECT_METRIC = 'iou'       # pick the exported model by val pixel IoU at its val-picked threshold (10 m)
CAL_N = 256                 # INT8 calibration chips (train split)
WORKERS = None              # DataLoader workers; None = min(4, cpu_count)

# ---- names ----
TRACKIO_PROJECT = 'methane-se2-fullres'
TRACKIO_SPACE = 'sighmon/methane-se2-fullres-trackio'   # new Space; leaves sighmon/methane-se2-trackio untouched
HF_REPO = 'sighmon/methane-plume-se2-fullres'           # new model repo; leaves sighmon/methane-plume-se2 untouched
WORK = None                 # local working dir; None = /content/methane-fullres on Colab, ./methane-fullres elsewhere
''', tags=['parameters'])

md('## 2. Setup\nThis installs the packages, picks the device and sets the folders. On Colab it uses the preinstalled CUDA PyTorch and only adds the rest.')
code(r'''
import os, sys, subprocess, json, time, shutil, glob, platform
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
if IN_COLAB:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.6.0', 'trackio',
                    'rasterio', 'onnx', 'onnxruntime', 'onnxscript', 'fvcore', 'safetensors', 'psutil',
                    'huggingface_hub>=0.34', 'pyarrow'], check=True)
import numpy as np, pandas as pd, torch
if SMOKE_TEST:
    BASE_EPOCHS = NEG_FT_EPOCHS = LONG_EPOCHS = SOUP_EPOCHS = 1; SOUP_SEEDS = SOUP_SEEDS[:2]; CAL_N = 8; BATCH_SIZE = 4
WORK = WORK or ('/content/methane-fullres' if IN_COLAB else os.path.abspath('methane-fullres'))
DATA, RUNS, OUT = f'{WORK}/data', f'{WORK}/runs', f'{WORK}/hf_upload'
for d in (DATA, RUNS, OUT): os.makedirs(d, exist_ok=True)
DRIVE = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive; drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/methane-se2-fullres'; os.makedirs(DRIVE, exist_ok=True)
os.environ['TRACKIO_DIR'] = f'{DRIVE or WORK}/trackio'; os.makedirs(os.environ['TRACKIO_DIR'], exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = WORKERS if WORKERS is not None else min(4, os.cpu_count() or 2)
GPU = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'none (CPU)'
print(f'torch {torch.__version__} | device {DEV} | GPU {GPU} | cpus {os.cpu_count()} | workers {WORKERS} | smoke {SMOKE_TEST}')
print('work dir', WORK, '| trackio dir', os.environ['TRACKIO_DIR'])
if DEV.type != 'cuda' and not SMOKE_TEST:
    print('WARNING: no GPU. Runtime -> Change runtime type -> GPU. A full run on CPU would take many hours.')
''')

md('## 3. Training code\nThis cell writes `methane_fullres.py`, which holds the data cache, pair building and augmentation, the model, the metrics and the training loop. The file is also uploaded as `src/` with the model.')
code('%%writefile methane_fullres.py\n' + MOD)
code(r'''
import importlib
sys.path.insert(0, os.getcwd()); import methane_fullres as mf; importlib.reload(mf)
shutil.copy('methane_fullres.py', f'{WORK}/methane_fullres.py')
''')

md('## 4. Hugging Face login\nThis uses the Colab secret `HF_TOKEN` and falls back to an interactive login. The token is never printed or stored in the notebook. The dataset is public, so the token is only needed to upload.')
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
    elif UPLOAD or SYNC_TRACKIO:
        from huggingface_hub import notebook_login; notebook_login(); HF_OK = True
    if HF_OK: print('logged in as', whoami()['name'])
    else: print('no HF token: fine for training; needed only for UPLOAD / SYNC_TRACKIO')
    del tok
except Exception as e:
    print('HF login skipped:', type(e).__name__, e)
''')

md('## 5. Download MethaneSET (Sentinel-2 finetune, v1.4.0)\nThis downloads to local disk. Smoke mode only fetches the 7 smallest country files (~20 MB).')
code(r'''
from huggingface_hub import hf_hub_download
REPO, SUB = 'tacofoundation/methaneset', 'methaneset-s2-finetune'
CACHE = f'{DATA}/cache10m.npy'
def drive_has(p): return DRIVE and os.path.exists(f'{DRIVE}/{p}')
meta_files = {}
for f in ['level0.parquet', 'level1.parquet', 'COLLECTION.json']:
    meta_files[f] = hf_hub_download(REPO, f'{SUB}/.tacocat/{f}', repo_type='dataset', local_dir=DATA)
l0 = pd.read_parquet(meta_files['level0.parquet']).reset_index(drop=True)
l1 = pd.read_parquet(meta_files['level1.parquet'])
SMALL = ['Oman', 'Poland', 'Russian_Federation', 'Venezuela', 'Argentina', 'China', 'Jordan']
if SMOKE_TEST:
    l0 = l0[l0['internal:source_file'].isin([f'{SUB}_{c}.tacozip' for c in SMALL])].reset_index(drop=True)
zips = sorted(l0['internal:source_file'].unique())
ZIPDIR = f'{DATA}/{SUB}'
need_zips = not (os.path.exists(CACHE) or drive_has('cache10m.npy'))
t = time.time()
if need_zips:
    for z in zips:
        hf_hub_download(REPO, f'{SUB}/{z}', repo_type='dataset', local_dir=DATA)
    print(f'downloaded {len(zips)} tacozips in {time.time() - t:.0f}s')
else:
    print('prepared cache found, skipping the tacozip download')
print(len(l0), 'chips;', l0.split.value_counts().to_dict())
''')

md('## 6. Build the 10 m cache\nThe cache is float16, `N×31×200×200`: the mask, the target, and bg0–bg3, each with B2, B3, B4, B8A, B11 and B12 as DN/10000. It is 8.9 GB for the full set.')
code(r'''
meta = l0[[c for c in mf.META_COLS if c in l0.columns]].copy()
if SMOKE_TEST:  # the small countries have no validation chips: assign smoke splits round-robin
    meta['split'] = [['train', 'train', 'validation', 'test'][i % 4] for i in range(len(meta))]
meta.to_csv(f'{DATA}/meta.csv', index=False)
if not os.path.exists(CACHE):
    if drive_has('cache10m.npy') and not SMOKE_TEST:
        t = time.time(); shutil.copy(f'{DRIVE}/cache10m.npy', CACHE); print(f'copied cache from Drive in {time.time() - t:.0f}s')
    else:
        mf.build_cache(l0, l1, ZIPDIR, CACHE, workers=os.cpu_count())
        if DRIVE and not SMOKE_TEST: shutil.copy(CACHE, f'{DRIVE}/cache10m.npy'); print('cache saved to Drive')
X = mf.load_cache(CACHE)
assert X.shape[1:] == (31, 200, 200) and len(X) == len(meta)
pf = float(np.mean(X[:min(len(X), 200), 0] >= 0.5))
print('plume-pixel fraction (first 200 chips, 10 m):', round(pf, 4))
tr = np.where(meta.split == 'train')[0]; va = np.where(meta.split == 'validation')[0]; te = np.where(meta.split == 'test')[0]
print('train/val/test', len(tr), len(va), len(te))
''')

md('## 7. Experiments\nEvery run logs to Trackio (`methane-se2-fullres`): loss, lr, val metrics at 10 m and 20 m-equivalent, and preview images every 3 epochs. To open the dashboard live, run `import trackio; trackio.show(project=TRACKIO_PROJECT)` in a spare cell.')
code(r'''
def sync_run(name):
    if DRIVE and not SMOKE_TEST: shutil.copytree(f'{RUNS}/{name}', f'{DRIVE}/runs/{name}', dirs_exist_ok=True)
def pull_runs():
    if DRIVE and os.path.isdir(f'{DRIVE}/runs') and not SMOKE_TEST:
        shutil.copytree(f'{DRIVE}/runs', RUNS, dirs_exist_ok=True)
pull_runs()
COMMON = dict(encoder=ENCODER, bs=BATCH_SIZE, amp=AMP, workers=WORKERS, project=TRACKIO_PROJECT,
              skip_done=SKIP_DONE, preview_every=1 if SMOKE_TEST else 3)
RUN_TIMES = {}
def run(name, **kw):
    t = time.time(); ck = mf.train_run(name, X, meta, DEV, RUNS, **COMMON, **kw); RUN_TIMES[name] = time.time() - t
    sync_run(name); print(f'== {name} done in {RUN_TIMES[name] / 60:.1f} min'); return ck
''')
md('### (a) Full-res baseline, using the CPU `base` recipe')
code("CK_BASE = run('fr_base', epochs=BASE_EPOCHS, lr=LR, aug=1, neg_p=0.0)")
md('### (b) Full-res + plume-free negatives, using the CPU `ft_neg` recipe (fine-tune of a)')
code("CK_NEG = run('fr_neg', init=CK_BASE, epochs=NEG_FT_EPOCHS, lr=FT_LR, seed=3, aug=1, neg_p=NEG_P)")
md('### (c) Longer training + flips/rot90 + brightness jitter + negatives + extra references')
code(r'''
LONG = dict(aug=2, neg_p=NEG_P, extra_refs=EXTRA_REFS, extra_ref_p=EXTRA_REF_P, neg_max_methane=NEG_MAX_METHANE)
CK_LONG = run('fr_long', epochs=LONG_EPOCHS, lr=LR, seed=0, **LONG)
''')
md('### (d) Soup ingredients: fine-tunes of (c) with different seeds')
code(r'''
SOUP_RUNS = []
for s in SOUP_SEEDS:
    run(f'fr_ft_s{s}', init=CK_LONG, epochs=SOUP_EPOCHS, lr=FT_LR, seed=100 + s, **LONG); SOUP_RUNS.append(f'fr_ft_s{s}')
print(SOUP_RUNS)
''')

md('## 8. Soups, threshold sweep and the val/test evaluation\nThe greedy soup ([Wortsman et al. 2022](https://arxiv.org/abs/2203.05482)) adds fine-tunes in order of val IoU and keeps each one only if val IoU at 0.5 (10 m) improves. Test is only read after every choice has been made on val.')
code(r'''
load = lambda n: torch.load(f'{RUNS}/{n}/final.pt', map_location='cpu')
SD = {n: load(n) for n in ['fr_base', 'fr_neg', 'fr_long'] + SOUP_RUNS}
VAL = {}
def val(n, state=None):
    if n not in VAL: VAL[n] = mf.evaluate_state(state if state is not None else SD[n], ENCODER, X, va, DEV, WORKERS, AMP)
    return VAL[n]
for n in SD: v = mf.at(val(n), 0.5); print(f'{n:12s} val IoU {v["iou"]:.3f} F1 {v["f1"]:.3f} FA {v["false_alarm"]:.3f}')
SD['uniform_soup'] = mf.average([SD[n] for n in SOUP_RUNS]); val('uniform_soup', SD['uniform_soup'])
order = sorted(SOUP_RUNS, key=lambda n: -mf.at(val(n), 0.5)['iou']); greedy = [order[0]]
best = mf.at(val(order[0]), 0.5)['iou']
for n in order[1:]:
    c = mf.evaluate_state(mf.average([SD[g] for g in greedy + [n]]), ENCODER, X, va, DEV, WORKERS, AMP)
    if mf.at(c, 0.5)['iou'] > best: greedy.append(n); best = mf.at(c, 0.5)['iou']
SD['greedy_soup'] = mf.average([SD[g] for g in greedy]); VAL.pop('greedy_soup', None); val('greedy_soup', SD['greedy_soup'])
print('greedy soup ingredients:', greedy)
torch.save(SD['uniform_soup'], f'{RUNS}/uniform_soup.pt'); torch.save(SD['greedy_soup'], f'{RUNS}/greedy_soup.pt')

RES = {}
for n in SD:
    thr = mf.pick_threshold(VAL[n], THRESHOLD_METRIC)
    tc = mf.evaluate_state(SD[n], ENCODER, X, te, DEV, WORKERS, AMP)
    RES[n] = dict(threshold=thr, ingredients=greedy if n == 'greedy_soup' else (SOUP_RUNS if n == 'uniform_soup' else None),
                  **{f'{s}@{lab}/{g}': mf.at(c, t, g) for s, c in (('val', VAL[n]), ('test', tc))
                     for lab, t in (('0.5', 0.5), ('thr', thr)) for g in ('10m', '20m')},
                  sweep={s: {g: {k: v.tolist() for k, v in c[g].items()} for g in c} for s, c in (('val', VAL[n]), ('test', tc))})
BEST = max(RES, key=lambda n: RES[n][f'val@thr/10m'][SELECT_METRIC])
BEST_THR = RES[BEST]['threshold']
print('selected for export (val only):', BEST, '| threshold', BEST_THR)
json.dump(dict(thresholds=mf.THRS.tolist(), results=RES, selected=BEST, selected_threshold=BEST_THR,
               run_minutes={k: v / 60 for k, v in RUN_TIMES.items()}, gpu=GPU, smoke_test=SMOKE_TEST),
          open(f'{RUNS}/results.json', 'w'), indent=1)
''')

md('## 9. Comparison with the CPU baseline\nThe CPU numbers come from the published model card (20 m, threshold 0.5). Compare them with the **20 m-equivalent** columns.')
code(r'''
CPU = {  # sighmon/methane-plume-se2, test split, 20 m, threshold 0.5
    'CPU greedy soup (released)': dict(iou=0.509, f1=0.675, precision=0.710, recall=0.643, det_rate=0.899, false_alarm=0.825),
    'CPU ft_neg':                 dict(iou=0.483, f1=0.651, precision=0.745, recall=0.578, det_rate=0.814, false_alarm=0.562)}
K = ['iou', 'f1', 'precision', 'recall', 'det_rate', 'false_alarm']
rows = [dict(model=k, grid='20m (CPU)', thr=0.5, **v) for k, v in CPU.items()]
for n in RES:
    for lab in ('0.5', 'thr'):
        for g in ('20m', '10m'):
            rows.append(dict(model=n, grid=g, thr=0.5 if lab == '0.5' else RES[n]['threshold'], **RES[n][f'test@{lab}/{g}']))
TABLE = pd.DataFrame(rows).round(3)
TABLE['Δiou_vs_cpu_soup'] = (TABLE.iou - CPU['CPU greedy soup (released)']['iou']).round(3)
TABLE['ΔFA_vs_cpu_soup'] = (TABLE.false_alarm - CPU['CPU greedy soup (released)']['false_alarm']).round(3)
TABLE.to_csv(f'{RUNS}/comparison_test.csv', index=False)
pd.set_option('display.width', 200); print('TEST split' + ('  [SMOKE TEST: meaningless numbers]' if SMOKE_TEST else ''))
display(TABLE) if 'display' in globals() else print(TABLE.to_string())
VALT = pd.DataFrame([dict(model=n, thr=RES[n]['threshold'], **RES[n]['val@thr/10m']) for n in RES]).round(3)
print('VAL split, 10 m, at each model\'s val-picked threshold'); print(VALT.to_string(index=False))
''')

md('## 10. Figures\nThis makes the training curves for every run, threshold sweeps on val, and test predictions from the selected model.')
code(r'''
import matplotlib; import matplotlib.pyplot as plt
FIG = OUT
f, ax = plt.subplots(1, 4, figsize=(17, 3.8))
offs = {'fr_base': 0, 'fr_neg': BASE_EPOCHS, 'fr_long': 0}
for p in sorted(glob.glob(f'{RUNS}/*/history.json')):
    n = p.split('/')[-2]; h = json.load(open(p))['history']; off = offs.get(n, LONG_EPOCHS)
    for a, k in zip(ax, ['train_loss', 'val/iou', 'val/f1', 'val/false_alarm']):
        e = [r['epoch'] + off for r in h if k in r]; a.plot(e, [r[k] for r in h if k in r], label=n, lw=1.5, marker='o', ms=2.5)
for a, k in zip(ax, ['train loss (Dice+BCE)', 'val IoU (10 m, thr 0.5)', 'val F1 (10 m, thr 0.5)', 'val chip false-alarm rate']):
    a.set_title(k); a.set_xlabel('epoch (fine-tunes continue from their init)'); a.grid(alpha=.2)
ax[0].legend(fontsize=7); f.tight_layout(); f.savefig(f'{FIG}/training_curves.png', dpi=120); plt.close(f)

f, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for n in RES:
    s = RES[n]['sweep']['val']['10m']
    for a, k in zip(ax, ['f1', 'iou', 'false_alarm']): a.plot(mf.THRS, s[k], label=n, lw=1.3)
for a, k in zip(ax, ['val F1 (10 m)', 'val IoU (10 m)', 'val false-alarm rate']): a.set_title(k); a.set_xlabel('threshold'); a.grid(alpha=.2)
ax[0].legend(fontsize=7); f.tight_layout(); f.savefig(f'{FIG}/threshold_sweep_val.png', dpi=120); plt.close(f)

m = mf.MethaneNet(ENCODER, weights=None); m.load_state_dict(SD[BEST]); m.eval()
rng = np.random.default_rng(3); pick = np.sort(rng.choice(te, min(6, len(te)), replace=False))
x, y, _ = mf.eval_batch(torch.from_numpy(np.array(X[pick, :mf.NCH_BASIC])))
with torch.no_grad(): p = torch.sigmoid(m(x)).numpy()
ratio = (np.log((x[:, 5] + 1e-3) / (x[:, 4] + 1e-3)) - np.log((x[:, 7] + 1e-3) / (x[:, 6] + 1e-3))).numpy()
f, ax = plt.subplots(len(pick), 4, figsize=(10, 2.6 * len(pick)), squeeze=False)
for i in range(len(pick)):
    ax[i, 0].imshow(np.clip(x[i, [2, 1, 0]].permute(1, 2, 0).numpy() * 3.5, 0, 1))
    ax[i, 1].imshow(ratio[i] - np.median(ratio[i]), cmap='RdBu', vmin=-0.08, vmax=0.08)
    ax[i, 2].imshow(y[i, 0], cmap='gray'); ax[i, 3].imshow(p[i, 0] >= BEST_THR, cmap='gray')
    if y[i, 0].max() > 0: ax[i, 3].contour(y[i, 0], levels=[0.5], colors='r', linewidths=0.6)
    ax[i, 0].set_ylabel(str(meta.loc[pick[i], 'detection:case_study'])[:18], fontsize=8)
for j, t in enumerate(['RGB (B4,B3,B2)', 'SWIR ratio vs ref', 'ground truth', f'prediction @ {BEST_THR} (red = GT)']):
    ax[0, j].set_title(t, fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
f.tight_layout(); f.savefig(f'{FIG}/example_predictions.png', dpi=110); plt.close(f)
print('figures written to', FIG, os.listdir(FIG))
''')

md('## 11. Export: safetensors, ONNX (opset 17, static 1×8×200×200), onnxruntime parity, CPU latency, INT8 calibration set, TensorRT script')
code(r'''
import onnx, onnxruntime as ort
from safetensors.torch import save_file
m = mf.MethaneNet(ENCODER, weights=None); m.load_state_dict(SD[BEST]); m.eval()
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f'{OUT}/model.safetensors')
ONNX_NAME = f'methane_se2_fullres_unet_{ENCODER.replace("-", "")}.onnx'; onnx_p = f'{OUT}/{ONNX_NAME}'
x1, _, _ = mf.eval_batch(torch.from_numpy(np.array(X[te[:1], :mf.NCH_BASIC])))
torch.onnx.export(m, x1, onnx_p, opset_version=17, input_names=['reflectance'], output_names=['plume_logits'],
                  dynamo=False, do_constant_folding=True)
onnx.checker.check_model(onnx_p)
INFO = dict(model=BEST, encoder=ENCODER, threshold=BEST_THR, params=sum(p.numel() for p in m.parameters()),
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
    mg = mf.MethaneNet(ENCODER, weights=None); mg.load_state_dict(SD[BEST]); mg.eval().to(DEV).half()
    xg = x1.to(DEV).half()
    def g():
        with torch.no_grad(): mg(xg)
        torch.cuda.synchronize()
    INFO['torch_gpu_fp16_ms_bs1'] = bench(g, 100); INFO['gpu'] = GPU
INFO['cpu'] = platform.processor() or platform.machine()
cal = np.random.default_rng(0).choice(tr, min(CAL_N, len(tr)), replace=False)
np.save(f'{OUT}/calibration_train{len(cal)}.npy', mf.eval_batch(torch.from_numpy(np.array(X[np.sort(cal), :mf.NCH_BASIC])))[0].numpy().astype(np.float16))
INFO['calibration'] = f'calibration_train{len(cal)}.npy'
json.dump(INFO, open(f'{OUT}/export_info.json', 'w'), indent=1)
json.dump(dict(input_name='reflectance', input_shape=[1, 8, 200, 200], input_bands=mf.INPUT_BANDS, scale='L1C DN / 10000',
               resolution_m=10, output_name='plume_logits', threshold=BEST_THR, min_chip_pixels=40, encoder=ENCODER,
               model=BEST), open(f'{OUT}/config.json', 'w'), indent=1)
print(json.dumps(INFO, indent=1))
''')
code(r'''
CALF = INFO['calibration']
open(f'{OUT}/build_trt.sh', 'w').write(f"""#!/usr/bin/env bash
# Build TensorRT engines for the Spiral Blue SE-2 (Jetson Orin NX 16GB, JetPack 6.2, TensorRT 10.3).
# UNTESTED ON JETSON: written for JetPack 6.2 but never run on an Orin. Run ON the Jetson; engines are not portable.
# Full-resolution model: static input reflectance 1x8x200x200 (10 m, 2 km chip).
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
# usage: python infer.py target.tif reference.tif [--onnx {ONNX_NAME}] [--thr {BEST_THR}]
import argparse, numpy as np, rasterio, onnxruntime as ort
BANDS = [2, 3, 4, 9, 12, 13]  # 1-based in the 13-band L1C stack: B2,B3,B4,B8A,B11,B12
def load(path):
    with rasterio.open(path) as d:
        a = d.read(BANDS).astype(np.float32) / 10000.0
    assert a.shape[1:] == (200, 200), 'expects a 200x200 10 m chip; tile larger scenes'
    return a
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('target'); ap.add_argument('reference')
    ap.add_argument('--onnx', default='{ONNX_NAME}'); ap.add_argument('--thr', type=float, default={BEST_THR})
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
os.makedirs(f'{OUT}/src', exist_ok=True); os.makedirs(f'{OUT}/training_history', exist_ok=True)
shutil.copy(f'{WORK}/methane_fullres.py', f'{OUT}/src/methane_fullres.py')
for p in glob.glob(f'{RUNS}/*/history.json'): shutil.copy(p, f'{OUT}/training_history/{p.split("/")[-2]}.json')
shutil.copy(f'{RUNS}/results.json', f'{OUT}/results.json'); shutil.copy(f'{RUNS}/comparison_test.csv', f'{OUT}/comparison_test.csv')
open(f'{OUT}/requirements.txt', 'w').write('torch\nsegmentation-models-pytorch==0.6.0\nonnxruntime\nrasterio\nnumpy\nsafetensors\n')
# sanity: run infer.py's preprocessing path on a real chip pair through onnxruntime
if SMOKE_TEST or True:
    xi = mf.eval_batch(torch.from_numpy(np.array(X[te[:1], :mf.NCH_BASIC])))[0].numpy()
    print('infer-path logits shape', sess.run(None, {'reflectance': xi})[0].shape)
print(sorted(os.listdir(OUT)))
''')

md('## 12. Model card\nThe card is generated from the measured numbers. It says when results come from a smoke test. Read it before uploading.')
code(r'''
def r3(v): return f'{v:.3f}'
def row(n, d): return f"| {n} | " + ' | '.join(r3(d[k]) for k in K) + ' |'
H = '| model | IoU | F1 | P | R | det | FA |\n|---|---|---|---|---|---|---|'
cmp20 = '\n'.join([H] + [row(k + ' (20 m)', v) for k, v in CPU.items()] +
                  [row(f'{n} @0.5', RES[n]['test@0.5/20m']) for n in RES] +
                  [row(f'**{BEST} @{BEST_THR}** (selected)', RES[BEST]['test@thr/20m'])])
t10 = '\n'.join([H] + [row(f'{n} @{RES[n]["threshold"]}', RES[n]['test@thr/10m']) for n in RES])
v10 = '\n'.join([H] + [row(f'{n} @{RES[n]["threshold"]}', RES[n]['val@thr/10m']) for n in RES])
lat = ', '.join(f'{k}: {v:.1f} ms' for k, v in INFO.items() if k.endswith('_ms') or '_ms_' in k)
warn = ('\n> **SMOKE TEST BUILD: these numbers come from a 1-epoch run on a handful of chips and mean nothing. '
        'Do not upload this card.**\n') if SMOKE_TEST else ''
card = f"""---
license: cc-by-nc-sa-4.0
library_name: segmentation-models-pytorch
pipeline_tag: image-segmentation
tags: [methane, sentinel-2, remote-sensing, edge-ai, jetson, tensorrt, onnx, model-soup]
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

## Input contract

- `reflectance`: float32 `1×8×200×200`, L1C TOA reflectance (DN / 10000) at 10 m (2 km × 2 km). Channels: target B2, B3, B4, B8A,
  B11, B12, then reference B11, B12. The 20 m bands are used as MethaneSET provides them, resampled to 10 m.
- `plume_logits`: float32 `1×1×200×200`. Apply a sigmoid, then threshold at **{BEST_THR}**, which was picked on val by {THRESHOLD_METRIC.upper()}. A chip
  counts as flagged at ≥40 pixels (0.4 ha).
- A SWIR ratio channel `log(B12/B11)_target − log(B12/B11)_reference` (mean-centred) is computed inside the graph in fp32.
  The input is reflect-padded to 224 inside the graph and cropped back to 200.
- The channels and scaling are the same as the 20 m model. Only the spatial size differs.

## Training

The data is [MethaneSET](https://huggingface.co/datasets/tacofoundation/methaneset) `methaneset-s2-finetune` v1.4.0. It has 3,605 chips (200×200 at
10 m) in the official site-grouped splits of 2,362 train, 193 val and 1,050 test, and is licensed CC BY-NC-SA 4.0. It was trained on {GPU} with AMP={AMP}, with the SWIR head in fp32.

- `fr_base`: ImageNet init, {BASE_EPOCHS} epochs, lr {LR}, rot90 and flips. This is the CPU `base` recipe at 10 m.
- `fr_neg`: a fine-tune of `fr_base` for {NEG_FT_EPOCHS} epochs at lr {FT_LR}. {int(NEG_P * 100)} % of chips are replaced by plume-free pairs (bg1 vs bg0).
- `fr_long`: {LONG_EPOCHS} epochs with rot90, flips, ±10 % brightness jitter and {int(NEG_P * 100)} % negatives. Extra references were {'on' if EXTRA_REFS else 'off'}: bg0 was swapped for
  bg1–bg3 with p={EXTRA_REF_P}, negatives drew any reference against another, and references whose own enhancement exceeds {NEG_MAX_METHANE} were not
  used as negatives.
- `fr_ft_s*`: {len(SOUP_RUNS)} fine-tunes of `fr_long` with different seeds, {SOUP_EPOCHS} epochs each at lr {FT_LR}. These are averaged as a uniform soup and as a greedy soup on val.
  The greedy soup kept {RES['greedy_soup']['ingredients']}.
- The released weights are `{BEST}`, selected on val by {SELECT_METRIC} at each model's val-picked threshold.

## Results (test split, measured)

Like-for-like with the CPU model, on the **20 m-equivalent** grid. The 10 m probabilities are 2×2-mean-pooled and scored
against the CPU model's 20 m labels (pooled fraction ≥0.5); det and FA use ≥10 px (0.4 ha).

{cmp20}

Native 10 m, at each model's val-picked threshold:

{t10}

Val split, 10 m, at each model's val-picked threshold (all selection was made here):

{v10}

- `det` is the share of plume chips with ≥0.4 ha predicted that overlaps the label. `FA` is the share of plume-free pairs (bg1 vs bg0 of the same site)
  with ≥0.4 ha predicted. The references come from known super-emitter sites and may hold real, unlabelled plumes, so FA is an
  upper bound on true false alarms.
- These are single runs with no confidence intervals. Treat differences of about 1 IoU point as noise.

## Size and speed

- {INFO['params']:,} parameters; {INFO.get('gmacs', float('nan')):.2f} GMACs per 200×200 chip (fvcore); ONNX {INFO['onnx_bytes'] / 1e6:.1f} MB (opset 17, static).
- ONNX vs PyTorch: max abs logit diff {INFO['onnx_max_abs_diff_logits']:.1e}, mask agreement {INFO['onnx_mask_agreement'] * 100:.2f} % on {INFO['parity_chips']} test chips.
- Latency (batch 1): {lat}. CPU: {INFO['cpu']}. Jetson Orin NX FP16/INT8 was **not measured**; see `build_trt.sh`.
- The 10 m model does about 3× the compute of the 20 m model per 2 km chip.

## Files

`model.safetensors`, `{ONNX_NAME}`, `config.json` (input contract and threshold), `infer.py`, `build_trt.sh` (untested on Jetson),
`{INFO['calibration']}` (INT8 calibration inputs, float16), `results.json` (with full threshold sweeps), `comparison_test.csv`,
`training_history/`, `training_curves.png`, `threshold_sweep_val.png`, `example_predictions.png`, and `src/methane_fullres.py`.

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
print(card[:3000])
''')

md(f'## 13. Publish the Trackio dashboard (off by default)\nThis creates or updates the **new** static Space `sighmon/methane-se2-fullres-trackio`, which is public. A free account cannot host the live Gradio dashboard, so it is published as a static Space and has to be re-synced after new runs.')
code(r'''
if SYNC_TRACKIO and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    subprocess.run(['trackio', 'sync', '--project', TRACKIO_PROJECT, '--space-id', TRACKIO_SPACE, '--sdk', 'static', '--force'], check=True)
    print('dashboard:', f'https://huggingface.co/spaces/{TRACKIO_SPACE}')
else:
    print('SYNC_TRACKIO is False (or smoke test): nothing published. Local Trackio DB:', os.environ['TRACKIO_DIR'])
''')

md('## 14. Upload to a new model repo (off by default)\nThis creates the **new** repo `sighmon/methane-plume-se2-fullres`, which is public, and uploads `hf_upload/`. The existing `sighmon/methane-plume-se2` is not touched. Check the comparison table and the model card first, then set `UPLOAD = True` and run this cell.')
code(r'''
if UPLOAD and not SMOKE_TEST:
    from huggingface_hub import HfApi
    assert HF_OK, 'log in first (section 4)'
    api = HfApi(); api.create_repo(HF_REPO, repo_type='model', exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=HF_REPO, repo_type='model',
                      commit_message=f'Full-resolution (10 m) model: {BEST}, threshold {BEST_THR}')
    print('uploaded:', f'https://huggingface.co/{HF_REPO}')
else:
    print('UPLOAD is False (or smoke test): nothing uploaded. Files ready in', OUT)
    print(sorted(os.listdir(OUT)))
''')

nb.cells = C
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
               'language_info': {'name': 'python'}, 'accelerator': 'GPU',
               'colab': {'provenance': [], 'gpuType': 'T4', 'machine_shape': 'hm'}}
nbf.validate(nb)
nbf.write(nb, '/workspace/methane/colab/methane_fullres_colab.ipynb'); print('written', len(C), 'cells')
