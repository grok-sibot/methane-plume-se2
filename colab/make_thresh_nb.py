"""Generate fr_neg_thresholds_colab.ipynb: val-calibrated operating thresholds for the released full-res model (fr_neg).
Run from this folder:
    python make_thresh_nb.py        -> writes fr_neg_thresholds_colab.ipynb next to it
Embeds methane_fullres.py, synplume.py and thresholds.py (all in this folder) with %%writefile. No training, no
synthetic caches, no uploads, no Trackio."""
import os, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
MODS = {f: open(f'{HERE}/{f}').read() for f in ('methane_fullres.py', 'synplume.py', 'thresholds.py')}
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s, tags=None):
    c = nbf.v4.new_code_cell(s.strip('\n'))
    if tags: c.metadata['tags'] = tags
    C.append(c)

md(r'''
# Operating thresholds for the released full-res model (`fr_neg`) (Colab, T4 or CPU)

The released model ([`sighmon/methane-plume-se2-fullres`](https://huggingface.co/sighmon/methane-plume-se2-fullres)) is documented at one
pixel threshold, 0.5. At 0.5 it flags about a third of real plume-free 2024 scenes (round 2: 35.6 % of the 8,740 `real2024_strict`
pairs). This notebook picks **operating thresholds on 2021 val only** and reports what they do on the 2024 test split. Nothing is retrained.

**Two thresholds, one network output.** The contract is unchanged (`reflectance` 1×8×200×200 → `plume_logits` 1×1×200×200).
```
p  = sigmoid(plume_logits);  pp = avg_pool2d(p, 2)      # 20 m grid
scene_score = max(pp)                                    # stage 1: plume present?  scene_score >= T_scene
mask = (pp >= t_px) if scene_score >= T_scene else empty # stage 2: where?
```
* `T_scene`: the threshold giving 10 / 5 / 1 % false alarms on 2,000 **2021 val** plume-free pairs (same pairs as round 2).
* `t_px`: the pixel threshold with the best pixel IoU on the 193 **2021 val** plume chips (20 m grid).

Why two: the scene score is the *maximum* of ~10,000 pooled pixels, so false-alarm-calibrated values sit near 0.99. Used as a pixel
threshold, 0.9966 shrank every mask to its few most confident pixels (round 2: unseen IoU 0.03). Gating by scene and masking at a
normal pixel threshold keeps the mask quality and cuts the false alarms.

**Reported on test (2024), never used to choose anything:** scene detect and hit rates on all / seen / unseen-site plumes, the real
false-alarm rate on the 2024 plume-free sets, two-stage mask IoU (end-to-end: an unflagged chip predicts an empty mask; and on flagged
chips only), and site-clustered 95 % bootstrap intervals. Also checks that the round-2 `fr_neg` numbers are reproduced.

**Limits (stated in the output too).** The split is temporal, not site-grouped: 799 of the 1,050 test plumes are at sites seen in
training. The val false-alarm pairs are 2021 scenes at any non-unseen-test site, the test ones are 2024 scenes at sites in no
training split, so the val FA rate need not transfer (round 2: 5 % on val gave 2.8 % on `real2024_strict`). 193 val plumes and 251
unseen-site test plumes make every rate uncertain by several points.

**How to run:** Runtime → Change runtime type → **T4 GPU** (CPU works, slower). No token needed: the weights and data are public.
*Runtime → Run all.* Results: `/content/methane-thresholds/out/thresholds.json` (send that file) and `config_thresholds.json`.

| step | measured: box CPU, 8 vCPU, fp32 (25 Nov 2026) | T4 (estimate) |
|---|---|---|
| fine-tune data: 6.2 GB zips → 8.9 GB cache | reused an existing cache | ~4–6 min (skipped if a cache exists) |
| plume-free pair caches: 2,000 val + 10,499 test pairs, HTTP range reads, 8.1 GB downloaded, 8.0 GB on disk | 3.5 min (60 pairs/s) | ~4–10 min (round 2 on Colab: ~21 pairs/s) |
| scoring 193 val + 1,050 test chips (each with its plume-free pair) and 12,499 pairs, fp32 | 7.4 min | ~2 min |
| **total** | **11.3 min** | **~10–20 min**, ~17 GB peak disk |

The box run reproduced the round-2 `fr_neg` numbers (val thresholds to 5 decimals, test IoU at 0.5 to 0.0001).
''')

md('## 1. Config\nThe papermill `parameters` cell. `SMOKE_TEST = True` runs every cell on a few chips (CPU is fine); its numbers mean nothing.')
code(r'''
SMOKE_TEST = False          # True: tiny slices; proves the notebook runs. Numbers are meaningless.
USE_DRIVE = False           # True: keep the caches on Google Drive too (resume after a disconnect)
WEIGHTS = 'released'        # 'released': model.safetensors from HF_MODEL (public); or a local path to a .safetensors / .pt state dict
HF_MODEL = 'sighmon/methane-plume-se2-fullres'
CACHE_PATH = None           # optional existing 10 m fine-tune cache (cache10m.npy, 31 channels)
VAL_FA_N = 2000             # 2021 plume-free pairs for T_scene; 2000 with random_state 0 is the round-2 set
FA_N = None                 # None: every 2024 pair of the test FA sets (8,740 strict + 1,759 more at unseen-test sites)
VAL_FAS = [0.10, 0.05, 0.01]  # val false-alarm targets for T_scene
RECOMMENDED = '5%'          # which operating point config_thresholds.json marks as the default
PX_GRID = [round(0.05 * i, 2) for i in range(1, 20)]   # candidate pixel thresholds (0.05 ... 0.95)
PX_PICK_GRID = '20m'        # t_px is picked by val pixel IoU on this grid
AMP = False                 # fp32 by default: thresholds near 0.997 should be read at full precision, like the fp32 ONNX
BOOT_N = 1000               # site-clustered bootstrap resamples for the 95 % intervals
READ_THREADS = 128          # concurrent HTTP range reads
WORKERS = None              # DataLoader workers; None: min(4, cpu_count)
DISK_RESERVE_GB = 5
DELETE_ZIPS_AFTER_USE = True
WORK = None                 # None: /content/methane-thresholds on Colab, ./methane-thresholds elsewhere
FT_WORK = None              # where an earlier run left data/cache10m.npy; None: /content/methane-fullres on Colab
''', tags=['parameters'])

md('## 2. Setup')
code(r'''
import os, sys, subprocess, json, time, shutil, glob, importlib, importlib.util, hashlib
T0 = time.time(); TIMES = {}
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
PKGS = {'segmentation_models_pytorch': 'segmentation-models-pytorch==0.6.0', 'rasterio': 'rasterio', 'safetensors': 'safetensors',
        'psutil': 'psutil', 'huggingface_hub': 'huggingface_hub>=0.34', 'pyarrow': 'pyarrow', 'matplotlib': 'matplotlib', 'requests': 'requests'}
MISSING = [p for m, p in PKGS.items() if importlib.util.find_spec(m) is None]
if MISSING:
    print('installing missing packages:', MISSING); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *MISSING], check=True)
else: print('all packages already importable: skipping pip')
import numpy as np, pandas as pd, torch, psutil
if SMOKE_TEST: VAL_FA_N, FA_N, BOOT_N = 16, 24, 20
WORK = WORK or ('/content/methane-thresholds' if IN_COLAB else os.path.abspath('methane-thresholds'))
FT_WORK = FT_WORK or ('/content/methane-fullres' if IN_COLAB else os.path.abspath('methane-fullres'))
DATA, OUT = f'{WORK}/data', f'{WORK}/out'
for d in (DATA, OUT): os.makedirs(d, exist_ok=True)
DRIVE = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive; drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/methane-thresholds'; os.makedirs(f'{DRIVE}/data', exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = WORKERS if WORKERS is not None else min(4, os.cpu_count() or 2)
GPU = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'none (CPU)'
print(f'torch {torch.__version__} | device {DEV} | GPU {GPU} | cpus {os.cpu_count()} | RAM {psutil.virtual_memory().total / 1e9:.0f} GB '
      f'| free disk {shutil.disk_usage(WORK).free / 1e9:.0f} GB | smoke {SMOKE_TEST}')
def drive_get(name):
    if DRIVE and not os.path.exists(f'{DATA}/{name}'):
        for f in glob.glob(f'{DRIVE}/data/{name}*'): print('restoring', os.path.basename(f), 'from Drive'); shutil.copy(f, DATA)
def drive_put(name):
    if DRIVE:
        for f in glob.glob(f'{DATA}/{name}*'):
            if not f.endswith('.tmp') and not os.path.exists(f'{DRIVE}/data/{os.path.basename(f)}'): shutil.copy(f, f'{DRIVE}/data/'); print('saved', os.path.basename(f), 'to Drive')
def gb(p): return os.path.getsize(p) / 1e9 if os.path.exists(p) else 0.0
def free_gb(): return shutil.disk_usage(WORK).free / 1e9
def need_disk(need, what):
    f = free_gb(); ok = f >= need + DISK_RESERVE_GB
    print(f'disk check, {what}: needs ~{need:.1f} GB + {DISK_RESERVE_GB} GB reserve, {f:.1f} GB free -> {"ok" if ok else "NOT ENOUGH"}'); return ok
''')

md('## 3. Code\n`methane_fullres.py` (model, pair building, pixel metrics) and `synplume.py` (range reads, pair caches, scene scores) as in round 2, and `thresholds.py` (the two-stage read-out and its metrics).')
for f, s in MODS.items(): code(f'%%writefile {f}\n' + s)
code(r'''
sys.path.insert(0, os.getcwd())
import methane_fullres as mf; importlib.reload(mf)
import synplume as sp; importlib.reload(sp)
import thresholds as th; importlib.reload(th)
for f in ('methane_fullres.py', 'synplume.py', 'thresholds.py'): shutil.copy(f, f'{OUT}/{f}')
''')

md('## 4. Fine-tune data (MethaneSET S2 finetune v1.4.0): 10 m cache, seen / unseen test sites\nAs round 2 section 5, without the `ch4` leaf. Reuses an existing `cache10m.npy` (`CACHE_PATH` or `FT_WORK/data`). Seen = a train chip has the same `site:location_name` and an emission point within 1 km.')
code(r'''
from huggingface_hub import hf_hub_download
t = time.time()
REPO, SUB = 'tacofoundation/methaneset', 'methaneset-s2-finetune'
FT_DATA = f'{FT_WORK}/data' if os.path.exists(f'{FT_WORK}/data') else DATA
CACHE = CACHE_PATH or (f'{FT_DATA}/cache10m.npy' if os.path.exists(f'{FT_DATA}/cache10m.npy') else f'{DATA}/cache10m.npy')
if not SMOKE_TEST: drive_get('cache10m.npy')
meta_files = {f: hf_hub_download(REPO, f'{SUB}/.tacocat/{f}', repo_type='dataset', local_dir=DATA) for f in ['level0.parquet', 'level1.parquet']}
l0_full = pd.read_parquet(meta_files['level0.parquet']).reset_index(drop=True)
l1 = pd.read_parquet(meta_files['level1.parquet'])
def hav(a, b, c, d):
    p = np.pi / 180; x = np.sin((c - a) * p / 2) ** 2 + np.cos(a * p) * np.cos(c * p) * np.sin((d - b) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))
_tr = l0_full[l0_full.split == 'train']; _te = l0_full[l0_full.split == 'test']
SEEN = {r['id']: bool(((_tr['site:location_name'].values == r['site:location_name']) &
                       (hav(r['emission:lat'], r['emission:lon'], _tr['emission:lat'].values, _tr['emission:lon'].values) <= 1.0)).any())
        for _, r in _te.iterrows()}
UNSEEN_SITES = set(_te[~_te.id.map(SEEN)]['site:location_name'])
print(f'test chips {len(_te)}: unseen {sum(not v for v in SEEN.values())} at {len(UNSEEN_SITES)} sites')
l0 = l0_full
SMALL = ['Oman', 'Poland', 'Russian_Federation', 'Venezuela', 'Argentina', 'China', 'Jordan']
if SMOKE_TEST:
    l0 = l0[l0['internal:source_file'].isin([f'{SUB}_{c}.tacozip' for c in SMALL])].reset_index(drop=True)
    CACHE = f'{DATA}/cache10m_smoke.npy'
ZIPDIR = f'{FT_DATA}/{SUB}' if os.path.exists(f'{FT_DATA}/{SUB}') else f'{DATA}/{SUB}'
meta = l0[[c for c in mf.META_COLS if c in l0.columns]].copy()
if SMOKE_TEST: meta['split'] = [['train', 'train', 'validation', 'test'][i % 4] for i in range(len(meta))]
if not os.path.exists(CACHE):
    if not need_disk(1.0 if SMOKE_TEST else 15.1, 'fine-tune zips (6.2 GB) + 10 m cache (8.9 GB)'):
        raise SystemExit('Not enough disk for the fine-tune cache. Free space, or point CACHE_PATH / FT_WORK at an existing cache10m.npy.')
    for z in sorted(l0['internal:source_file'].unique()):
        if not os.path.exists(f'{ZIPDIR}/{z}'): hf_hub_download(REPO, f'{SUB}/{z}', repo_type='dataset', local_dir=os.path.dirname(ZIPDIR))
    mf.build_cache(l0, l1, ZIPDIR, CACHE, workers=os.cpu_count())
    if not SMOKE_TEST: drive_put('cache10m.npy')
    if DELETE_ZIPS_AFTER_USE and ZIPDIR.startswith(DATA) and os.path.isdir(ZIPDIR): shutil.rmtree(ZIPDIR); print('deleted the fine-tune zips: the cache is built')
XFT = mf.load_cache(CACHE, to_ram=False)
assert XFT.shape[1:] == (31, 200, 200) and len(XFT) == len(meta), (XFT.shape, len(meta))
va = np.where(meta.split == 'validation')[0]; te = np.where(meta.split == 'test')[0]
seen_flag = meta.id.map(SEEN)
TE = {'all': te, 'seen': te[seen_flag.values[te] == True], 'unseen': te[seen_flag.values[te] == False]}
if SMOKE_TEST and len(TE['unseen']) == 0: TE['unseen'] = te[::2]; TE['seen'] = te[1::2]; print('smoke: no real unseen chips in the slice, using every 2nd test chip as a stand-in')
SITE = meta['site:location_name'].astype(str).values
print('fine-tune val/test', len(va), len(te), '| test seen/unseen', len(TE['seen']), len(TE['unseen']), f'| cache {gb(CACHE):.1f} GB')
TIMES['finetune data'] = time.time() - t
''')

md('## 5. Plume-free pairs (MethaneSET S2 pretraining): val and test false-alarm sets\nExactly the round-2 sets.\n* **val FA** (chooses `T_scene`): 2021 pairs, not at unseen-test sites, a random `VAL_FA_N` (`random_state=0`).\n* **test FA** (reporting only): 2024 pairs at sites in neither the fine-tune train split nor pretraining train (`real2024_strict`, 8,740; `real2024_strict_unseen` = those at unseen-test sites), plus every other 2024 pair at an unseen-test site (`real2024_unseen_all`, 3,066).')
code(r'''
t = time.time()
PSUB = 'methaneset-s2-pretraining'
pf = {f: hf_hub_download(REPO, f'{PSUB}/.tacocat/{f}', repo_type='dataset', local_dir=DATA) for f in ['level0.parquet', 'level1.parquet']}
p0 = pd.read_parquet(pf['level0.parquet']).reset_index(drop=True)
P1 = sp.leaf_index(pd.read_parquet(pf['level1.parquet'], columns=['id', 'internal:offset', 'internal:size', 'internal:source_file', 'internal:relative_path']))
atU = p0['site:location_name'].isin(UNSEEN_SITES)
ptr_all = p0[p0.split == 'train']
pva = p0[(p0.split == 'validation') & ~atU]
VAL_ROWS = pva.sample(min(VAL_FA_N, len(pva)), random_state=0).reset_index(drop=True)
pte = p0[p0.split == 'test']
TRAINED_SITES = set(l0_full[l0_full.split == 'train']['site:location_name']) | set(ptr_all['site:location_name'])
strict = pte[~pte['site:location_name'].isin(TRAINED_SITES)]
extraU = pte[pte['site:location_name'].isin(UNSEEN_SITES) & ~pte.index.isin(strict.index)]
print(f'2024 FA pairs: strict {len(strict)} (at unseen-test sites {strict["site:location_name"].isin(UNSEEN_SITES).sum()}, '
      f'{strict["site:location_name"].nunique()} sites) + {len(extraU)} more at unseen-test sites = {len(strict) + len(extraU)}')
if FA_N:
    k = min(FA_N, len(strict) + len(extraU)); ns = max(1, int(round(k * len(strict) / (len(strict) + len(extraU)))))
    strict = pd.concat([strict[strict['site:location_name'].isin(UNSEEN_SITES)].head(max(1, ns // 2)),
                        strict[~strict['site:location_name'].isin(UNSEEN_SITES)].head(ns - max(1, ns // 2))]); extraU = extraU.head(max(1, k - ns))
FA_ROWS = pd.concat([strict.assign(_strict=True), extraU.assign(_strict=False)]).reset_index(drop=True)
FA_ROWS['_unseen'] = FA_ROWS['site:location_name'].isin(UNSEEN_SITES)
assert set(FA_ROWS.split) == {'test'} and set(VAL_ROWS.split) == {'validation'}
PAIR_GB = 8 * 200 * 200 * 2 / 1e9; sfx = '_smoke' if SMOKE_TEST else ''
need = sum(0.0 if os.path.exists(f'{DATA}/{n}{sfx}_{len(r)}.npy') else len(r) * PAIR_GB for n, r in (('pairs_valfa', VAL_ROWS), ('pairs_testfa', FA_ROWS)))
if not need_disk(need, f'pair caches ({len(VAL_ROWS)} val FA + {len(FA_ROWS)} test FA)'): raise SystemExit('Not enough disk for the pair caches.')
RD = sp.LeafReader(local_dirs=[DATA]); RD.bytes = RD.remote = 0
CACHES = {}
for name, rows in (('pairs_valfa', VAL_ROWS), ('pairs_testfa', FA_ROWS)):
    p = f'{DATA}/{name}{sfx}_{len(rows)}.npy'
    if not SMOKE_TEST: drive_get(os.path.basename(p))
    CACHES[name] = sp.build_pair_cache(rows, P1, PSUB, p, RD, workers=READ_THREADS)
    if not SMOKE_TEST: drive_put(os.path.basename(p))
XV, bad_v = CACHES['pairs_valfa']; XF, bad_f = CACHES['pairs_testfa']
VAL_NEG = np.where(~bad_v)[0]
FA_SETS = {'real2024_strict': np.where(FA_ROWS._strict.values & ~bad_f)[0],
           'real2024_strict_unseen': np.where(FA_ROWS._strict.values & FA_ROWS._unseen.values & ~bad_f)[0],
           'real2024_unseen_all': np.where(FA_ROWS._unseen.values & ~bad_f)[0]}
TIMES['pair caches'] = time.time() - t; DL_GB = RD.bytes / 1e9
print({k: len(v) for k, v in FA_SETS.items()}, f'| val FA {len(VAL_NEG)} | downloaded {DL_GB:.2f} GB in {TIMES["pair caches"] / 60:.1f} min')
''')

md('## 6. Weights\n`WEIGHTS = \'released\'`: the public `model.safetensors` of the released model, so the thresholds belong to exactly the file people download. Its SHA-256 is recorded.')
code(r'''
if WEIGHTS == 'released': WPATH = hf_hub_download(HF_MODEL, 'model.safetensors', local_dir=f'{DATA}/released')
else: WPATH = WEIGHTS
if WPATH.endswith('.safetensors'):
    from safetensors.torch import load_file; SD = load_file(WPATH)
else: SD = torch.load(WPATH, map_location='cpu')
W_SHA = hashlib.sha256(open(WPATH, 'rb').read()).hexdigest()
MODEL = sp.load_model('plain', SD, DEV)
AMP_DTYPE = mf.amp_dtype_for(DEV) if AMP else None
print('weights', WPATH, '| sha256', W_SHA[:16], '| params', sp.n_params(MODEL), '| precision', AMP_DTYPE or 'fp32')
''')

md('## 7. Choose on val (2021) only\n`T_scene` at each `VAL_FAS` target from the val plume-free pairs; `t_px` = best val pixel IoU. Written to `out/val_choice.json` **before** any test chip or 2024 pair is scored.')
code(r'''
t = time.time()
VAL_TAB = th.chip_table(MODEL, XFT, va, DEV, AMP_DTYPE, workers=WORKERS, px_grid=PX_GRID)
VAL_NEG_S = sp.scene_scores(MODEL, XV, VAL_NEG, DEV, AMP_DTYPE)['score']
T_SCENE = {f'{int(round(f * 100))}%': sp.thr_at_fa(VAL_NEG_S, f) for f in VAL_FAS}
T_PX, VAL_IOU_CURVE = th.pick_px(VAL_TAB, PX_PICK_GRID)
VAL_SUMMARY = {op: dict(T_scene=T, T_scene_logit_of_pooled_prob=th.logit(T), val_fa=float((VAL_NEG_S >= T).mean()), **th.scene_rates(VAL_TAB, T),
                        mask=th.mask_metrics(VAL_TAB, T, T_PX, PX_PICK_GRID)) for op, T in T_SCENE.items()}
VAL_SUMMARY['0.5'] = dict(T_scene=0.5, val_fa=float((VAL_NEG_S >= 0.5).mean()), **th.scene_rates(VAL_TAB, 0.5), mask=th.mask_metrics(VAL_TAB, 0.5, 0.5, PX_PICK_GRID))
VAL_CHOICE = dict(written=time.strftime('%Y-%m-%d %H:%M:%S %Z'), smoke_test=SMOKE_TEST, weights_sha256=W_SHA, n_val_plumes=int(len(va)), n_val_fa=int(len(VAL_NEG_S)),
                  T_scene=T_SCENE, t_px=T_PX, t_px_grid=PX_PICK_GRID, val_iou_by_t_px=VAL_IOU_CURVE, val=VAL_SUMMARY, precision=str(AMP_DTYPE or 'fp32'))
json.dump(VAL_CHOICE, open(f'{OUT}/val_choice.json', 'w'), indent=1, default=float)
print('T_scene (val FA target -> threshold):', {k: round(v, 5) for k, v in T_SCENE.items()}, f'| t_px = {T_PX} (val IoU {VAL_IOU_CURVE[T_PX]:.3f}; at 0.5: {VAL_IOU_CURVE.get(0.5, float("nan")):.3f})')
for op, v in VAL_SUMMARY.items():
    print(f'  val {op:>4s}: FA {v["val_fa"]:.3f} detect {v["detect"]:.3f} hit {v["hit"]:.3f} | mask IoU end-to-end {v["mask"]["end_to_end"]["iou"]:.3f}')
TIMES['val'] = time.time() - t
''')

md('## 8. Test (2024): score once, then read every operating point\nRows: the current card (one threshold, 0.5), then each val-chosen `T_scene` with the val-chosen `t_px` (and with 0.5 for comparison). 95 % intervals resample **sites** (test plumes by `site:location_name`, plume-free pairs by their site).')
code(r'''
t = time.time()
TEST_TAB = th.chip_table(MODEL, XFT, te, DEV, AMP_DTYPE, workers=WORKERS, px_grid=PX_GRID)
pos_of = {i: k for k, i in enumerate(te)}
SUBTAB = {s: th.subset(TEST_TAB, np.array([pos_of[i] for i in idx], dtype=int)) for s, idx in TE.items()}
SUBSITE = {s: SITE[idx] for s, idx in TE.items()}
NEG = {fs: (sp.scene_scores(MODEL, XF, idx, DEV, AMP_DTYPE)['score'] if len(idx) else np.zeros(0)) for fs, idx in FA_SETS.items()}
NEG_SITE = {fs: FA_ROWS['site:location_name'].astype(str).values[idx] for fs, idx in FA_SETS.items()}
TIMES['test scoring'] = time.time() - t
OPS = [('0.5 (current card)', 0.5, 0.5)] + [(f'val {op} FA', T, T_PX) for op, T in T_SCENE.items()] + \
      ([(f'val {op} FA, mask 0.5', T, 0.5) for op, T in T_SCENE.items()] if T_PX != 0.5 else [])
def op_row(name, T, tp):
    r = dict(op=name, T_scene=float(T), t_px=float(tp), subsets={}, real_fa={})
    for s, tab in SUBTAB.items():
        sr = th.scene_rates(tab, T); m20 = th.mask_metrics(tab, T, tp, '20m'); m10 = th.mask_metrics(tab, T, tp, '10m')
        r['subsets'][s] = dict(**sr, iou20_end_to_end=m20['end_to_end']['iou'], iou20_flagged=m20['flagged']['iou'], f1_20_end_to_end=m20['end_to_end']['f1'],
                               precision20_end_to_end=m20['end_to_end']['precision'], recall20_end_to_end=m20['end_to_end']['recall'],
                               iou10_end_to_end=m10['end_to_end']['iou'], n_flagged=m20['n_flagged'], chip_fa_bg1=m20['chip_fa'], chip_fa_bg1_scene_only=m20['chip_fa_scene_only'],
                               hit_ci95=th.cluster_boot(SUBSITE[s], lambda rows, tab=tab: th.scene_rates(th.subset(tab, rows), T)['hit'], BOOT_N),
                               iou20_end_to_end_ci95=th.cluster_boot(SUBSITE[s], lambda rows, tab=tab: th.mask_metrics(th.subset(tab, rows), T, tp, '20m')['end_to_end']['iou'], BOOT_N))
    for fs, neg in NEG.items():
        if len(neg): r['real_fa'][fs] = dict(rate=float((neg >= T).mean()), n=int(len(neg)),
                                             ci95=th.cluster_boot(NEG_SITE[fs], lambda rows, neg=neg: float((neg[rows] >= T).mean()), BOOT_N))
    return r
RESULTS = [op_row(*o) for o in OPS]
TIMES['test metrics'] = time.time() - t - TIMES['test scoring']
rows = []
for r in RESULTS:
    for s, v in r['subsets'].items():
        rows.append(dict(op=r['op'], T_scene=r['T_scene'], t_px=r['t_px'], plumes=s, n=v['n'], detect=v['detect'], hit=v['hit'],
                         hit_lo=v['hit_ci95'][0], hit_hi=v['hit_ci95'][1], iou20_e2e=v['iou20_end_to_end'], iou_lo=v['iou20_end_to_end_ci95'][0],
                         iou_hi=v['iou20_end_to_end_ci95'][1], iou20_flagged=v['iou20_flagged'], iou10_e2e=v['iou10_end_to_end'], chip_fa_bg1=v['chip_fa_bg1'],
                         real_fa_strict=r['real_fa'].get('real2024_strict', {}).get('rate', np.nan), real_fa_unseen_all=r['real_fa'].get('real2024_unseen_all', {}).get('rate', np.nan)))
TABLE = pd.DataFrame(rows); TABLE.to_csv(f'{OUT}/thresholds_table.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(TABLE.round(3).to_string(index=False))
''')

md('## 9. Reproduction check against round 2\nThe round-2 run (`syn_results.json`, Colab T4, AMP fp16, `fr_neg/final.pt`) recorded these `fr_neg` numbers. Small differences are expected from fp32 here vs fp16 there; a large one means different weights or data.')
code(r'''
R2 = {'val T_scene 10%': 0.9943740963935852, 'val T_scene 5%': 0.9966223239898682, 'val T_scene 1%': 0.9983797669410706,
      'test IoU@0.5 20m all': 0.5206476378975718, 'test IoU@0.5 20m unseen': 0.4354487470015365,
      'real FA@0.5 real2024_strict': 0.3562929061784897, 'hit@0.5 unseen': 0.7370517928286853,
      'hit@val5% unseen': 0.3107569721115538, 'real FA@val5% real2024_strict': 0.028146453089244853}
r05 = RESULTS[0]; r5 = next(r for r in RESULTS if r['op'] == 'val 5% FA')
NOW = {'val T_scene 10%': T_SCENE.get('10%'), 'val T_scene 5%': T_SCENE.get('5%'), 'val T_scene 1%': T_SCENE.get('1%'),
       'test IoU@0.5 20m all': r05['subsets']['all']['iou20_end_to_end'], 'test IoU@0.5 20m unseen': r05['subsets']['unseen']['iou20_end_to_end'],
       'real FA@0.5 real2024_strict': r05['real_fa'].get('real2024_strict', {}).get('rate'), 'hit@0.5 unseen': r05['subsets']['unseen']['hit'],
       'hit@val5% unseen': r5['subsets']['unseen']['hit'], 'real FA@val5% real2024_strict': r5['real_fa'].get('real2024_strict', {}).get('rate')}
REPRO = {k: dict(round2=v, now=NOW[k], diff=(NOW[k] - v) if NOW[k] is not None else None) for k, v in R2.items()}
for k, v in REPRO.items(): print(f'{k:32s} round2 {v["round2"]:.4f} | now {v["now"] if v["now"] is None else round(v["now"], 4)} | diff {v["diff"] if v["diff"] is None else round(v["diff"], 4)}')
REPRO_OK = all(v['diff'] is not None and abs(v['diff']) <= (0.0005 if 'T_scene' in k else 0.01) for k, v in REPRO.items())
print('reproduces round 2:', REPRO_OK, '(smoke test: not expected)' if SMOKE_TEST else '')
''')

md('## 10. Plots')
code(r'''
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
neg = NEG['real2024_strict']
f, ax = plt.subplots(1, 3, figsize=(15, 4))
for s, c in zip(('all', 'seen', 'unseen'), ('k', 'tab:blue', 'tab:red')):
    if not len(neg) or not len(SUBTAB[s]['score']): continue
    ts = np.unique(np.r_[np.quantile(neg, np.linspace(0, 1, 300)), 0.5, list(T_SCENE.values())])
    ax[0].plot([(neg >= x).mean() for x in ts], [(SUBTAB[s]['score_in'] >= x).mean() for x in ts], color=c, label=f'{s} test plumes')
    for op, T in T_SCENE.items(): ax[0].plot([(neg >= T).mean()], [(SUBTAB[s]['score_in'] >= T).mean()], 'o', color=c, ms=5)
    ax[0].plot([(neg >= 0.5).mean()], [(SUBTAB[s]['score_in'] >= 0.5).mean()], 's', color=c, ms=5)
ax[0].set_xscale('log'); ax[0].set_xlabel('real 2024 false-alarm rate (real2024_strict)'); ax[0].set_ylabel('scene hit rate')
ax[0].set_title('dots: val-chosen T_scene (10/5/1 %), squares: 0.5', fontsize=9); ax[0].legend(fontsize=8)
for s, c in zip(('all', 'seen', 'unseen'), ('k', 'tab:blue', 'tab:red')):
    ax[1].plot(PX_GRID, [th.mask_metrics(SUBTAB[s], -1, x, '20m')['end_to_end']['iou'] for x in PX_GRID], color=c, label=f'test {s}')
ax[1].plot(list(VAL_IOU_CURVE), list(VAL_IOU_CURVE.values()), 'g--', label='val (chooses t_px)'); ax[1].axvline(T_PX, color='g', lw=.6)
ax[1].set_xlabel('pixel threshold t_px (20 m, no scene gate)'); ax[1].set_ylabel('pixel IoU'); ax[1].legend(fontsize=8)
Ts = np.r_[0.5, np.linspace(0.98, 0.9995, 40)]
for s, c in zip(('all', 'unseen'), ('k', 'tab:red')):
    ax[2].plot(Ts[1:], [th.mask_metrics(SUBTAB[s], T, T_PX, '20m')['end_to_end']['iou'] for T in Ts[1:]], color=c, label=f'{s}: end-to-end')
    ax[2].plot(Ts[1:], [th.mask_metrics(SUBTAB[s], T, T_PX, '20m')['flagged']['iou'] for T in Ts[1:]], color=c, ls=':', label=f'{s}: flagged chips only')
for T in T_SCENE.values(): ax[2].axvline(T, color='grey', lw=.5, ls='--')
ax[2].set_xlabel(f'T_scene (mask at t_px = {T_PX})'); ax[2].set_ylabel('pixel IoU (20 m)'); ax[2].legend(fontsize=7)
f.tight_layout(); f.savefig(f'{OUT}/thresholds.png', dpi=90); plt.close(f)
from IPython.display import Image, display; display(Image(f'{OUT}/thresholds.png'))
''')

md('## 11. Write `out/thresholds.json` and `out/config_thresholds.json`\n`config_thresholds.json` is a snippet for the model\'s `config.json`; nothing is uploaded anywhere.')
code(r'''
CONFIG = {'threshold': 0.5,
          'readout': 'p = sigmoid(plume_logits); pp = avg_pool2d(p, 2) (20 m); scene_score = max(pp); plume_present = scene_score >= scene_threshold; '
                     'mask = (pp >= mask_threshold) if plume_present else empty',
          'scene_threshold': T_SCENE.get(RECOMMENDED), 'mask_threshold': T_PX, 'mask_grid': '20m (2x2 mean-pooled probabilities)', 'min_mask_pixels_20m': 10,
          'operating_points': {op: {'scene_threshold': T, 'val_false_alarm_target': float(op.rstrip('%')) / 100,
                                    'test_real2024_strict_false_alarm': next(r for r in RESULTS if r['op'] == f'val {op} FA')['real_fa'].get('real2024_strict', {}).get('rate')}
                               for op, T in T_SCENE.items()},
          'recommended_operating_point': RECOMMENDED,
          'note': 'threshold (0.5) is the legacy single pixel threshold. Scene thresholds were chosen on 2021 val plume-free pairs only; '
                  'they are for scene_score (max of pooled probabilities), not for single pixels.'}
TIMES['total'] = time.time() - T0
SUMMARY = dict(notebook='fr_neg_thresholds_colab.ipynb', smoke_test=SMOKE_TEST, gpu=GPU, precision=str(AMP_DTYPE or 'fp32'), weights=WEIGHTS, weights_sha256=W_SHA,
               val_choice=VAL_CHOICE, test=RESULTS, reproduction_vs_round2=REPRO, reproduces_round2=REPRO_OK, config_snippet=CONFIG,
               n=dict(val_plumes=int(len(va)), val_fa=int(len(VAL_NEG_S)), test=dict((s, int(len(i))) for s, i in TE.items()), test_fa=dict((k, int(len(v))) for k, v in FA_SETS.items())),
               boot_n=BOOT_N, times_s=TIMES, downloaded_gb=DL_GB,
               caveats=['temporal split (val 2021, test 2024), not site-grouped: 799 of 1,050 test plumes are at sites seen in training',
                        'val FA pairs (2021, any non-unseen-test site) differ from the test FA pairs (2024, sites in no training split): the val FA rate need not transfer',
                        't_px picked on 193 val plume chips; T_scene at 1 % rests on ~20 val plume-free pairs above it',
                        'intervals are site-clustered percentile bootstraps over the test set only; they do not cover the val choice'])
json.dump(SUMMARY, open(f'{OUT}/thresholds.json', 'w'), indent=1, default=float)
json.dump(CONFIG, open(f'{OUT}/config_thresholds.json', 'w'), indent=1, default=float)
print(json.dumps(CONFIG, indent=1, default=float))
print({k: round(v / 60, 1) for k, v in TIMES.items()}, 'min')
print(f'wrote {OUT}/thresholds.json  <- send this file')
''')

nb['cells'] = C
nb.metadata.update({'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'accelerator': 'GPU',
                    'colab': {'provenance': [], 'gpuType': 'T4'}, 'language_info': {'name': 'python'}})
p = f'{HERE}/fr_neg_thresholds_colab.ipynb'; nbf.write(nb, p); print('wrote', p, len(C), 'cells')
