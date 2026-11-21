"""Generate synthetic_plumes_colab.ipynb (colab/PLAN_synthetic_plumes.md). Run from this folder:
    python make_syn_nb.py            -> writes synthetic_plumes_colab.ipynb next to it
The notebook embeds methane_fullres.py and synplume.py (both in this folder) with %%writefile."""
import os, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
MOD_FR = open(f'{HERE}/methane_fullres.py').read()
MOD_SYN = open(f'{HERE}/synplume.py').read()
nb = nbf.v4.new_notebook(); C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
def code(s, tags=None):
    c = nbf.v4.new_code_cell(s.strip('\n'))
    if tags: c.metadata['tags'] = tags
    C.append(c)

md(r'''
# Synthetic-plume training for the full-res methane detector (Colab, T4 + High-RAM)

Implements [`colab/PLAN_synthetic_plumes.md`](https://github.com/grok-sibot/methane-plume-se2/blob/main/colab/PLAN_synthetic_plumes.md).
Goal: better detection at **sites the model has never seen**, and an honest false-alarm rate measured on **real plume-free scenes**,
with the onboard contract unchanged (`reflectance` float32 1×8×200×200 → `plume_logits` 1×1×200×200).

| stage | what | stops the notebook if |
|---|---|---|
| 0 | data: fine-tune 10 m cache, plume-free pretraining pairs (HTTP range reads, no 36.9 GB download), WRF-LES plume bank | |
| 1 | **gate**: inject 3 t/h plumes into plume-free scenes and compare the B12/B11 dip, MBMP and label size with real plumes of 2–4.5 t/h | the injected plumes do not look like real ones |
| 2 | train `syn` (injection + enhancement-weighted loss) and `syn_mbmp` (+ in-graph MBMP channel), both from `fr_neg` | |
| 3 | evaluate `fr_neg`, `syn`, `syn_mbmp` identically: unseen / seen / all test plumes, real 2024 plume-free pairs, PR / AP, operating points at 10 / 5 / 1 % real FA; go / no-go | |
| 4 | only on **GO**: ONNX export + candidate card. Upload only with `UPLOAD = True` | |

**Leak control.** Training uses pretraining *train* pairs (2018–20, 2022–23) only, minus every pair at one of the 92 sites of the
251 unseen-site test chips. Selection uses 2021 (val) only. Nothing from 2024 is used for training or selection.

**Safe defaults.** `UPLOAD = False`, `SYNC_TRACKIO = False`, `SAVE_RUNS_PRIVATE = False`. pip is skipped when the packages already
import. Caches are memory-mapped, never copied into RAM. The HF token is never printed.

**How to run:** Runtime → Change runtime type → **T4 GPU** (High-RAM helps but is not needed). Add the `HF_TOKEN` Colab secret (read access to
`sighmon/methane-fullres-runs` is needed to pull `fr_neg`). Then *Runtime → Run all*. Set `USE_DRIVE = True` to survive a disconnect.

TIME_TABLE_PLACEHOLDER
''')

md('## 1. Config\nThis is the papermill `parameters` cell. `SMOKE_TEST = True` runs every stage on a few chips (CPU is fine); its numbers mean nothing.')
code(r'''
# ---- run mode ----
SMOKE_TEST = False          # True: tiny slices, 1 epoch of a few steps, never uploads. Proves the notebook runs; numbers are meaningless.
USE_DRIVE = False           # True: mount Google Drive; runs and Trackio live there (resume after disconnects)
DRIVE_CACHES = True         # with USE_DRIVE: also copy the built caches to Drive (~27 GB: needs that much Drive space)
SKIP_DONE = True            # reuse finished caches and runs (final.pt + best.pt)
UPLOAD = False              # True (and GO): upload the candidate to HF_CANDIDATE_REPO (private). Never touches the released model.
SYNC_TRACKIO = False        # True: publish the Trackio dashboard as a static Space (public)
SAVE_RUNS_PRIVATE = False   # True: also push syn/syn_mbmp checkpoints to the private RUNS_REPO under syn/
PULL_PRIVATE_RUNS = True    # fetch fr_neg/final.pt from the private RUNS_REPO when it is not on disk
CACHE_PATH = None           # optional existing 10 m fine-tune cache (cache10m.npy, 31 channels); reused and memory-mapped
INIT_CKPT = None            # optional local path to fr_neg/final.pt (skips the pull)
GATE_ONLY = False           # True: stop after the sanity gate (cheap first look)
SMOKE_FORCE_GO = False      # smoke test only: take the GO branch anyway so the export + card code is exercised (ignored when SMOKE_TEST is False)

# ---- data sizes ----
PRETRAIN_N = 12000          # plume-free train pairs cached (of 24,918 eligible): 7.7 GB on disk, ~7.8 GB downloaded
VAL_FA_N = 2000             # 2021 plume-free pairs for selection (val false alarms)
FA_N = None                 # None: every 2024 pair of the FA sets (8,740 strict + 1,759 more at unseen sites), 6.7 GB
GATE_N = 64                 # scenes for the injection sanity gate
BANK_K = 3                  # plumes pre-assigned per training scene (nearest sun geometry); one is drawn per use
MIN_CLEAR = 95.0            # training pairs only: minimum quality:percentage_clear
DOWNLOAD_BANK_ZIPS = True   # download the 5 GB bank once (fast) instead of ~30k small range reads; smoke uses range reads
READ_THREADS = 128          # concurrent HTTP range reads (measured ~58 pairs/s with 2 vCPUs)

# ---- injection ----
Q_MIN, Q_MAX = 500.0, 20000.0   # kg/h, log-uniform; also capped per plume so the LES downwind cut stays below the scene noise
LABEL_FLOOR_SIGMA = 'auto'  # label: enhancement at least this many x scene noise (ppb); auto: matched to real label areas in the gate
INJ_P = 0.7                 # of the plume-free half of each batch: 0.7 injected, 0.3 untouched (35 % / 15 % of the batch)

# ---- training ----
EPOCHS = 12                 # one epoch: one pass over the 2,362 real train chips (+ as many plume-free pairs)
LR = 3e-4                   # fine-tune from fr_neg
BS_REAL, BS_SYN = 8, 8      # 16 chips per step, half real
NEG_P = 0.3                 # fr_neg's recipe on the real half: 30 % become bg1-vs-bg0 plume-free pairs
ENH_WEIGHTING = True        # enhancement-weighted BCE positives (real: ch4 leaf, synthetic: injected ppb)
RUNS_TO_TRAIN = ['syn', 'syn_mbmp']
MAX_STEPS = None            # cap steps per epoch (debugging); smoke test uses 3
AMP = True
WORKERS = None              # DataLoader workers; None: min(4, cpu_count)

# ---- names ----
RUNS_REPO = 'sighmon/methane-fullres-runs'              # private, holds fr_neg/final.pt
HF_CANDIDATE_REPO = 'sighmon/methane-plume-se2-syn'     # NEW private repo, only used with UPLOAD True and a GO
TRACKIO_PROJECT = 'methane-se2-syn'
TRACKIO_SPACE = 'sighmon/methane-se2-syn-trackio'
WORK = None                 # None: /content/methane-syn on Colab, ./methane-syn elsewhere
FT_WORK = None              # where an earlier full-res run left data/ (cache10m.npy, tacozips); None: /content/methane-fullres on Colab
''', tags=['parameters'])

md('## 2. Setup')
code(r'''
import os, sys, subprocess, json, time, shutil, glob, platform, importlib, importlib.util
T0 = time.time(); TIMES = {}
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
PKGS = {'segmentation_models_pytorch': 'segmentation-models-pytorch==0.6.0', 'trackio': 'trackio', 'rasterio': 'rasterio',
        'onnx': 'onnx', 'onnxruntime': 'onnxruntime', 'onnxscript': 'onnxscript', 'safetensors': 'safetensors', 'psutil': 'psutil',
        'huggingface_hub': 'huggingface_hub>=0.34', 'pyarrow': 'pyarrow', 'matplotlib': 'matplotlib', 'requests': 'requests'}
MISSING = [p for m, p in PKGS.items() if importlib.util.find_spec(m) is None]
if MISSING:
    print('installing missing packages:', MISSING); subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *MISSING], check=True)
else: print('all packages already importable: skipping pip')
import numpy as np, pandas as pd, torch, psutil
if SMOKE_TEST:
    PRETRAIN_N, VAL_FA_N, FA_N, GATE_N, BANK_K = 32, 16, 24, 24, 2
    EPOCHS, BS_REAL, BS_SYN = 1, 4, 4; DOWNLOAD_BANK_ZIPS = False; UPLOAD = SYNC_TRACKIO = SAVE_RUNS_PRIVATE = False
MAX_STEPS = 3 if SMOKE_TEST else MAX_STEPS
WORK = WORK or ('/content/methane-syn' if IN_COLAB else os.path.abspath('methane-syn'))
FT_WORK = FT_WORK or ('/content/methane-fullres' if IN_COLAB else os.path.abspath('methane-fullres'))
DATA, OUT = f'{WORK}/data', f'{WORK}/out'
for d in (DATA, OUT): os.makedirs(d, exist_ok=True)
DRIVE = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive; drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/methane-se2-syn'; os.makedirs(f'{DRIVE}/data', exist_ok=True)
RUNS = f'{DRIVE or WORK}/runs'; os.makedirs(RUNS, exist_ok=True)
os.environ['TRACKIO_DIR'] = f'{DRIVE or WORK}/trackio'; os.makedirs(os.environ['TRACKIO_DIR'], exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = WORKERS if WORKERS is not None else min(4, os.cpu_count() or 2)
GPU = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'none (CPU)'
print(f'torch {torch.__version__} | device {DEV} | GPU {GPU} | cpus {os.cpu_count()} | RAM {psutil.virtual_memory().total / 1e9:.0f} GB '
      f'| free disk {shutil.disk_usage(WORK).free / 1e9:.0f} GB | smoke {SMOKE_TEST}')
if DEV.type != 'cuda' and not SMOKE_TEST: print('WARNING: no GPU. Runtime -> Change runtime type -> T4 GPU.')
def drive_get(name):
    """Copy a finished cache back from Drive (local disk is much faster to memory-map than Drive)."""
    if DRIVE and DRIVE_CACHES and not os.path.exists(f'{DATA}/{name}'):
        for f in glob.glob(f'{DRIVE}/data/{name}*'):
            print('restoring', os.path.basename(f), 'from Drive'); shutil.copy(f, DATA)
def drive_put(name):
    if DRIVE and DRIVE_CACHES:
        for f in glob.glob(f'{DATA}/{name}*'):
            if not f.endswith('.tmp') and not os.path.exists(f'{DRIVE}/data/{os.path.basename(f)}'):
                shutil.copy(f, f'{DRIVE}/data/'); print('saved', os.path.basename(f), 'to Drive')
def gb(p): return os.path.getsize(p) / 1e9 if os.path.exists(p) else 0.0
''')

md('## 3. Code\nThe first cell writes the full-res training module unchanged (model, pair building, pixel metrics). The second writes `synplume.py`: range reads, caches, transmittance, GPU injection, the MBMP model, the training loop, scene-level scoring and the go / no-go rule.')
code('%%writefile methane_fullres.py\n' + MOD_FR)
code('%%writefile synplume.py\n' + MOD_SYN)
code(r'''
sys.path.insert(0, os.getcwd())
import methane_fullres as mf; importlib.reload(mf)
import synplume as sp; importlib.reload(sp)
for f in ('methane_fullres.py', 'synplume.py'): shutil.copy(f, f'{OUT}/{f}')
''')

md('## 4. Hugging Face login\nUses the Colab secret `HF_TOKEN` (or the `HF_TOKEN` environment variable). The token is never printed. The datasets are public; the token is needed to pull `fr_neg` from the private repo and for the optional uploads.')
code(r'''
HF_OK = False
try:
    from huggingface_hub import login, whoami
    tok = None
    try:
        from google.colab import userdata; tok = userdata.get('HF_TOKEN')
    except Exception: pass
    if tok: login(token=tok, add_to_git_credential=False); HF_OK = True
    elif os.environ.get('HF_TOKEN'): HF_OK = True
    print('logged in as', whoami()['name']) if HF_OK else print('no HF token: PULL_PRIVATE_RUNS and uploads will not work')
    del tok
except Exception as e: print('HF login skipped:', type(e).__name__, e)
''')

md('## 5. Fine-tune data (MethaneSET S2 finetune v1.4.0): 10 m cache, `ch4` leaf, seen / unseen test sites\nReuses an earlier run\'s cache (`CACHE_PATH`, or `FT_WORK/data/cache10m.npy`) when it exists. Seen = a train chip has the same `site:location_name` and an emission point within 1 km (the rule behind `test_seen_unseen.csv`).')
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
# --- unseen-site rule on the FULL catalog (always, also in smoke mode)
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
    for z in sorted(l0['internal:source_file'].unique()):
        if not os.path.exists(f'{ZIPDIR}/{z}'): hf_hub_download(REPO, f'{SUB}/{z}', repo_type='dataset', local_dir=os.path.dirname(ZIPDIR))
    mf.build_cache(l0, l1, ZIPDIR, CACHE, workers=os.cpu_count())
    if not SMOKE_TEST: drive_put('cache10m.npy')
XFT = mf.load_cache(CACHE, to_ram=False)
assert XFT.shape[1:] == (31, 200, 200) and len(XFT) == len(meta), (XFT.shape, len(meta))
# --- MBMP `ch4` leaf (loss weights only; never a model input)
RD = sp.LeafReader(local_dirs=[os.path.dirname(ZIPDIR), DATA])
CH4P = f'{DATA}/ch4_10m{"_smoke" if SMOKE_TEST else ""}.npy'
if not SMOKE_TEST: drive_get(os.path.basename(CH4P))
CH4 = sp.build_leaf_cache(l0, sp.leaf_index(l1), 'ch4', SUB, CH4P, RD, workers=32) if ENH_WEIGHTING else None
if not SMOKE_TEST: drive_put(os.path.basename(CH4P))
if CH4 is not None:
    s_ = np.array(CH4[:min(64, len(CH4))]); print('ch4 leaf sample: p1/p50/p99 =', np.nanpercentile(s_, [1, 50, 99]).round(1), '(units not stated by MethaneSET; only per-chip ranks are used)')
tr = np.where(meta.split == 'train')[0]; va = np.where(meta.split == 'validation')[0]; te = np.where(meta.split == 'test')[0]
seen_flag = meta.id.map(SEEN)
TE = {'all': te, 'seen': te[seen_flag.values[te] == True], 'unseen': te[seen_flag.values[te] == False]}
if SMOKE_TEST and len(TE['unseen']) == 0: TE['unseen'] = te[::2]; print('smoke: no real unseen chips in the slice, using every 2nd test chip as a stand-in')
print('fine-tune train/val/test', len(tr), len(va), len(te), '| test seen/unseen', len(TE['seen']), len(TE['unseen']), f'| cache {gb(CACHE):.1f} GB')
TIMES['finetune data'] = time.time() - t
''')

md('## 6. Plume-free pretraining pairs: leak control and sampling\n* **train**: pretraining `train` (2018–20, 2022–23), not at any of the unseen-test sites, `quality:percentage_clear >= MIN_CLEAR`; a random `PRETRAIN_N`.\n* **val FA** (selection only): 2021 pairs, not at unseen-test sites; a random `VAL_FA_N`.\n* **test FA** (never used before stage 3): 2024 pairs at sites in **neither** the fine-tune train split **nor** pretraining train (`real2024_strict`, 8,740 pairs; `real2024_strict_unseen` = the ones at unseen-test sites), plus every other 2024 pair at an unseen-test site (`real2024_unseen_all`, 3,066). Our training drops all unseen-site pairs, so none of these were trained on.')
code(r'''
t = time.time()
PSUB = 'methaneset-s2-pretraining'
pf = {f: hf_hub_download(REPO, f'{PSUB}/.tacocat/{f}', repo_type='dataset', local_dir=DATA) for f in ['level0.parquet', 'level1.parquet']}
p0 = pd.read_parquet(pf['level0.parquet']).reset_index(drop=True)
P1 = sp.leaf_index(pd.read_parquet(pf['level1.parquet'], columns=['id', 'internal:offset', 'internal:size', 'internal:source_file', 'internal:relative_path']))
site = p0['site:location_name']; atU = site.isin(UNSEEN_SITES)
ptr_all = p0[(p0.split == 'train')]; ptr = ptr_all[~ptr_all['site:location_name'].isin(UNSEEN_SITES) & (ptr_all['quality:percentage_clear'] >= MIN_CLEAR)]
print(f'pretraining train {len(ptr_all)} -> dropped {ptr_all["site:location_name"].isin(UNSEEN_SITES).sum()} at '
      f'{ptr_all[ptr_all["site:location_name"].isin(UNSEEN_SITES)]["site:location_name"].nunique()} unseen-test sites, '
      f'{(ptr_all["quality:percentage_clear"] < MIN_CLEAR).sum()} below {MIN_CLEAR}% clear -> {len(ptr)} eligible')
TRAIN_ROWS = ptr.sample(min(PRETRAIN_N, len(ptr)), random_state=0).reset_index(drop=True)
pva = p0[(p0.split == 'validation') & ~atU]
VAL_ROWS = pva.sample(min(VAL_FA_N, len(pva)), random_state=0).reset_index(drop=True)
pte = p0[p0.split == 'test']
TRAINED_SITES = set(l0_full[l0_full.split == 'train']['site:location_name']) | set(ptr_all['site:location_name'])
strict = pte[~pte['site:location_name'].isin(TRAINED_SITES)]
extraU = pte[pte['site:location_name'].isin(UNSEEN_SITES) & ~pte.index.isin(strict.index)]
print(f'2024 FA pairs: strict {len(strict)} (at unseen-test sites {strict["site:location_name"].isin(UNSEEN_SITES).sum()}, '
      f'{strict["site:location_name"].nunique()} sites) + {len(extraU)} more at unseen-test sites = {len(strict) + len(extraU)}')
if FA_N:  # smoke / quick runs: keep the same proportions
    k = min(FA_N, len(strict) + len(extraU)); ns = max(1, int(round(k * len(strict) / (len(strict) + len(extraU)))))
    strict = pd.concat([strict[strict['site:location_name'].isin(UNSEEN_SITES)].head(max(1, ns // 2)),
                        strict[~strict['site:location_name'].isin(UNSEEN_SITES)].head(ns - max(1, ns // 2))]); extraU = extraU.head(max(1, k - ns))
FA_ROWS = pd.concat([strict.assign(_strict=True), extraU.assign(_strict=False)]).reset_index(drop=True)
FA_ROWS['_unseen'] = FA_ROWS['site:location_name'].isin(UNSEEN_SITES)
assert not set(TRAIN_ROWS['site:location_name']) & UNSEEN_SITES and not TRAIN_ROWS.split.ne('train').any()
assert set(FA_ROWS.split) == {'test'} and set(VAL_ROWS.split) == {'validation'}
print(f'cache plan: train {len(TRAIN_ROWS)}, val FA {len(VAL_ROWS)}, test FA {len(FA_ROWS)} pairs '
      f'= {(len(TRAIN_ROWS) + len(VAL_ROWS) + len(FA_ROWS)) * 8 * 200 * 200 * 2 / 1e9:.1f} GB float16')
TIMES['pretraining catalog'] = time.time() - t
''')

md('## 7. Build the pair caches (HTTP range reads)\nEach pair is two ~300 KB GeoTIFF leaves read straight out of the `.tacozip` files on the hub. Resumable: rerun the cell after a disconnect and it continues where it stopped.')
code(r'''
t = time.time(); RD.bytes = RD.remote = 0
sfx = '_smoke' if SMOKE_TEST else ''
CACHES = {}
for name, rows in (('pairs_train', TRAIN_ROWS), ('pairs_valfa', VAL_ROWS), ('pairs_testfa', FA_ROWS)):
    p = f'{DATA}/{name}{sfx}_{len(rows)}.npy'
    if not SMOKE_TEST: drive_get(os.path.basename(p))
    CACHES[name] = sp.build_pair_cache(rows, P1, PSUB, p, RD, workers=READ_THREADS)
    if not SMOKE_TEST: drive_put(os.path.basename(p))
XP, bad_tr = CACHES['pairs_train']; XV, bad_v = CACHES['pairs_valfa']; XF, bad_f = CACHES['pairs_testfa']
TR_PAIRS = np.where(~bad_tr)[0]; VAL_NEG = np.where(~bad_v)[0]
FA_SETS = {'real2024_strict': np.where(FA_ROWS._strict.values & ~bad_f)[0],
           'real2024_strict_unseen': np.where(FA_ROWS._strict.values & FA_ROWS._unseen.values & ~bad_f)[0],
           'real2024_unseen_all': np.where(FA_ROWS._unseen.values & ~bad_f)[0]}
TIMES['pair caches'] = time.time() - t
DL_GB = RD.bytes / 1e9
print({k: len(v) for k, v in FA_SETS.items()}, f'| downloaded {DL_GB:.2f} GB in {TIMES["pair caches"] / 60:.1f} min '
      f'| caches {sum(gb(f"{DATA}/{n}{sfx}_{len(r)}.npy") for n, r in (("pairs_train", TRAIN_ROWS), ("pairs_valfa", VAL_ROWS), ("pairs_testfa", FA_ROWS))):.2f} GB')
''')

md('## 8. Plume bank (WRF-LES, ppb at 3000 kg/h) and the transmittance tables\nFor every training scene, `BANK_K` random physical plumes, each at the bank sun position nearest to the scene (`target:sza`, `target:raa`; MethaneSET v1.2.0+ defines `target:raa` with the bank\'s convention, checked below against `target:saa` and the wind). The gate scenes are the first `GATE_N` training scenes.')
code(r'''
t = time.time(); BSUB = 'methaneset-bank'
bank = pd.read_parquet(hf_hub_download(REPO, f'{BSUB}/.tacocat/level0.parquet', repo_type='dataset', local_dir=DATA))
TJ = hf_hub_download(REPO, 'integrated_transmittances.json', repo_type='dataset', local_dir=DATA)
th = np.degrees(np.arctan2(TRAIN_ROWS['meteo:wind_v'], TRAIN_ROWS['meteo:wind_u'])); raa_chk = (90 - TRAIN_ROWS['target:saa'] - th) % 360
d = ((raa_chk - TRAIN_ROWS['target:raa'] + 180) % 360 - 180).abs()
print(f'target:raa vs (90 - SAA - wind direction): median |diff| {d.median():.2e} deg, max {d.max():.2e} (0 = bank convention confirmed)')
ASSIGN_RAW = sp.assign_bank(TRAIN_ROWS, bank, BANK_K, seed=0)
UB = np.unique(ASSIGN_RAW); REMAP = {int(v): i for i, v in enumerate(UB)}; ASSIGN = np.vectorize(REMAP.get)(ASSIGN_RAW)
BROWS = bank.iloc[UB].reset_index(drop=True); EDGE = BROWS['edge:downwind_ppb'].values.astype(np.float32)
if DOWNLOAD_BANK_ZIPS and not SMOKE_TEST:
    for z in sorted(BROWS['internal:source_file'].unique()):
        if not os.path.exists(f'{DATA}/{BSUB}/{z}'): hf_hub_download(REPO, f'{BSUB}/{z}', repo_type='dataset', local_dir=DATA)
BP = f'{DATA}/bank{sfx}_{len(TRAIN_ROWS)}x{BANK_K}.npy'
if not SMOKE_TEST: drive_get(os.path.basename(BP).replace('.npy', ''))
CANVAS, CAN_SRC, LOST = sp.build_bank_cache(BROWS, BP, RD, BSUB, workers=READ_THREADS)
if not SMOKE_TEST: drive_put(os.path.basename(BP).replace('.npy', ''))
PMETA = sp.scene_meta(TRAIN_ROWS)
TM = sp.Transmittance(TJ)
print(f'bank: {len(UB)} rows from {BROWS["methane:plume_uid"].nunique()} physical plumes for {len(TRAIN_ROWS)} scenes | {gb(BP):.2f} GB')
TIMES['bank'] = time.time() - t
''')

md('## 9. Initial weights: `fr_neg`\nOrder: `INIT_CKPT`, then `RUNS/fr_neg/final.pt`, then the private `RUNS_REPO` (`PULL_PRIVATE_RUNS`).')
code(r'''
CK0 = INIT_CKPT or f'{RUNS}/fr_neg/final.pt'
if not os.path.exists(CK0) and PULL_PRIVATE_RUNS:
    hf_hub_download(RUNS_REPO, 'fr_neg/final.pt', local_dir=RUNS); print('pulled fr_neg/final.pt from', RUNS_REPO)
assert os.path.exists(CK0), f'no fr_neg weights at {CK0}: set INIT_CKPT, or PULL_PRIVATE_RUNS = True with an HF token that can read {RUNS_REPO}'
SD0 = torch.load(CK0, map_location='cpu'); print('fr_neg weights:', CK0, '|', sum(v.numel() for k, v in SD0.items() if 'running' not in k and 'num_batches' not in k), 'tensor elements')
''')

md('## 10. Stage 1: injection sanity gate\nInject Q = 3000 kg/h into `GATE_N` plume-free scenes and compare with **real** fine-tune *train* plumes of 2–4.5 t/h, using the same estimator (mean inside the mask minus the median outside) for ln(B12/B11) and MBMP. Also checks the transmittance units: B11 is a plain transmittance (~0.99) but B12 is tabulated on a ~64 scale; its `_bg` column is the B12 value at 1,800 ppb, so the injection uses T(1800 + Δ) / T(1800) at the scene AMF and the scale cancels. **The notebook stops here if the gate fails.**')
code(r'''
t = time.time()
GATE_IDX = np.arange(min(GATE_N, len(TRAIN_ROWS)))
GATE_OK, GATE = sp.sanity_gate(XP, GATE_IDX, ASSIGN, CANVAS, CAN_SRC, EDGE, PMETA, TM, XFT, meta, tr, DEV, f'{OUT}/gate_injection.png',
                               floor_sigma=LABEL_FLOOR_SIGMA)
json.dump(GATE, open(f'{OUT}/gate_report.json', 'w'), indent=1, default=float)
FLOOR = GATE['label_floor']['chosen_sigma']
print(json.dumps({k: GATE[k] for k in ('units', 'real', 'synthetic', 'ratio_enhancement', 'mbmp_enhancement', 'area', 'label_floor')}, indent=1, default=float))
from IPython.display import Image, display
display(Image(f'{OUT}/gate_injection.png')); display(Image(f'{OUT}/gate_injection_hist.png'))
TIMES['gate'] = time.time() - t
if not GATE_OK:
    msg = 'STOP: injection sanity gate FAILED: ' + '; '.join(GATE['why'])
    if SMOKE_TEST: print(msg, '(smoke test: continuing anyway, the slice is tiny)')
    else: raise SystemExit(msg + '. No training was run. See gate_report.json and gate_injection.png.')
else: print(f'GATE PASS: label floor = {FLOOR} x scene noise')
if GATE_ONLY: raise SystemExit('GATE_ONLY = True: stopping after the gate.')
''')

md('## 11. Stage 2: train `syn` and `syn_mbmp` (from `fr_neg`)\nEach step: 8 real fine-tune chips (fr_neg recipe) + 8 plume-free pairs, 70 % of them with a plume injected on the GPU (nearest sun geometry, rotated to the scene wind ±15°, source within ±600 m of the chip centre, Q log-uniform 0.5–20 t/h capped by the LES cut limit). Loss: BCE + Dice with positives weighted by enhancement. Selection: best epoch by **val** hit rate at 5 % val false alarms (2021 only). Trackio logs every epoch.')
code(r'''
import trackio
TRAIN_TIMES = {}
for name in RUNS_TO_TRAIN:
    t = time.time()
    sp.train_syn(name, 'mbmp' if name.endswith('mbmp') else 'plain', XFT, meta, CH4, XP, TR_PAIRS, ASSIGN, CANVAS, CAN_SRC, EDGE, PMETA,
                 TM.to(DEV), DEV, RUNS, init_sd=SD0, XV=XV, val_neg=VAL_NEG, epochs=EPOCHS, lr=LR, bs_real=BS_REAL, bs_syn=BS_SYN,
                 inj_p=INJ_P, neg_p=NEG_P, q_min=Q_MIN, q_max=Q_MAX, floor_sigma=FLOOR, enh_weighting=ENH_WEIGHTING, amp=AMP,
                 workers=WORKERS, project=TRACKIO_PROJECT, skip_done=SKIP_DONE, max_steps=MAX_STEPS, preview_every=1 if SMOKE_TEST else 3)
    TRAIN_TIMES[name] = time.time() - t
print({k: f'{v / 60:.1f} min' for k, v in TRAIN_TIMES.items()}); TIMES['training'] = sum(TRAIN_TIMES.values())
''')

md('## 12. Stage 3: evaluate `fr_neg`, `syn`, `syn_mbmp` identically\nScene score = max of the 2×2-pooled probability (20 m). *detect* = score ≥ threshold; *hit* = a flagged 20 m pixel inside the plume mask. Operating thresholds are set on the real 2024 plume-free pairs (`real2024_strict`) at 10 / 5 / 1 % false alarms. Pixel IoU/F1 at 0.5 (20 m grid) as on the cards. The candidate is picked on **val** only, then judged by the plan\'s go / no-go rule.')
code(r'''
t = time.time()
amp_dtype = mf.amp_dtype_for(DEV) if AMP else None
MODELS = {'fr_neg': ('plain', SD0)}
for name in RUNS_TO_TRAIN:
    MODELS[name] = ('mbmp' if name.endswith('mbmp') else 'plain', torch.load(f'{RUNS}/{name}/best.pt', map_location='cpu'))
RES, RAW = {}, {}
for name, (kind, sd) in MODELS.items():
    m = sp.load_model(kind, sd, DEV)
    RES[name], RAW[name] = sp.evaluate(name, m, XFT, TE, XF, FA_SETS, XV, VAL_NEG, va, DEV, amp_dtype, WORKERS)
    v = RES[name]['val'].get('ops', {}).get('5%', {}); s = RES[name]['scene'].get('unseen|real2024_strict', {})
    print(f'{name:9s} val hit@5%FA {v.get("hit", float("nan")):.3f} | unseen: IoU@0.5 {RES[name]["pixel"]["unseen"]["20m"]["iou"]:.3f} '
          f'hit@5%FA {s.get("ops", {}).get("5%", {}).get("hit", float("nan")):.3f} AP {s.get("AP", float("nan")):.3f} '
          f'| real FA@0.5 {s.get("real_fa_at_0p5", float("nan")):.3f}')
CANDS = [n for n in RUNS_TO_TRAIN]
CAND = max(CANDS, key=lambda n: RES[n]['val'].get('ops', {}).get('5%', {}).get('hit', -1)) if CANDS else None
print('val-selected candidate:', CAND)
TIMES['evaluation'] = time.time() - t
# --- tables
rows = []
for name, r in RES.items():
    for sub, px in r['pixel'].items():
        for fs in FA_SETS:
            s = r['scene'].get(f'{sub}|{fs}')
            if not s: continue
            row = dict(model=name, plumes=sub, n_plumes=s['n_pos'], fa_set=fs, n_fa=s['n_neg'], iou_0p5_20m=px['20m']['iou'], f1_0p5_20m=px['20m']['f1'],
                       det_0p5_20m=px['20m']['det_rate'], bg1_fa_0p5_20m=px['20m']['false_alarm'], AP=s['AP'], real_fa_0p5=s['real_fa_at_0p5'],
                       hit_0p5=s['hit_at_0p5'])
            for op, o in s['ops'].items(): row.update({f'thr@{op}': o['threshold'], f'hit@{op}': o['hit'], f'detect@{op}': o['detect']})
            rows.append(row)
CMP = pd.DataFrame(rows); CMP.to_csv(f'{OUT}/syn_comparison.csv', index=False, float_format='%.4f')
show = CMP[CMP.fa_set == 'real2024_strict'][['model', 'plumes', 'n_plumes', 'iou_0p5_20m', 'AP', 'real_fa_0p5', 'hit@10%', 'hit@5%', 'hit@1%']]
print(show.round(3).to_string(index=False))
''')
code(r'''
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, sub in zip(ax, ('unseen', 'seen', 'all')):
    for name in RES:
        if sub not in RAW[name]['pos'] or not len(RAW[name]['neg']['real2024_strict']): continue
        ap, (P, R) = sp.average_precision(RAW[name]['pos'][sub]['score'], RAW[name]['neg']['real2024_strict'])
        a.step(R, P, where='post', label=f'{name} AP {ap:.3f}')
    a.set_title(f'{sub} test plumes vs real 2024 plume-free pairs', fontsize=9); a.set_xlabel('recall (detect)'); a.set_ylabel('precision'); a.legend(fontsize=8)
f.tight_layout(); f.savefig(f'{OUT}/pr_curves.png', dpi=90); plt.close(f)
f, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, sub in zip(ax, ('unseen', 'all')):
    for name in RES:
        neg = RAW[name]['neg']['real2024_strict']; pos = RAW[name]['pos'].get(sub)
        if pos is None or not len(neg): continue
        ts = np.unique(np.r_[np.quantile(neg, np.linspace(0, 1, 200)), 0.5]); fa = [(neg >= x).mean() for x in ts]; hit = [(pos['score_in'] >= x).mean() for x in ts]
        a.plot(fa, hit, label=name)
    for x in (0.01, 0.05, 0.10): a.axvline(x, color='grey', lw=.5, ls='--')
    a.set_xscale('log'); a.set_xlabel('real false-alarm rate (2024 plume-free pairs)'); a.set_ylabel('hit rate'); a.set_title(f'{sub} test plumes', fontsize=9); a.legend(fontsize=8)
f.tight_layout(); f.savefig(f'{OUT}/hit_vs_real_fa.png', dpi=90); plt.close(f)
from IPython.display import Image, display
display(Image(f'{OUT}/pr_curves.png')); display(Image(f'{OUT}/hit_vs_real_fa.png'))
''')

md('## 13. Go / no-go (plan section "Go / no-go")\nLatency: both models exported to ONNX (opset 17, static 1×8×200×200) and timed with onnxruntime on this CPU, one thread.')
code(r'''
import onnx, onnxruntime as ort
def export(kind, sd, path):
    m = sp.make_model(kind); m.load_state_dict(sd); m.eval()
    x1, _, _ = mf.eval_batch(torch.from_numpy(np.array(XFT[te[:1], :mf.NCH_BASIC])))
    torch.onnx.export(m, x1, path, opset_version=17, input_names=['reflectance'], output_names=['plume_logits'], dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(path); g = onnx.load(path).graph
    shape = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
    return m, x1, shape == [1, 8, 200, 200] and g.input[0].name == 'reflectance' and g.output[0].name == 'plume_logits' and len(g.input) == 1
def latency(path, x, n):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    xi = x.numpy()
    for _ in range(3): s.run(None, {'reflectance': xi})
    ts = []
    for _ in range(n): t_ = time.perf_counter(); s.run(None, {'reflectance': xi}); ts.append(time.perf_counter() - t_)
    return float(np.median(ts) * 1000)
TMP = f'{WORK}/onnx_tmp'; os.makedirs(TMP, exist_ok=True); NB = 5 if SMOKE_TEST else 50
GO, CRIT, LAT = False, {}, {}
if CAND:
    kind_c, sd_c = MODELS[CAND]
    m0, x1, ok0 = export('plain', SD0, f'{TMP}/fr_neg.onnx'); mc, _, okc = export(kind_c, sd_c, f'{TMP}/{CAND}.onnx')
    for _ in range(2):  # interleave to even out CPU noise
        for n_, p_ in (('fr_neg', f'{TMP}/fr_neg.onnx'), (CAND, f'{TMP}/{CAND}.onnx')): LAT.setdefault(n_, []).append(latency(p_, x1, NB))
    LAT = {k: min(v) for k, v in LAT.items()}; lat_ratio = LAT[CAND] / LAT['fr_neg']
    GO, CRIT = sp.go_no_go(RES[CAND], RES['fr_neg'], lat_ratio, ok0 and okc, sp.n_params(mc), sp.n_params(m0), fa_key='real2024_strict')
    print(f'candidate {CAND}: params {sp.n_params(mc):,} vs fr_neg {sp.n_params(m0):,} | ORT CPU 1 thread {LAT[CAND]:.1f} ms vs {LAT["fr_neg"]:.1f} ms')
    for k, v in CRIT.items(): print(('  PASS ' if v['ok'] else '  FAIL ') + k + f'  ({v["value"]:.4f})' if not isinstance(v['value'], bool) else ('  PASS ' if v['ok'] else '  FAIL ') + k)
if SMOKE_TEST:
    print('smoke test: the decision below is meaningless')
    if SMOKE_FORCE_GO and CAND: GO = True; print('SMOKE_FORCE_GO: forcing the GO branch to exercise the export code')
print('DECISION:', 'GO' if GO else 'NO-GO (keep fr_neg)')
SUMMARY = dict(smoke_test=SMOKE_TEST, gpu=GPU, decision='GO' if GO else 'NO-GO', candidate=CAND, criteria=CRIT, latency_ms_ort_cpu_1t=LAT,
               gate=GATE, label_floor_sigma=FLOOR, results=RES, fa_sets={k: int(len(v)) for k, v in FA_SETS.items()},
               n_train_pairs=int(len(TR_PAIRS)), n_val_fa=int(len(VAL_NEG)), bank_rows=int(len(UB)),
               config=dict(EPOCHS=EPOCHS, LR=LR, BS_REAL=BS_REAL, BS_SYN=BS_SYN, INJ_P=INJ_P, NEG_P=NEG_P, Q_MIN=Q_MIN, Q_MAX=Q_MAX,
                           PRETRAIN_N=PRETRAIN_N, VAL_FA_N=VAL_FA_N, MIN_CLEAR=MIN_CLEAR, BANK_K=BANK_K, ENH_WEIGHTING=ENH_WEIGHTING),
               times_s=TIMES, train_times_s=TRAIN_TIMES, downloaded_gb=DL_GB)
json.dump(SUMMARY, open(f'{OUT}/syn_results.json', 'w'), indent=1, default=float)
print('wrote', f'{OUT}/syn_results.json')
''')

md('## 14. Only on GO: export the candidate and write its card\nSafetensors, ONNX (parity-checked against PyTorch), `config.json` and `CANDIDATE_CARD.md`. Nothing is written on NO-GO.')
code(r'''
if GO:
    from safetensors.torch import save_file
    shutil.copy(f'{TMP}/{CAND}.onnx', f'{OUT}/methane_se2_fullres_{CAND}.onnx')
    save_file({k: v.contiguous() for k, v in mc.state_dict().items()}, f'{OUT}/model.safetensors')
    s = ort.InferenceSession(f'{OUT}/methane_se2_fullres_{CAND}.onnx', providers=['CPUExecutionProvider'])
    xb, _, _ = mf.eval_batch(torch.from_numpy(np.array(XFT[np.sort(te[:16]), :mf.NCH_BASIC]))); d_, ag = [], []
    with torch.no_grad():
        for i in range(len(xb)):
            pt = mc(xb[i:i + 1]).numpy(); ox = s.run(None, {'reflectance': xb[i:i + 1].numpy()})[0]
            d_.append(float(np.abs(pt - ox).max())); ag.append(float(((pt > 0) == (ox > 0)).mean()))
    assert max(d_) < 1e-2 and np.mean(ag) > 0.999, 'ONNX parity check failed'
    json.dump(dict(input_name='reflectance', input_shape=[1, 8, 200, 200], input_bands=mf.INPUT_BANDS, scale='L1C DN / 10000', resolution_m=10,
                   output_name='plume_logits', threshold=0.5, model=CAND, init='fr_neg', status='candidate (not released)'), open(f'{OUT}/config.json', 'w'), indent=1)
    R = RES[CAND]; B = RES['fr_neg']; op = lambda r, sub, k='hit', o='5%': r['scene'][f'{sub}|real2024_strict']['ops'][o][k]
    lines = '\n'.join(f'| {n} | {RES[n]["pixel"][sub]["20m"]["iou"]:.3f} | {RES[n]["scene"][f"{sub}|real2024_strict"]["AP"]:.3f} | '
                      f'{op(RES[n], sub, o="10%"):.3f} | {op(RES[n], sub):.3f} | {op(RES[n], sub, o="1%"):.3f} |'
                      for sub in ('unseen', 'seen', 'all') if f'{sub}|real2024_strict' in RES[CAND]['scene'] for n in ('fr_neg', CAND))
    crit = '\n'.join(f'| {k} | {"pass" if v["ok"] else "fail"} | {v["value"] if isinstance(v["value"], bool) else round(v["value"], 4)} |' for k, v in CRIT.items())
    card = f"""# Candidate: `{CAND}` (synthetic-plume training), not released

Trained in Colab ({GPU}) from `fr_neg` with plumes from the MethaneSET WRF-LES bank injected into real plume-free Sentinel-2 pairs
(`colab/PLAN_synthetic_plumes.md`). Same input contract as the released model: `reflectance` float32 1x8x200x200 -> `plume_logits`.
It passed the plan's go / no-go rule against `fr_neg` evaluated identically. **It replaces nothing until Simon decides to release it.**

## Results (test plumes vs {len(FA_SETS['real2024_strict'])} real 2024 plume-free pairs at sites in no training split)

Rows: plume subset / model. IoU at 0.5 on the 20 m grid. Hit rate = a flagged 20 m pixel inside the plume mask, at the threshold where the real
false-alarm rate is 10 / 5 / 1 %.

| subset / model | IoU@0.5 | AP | hit @10% FA | hit @5% FA | hit @1% FA |
|---|---:|---:|---:|---:|---:|
{lines}

Unseen = {len(TE['unseen'])} test chips at 92 sites with no train chip within 1 km. 2021 val (selection) hit rate at 5 % val FA:
`fr_neg` {B['val']['ops']['5%']['hit']:.3f}, `{CAND}` {R['val']['ops']['5%']['hit']:.3f}.

## Go / no-go

| criterion | result | value |
|---|---|---|
{crit}

## How it was trained

- Real half: the 2,362 fine-tune train chips with fr_neg's recipe (30 % bg1-vs-bg0 plume-free pairs).
- Plume-free half: {len(TR_PAIRS)} pretraining train pairs (2018-20, 2022-23), none at the 92 unseen-test sites; {INJ_P:.0%} get a plume:
  nearest bank sun geometry, rotated to the scene wind, Q log-uniform {Q_MIN / 1000:g}-{Q_MAX / 1000:g} t/h, multiplicative B11/B12 transmittance.
- Label floor: {FLOOR} x scene noise (matched to real label areas in the sanity gate). Loss: BCE + Dice, positives weighted by enhancement.
- {EPOCHS} epochs, lr {LR}, best epoch by 2021 val hit rate at 5 % val false alarms.

## Caveats

- Plume-free means "no MARS detection", so the real false-alarm rates are slightly pessimistic.
- Chips are centred on known emitter sites; onboard use needs tiling and a co-registered reference.
- Not tested on a Jetson. INT8 may erase the ~1 % SWIR signal; validate any engine against the ONNX.
- Data CC BY-NC-SA 4.0 (MethaneSET) and CC BY 4.0 (bank): non-commercial.
"""
    open(f'{OUT}/CANDIDATE_CARD.md', 'w').write(card); print(card[:2500])
else:
    print('NO-GO (or no candidate): nothing exported. fr_neg stays the release. Results are in', f'{OUT}/syn_results.json')
''')

md('## 15. Trackio dashboard (off by default)\nPublishes the static Space `sighmon/methane-se2-syn-trackio` (public).')
code(r'''
if SYNC_TRACKIO and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    subprocess.run(['trackio', 'sync', '--project', TRACKIO_PROJECT, '--space-id', TRACKIO_SPACE, '--sdk', 'static', '--force'], check=True)
    print('dashboard:', f'https://huggingface.co/spaces/{TRACKIO_SPACE}')
else: print('SYNC_TRACKIO is False (or smoke test): nothing published. Local Trackio DB:', os.environ['TRACKIO_DIR'])
''')

md('## 16. Uploads (off by default)\n`UPLOAD = True` and a **GO**: create the **private** repo `HF_CANDIDATE_REPO` and upload `out/`. The released `sighmon/methane-plume-se2-fullres` is never touched here. `SAVE_RUNS_PRIVATE = True`: push the `syn*` checkpoints to the private `RUNS_REPO` under `syn/`.')
code(r'''
from huggingface_hub import HfApi
if UPLOAD and GO and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    api = HfApi(); api.create_repo(HF_CANDIDATE_REPO, repo_type='model', private=True, exist_ok=True)
    api.upload_folder(folder_path=OUT, repo_id=HF_CANDIDATE_REPO, repo_type='model', commit_message=f'Synthetic-plume candidate {CAND} (GO vs fr_neg)')
    print('uploaded (private):', f'https://huggingface.co/{HF_CANDIDATE_REPO}')
else: print('UPLOAD off, NO-GO or smoke test: nothing uploaded. Files in', OUT, sorted(os.listdir(OUT)))
if SAVE_RUNS_PRIVATE and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    HfApi().upload_folder(folder_path=RUNS, repo_id=RUNS_REPO, path_in_repo='syn', allow_patterns=['syn*/*.pt', 'syn*/*.json'],
                          commit_message='Synthetic-plume checkpoints (private)')
    print('checkpoints pushed to', RUNS_REPO, '(private) under syn/')
''')

md('## 17. Time and disk summary')
code(r'''
TIMES['total'] = time.time() - T0
du = lambda p: sum(os.path.getsize(f) for f in glob.glob(f'{p}/**', recursive=True) if os.path.isfile(f)) / 1e9 if os.path.exists(p) else 0
print({k: f'{v / 60:.1f} min' for k, v in TIMES.items()})
print(f'disk: {WORK} {du(WORK):.1f} GB, fine-tune data {du(FT_DATA):.1f} GB | peak RSS of this kernel {psutil.Process().memory_info().rss / 1e9:.1f} GB '
      f'| RAM total {psutil.virtual_memory().total / 1e9:.0f} GB')
json.dump(dict(times_s=TIMES, disk_gb=dict(work=du(WORK), finetune_data=du(FT_DATA))), open(f'{OUT}/run_times.json', 'w'), indent=1)
''')

nb.cells = C
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'},
               'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4', 'machine_shape': 'hm'}}
TIME_TABLE = open(f'{HERE}/time_table.md').read() if os.path.exists(f'{HERE}/time_table.md') else ''
nb.cells[0].source = nb.cells[0].source.replace('TIME_TABLE_PLACEHOLDER', TIME_TABLE)
nbf.validate(nb)
nbf.write(nb, f'{HERE}/synthetic_plumes_colab.ipynb'); print('written', len(C), 'cells')
