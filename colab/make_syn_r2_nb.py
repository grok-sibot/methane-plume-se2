"""Generate synthetic_plumes_r2_colab.ipynb: round 2 of the synthetic-plume experiment (colab/PLAN_synthetic_plumes.md,
section "Round 2"). Run from this folder:
    python make_syn_r2_nb.py         -> writes synthetic_plumes_r2_colab.ipynb next to it
Round 1 (make_syn_nb.py, synthetic_plumes_colab.ipynb) is left as it was. Both embed methane_fullres.py and synplume.py
(both in this folder) with %%writefile."""
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


RULE = "**Go / no-go rule (fixed before the run; same as round 1 plus 2b).** The candidate is the trained variant with the best\n2021 val hit rate at 5 % val false alarms (val only, never test). It is compared with `fr_neg`, evaluated identically. **GO** only if all hold:\n1. unseen-site test plumes: hit rate at 5 % real false alarms (thresholds set per model on the 8,740 `real2024_strict` plume-free pairs) improves by **>= 0.10**;\n2. unseen-site pixel IoU (20 m grid) at threshold 0.5 drops by **<= 0.01**;\n2b. unseen-site pixel IoU (20 m grid) at each model's **val-calibrated** threshold (the threshold giving 5 % false alarms on the 2021 val plume-free pairs) drops by **<= 0.01**;\n3. 2021 val goes the same way (val hit rate at 5 % val false alarms is higher);\n4. ONNX input contract 1x8x200x200 unchanged, parameter count within 0.01 %, onnxruntime CPU latency within +10 %;\n5. all test plumes: hit rate at 5 % real false alarms not lower.\n\nOn GO the candidate's `config.json` carries its val-calibrated threshold (and 0.5 for reference). Nothing is uploaded unless `UPLOAD = True`, and then only to a private repo."

md(r'''
# Round 2: synthetic-plume training for the full-res methane detector (Colab, T4)

Round 2 of [`colab/PLAN_synthetic_plumes.md`](https://github.com/grok-sibot/methane-plume-se2/blob/main/colab/PLAN_synthetic_plumes.md).
Round 1 (`synthetic_plumes_colab.ipynb`) is unchanged. Same goal: better detection at **sites the model has never seen**, with false alarms
measured on **real plume-free scenes**. The onboard contract is the same: `reflectance` float32 1×8×200×200 → `plume_logits` 1×1×200×200.

**Round 1 result (Colab T4, 21 Nov 2026): NO-GO.** `syn` against `fr_neg`:
- unseen-site hit rate at 5 % real false alarms: 0.438 against 0.402 (+0.036; the rule needs +0.10);
- AP: 0.614 against 0.523 on all test plumes, and 0.246 against 0.192 on unseen-site plumes;
- hit rate at 1 % real false alarms: 0.343 against 0.204 on all test plumes;
- worse at the fixed 0.5 threshold: false alarms on the test chips' bg1-vs-bg0 plume-free pairs (≥ 10 flagged 20 m pixels) 0.68 against 0.55, and seen IoU 0.536 against 0.543;
- `syn_mbmp` was no better.

Settings were 12 epochs, 12,000 of the ~24.9k eligible plume-free pairs, `INJ_P` 0.7, `NEG_P` 0.3, `BANK_K` 3, Q 0.5–20 t/h, and about 11 min of training per run.

**What round 2 changes, and why**
| change | why |
|---|---|
| `syn_mbmp` dropped. Up to three `syn` variants: `syn2` (round-1 recipe), `syn2_neg` (`INJ_P` 0.5, `NEG_P` 0.5: more clean and real negatives) and `syn2_hn` (`syn2_neg` + hard negatives) | MBMP did not help. The extra negatives target the false alarms at 0.5 |
| 20 epochs; every eligible plume-free pair (`PRETRAIN_N = None`, ~24.9k at ≥ 95 % clear) | round 1 used 12 epochs and 12k pairs. There are not 30k eligible train pairs: 26,102 are ≥ 95 % clear, and 24,918 remain after dropping the unseen-test sites |
| **unseen-like sampling weights** for the plume-free pairs (section 8b) | push the synthetic half toward sites, regions and surfaces the fine-tune train split covers least, using **training-split information only** |
| **hard negatives** (`syn2_hn`): the 10 % of training pairs that `fr_neg` scores highest are never injected and make up 25 % of the plume-free draws | aims directly at the bright, plume-like false alarms |
| **calibrated threshold**: per model, the threshold giving 5 % false alarms on the 2021 val plume-free pairs; every test metric is also reported at it, including pixel IoU/F1 on seen and unseen plumes | 0.5 is not a meaningful operating point for these models (round-1 5 %-FA thresholds were ~0.99) |
| disk checks before every cache; source zips deleted once their cache is built; a time budget skips a variant that would not fit | fresh Colab runtime, ~2.5 h |

RULE_MD

**Leak control (as round 1).** Training uses pretraining *train* pairs (2018–20, 2022–23) only, minus every pair at one of the 92 sites of
the 251 unseen-site test chips. Sampling weights use only the training pool and the fine-tune *train* site list. Selection and
calibration use 2021 (val) only. Nothing from 2024 is used for training, weighting, selection or calibration. The section-12 diagnostic
compares the weighted pool with the unseen test chips only *after* training, and nothing feeds back from it.

**Safe defaults.** `UPLOAD = False`, `SYNC_TRACKIO = False`, `SAVE_RUNS_PRIVATE = False`. Every upload target is a private repo. pip is
skipped when the packages already import. Caches are memory-mapped. The HF token is never printed.

**How to run (fresh runtime):** Runtime → Change runtime type → **T4 GPU**. Add the `HF_TOKEN` Colab secret, with read access to
`sighmon/methane-fullres-runs` for `fr_neg`. Then *Runtime → Run all*. Set `USE_DRIVE = True` to survive a disconnect: caches and runs
then also go to Drive, and a rerun resumes.

TIME_TABLE_PLACEHOLDER
''')

md('## 1. Config\nThis is the papermill `parameters` cell. `SMOKE_TEST = True` runs every stage on a few chips (CPU is fine); its numbers mean nothing.')
code(r'''
# ---- run mode ----
SMOKE_TEST = False          # True: tiny slices, 1 epoch of a few steps, never uploads. Proves the notebook runs; numbers are meaningless.
USE_DRIVE = False           # True: mount Google Drive; runs, Trackio and (DRIVE_CACHES) caches live there too (resume after disconnects)
DRIVE_CACHES = True         # with USE_DRIVE: also copy the built caches to Drive (~33 GB: needs that much Drive space)
SKIP_DONE = True            # reuse finished caches and runs (final.pt + best.pt)
UPLOAD = False              # True (and GO): upload the candidate to HF_CANDIDATE_REPO (private). Never touches the released model.
SYNC_TRACKIO = False        # True: sync the Trackio dashboard to a NEW private Space (TRACKIO_SPACE)
SAVE_RUNS_PRIVATE = False   # True: also push the syn2* checkpoints to the private RUNS_REPO under syn_r2/
PULL_PRIVATE_RUNS = True    # fetch fr_neg/final.pt from the private RUNS_REPO when it is not on disk
CACHE_PATH = None           # optional existing 10 m fine-tune cache (cache10m.npy, 31 channels); reused and memory-mapped
INIT_CKPT = None            # optional local path to fr_neg/final.pt (skips the pull)
GATE_ONLY = False           # True: stop after the sanity gate (cheap first look)
SMOKE_FORCE_GO = False      # smoke test only: take the GO branch anyway so the export + card code is exercised (ignored when SMOKE_TEST is False)

# ---- time and disk ----
TIME_BUDGET_MIN = 150       # skip a remaining variant if elapsed + its estimated time + EVAL_RESERVE_MIN would pass this (None: never skip)
EVAL_RESERVE_MIN = 12       # kept free for evaluation, export and the summary
EPOCH_S_EST = 60            # T4 seconds per epoch before the first run has been timed (round 1: 57 s)
DISK_RESERVE_GB = 5         # every cache build first checks that its size plus this reserve is free
DELETE_ZIPS_AFTER_USE = True  # delete downloaded fine-tune / bank zips once their caches are built (-11 GB)
MIN_TRAIN_PAIRS = 4000      # if disk forces the training pool smaller than this, stop instead

# ---- data sizes ----
PRETRAIN_N = None           # plume-free train pairs cached; None: every eligible one (24,918 at MIN_CLEAR 95), ~16 GB on disk
VAL_FA_N = 2000             # 2021 plume-free pairs for selection and threshold calibration
FA_N = None                 # None: every 2024 pair of the FA sets (8,740 strict + 1,759 more at unseen sites), 6.7 GB
GATE_N = 64                 # scenes for the injection sanity gate
BANK_K = 3                  # plumes pre-assigned per training scene (nearest sun geometry); one is drawn per use
MIN_CLEAR = 95.0            # training pairs only: minimum quality:percentage_clear
DOWNLOAD_BANK_ZIPS = True   # download the 5 GB bank once (fast) instead of ~50k small range reads; smoke uses range reads
READ_THREADS = 128          # concurrent HTTP range reads (round 1 on Colab: ~21 pairs/s)

# ---- injection ----
Q_MIN, Q_MAX = 500.0, 20000.0   # kg/h, log-uniform; also capped per plume so the LES downwind cut stays below the scene noise
LABEL_FLOOR_SIGMA = 'auto'  # label: enhancement at least this many x scene noise (ppb); auto: matched to real label areas in the gate

# ---- unseen-like sampling of the plume-free pairs (section 8b; training-split information only) ----
UNSEEN_WEIGHTING = True     # False: uniform sampling as in round 1
NOVEL_BOOST = 2.0           # x weight for pairs at sites with no fine-tune train chip
A_SITE, A_COUNTRY, A_ALBEDO = 0.5, 0.5, 0.5   # weight ~ count(site)^-a * count(country)^-a * count(albedo bin)^-a
WEIGHT_MIX = 0.5            # final weight is mix * w + (1 - mix): half of the draws stay uniform
HN_TOP_FRAC = 0.10          # hard negatives: the top 10 % of training pairs by fr_neg scene score ...
HN_SHARE = 0.25             # ... make up 25 % of the plume-free draws, and are never injected

# ---- training ----
EPOCHS = 20                 # one epoch: one pass over the 2,362 real train chips (+ as many plume-free pairs)
LR = 3e-4                   # fine-tune from fr_neg
BS_REAL, BS_SYN = 8, 8      # 16 chips per step, half real
ENH_WEIGHTING = True        # enhancement-weighted BCE positives (real: ch4 leaf, synthetic: injected ppb)
# RUN_SPECS: inj_p is the share of plume-free pairs injected; neg_p the share of the real half swapped for bg1-vs-bg0 plume-free pairs;
# weighted uses the section-8b sampling weights; hard adds hard negatives (section 9)
RUN_SPECS = {'syn2': {'inj_p': 0.7, 'neg_p': 0.3, 'weighted': True, 'hard': False}, 'syn2_neg': {'inj_p': 0.5, 'neg_p': 0.5, 'weighted': True, 'hard': False}, 'syn2_hn': {'inj_p': 0.5, 'neg_p': 0.5, 'weighted': True, 'hard': True}}
RUNS_TO_TRAIN = ['syn2', 'syn2_neg', 'syn2_hn']   # in this order; the time budget can skip the last ones
CAL_FA = 0.05               # calibrated threshold: false-alarm rate on the 2021 val plume-free pairs
MAX_STEPS = None            # cap steps per epoch (debugging); smoke test uses 3
AMP = True
WORKERS = None              # DataLoader workers; None: min(4, cpu_count)

# ---- names ----
RUNS_REPO = 'sighmon/methane-fullres-runs'              # private, holds fr_neg/final.pt
HF_CANDIDATE_REPO = 'sighmon/methane-plume-se2-syn'     # private, only used with UPLOAD True and a GO
TRACKIO_PROJECT = 'methane-se2-syn-r2'
TRACKIO_SPACE = 'sighmon/methane-se2-syn-r2-trackio'    # created private by SYNC_TRACKIO; refuses to sync to an existing public Space
WORK = None                 # None: /content/methane-syn-r2 on Colab, ./methane-syn-r2 elsewhere
FT_WORK = None              # where an earlier full-res run left data/ (cache10m.npy, tacozips); None: /content/methane-fullres on Colab
''', tags=['parameters'])
md('## 2. Setup\nAlso writes the go / no-go rule and this config to `out/preregistered_rule.json` before any data is touched (kept from the first run if the cell is rerun).')
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
    PRETRAIN_N, VAL_FA_N, FA_N, GATE_N, BANK_K, MIN_TRAIN_PAIRS = 32, 16, 24, 24, 2, 8
    EPOCHS, BS_REAL, BS_SYN = 1, 4, 4; DOWNLOAD_BANK_ZIPS = False; UPLOAD = SYNC_TRACKIO = SAVE_RUNS_PRIVATE = False; TIME_BUDGET_MIN = None
MAX_STEPS = 3 if SMOKE_TEST else MAX_STEPS
assert set(RUNS_TO_TRAIN) <= set(RUN_SPECS) and RUNS_TO_TRAIN, f'RUNS_TO_TRAIN must name entries of RUN_SPECS: {list(RUN_SPECS)}'
WORK = WORK or ('/content/methane-syn-r2' if IN_COLAB else os.path.abspath('methane-syn-r2'))
FT_WORK = FT_WORK or ('/content/methane-fullres' if IN_COLAB else os.path.abspath('methane-fullres'))
DATA, OUT = f'{WORK}/data', f'{WORK}/out'
for d in (DATA, OUT): os.makedirs(d, exist_ok=True)
DRIVE = None
if USE_DRIVE and IN_COLAB:
    from google.colab import drive; drive.mount('/content/drive')
    DRIVE = '/content/drive/MyDrive/methane-se2-syn-r2'; os.makedirs(f'{DRIVE}/data', exist_ok=True)
RUNS = f'{DRIVE or WORK}/runs'; os.makedirs(RUNS, exist_ok=True)
os.environ['TRACKIO_DIR'] = f'{DRIVE or WORK}/trackio'; os.makedirs(os.environ['TRACKIO_DIR'], exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = WORKERS if WORKERS is not None else min(4, os.cpu_count() or 2)
GPU = torch.cuda.get_device_name(0) if DEV.type == 'cuda' else 'none (CPU)'
print(f'torch {torch.__version__} | pandas {pd.__version__} | device {DEV} | GPU {GPU} | cpus {os.cpu_count()} | RAM {psutil.virtual_memory().total / 1e9:.0f} GB '
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
# --- disk: check before every cache build; delete what is no longer needed
RECLAIM = []   # (path, what): safe to delete once its cache exists (used when DELETE_ZIPS_AFTER_USE is False and disk runs short)
def free_gb(): return shutil.disk_usage(WORK).free / 1e9
def allocated_gb(p):
    return sum(os.stat(f).st_blocks * 512 for f in glob.glob(p + '*') if os.path.isfile(f)) / 1e9
def drop(path, what):
    if os.path.exists(path):
        g = allocated_gb(path) if os.path.isfile(path) else sum(os.stat(f).st_blocks * 512 for f in glob.glob(f'{path}/**', recursive=True) if os.path.isfile(f)) / 1e9
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path); print(f'deleted {what} ({g:.1f} GB): its cache is built')
def need_disk(need, what):
    """True if `need` GB (+ DISK_RESERVE_GB) is free, after deleting reclaimable files if necessary."""
    if free_gb() < need + DISK_RESERVE_GB:
        while RECLAIM and free_gb() < need + DISK_RESERVE_GB: drop(*RECLAIM.pop(0))
    f = free_gb(); ok = f >= need + DISK_RESERVE_GB
    print(f'disk check, {what}: needs ~{need:.1f} GB + {DISK_RESERVE_GB} GB reserve, {f:.1f} GB free -> {"ok" if ok else "NOT ENOUGH"}')
    return ok
def done_with(path, what):
    if DELETE_ZIPS_AFTER_USE: drop(path, what)
    else: RECLAIM.append((path, what))
# --- the rule, fixed before the run
RULE = "**Go / no-go rule (fixed before the run; same as round 1 plus 2b).** The candidate is the trained variant with the best\n2021 val hit rate at 5 % val false alarms (val only, never test). It is compared with `fr_neg`, evaluated identically. **GO** only if all hold:\n1. unseen-site test plumes: hit rate at 5 % real false alarms (thresholds set per model on the 8,740 `real2024_strict` plume-free pairs) improves by **>= 0.10**;\n2. unseen-site pixel IoU (20 m grid) at threshold 0.5 drops by **<= 0.01**;\n2b. unseen-site pixel IoU (20 m grid) at each model's **val-calibrated** threshold (the threshold giving 5 % false alarms on the 2021 val plume-free pairs) drops by **<= 0.01**;\n3. 2021 val goes the same way (val hit rate at 5 % val false alarms is higher);\n4. ONNX input contract 1x8x200x200 unchanged, parameter count within 0.01 %, onnxruntime CPU latency within +10 %;\n5. all test plumes: hit rate at 5 % real false alarms not lower.\n\nOn GO the candidate's `config.json` carries its val-calibrated threshold (and 0.5 for reference). Nothing is uploaded unless `UPLOAD = True`, and then only to a private repo."
PRE = f'{OUT}/preregistered_rule.json'
if not os.path.exists(PRE):
    json.dump(dict(written=time.strftime('%Y-%m-%d %H:%M:%S %Z'), smoke_test=SMOKE_TEST, rule=RULE, run_specs=RUN_SPECS, runs_to_train=RUNS_TO_TRAIN,
                   config=dict(EPOCHS=EPOCHS, LR=LR, PRETRAIN_N=PRETRAIN_N, VAL_FA_N=VAL_FA_N, MIN_CLEAR=MIN_CLEAR, BANK_K=BANK_K, CAL_FA=CAL_FA,
                               UNSEEN_WEIGHTING=UNSEEN_WEIGHTING, NOVEL_BOOST=NOVEL_BOOST, A_SITE=A_SITE, A_COUNTRY=A_COUNTRY, A_ALBEDO=A_ALBEDO,
                               WEIGHT_MIX=WEIGHT_MIX, HN_TOP_FRAC=HN_TOP_FRAC, HN_SHARE=HN_SHARE, Q_MIN=Q_MIN, Q_MAX=Q_MAX)), open(PRE, 'w'), indent=1)
print(RULE); print('rule written to', PRE, '(before any data or training)')
''')
md('## 3. Code\nThe first cell writes the full-res training module unchanged (model, pair building, pixel metrics). The second writes `synplume.py`: range reads, caches, transmittance, GPU injection, the training loop, scene-level scoring and the go / no-go rule, the calibrated evaluation and the round-2 sampling weights.')
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
    if not need_disk(1.0 if SMOKE_TEST else 15.7, 'fine-tune zips (6.2 GB) + 10 m cache (8.9 GB) + ch4 (0.6 GB)'):
        raise SystemExit('Not enough disk for the fine-tune cache. Free space, or point CACHE_PATH / FT_WORK at an existing cache10m.npy.')
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
if os.path.exists(CACHE) and (CH4 is not None or not ENH_WEIGHTING) and ZIPDIR.startswith(DATA) and os.path.isdir(ZIPDIR):
    done_with(ZIPDIR, 'fine-tune tacozips')   # only zips this notebook downloaded; an earlier run's FT_WORK copy is left alone
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
TRAIN_ROWS = ptr.sample(min(PRETRAIN_N or len(ptr), len(ptr)), random_state=0).reset_index(drop=True)
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
# --- disk plan: pair caches + bank (zips are deleted after the bank cache is built); shrink the train pool only if it cannot fit
PAIR_GB = 8 * 200 * 200 * 2 / 1e9; CANVAS_GB = 160 * 192 * 2 / 1e9; sfx = '_smoke' if SMOKE_TEST else ''
def cache_left(name, n):   # GB still to be written for a (possibly partial) cache
    p = f'{DATA}/{name}{sfx}_{n}.npy'
    return 0.0 if os.path.exists(p) else max(0.0, n * PAIR_GB - allocated_gb(p + '.tmp'))
BANK_NEED = lambda n: (0 if SMOKE_TEST or not DOWNLOAD_BANK_ZIPS else 5.0) + min(n * BANK_K, 279 * 855) * CANVAS_GB * 0.8
need = cache_left('pairs_valfa', len(VAL_ROWS)) + cache_left('pairs_testfa', len(FA_ROWS)) + cache_left('pairs_train', len(TRAIN_ROWS)) + BANK_NEED(len(TRAIN_ROWS))
SHRUNK = None
if not need_disk(need, f'pair caches ({len(TRAIN_ROWS)} train + {len(VAL_ROWS)} val FA + {len(FA_ROWS)} test FA) + bank'):
    room = free_gb() - DISK_RESERVE_GB - cache_left('pairs_valfa', len(VAL_ROWS)) - cache_left('pairs_testfa', len(FA_ROWS)) - BANK_NEED(0)
    n_fit = int(room / (PAIR_GB + BANK_K * CANVAS_GB * 0.8))
    if n_fit < MIN_TRAIN_PAIRS: raise SystemExit(f'Not enough disk: room for only {max(n_fit, 0)} training pairs. Free space (or use a larger runtime) and rerun.')
    SHRUNK = (len(TRAIN_ROWS), n_fit); TRAIN_ROWS = TRAIN_ROWS.head(n_fit).reset_index(drop=True)
    print(f'*** DISK: training pool shrunk from {SHRUNK[0]} to {n_fit} pairs to fit (recorded in syn_results.json) ***')
print(f'cache plan: train {len(TRAIN_ROWS)}, val FA {len(VAL_ROWS)}, test FA {len(FA_ROWS)} pairs '
      f'= {(len(TRAIN_ROWS) + len(VAL_ROWS) + len(FA_ROWS)) * PAIR_GB:.1f} GB float16 | free disk {free_gb():.0f} GB')
TIMES['pretraining catalog'] = time.time() - t
''')
md('## 7. Build the pair caches (HTTP range reads)\nEach pair is two ~300 KB GeoTIFF leaves read straight out of the `.tacozip` files on the hub. Resumable: rerun the cell after a disconnect and it continues where it stopped.')
code(r'''
t = time.time(); RD.bytes = RD.remote = 0
CACHES = {}
for name, rows in (('pairs_valfa', VAL_ROWS), ('pairs_testfa', FA_ROWS), ('pairs_train', TRAIN_ROWS)):
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
BP = f'{DATA}/bank{sfx}_{len(TRAIN_ROWS)}x{BANK_K}.npy'
if not os.path.exists(BP) and not need_disk((5.0 if DOWNLOAD_BANK_ZIPS and not SMOKE_TEST else 0) + len(BROWS) * CANVAS_GB, f'bank ({len(BROWS)} canvases)'):
    raise SystemExit('Not enough disk for the plume bank. Set DOWNLOAD_BANK_ZIPS = False (range reads, slower) or free space.')
if DOWNLOAD_BANK_ZIPS and not SMOKE_TEST and not os.path.exists(BP):
    for z in sorted(BROWS['internal:source_file'].unique()):
        if not os.path.exists(f'{DATA}/{BSUB}/{z}'): hf_hub_download(REPO, f'{BSUB}/{z}', repo_type='dataset', local_dir=DATA)
if not SMOKE_TEST: drive_get(os.path.basename(BP).replace('.npy', ''))
CANVAS, CAN_SRC, LOST = sp.build_bank_cache(BROWS, BP, RD, BSUB, workers=READ_THREADS)
if not SMOKE_TEST: drive_put(os.path.basename(BP).replace('.npy', ''))
if os.path.isdir(f'{DATA}/{BSUB}') and glob.glob(f'{DATA}/{BSUB}/*.tacozip'): done_with(f'{DATA}/{BSUB}', 'bank zips')
PMETA = sp.scene_meta(TRAIN_ROWS)
TM = sp.Transmittance(TJ)
print(f'bank: {len(UB)} rows from {BROWS["methane:plume_uid"].nunique()} physical plumes for {len(TRAIN_ROWS)} scenes | {gb(BP):.2f} GB')
TIMES['bank'] = time.time() - t
''')
md('''## 8b. Unseen-like sampling weights for the plume-free pairs
Unseen test sites are, by definition, sites with no fine-tune train chip. These weights push the synthetic half of each batch toward the conditions the fine-tune **train** split covers least. They use only the training pool and the fine-tune train site list, with **no val or test statistics**:
* **novel site**: a pair at a site with no fine-tune train chip gets × `NOVEL_BOOST`. 82 % of the 24,918 eligible pairs sit at fine-tune train sites (the other 4,431 are at novel sites);
* **site balance**: count(site)^−`A_SITE`, so the large, often-revisited sites (mostly Algeria) stop dominating;
* **region balance**: count(country)^−`A_COUNTRY`;
* **surface balance**: count(albedo bin)^−`A_ALBEDO`. Albedo here is the mean target B11/B12 reflectance of the chip (a SWIR-brightness proxy; MethaneSET has no land-cover field), in 10 equal-width bins between the pool's 1st and 99th percentiles, so unusually dark or bright surfaces are drawn more often.

The weights are normalised and capped at 20×, then mixed 50/50 with uniform (`WEIGHT_MIX`), so half of the draws are still round-1 sampling. The 92 unseen-test sites stay excluded from the pool (section 6). This is a heuristic, not a fitted density ratio: nothing guarantees that the conditions are the ones that matter at the unseen sites. The section-12 diagnostic checks that after training.''')
code(r'''
t = time.time()
FT_TRAIN_SITES = set(l0_full[l0_full.split == 'train']['site:location_name'])   # training split only
ALBP = f'{DATA}/albedo_train{sfx}_{len(TRAIN_ROWS)}.npy'
if os.path.exists(ALBP): ALB = np.load(ALBP)
else: ALB = sp.swir_albedo(XP, TR_PAIRS, c0=4); np.save(ALBP, ALB)
RT = TRAIN_ROWS.iloc[TR_PAIRS]
W_POOL, WREP = sp.unseen_like_weights(RT['site:location_name'].values, RT['site:country'].values, ALB, FT_TRAIN_SITES,
                                      novel_boost=NOVEL_BOOST, a_site=A_SITE, a_country=A_COUNTRY, a_albedo=A_ALBEDO, mix=WEIGHT_MIX)
if not UNSEEN_WEIGHTING: W_POOL = np.ones(len(TR_PAIRS)); print('UNSEEN_WEIGHTING = False: uniform sampling')
print(f'pool {WREP["n"]} pairs at {WREP["n_sites"]} sites | effective sample size {WREP["effective_sample_size"]:.0f} | weight range {W_POOL.min():.2f}-{W_POOL.max():.2f}')
print(f'novel-site share of draws: uniform {WREP["novel_site_share"]:.3f} -> weighted {WREP["weighted"]["novel_site_share"]:.3f}')
print('country share, uniform: ', WREP['country_share']); print('country share, weighted:', WREP['weighted']['country_share'])
print(f'albedo quartiles, uniform {np.round(WREP["albedo_q25_q50_q75"], 3)} -> weighted {np.round(WREP["weighted"]["albedo_q25_q50_q75"], 3)}')
json.dump(WREP, open(f'{OUT}/sampling_weights.json', 'w'), indent=1, default=float)
TIMES['sampling weights'] = time.time() - t
''')
md('## 9. Initial weights: `fr_neg`, and hard negatives\nOrder: `INIT_CKPT`, then `RUNS/fr_neg/final.pt`, then the private `RUNS_REPO` (`PULL_PRIVATE_RUNS`). If any variant uses hard negatives, `fr_neg` scores every training pair once (no augmentation). The top `HN_TOP_FRAC` become hard negatives: never injected, and `HN_SHARE` of the plume-free draws. Training pairs only.')
code(r'''
CK0 = INIT_CKPT or f'{RUNS}/fr_neg/final.pt'
if not os.path.exists(CK0) and PULL_PRIVATE_RUNS:
    hf_hub_download(RUNS_REPO, 'fr_neg/final.pt', local_dir=RUNS); print('pulled fr_neg/final.pt from', RUNS_REPO)
assert os.path.exists(CK0), f'no fr_neg weights at {CK0}: set INIT_CKPT, or PULL_PRIVATE_RUNS = True with an HF token that can read {RUNS_REPO}'
SD0 = torch.load(CK0, map_location='cpu'); print('fr_neg weights:', CK0, '|', sum(v.numel() for k, v in SD0.items() if 'running' not in k and 'num_batches' not in k), 'tensor elements')
t = time.time()
HARD = np.zeros(len(XP), bool); W_HN = None; HN = {}
if any(RUN_SPECS[n]['hard'] for n in RUNS_TO_TRAIN):
    SCP = f'{RUNS}/fr_neg_scores_train{sfx}_{len(TRAIN_ROWS)}.npy'
    if os.path.exists(SCP): S_TR = np.load(SCP)
    else:
        m0 = sp.load_model('plain', SD0, DEV)
        S_TR = sp.scene_scores(m0, XP, TR_PAIRS, DEV, mf.amp_dtype_for(DEV) if AMP else None)['score']; np.save(SCP, S_TR); del m0
    h, thr_h = sp.hard_negatives(S_TR, HN_TOP_FRAC); HARD[TR_PAIRS[h]] = True
    W_HN = sp.boost_share(W_POOL, h, HN_SHARE)
    HN = dict(n_hard=int(h.sum()), score_threshold=thr_h, fr_neg_score_median=float(np.median(S_TR)), share_of_draws=HN_SHARE,
              fr_neg_fa_at_0p5_on_train_pairs=float((S_TR >= 0.5).mean()))
    print(f'hard negatives: {h.sum()} of {len(h)} training pairs (fr_neg scene score >= {thr_h:.4f}); fr_neg flags {HN["fr_neg_fa_at_0p5_on_train_pairs"]:.1%} '
          f'of the training pairs at 0.5 | {time.time() - t:.0f}s')
TIMES['hard negatives'] = time.time() - t
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
md('## 11. Stage 2: train the `syn2` variants (from `fr_neg`)\nEach step: 8 real fine-tune chips (fr_neg recipe, `neg_p` of them swapped for bg1-vs-bg0 plume-free pairs) + 8 plume-free pairs drawn with the section-8b weights, `inj_p` of them with a plume injected on the GPU (nearest sun geometry, rotated to the scene wind ±15°, source within ±600 m of the chip centre, Q log-uniform 0.5–20 t/h capped by the LES cut limit). Hard negatives are never injected. Loss: BCE + Dice with positives weighted by enhancement. Selection: best epoch by **val** hit rate at 5 % val false alarms (2021 only). Trackio logs every epoch. Before each variant the time budget is checked; a skipped variant is reported, never silently dropped.')
code(r'''
import trackio
TRAIN_TIMES, SKIPPED = {}, {}
for name in RUNS_TO_TRAIN:
    spec = RUN_SPECS[name]; t = time.time()
    finished = os.path.exists(f'{RUNS}/{name}/final.pt') and os.path.exists(f'{RUNS}/{name}/best.pt')
    est = (max(TRAIN_TIMES.values()) if TRAIN_TIMES else EPOCHS * EPOCH_S_EST)
    left = (TIME_BUDGET_MIN - EVAL_RESERVE_MIN) * 60 - (time.time() - T0) if TIME_BUDGET_MIN else float('inf')
    if not (finished and SKIP_DONE) and TRAIN_TIMES and est > left:
        SKIPPED[name] = f'time budget: ~{est / 60:.0f} min needed, {left / 60:.0f} min left of TIME_BUDGET_MIN={TIME_BUDGET_MIN}'
        print(f'[{name}] SKIPPED ({SKIPPED[name]}). Rerun this cell later (SKIP_DONE keeps finished runs) or raise TIME_BUDGET_MIN.'); continue
    w = (W_HN if spec['hard'] else W_POOL) if spec['weighted'] else None
    sp.train_syn(name, 'plain', XFT, meta, CH4, XP, TR_PAIRS, ASSIGN, CANVAS, CAN_SRC, EDGE, PMETA, TM, DEV, RUNS, init_sd=SD0, XV=XV,
                 val_neg=VAL_NEG, epochs=EPOCHS, lr=LR, bs_real=BS_REAL, bs_syn=BS_SYN, inj_p=spec['inj_p'], neg_p=spec['neg_p'],
                 q_min=Q_MIN, q_max=Q_MAX, floor_sigma=FLOOR, enh_weighting=ENH_WEIGHTING, amp=AMP, workers=WORKERS,
                 project=TRACKIO_PROJECT, skip_done=SKIP_DONE, max_steps=MAX_STEPS, preview_every=1 if SMOKE_TEST else 5,
                 pair_weights=w, hard=HARD if spec['hard'] else None,
                 extra_cfg=dict(variant=spec, unseen_weighting=bool(spec['weighted'] and UNSEEN_WEIGHTING), hn_share=HN_SHARE if spec['hard'] else 0))
    if not (finished and SKIP_DONE): TRAIN_TIMES[name] = time.time() - t
    print(f'[{name}] {(time.time() - t) / 60:.1f} min | elapsed since setup {(time.time() - T0) / 60:.0f} min')
TIMES['training'] = sum(TRAIN_TIMES.values())
TRAINED = [n for n in RUNS_TO_TRAIN if os.path.exists(f'{RUNS}/{n}/best.pt')]
print('trained:', TRAINED, '| skipped:', SKIPPED or 'none')
''')
md('''## 12. Stage 3: evaluate `fr_neg` and every `syn2` variant identically
Scene score = max of the 2×2-pooled probability (20 m). *Detect* = score ≥ threshold. *Hit* = a flagged 20 m pixel inside the plume mask.

**Three kinds of threshold:**
* **fixed 0.5**, as on the released card;
* **test operating points**: per model, the thresholds giving 10 / 5 / 1 % false alarms on the 8,740 real 2024 plume-free pairs (`real2024_strict`). These are what criteria 1 and 5 use;
* **val-calibrated** (new): per model, the threshold giving `CAL_FA` = 5 % false alarms on the 2021 val plume-free pairs. It is chosen without any test data, so it is the threshold a model would actually ship with. At it we report pixel IoU/F1 (seen / unseen / all), scene detect and hit rates, and the false-alarm rate it really gives on the 2024 plume-free sets (does 5 % on val transfer?).

The candidate is picked on **val** only.''')
code(r'''
t = time.time()
amp_dtype = mf.amp_dtype_for(DEV) if AMP else None
MODELS = {'fr_neg': ('plain', SD0)}
for name in TRAINED: MODELS[name] = ('plain', torch.load(f'{RUNS}/{name}/best.pt', map_location='cpu'))
RES, RAW = {}, {}
for name, (kind, sd) in MODELS.items():
    m = sp.load_model(kind, sd, DEV)
    RES[name], RAW[name] = sp.evaluate(name, m, XFT, TE, XF, FA_SETS, XV, VAL_NEG, va, DEV, amp_dtype, WORKERS, cal_fa=CAL_FA)
    del m
    r = RES[name]; cal = r.get('calibrated', {}); s = r['scene'].get('unseen|real2024_strict', {}); cs = cal.get('scene', {}).get('unseen|real2024_strict', {})
    print(f'{name:9s} val hit@5%valFA {r["val"].get("ops", {}).get("5%", {}).get("hit", float("nan")):.3f} | thr_cal {cal.get("threshold", float("nan")):.4f} | '
          f'unseen: IoU@0.5 {r["pixel"]["unseen"]["20m"]["iou"]:.3f} IoU@cal {cal.get("pixel", {}).get("unseen", {}).get("20m", {}).get("iou", float("nan")):.3f} '
          f'hit@5%realFA {s.get("ops", {}).get("5%", {}).get("hit", float("nan")):.3f} hit@cal {cs.get("hit", float("nan")):.3f} AP {s.get("AP", float("nan")):.3f} '
          f'| real FA @0.5 {s.get("real_fa_at_0p5", float("nan")):.3f} @cal {cs.get("real_fa", float("nan")):.3f}')
CAND = max(TRAINED, key=lambda n: RES[n]['val'].get('ops', {}).get('5%', {}).get('hit', -1)) if TRAINED else None
print('val-selected candidate:', CAND)
TIMES['evaluation'] = time.time() - t
# --- tables
rows = []
for name, r in RES.items():
    cal = r.get('calibrated', {})
    for sub, px in r['pixel'].items():
        for fs in FA_SETS:
            s = r['scene'].get(f'{sub}|{fs}')
            if not s: continue
            cp = cal.get('pixel', {}).get(sub, {}).get('20m', {}); cs = cal.get('scene', {}).get(f'{sub}|{fs}', {})
            row = dict(model=name, plumes=sub, n_plumes=s['n_pos'], fa_set=fs, n_fa=s['n_neg'], iou_0p5_20m=px['20m']['iou'], f1_0p5_20m=px['20m']['f1'],
                       det_0p5_20m=px['20m']['det_rate'], bg1_fa_0p5_20m=px['20m']['false_alarm'], AP=s['AP'], real_fa_0p5=s['real_fa_at_0p5'],
                       hit_0p5=s['hit_at_0p5'], thr_cal=cal.get('threshold', np.nan), val_fa_cal=cal.get('val', {}).get('real_fa', np.nan),
                       iou_cal_20m=cp.get('iou', np.nan), f1_cal_20m=cp.get('f1', np.nan), bg1_fa_cal_20m=cp.get('false_alarm', np.nan),
                       detect_cal=cs.get('detect', np.nan), hit_cal=cs.get('hit', np.nan), real_fa_cal=cs.get('real_fa', np.nan))
            for op, o in s['ops'].items(): row.update({f'thr@{op}': o['threshold'], f'hit@{op}': o['hit'], f'detect@{op}': o['detect']})
            rows.append(row)
CMP = pd.DataFrame(rows); CMP.to_csv(f'{OUT}/syn_comparison.csv', index=False, float_format='%.4f')
pd.set_option('display.width', 250)
print('\n== test operating points (thresholds set on real2024_strict) ==')
print(CMP[CMP.fa_set == 'real2024_strict'][['model', 'plumes', 'n_plumes', 'iou_0p5_20m', 'AP', 'real_fa_0p5', 'hit@10%', 'hit@5%', 'hit@1%']].round(3).to_string(index=False))
print('\n== val-calibrated threshold (5 % FA on 2021 val plume-free pairs): pixel metrics on the 20 m grid, scene rates, real 2024 FA it gives ==')
print(CMP[CMP.fa_set == 'real2024_strict'][['model', 'plumes', 'thr_cal', 'val_fa_cal', 'iou_0p5_20m', 'iou_cal_20m', 'f1_0p5_20m', 'f1_cal_20m',
                                            'bg1_fa_0p5_20m', 'bg1_fa_cal_20m', 'detect_cal', 'hit_cal', 'real_fa_cal']].round(3).to_string(index=False))
print('\nreal FA at the calibrated threshold, per 2024 set:')
print(CMP[CMP.plumes == 'all'][['model', 'fa_set', 'n_fa', 'thr_cal', 'real_fa_cal', 'real_fa_0p5']].round(4).to_string(index=False))
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
    for k, name in enumerate(RES):
        neg = RAW[name]['neg']['real2024_strict']; pos = RAW[name]['pos'].get(sub)
        if pos is None or not len(neg): continue
        ts = np.unique(np.r_[np.quantile(neg, np.linspace(0, 1, 200)), 0.5]); fa = [(neg >= x).mean() for x in ts]; hit = [(pos['score_in'] >= x).mean() for x in ts]
        l, = a.plot(fa, hit, label=name)
        tc = RES[name].get('calibrated', {}).get('threshold')
        if tc is not None: a.plot([(neg >= tc).mean()], [(pos['score_in'] >= tc).mean()], 'o', color=l.get_color(), ms=5)
    for x in (0.01, 0.05, 0.10): a.axvline(x, color='grey', lw=.5, ls='--')
    a.set_xscale('log'); a.set_xlabel('real false-alarm rate (2024 plume-free pairs)'); a.set_ylabel('hit rate')
    a.set_title(f'{sub} test plumes (dot: val-calibrated threshold)', fontsize=9); a.legend(fontsize=8)
f.tight_layout(); f.savefig(f'{OUT}/hit_vs_real_fa.png', dpi=90); plt.close(f)
from IPython.display import Image, display
display(Image(f'{OUT}/pr_curves.png')); display(Image(f'{OUT}/hit_vs_real_fa.png'))
''')
md('**Diagnostic (after training; nothing here feeds back into training, weighting or selection):** how do the uniform and the weighted plume-free pools compare with the fine-tune train chips and the unseen-site test chips? It shows the share of chips at sites with no fine-tune train chip, the country mix and the SWIR albedo quartiles.')
code(r'''
def chips_report(idx):
    m_ = meta.iloc[idx]; a_ = sp.swir_albedo(XFT, idx, c0=5)
    return sp.pool_report(m_['site:location_name'].values, m_['site:country'].values, a_, ~m_['site:location_name'].isin(FT_TRAIN_SITES).values)
DIAG = {'fine-tune train chips': chips_report(tr), 'plume-free pool, uniform': {k: WREP[k] for k in ('country_share', 'novel_site_share', 'n_sites', 'albedo_q25_q50_q75')},
        'plume-free pool, weighted': WREP['weighted'], 'unseen-site test chips': chips_report(TE['unseen'])}
for k, v in DIAG.items():
    print(f'{k:28s} novel-site share {v["novel_site_share"]:.3f} | albedo q25/50/75 {np.round(v["albedo_q25_q50_q75"], 3)} | top countries {dict(list(v["country_share"].items())[:5])}')
json.dump(DIAG, open(f'{OUT}/pool_diagnostic.json', 'w'), indent=1, default=float)
''')
md('## 13. Go / no-go (the rule stated at the top and in `out/preregistered_rule.json`)\nLatency: both models exported to ONNX (opset 17, static 1×8×200×200) and timed with onnxruntime on this CPU, one thread.')
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
    GO, CRIT = sp.go_no_go(RES[CAND], RES['fr_neg'], lat_ratio, ok0 and okc, sp.n_params(mc), sp.n_params(m0), fa_key='real2024_strict', calibrated=True)
    print(f'candidate {CAND}: params {sp.n_params(mc):,} vs fr_neg {sp.n_params(m0):,} | ORT CPU 1 thread {LAT[CAND]:.1f} ms vs {LAT["fr_neg"]:.1f} ms')
    for k, v in CRIT.items(): print(('  PASS ' if v['ok'] else '  FAIL ') + k + f'  ({v["value"]:.4f})' if not isinstance(v['value'], bool) else ('  PASS ' if v['ok'] else '  FAIL ') + k)
if SMOKE_TEST:
    print('smoke test: the decision below is meaningless')
    if SMOKE_FORCE_GO and CAND: GO = True; print('SMOKE_FORCE_GO: forcing the GO branch to exercise the export code')
print('DECISION:', 'GO' if GO else 'NO-GO (keep fr_neg)')
SUMMARY = dict(round=2, smoke_test=SMOKE_TEST, gpu=GPU, decision='GO' if GO else 'NO-GO', candidate=CAND, criteria=CRIT, rule=RULE,
               latency_ms_ort_cpu_1t=LAT, gate=GATE, label_floor_sigma=FLOOR, results=RES, trained=TRAINED, skipped=SKIPPED,
               run_specs={n: RUN_SPECS[n] for n in RUNS_TO_TRAIN}, sampling_weights=WREP, hard_negatives=HN, pool_diagnostic=DIAG,
               fa_sets={k: int(len(v)) for k, v in FA_SETS.items()}, n_train_pairs=int(len(TR_PAIRS)), n_val_fa=int(len(VAL_NEG)), bank_rows=int(len(UB)),
               train_pool_shrunk_for_disk=SHRUNK,
               config=dict(EPOCHS=EPOCHS, LR=LR, BS_REAL=BS_REAL, BS_SYN=BS_SYN, Q_MIN=Q_MIN, Q_MAX=Q_MAX, PRETRAIN_N=PRETRAIN_N, VAL_FA_N=VAL_FA_N,
                           MIN_CLEAR=MIN_CLEAR, BANK_K=BANK_K, ENH_WEIGHTING=ENH_WEIGHTING, CAL_FA=CAL_FA, UNSEEN_WEIGHTING=UNSEEN_WEIGHTING,
                           NOVEL_BOOST=NOVEL_BOOST, A_SITE=A_SITE, A_COUNTRY=A_COUNTRY, A_ALBEDO=A_ALBEDO, WEIGHT_MIX=WEIGHT_MIX,
                           HN_TOP_FRAC=HN_TOP_FRAC, HN_SHARE=HN_SHARE, TIME_BUDGET_MIN=TIME_BUDGET_MIN),
               times_s=TIMES, train_times_s=TRAIN_TIMES, downloaded_gb=DL_GB)
json.dump(SUMMARY, open(f'{OUT}/syn_results.json', 'w'), indent=1, default=float)
print('wrote', f'{OUT}/syn_results.json')
''')
md('## 14. Only on GO: export the candidate and write its card\nSafetensors, ONNX (parity-checked against PyTorch), `config.json` (with the val-calibrated threshold) and `CANDIDATE_CARD.md`. Nothing is written on NO-GO.')
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
    TC = RES[CAND]['calibrated']['threshold']
    json.dump(dict(input_name='reflectance', input_shape=[1, 8, 200, 200], input_bands=mf.INPUT_BANDS, scale='L1C DN / 10000', resolution_m=10,
                   output_name='plume_logits', threshold=TC, threshold_note=f'val-calibrated: {CAL_FA:.0%} false alarms on 2021 val plume-free pairs; 0.5 is the round-1 / fr_neg default',
                   threshold_0p5_reference=0.5, model=CAND, init='fr_neg', status='candidate (not released)'), open(f'{OUT}/config.json', 'w'), indent=1)
    R = RES[CAND]; B = RES['fr_neg']; op = lambda r, sub, k='hit', o='5%': r['scene'][f'{sub}|real2024_strict']['ops'][o][k]
    lines = '\n'.join(f'| {sub} / {n} | {RES[n]["pixel"][sub]["20m"]["iou"]:.3f} | {RES[n]["scene"][f"{sub}|real2024_strict"]["AP"]:.3f} | '
                      f'{op(RES[n], sub, o="10%"):.3f} | {op(RES[n], sub):.3f} | {op(RES[n], sub, o="1%"):.3f} |'
                      for sub in ('unseen', 'seen', 'all') if f'{sub}|real2024_strict' in RES[CAND]['scene'] for n in ('fr_neg', CAND))
    cl = lambda r, sub: r['calibrated']
    lines_cal = '\n'.join(f'| {sub} / {n} | {cl(RES[n], sub)["threshold"]:.4f} | {cl(RES[n], sub)["pixel"][sub]["20m"]["iou"]:.3f} | {cl(RES[n], sub)["pixel"][sub]["20m"]["f1"]:.3f} | '
                          f'{cl(RES[n], sub)["scene"][f"{sub}|real2024_strict"]["hit"]:.3f} | {cl(RES[n], sub)["scene"][f"{sub}|real2024_strict"]["real_fa"]:.3f} |'
                          for sub in ('unseen', 'seen', 'all') if sub in RES[CAND]['calibrated']['pixel'] for n in ('fr_neg', CAND))
    crit = '\n'.join(f'| {k} | {"pass" if v["ok"] else "fail"} | {v["value"] if isinstance(v["value"], bool) else round(v["value"], 4)} |' for k, v in CRIT.items())
    spec = RUN_SPECS[CAND]
    card = f"""# Candidate: `{CAND}` (synthetic-plume training, round 2), not released

Trained in Colab ({GPU}) from `fr_neg` with plumes from the MethaneSET WRF-LES bank injected into real plume-free Sentinel-2 pairs
(`colab/PLAN_synthetic_plumes.md`, round 2). Same input contract as the released model: `reflectance` float32 1x8x200x200 -> `plume_logits`.
It passed the go / no-go rule (fixed before the run) against `fr_neg` evaluated identically. **It replaces nothing until Simon decides to release it.**

## Results (test plumes vs {len(FA_SETS['real2024_strict'])} real 2024 plume-free pairs at sites in no training split)

Rows: plume subset / model. IoU at 0.5 on the 20 m grid. Hit rate = a flagged 20 m pixel inside the plume mask, at the threshold where the real
false-alarm rate is 10 / 5 / 1 %.

| subset / model | IoU@0.5 | AP | hit @10% FA | hit @5% FA | hit @1% FA |
|---|---:|---:|---:|---:|---:|
{lines}

**At the val-calibrated threshold** (each model's threshold for {CAL_FA:.0%} false alarms on the 2021 val plume-free pairs; no test data used):

| subset / model | threshold | IoU (20 m) | F1 (20 m) | hit | real 2024 FA |
|---|---:|---:|---:|---:|---:|
{lines_cal}

Unseen = {len(TE['unseen'])} test chips at 92 sites with no train chip within 1 km. 2021 val (selection) hit rate at 5 % val FA:
`fr_neg` {B['val']['ops']['5%']['hit']:.3f}, `{CAND}` {R['val']['ops']['5%']['hit']:.3f}.

## Go / no-go

| criterion | result | value |
|---|---|---|
{crit}

## How it was trained

- Real half: the 2,362 fine-tune train chips with fr_neg's recipe ({spec['neg_p']:.0%} bg1-vs-bg0 plume-free pairs).
- Plume-free half: {len(TR_PAIRS)} pretraining train pairs (2018-20, 2022-23), none at the 92 unseen-test sites, drawn with unseen-like sampling weights
  (novel sites x{NOVEL_BOOST:g}, site / country / albedo-bin balance, mixed {WEIGHT_MIX:.0%} with uniform; training-split information only).
  {spec['inj_p']:.0%} get a plume: nearest bank sun geometry, rotated to the scene wind, Q log-uniform {Q_MIN / 1000:g}-{Q_MAX / 1000:g} t/h, multiplicative B11/B12 transmittance.
  {'Hard negatives: the ' + format(HN_TOP_FRAC, '.0%') + ' of training pairs fr_neg scores highest, never injected, ' + format(HN_SHARE, '.0%') + ' of the plume-free draws.' if spec['hard'] else 'No hard-negative mining.'}
- Label floor: {FLOOR} x scene noise (matched to real label areas in the sanity gate). Loss: BCE + Dice, positives weighted by enhancement.
- {EPOCHS} epochs, lr {LR}, best epoch by 2021 val hit rate at 5 % val false alarms.

## Caveats

- Plume-free means "no MARS detection", so the real false-alarm rates are slightly pessimistic.
- Chips are centred on known emitter sites; onboard use needs tiling and a co-registered reference.
- Not tested on a Jetson. INT8 may erase the ~1 % SWIR signal; validate any engine against the ONNX.
- Data CC BY-NC-SA 4.0 (MethaneSET) and CC BY 4.0 (bank): non-commercial.
"""
    open(f'{OUT}/CANDIDATE_CARD.md', 'w').write(card); print(card[:3000])
else:
    print('NO-GO (or no candidate): nothing exported. fr_neg stays the release. Results are in', f'{OUT}/syn_results.json')
''')
md('## 15. Trackio dashboard (off by default)\n`SYNC_TRACKIO = True` syncs to a **new private** Space `TRACKIO_SPACE`. It refuses if that Space already exists and is public.')
code(r'''
if SYNC_TRACKIO and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    from huggingface_hub import HfApi
    try: info = HfApi().space_info(TRACKIO_SPACE); assert info.private, f'{TRACKIO_SPACE} exists and is public: not syncing (pick another TRACKIO_SPACE)'
    except Exception as e:
        if isinstance(e, AssertionError): raise
    subprocess.run(['trackio', 'sync', '--project', TRACKIO_PROJECT, '--space-id', TRACKIO_SPACE, '--private', '--force'], check=True)
    assert HfApi().space_info(TRACKIO_SPACE).private, 'the Space is not private: make it private on the hub now'
    print('dashboard (private):', f'https://huggingface.co/spaces/{TRACKIO_SPACE}')
else: print('SYNC_TRACKIO is False (or smoke test): nothing published. Local Trackio DB:', os.environ['TRACKIO_DIR'])
''')

md('## 16. Uploads (off by default)\n`UPLOAD = True` and a **GO**: create the **private** repo `HF_CANDIDATE_REPO` and upload `out/`. The released `sighmon/methane-plume-se2-fullres` is never touched here. `SAVE_RUNS_PRIVATE = True`: push the `syn2*` checkpoints to the private `RUNS_REPO` under `syn_r2/`.')
code(r'''
from huggingface_hub import HfApi
if UPLOAD and GO and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    api = HfApi(); api.create_repo(HF_CANDIDATE_REPO, repo_type='model', private=True, exist_ok=True)
    assert api.model_info(HF_CANDIDATE_REPO).private, f'{HF_CANDIDATE_REPO} is not private: not uploading'
    api.upload_folder(folder_path=OUT, repo_id=HF_CANDIDATE_REPO, repo_type='model', path_in_repo=f'round2_{CAND}',
                      commit_message=f'Synthetic-plume round-2 candidate {CAND} (GO vs fr_neg)')
    print('uploaded (private):', f'https://huggingface.co/{HF_CANDIDATE_REPO}')
else: print('UPLOAD off, NO-GO or smoke test: nothing uploaded. Files in', OUT, sorted(os.listdir(OUT)))
if SAVE_RUNS_PRIVATE and not SMOKE_TEST:
    assert HF_OK, 'log in first (section 4)'
    assert HfApi().model_info(RUNS_REPO).private, f'{RUNS_REPO} is not private: not uploading'
    HfApi().upload_folder(folder_path=RUNS, repo_id=RUNS_REPO, path_in_repo='syn_r2', allow_patterns=['syn2*/*.pt', 'syn2*/*.json'],
                          commit_message='Synthetic-plume round-2 checkpoints (private)')
    print('checkpoints pushed to', RUNS_REPO, '(private) under syn_r2/')
''')

md('## 17. Time and disk summary')
code(r'''
TIMES['total'] = time.time() - T0
du = lambda p: sum(os.path.getsize(f) for f in glob.glob(f'{p}/**', recursive=True) if os.path.isfile(f)) / 1e9 if os.path.exists(p) else 0
print({k: f'{v / 60:.1f} min' for k, v in TIMES.items()})
print(f'disk: {WORK} {du(WORK):.1f} GB, fine-tune data {du(FT_DATA):.1f} GB, free {free_gb():.0f} GB | peak RSS of this kernel {psutil.Process().memory_info().rss / 1e9:.1f} GB '
      f'| RAM total {psutil.virtual_memory().total / 1e9:.0f} GB')
json.dump(dict(times_s=TIMES, disk_gb=dict(work=du(WORK), finetune_data=du(FT_DATA), free=free_gb())), open(f'{OUT}/run_times.json', 'w'), indent=1)
''')

nb.cells = C
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'},
               'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4', 'machine_shape': 'hm'}}
TIME_TABLE = open(f'{HERE}/time_table_r2.md').read() if os.path.exists(f'{HERE}/time_table_r2.md') else ''
nb.cells[0].source = nb.cells[0].source.replace('TIME_TABLE_PLACEHOLDER', TIME_TABLE).replace('RULE_MD', RULE)
nbf.validate(nb)
nbf.write(nb, f'{HERE}/synthetic_plumes_r2_colab.ipynb'); print('written', len(C), 'cells')
