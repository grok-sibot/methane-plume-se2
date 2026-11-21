"""Synthetic-plume training for the full-res methane detector (colab/PLAN_synthetic_plumes.md).

Written by synthetic_plumes_colab.ipynb next to methane_fullres.py, which it imports as `mf` (model, pair building,
pixel metrics). Nothing here changes the onboard input contract: `reflectance` float32 1x8x200x200 (L1C DN/10000,
10 m; target B2,B3,B4,B8A,B11,B12 then reference B11,B12) -> `plume_logits` 1x1x200x200.

Pieces
  * LeafReader: HTTP range reads of single GeoTIFF leaves out of the MethaneSET .tacozip files on the HF hub
    (or from a local copy of the zip), so the 36.9 GB pretraining set is never downloaded whole.
  * pair caches: plume-free pretraining pairs as the 8 model channels, float16, memory-mapped, resumable.
  * bank cache: WRF-LES XCH4 enhancements (ppb at 3000 kg/h, 20 m, wind -> +x) on a fixed canvas.
  * Transmittance: MethaneSET integrated_transmittances.json -> relative B11/B12 transmittance T(1800+d)/T(1800).
  * inject(): GPU, per batch. Rotate the plume to the scene wind, place it, scale to Q, multiply target B11/B12.
  * MethaneNetMBMP: MethaneNet plus a Varon MBMP channel computed in-graph (fp32); same 8-channel input.
  * train_syn(), scene scores, PR / AP and operating points at fixed real false-alarm rates.
"""
import os, io, json, time, math, threading, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import methane_fullres as mf

HF_DS = 'https://huggingface.co/datasets/tacofoundation/methaneset/resolve/main'
BANDS0 = mf.BANDS0                       # B2,B3,B4,B8A,B11,B12 (0-based in the 13-band stack)
CANVAS_H, CANVAS_W, CAN_R, CAN_C = 160, 192, 80.0, 32.0   # bank canvas at 20 m; source pixel centre at (row 80, col 32)
SENSORS = ['S2A', 'S2B', 'S2C']
BG_PPB = 1800.0

# ----------------------------------------------------------------------------------------- leaf reads
class LeafReader:
    """Read byte ranges (one GeoTIFF leaf) from tacozip files: local file if present, else HTTP range request.
    The signed CDN URL behind each HF `resolve` link is looked up once per file and refreshed on 403.
    HF auth headers (if any token is configured) are sent to huggingface.co only and never printed."""
    def __init__(self, local_dirs=()):
        self.local_dirs = [d for d in local_dirs if d]; self.loc = {}; self.tl = threading.local(); self.lock = threading.Lock()
        try:
            from huggingface_hub.utils import build_hf_headers; self.h = build_hf_headers()
        except Exception: self.h = {}
        self.bytes = 0; self.remote = 0; self.retries = {}
    def _s(self):
        import requests
        if not hasattr(self.tl, 's'): self.tl.s = requests.Session()
        return self.tl.s
    def _local(self, sub, f):
        for d in self.local_dirs:
            for p in (f'{d}/{sub}/{f}', f'{d}/{f}'):
                if os.path.exists(p): return p
        return None
    def _url(self, sub, f, refresh=False):
        k = (sub, f)
        if refresh or k not in self.loc:
            r = self._s().head(f'{HF_DS}/{sub}/{f}', headers=self.h, allow_redirects=False, timeout=60)
            u = r.headers.get('location') or f'{HF_DS}/{sub}/{f}'
            with self.lock: self.loc[k] = u
        return self.loc[k]
    def read(self, sub, f, off, size):
        if not isinstance(f, str) or not f.endswith('.tacozip'):
            raise TypeError(f'leaf file name must be a .tacozip str, got {type(f).__name__}: {str(f)[:80]!r}')
        off, size = int(off), int(size)
        p = self._local(sub, f)
        if p:
            with open(p, 'rb') as fh: fh.seek(off); b = fh.read(size)
        else:
            for attempt in range(6):
                try:
                    r = self._s().get(self._url(sub, f, refresh=attempt > 0), headers={'Range': f'bytes={off}-{off + size - 1}'}, timeout=120)
                    if r.status_code == 206 and len(r.content) == size: b = r.content; break
                    why = f'HTTP {r.status_code}'
                except Exception as e: why = type(e).__name__
                with self.lock: self.retries[why] = self.retries.get(why, 0) + 1
                time.sleep(min(30, 2 ** attempt) * (0.5 + np.random.rand()))
            else: raise IOError(f'range read failed: {sub}/{f} @{off}+{size}')
            self.remote += 1
        self.bytes += len(b); return b

def decode(b, bands=None):
    import rasterio
    from rasterio.io import MemoryFile
    with MemoryFile(b) as m, m.open() as d:
        return d.read([i + 1 for i in bands] if bands is not None else None).astype(np.float32)

def leaf_index(l1):
    """(folder id, leaf id) -> (offset, size, source_file) as plain Python int/int/str, in a dict.
    Built once, single-threaded. Worker threads only read the dict. (Concurrent DataFrame.loc lookups on an unsorted
    MultiIndex are not thread-safe in pandas 2.2, the Colab version: a few of them return a one-row DataFrame, so the
    "file name" became a pandas Series and the range read failed.) Duplicate keys are counted: identical duplicates
    keep the first row, conflicting duplicates raise."""
    pid = l1['internal:relative_path'].str.split('/').str[0].values
    lid = l1['id'].astype(str).values
    off = l1['internal:offset'].astype('int64').values; size = l1['internal:size'].astype('int64').values
    src = l1['internal:source_file'].astype(str).values
    out, ndup, conflict = {}, 0, []
    for k in zip(pid, lid, off, size, src):
        key, val = (str(k[0]), str(k[1])), (int(k[2]), int(k[3]), str(k[4]))
        if key in out:
            ndup += 1
            if out[key] != val: conflict.append((key, out[key], val))
            continue
        out[key] = val
    if ndup: print(f'leaf_index: {ndup} duplicate (folder, leaf) rows, {len(conflict)} conflicting')
    if conflict: raise ValueError(f'conflicting duplicate leaves, e.g. {conflict[:3]}')
    return out

def _parallel(fn, n, workers, label, every=500):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    t = time.time(); done = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, i) for i in range(n)]
        for f in as_completed(futs):
            f.result(); done += 1
            if done % every == 0 or done == n: print(f'  {label} {done}/{n}  {time.time() - t:.0f}s', flush=True)
    return time.time() - t

def build_pair_cache(rows, L1, sub, out_npy, reader, workers=64):
    """Plume-free pairs -> (N,8,200,200) float16 [target B2,B3,B4,B8A,B11,B12, reference B11,B12], DN/10000.
    Resumable: a sidecar .done.npy marks finished rows; rows that fail to read are marked in .bad.npy (excluded later)."""
    N = len(rows); tmp = out_npy + '.tmp'; dn = out_npy + '.done.npy'; bad_p = out_npy + '.bad.npy'
    if os.path.exists(out_npy):
        return np.load(out_npy, mmap_mode='r'), (np.load(bad_p) if os.path.exists(bad_p) else np.zeros(N, bool))
    X = np.lib.format.open_memmap(tmp, 'r+' if os.path.exists(tmp) and os.path.exists(dn) else 'w+', np.float16, (N, 8, 200, 200))
    done = np.load(dn) if os.path.exists(dn) else np.zeros(N, bool)
    bad = np.load(bad_p) if os.path.exists(bad_p) else np.zeros(N, bool)
    todo = np.where(~done)[0]; ids = [str(v) for v in rows['id'].values]; lock = threading.Lock(); cnt = [0]
    print(f'pair cache {out_npy}: {N} pairs, {len(todo)} to fetch')
    def work(k):
        i = int(todo[k])
        try:
            to, ts, tf = L1[(ids[i], 'target')]; ro, rs, rf = L1[(ids[i], 'reference')]
            a = decode(reader.read(sub, tf, to, ts), BANDS0)
            b = decode(reader.read(sub, rf, ro, rs), BANDS0[4:6])
            assert a.shape == (6, 200, 200) and b.shape == (2, 200, 200)
            X[i] = np.concatenate([a, b]) / 10000.0
        except Exception as e:
            bad[i] = True; print('  bad pair', i, type(e).__name__, str(e)[:120])
        with lock:
            done[i] = True; cnt[0] += 1
            if cnt[0] % 1000 == 0: X.flush(); np.save(dn, done); np.save(bad_p, bad)
    s = _parallel(work, len(todo), workers, 'pairs')
    X.flush(); del X; np.save(bad_p, bad); os.replace(tmp, out_npy)
    if os.path.exists(dn): os.remove(dn)
    print(f'  pair cache done in {s:.0f}s ({len(todo) / max(s, 1e-9):.1f} pairs/s), {bad.sum()} bad, '
          f'{os.path.getsize(out_npy) / 1e9:.2f} GB on disk')
    return np.load(out_npy, mmap_mode='r'), bad

def build_leaf_cache(l0, L1, leaf, sub, out_npy, reader, workers=32, band=1):
    """One single-band leaf per fine-tune sample (e.g. the MBMP `ch4` leaf) -> (N,200,200) float32.
    Resumable like the pair caches: a .done.npy sidecar marks finished rows. A partial .tmp without a sidecar (left by
    an older version) is discarded and rebuilt. Every key is checked before any read starts."""
    if os.path.exists(out_npy): return np.load(out_npy, mmap_mode='r')
    N = len(l0); ids = [str(v) for v in l0['id'].values]; tmp = out_npy + '.tmp'; dn = out_npy + '.done.npy'
    missing = [i for i in ids if (i, leaf) not in L1]
    if missing: raise KeyError(f'{len(missing)} samples have no {leaf!r} leaf, e.g. {missing[:3]}')
    resume = os.path.exists(tmp) and os.path.exists(dn)
    W = np.lib.format.open_memmap(tmp, 'r+' if resume else 'w+', np.float32, (N, 200, 200))
    done = np.load(dn) if resume else np.zeros(N, bool); todo = np.where(~done)[0]; lock = threading.Lock(); cnt = [0]
    print(f'{leaf} cache {out_npy}: {N} samples, {len(todo)} to read' + (' (resuming)' if resume else ''))
    def work(k):
        i = int(todo[k]); o, sz, f = L1[(ids[i], leaf)]
        W[i] = decode(reader.read(sub, f, o, sz), [band - 1])[0]
        with lock:
            done[i] = True; cnt[0] += 1
            if cnt[0] % 500 == 0: W.flush(); np.save(dn, done)
    try:
        s = _parallel(work, len(todo), workers, leaf, 1000)
    finally:
        W.flush(); np.save(dn, done)
    del W; os.replace(tmp, out_npy); os.remove(dn)
    print(f'  {leaf} cache done in {s:.0f}s'); return np.load(out_npy, mmap_mode='r')

# ---------------------------------------------------------------------------------------------- bank
def assign_bank(scenes, bank, k, seed=0, sza_max=71.03):
    """For each scene pick k random physical plumes and, for each, the bank row with the nearest sun position
    (distance between solar-parallax vectors tan(SZA)*(cos RAA, sin RAA), the bank's own 60 m spacing).
    Scenes without a RAA (calm wind) get a random RAA. Returns (n_scenes, k) bank row numbers."""
    rng = np.random.default_rng(seed)
    uids = bank['methane:plume_uid'].values
    first = bank[uids == uids[0]]
    sun = np.stack([np.tan(np.radians(first['sun:sza'].values)) * np.cos(np.radians(first['sun:raa'].values)),
                    np.tan(np.radians(first['sun:sza'].values)) * np.sin(np.radians(first['sun:raa'].values))], 1)
    sun_key = list(zip(first['sun:sza'].round(4), first['sun:raa'].round(4)))
    pos = {(u, s): i for i, (u, s) in enumerate(zip(uids, zip(bank['sun:sza'].round(4), bank['sun:raa'].round(4))))}
    plumes = np.unique(uids)
    sza = np.clip(scenes['target:sza'].values, 0, sza_max)
    raa = scenes['target:raa'].values.copy(); nr = ~np.isfinite(raa); raa[nr] = rng.uniform(0, 360, nr.sum())
    v = np.stack([np.tan(np.radians(sza)) * np.cos(np.radians(raa)), np.tan(np.radians(sza)) * np.sin(np.radians(raa))], 1)
    near = np.argmin(((v[:, None, :] - sun[None]) ** 2).sum(-1), 1)
    out = np.zeros((len(scenes), k), np.int64)
    for i in range(len(scenes)):
        for j, u in enumerate(rng.choice(plumes, k, replace=False)):
            out[i, j] = pos[(u, sun_key[near[i]])]
    return out

def build_bank_cache(bank_rows, out_npy, reader, sub='methaneset-bank', workers=64):
    """Bank rows -> (M,160,192) float16 canvas at 20 m, ppb at 3000 kg/h, wind -> +x (columns), +y up (row 0 = top).
    Each crop is pasted with an integer offset so that its source pixel centre lands at (CAN_R + fr, CAN_C + fc),
    fr, fc in [0, 1); the exact canvas source position is returned. The canvas keeps 3.2 km downwind, 1.6 km crosswind
    each side and 640 m upwind of the source; the fraction of plume mass that falls outside is returned as `lost`."""
    meta_p = out_npy.replace('.npy', '_meta.npz')
    if os.path.exists(out_npy) and os.path.exists(meta_p):
        z = np.load(meta_p); return np.load(out_npy, mmap_mode='r'), z['src'], z['lost']
    M = len(bank_rows); C = np.lib.format.open_memmap(out_npy + '.tmp', 'w+', np.float16, (M, CANVAS_H, CANVAS_W))
    src = np.zeros((M, 2), np.float32); lost = np.zeros(M, np.float32)
    R = [dict(f=str(a), o=int(b), s=int(c), sr=float(d), sc=float(e)) for a, b, c, d, e in zip(  # plain values: no pandas in threads
        bank_rows['internal:source_file'], bank_rows['internal:offset'], bank_rows['internal:size'], bank_rows['array:source_row'], bank_rows['array:source_col'])]
    def work(i):
        r = R[i]
        a = decode(reader.read(sub, r['f'], r['o'], r['s']))[0]
        a = np.nan_to_num(a, nan=0.0)
        sr, sc = r['sr'], r['sc']
        r0, c0 = int(np.floor(CAN_R - sr)), int(np.floor(CAN_C - sc))
        src[i] = (sr + r0, sc + c0)
        H, W = a.shape; rr0, cc0 = max(r0, 0), max(c0, 0); rr1, cc1 = min(r0 + H, CANVAS_H), min(c0 + W, CANVAS_W)
        if rr1 > rr0 and cc1 > cc0:
            C[i, rr0:rr1, cc0:cc1] = a[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
        tot = float(a.sum()); lost[i] = max(0.0, 1 - float(C[i].astype(np.float32).sum()) / tot) if tot > 0 else 0.0
    s = _parallel(work, M, workers, 'bank', 2000); C.flush(); del C; os.replace(out_npy + '.tmp', out_npy)
    np.savez(meta_p, src=src, lost=lost)
    print(f'  bank cache: {M} plumes in {s:.0f}s, mass outside canvas median {np.median(lost):.3%} '
          f'p95 {np.percentile(lost, 95):.3%}, {os.path.getsize(out_npy) / 1e9:.2f} GB')
    return np.load(out_npy, mmap_mode='r'), src, lost

# ------------------------------------------------------------------------------------- transmittance
class Transmittance:
    """MethaneSET integrated_transmittances.json. Per sensor, tables over 8 AMFs x 9 CH4 mixing ratios.
    Units: B11 is a plain transmittance (~0.99) but B12 is tabulated on another scale (~64; its `_bg` column is the
    constant 64.2987 = the B12 value at 1800 ppb for every AMF). Only the ratio T(1800 + d) / T(1800) at the same
    AMF enters the injection, so the scale cancels; check_units() verifies that the 1800 ppb interpolation equals
    the tabulated background for both bands at every AMF before anything is trusted.
    Beyond the last mixing ratio (~8,280 ppb), ln T is extrapolated linearly from the last segment."""
    def __init__(self, path):
        d = json.load(open(path)); self.raw = d
        self.amf = torch.tensor(d['amf_arr'], dtype=torch.float32)                      # (8,)
        self.mr = torch.tensor(d['mr_ch4_arr'], dtype=torch.float32)                    # (8,9)
        self.bg = float(d['background_concentration']); assert abs(self.bg - BG_PPB) < 1e-6
        self.T = {b: torch.stack([torch.tensor(d[s][f'transmittance_{b}'], dtype=torch.float32) for s in SENSORS])
                  for b in ('b11', 'b12')}                                               # (3,8,9)
        self.Tbg = {b: torch.stack([torch.tensor(np.ravel(d[s][f'transmittance_{b}_bg']), dtype=torch.float32)
                                    for s in SENSORS]) for b in ('b11', 'b12')}          # (3,8)
        self.with_ltoa = d.get('with_ltoa_correction')
        self._cache = {}
    def to(self, dev):
        """Kept for older notebooks. The master tables stay on the CPU; every call uses a copy on the device of its
        inputs (see _on), so the same object serves CPU and GPU callers."""
        return self
    def _on(self, dev):
        """(amf, mr, T) on `dev`, copied once per device and cached."""
        k = str(dev)
        if k not in self._cache:
            self._cache[k] = (self.amf.to(dev), self.mr.to(dev), {b: v.to(dev) for b, v in self.T.items()})
        return self._cache[k]
    @staticmethod
    def _interp1(x, xp, fp, log_extrap=True):
        """Batched 1-D interpolation. x (B,N), xp (B,K) increasing, fp (B,K). Linear in ln f, linear extrapolation."""
        K = xp.shape[1]; i = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, K - 1)
        x0, x1 = xp.gather(1, i - 1), xp.gather(1, i); lf = torch.log(fp)
        y0, y1 = lf.gather(1, i - 1), lf.gather(1, i)
        return torch.exp(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    def rows_at(self, sensor, amf):
        """Per-chip table rows at the chip's AMF (linear in AMF). sensor (B,) long, amf (B,). -> mr (B,9), T{b}: (B,9)."""
        A, MR, TT = self._on(amf.device); sensor = sensor.to(amf.device).long()
        a = amf.float().clamp(float(self.amf[0]), float(self.amf[-1])); K = len(self.amf)
        j = torch.searchsorted(A.contiguous(), a.contiguous()).clamp(1, K - 1)
        w = ((a - A[j - 1]) / (A[j] - A[j - 1]))[:, None]
        mr = MR[j - 1] * (1 - w) + MR[j] * w
        T = {b: TT[b][sensor, j - 1] * (1 - w) + TT[b][sensor, j] * w for b in TT}
        return mr, T
    def rel(self, sensor, amf, delta):
        """T(1800 + delta) / T(1800) for B11 and B12. delta (B,N) ppb >= 0. -> (rel11, rel12) each (B,N)."""
        amf = amf.to(delta.device); mr, T = self.rows_at(sensor, amf); B = delta.shape[0]
        out = []
        for b in ('b11', 'b12'):
            t = self._interp1(BG_PPB + delta, mr, T[b]); t0 = self._interp1(torch.full((B, 1), BG_PPB, device=delta.device), mr, T[b])
            out.append(t / t0)
        return out
    def check_units(self, tol=0.005):
        """Interpolated T(1800) vs the tabulated *_bg value, every sensor/AMF/band. Returns (max rel. error, report dict)."""
        rep = {}; worst = 0.0
        for si, s in enumerate(SENSORS):
            for b in ('b11', 'b12'):
                t0 = self._interp1(torch.full((len(self.amf), 1), BG_PPB), self.mr.cpu(), self.T[b][si].cpu())[:, 0]
                e = float(((t0 - self.Tbg[b][si].cpu()) / self.Tbg[b][si].cpu()).abs().max()); worst = max(worst, e)
                rep[f'{s}_{b}'] = dict(T_at_1800=[round(float(v), 5) for v in t0], tabulated_bg=[round(float(v), 5) for v in self.Tbg[b][si]], max_rel_err=e)
        return worst, rep
    def sensitivity(self, sensor, amf, d=1000.0):
        """d ln(B12/B11) per ppb at the chip AMF, from a +1000 ppb step (negative)."""
        B = amf.shape[0]; r11, r12 = self.rel(sensor, amf, torch.full((B, 1), d, device=amf.device))
        return (torch.log(r12) - torch.log(r11))[:, 0] / d

# ------------------------------------------------------------------------------------- SWIR features
def log_ratio(x):
    """Same as the model's fp32 head (before mean removal): ln(T12/T11) - ln(R12/R11). x (B,8,H,W)."""
    eps = 1e-3; x = x.float()
    return torch.log((x[:, 5:6] + eps) / (x[:, 4:5] + eps)) - torch.log((x[:, 7:8] + eps) / (x[:, 6:7] + eps))

def mbmp(x):
    """Varon-style multi-band multi-pass: (c*B12 - B11)/B11 per image, c fitted per chip by least squares
    (c = sum(B11*B12)/sum(B12^2)), target minus reference. Negative where methane absorbs B12."""
    x = x.float(); eps = 1e-6
    def mb(b11, b12):
        c = (b11 * b12).sum((2, 3), keepdim=True) / ((b12 * b12).sum((2, 3), keepdim=True) + eps)
        return (c * b12 - b11) / (b11 + 1e-3)
    return mb(x[:, 4:5], x[:, 5:6]) - mb(x[:, 6:7], x[:, 7:8])

def robust_sigma(z):
    """1.4826 * MAD over each chip. z (B,1,H,W) -> (B,)."""
    f = z.flatten(1); m = f.median(1, keepdim=True).values
    return 1.4826 * (f - m).abs().median(1).values

# ---------------------------------------------------------------------------------------- injection
def inject(x, canv, src, theta, sza, vza, sensor, edge_ppb, tm, gen, *, q_min=500., q_max=20000., q_fixed=None,
           floor_sigma=0.5, centre_jitter_px=60, wind_jitter_deg=15.0):
    """x (B,8,200,200) clean pair on device, float. canv (B,1,160,192) ppb at 3000 kg/h. src (B,2) canvas source (row,col).
    theta (B,) wind direction (radians, counter-clockwise from east, direction the wind blows towards; NaN = random).
    Returns x_inj, label (B,1,H,W), delta (B,1,H,W) ppb, info dict (Q, noise ppb, floor, ...).
    The plume source is placed uniformly within +-centre_jitter_px (10 m px) of the chip centre with a sub-pixel offset."""
    B, _, H, W = x.shape; dev = x.device
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)
    th = torch.where(torch.isfinite(theta), theta, r(B) * 2 * math.pi) + torch.deg2rad(torch.tensor(wind_jitter_deg, device=dev)) * (2 * r(B) - 1)
    si = (H - 1) / 2 + (2 * r(B) - 1) * centre_jitter_px; sj = (W - 1) / 2 + (2 * r(B) - 1) * centre_jitter_px
    ii = torch.arange(H, device=dev, dtype=torch.float32)[None, :, None]; jj = torch.arange(W, device=dev, dtype=torch.float32)[None, None, :]
    east = (jj - sj[:, None, None]) * 10.0; north = -(ii - si[:, None, None]) * 10.0
    c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
    xp = east * c + north * s; yp = -east * s + north * c                       # metres in the plume frame
    col = src[:, 1, None, None] + xp / 20.0; row = src[:, 0, None, None] - yp / 20.0
    Hc, Wc = canv.shape[-2:]
    grid = torch.stack([(2 * col + 1) / Wc - 1, (2 * row + 1) / Hc - 1], -1)
    base = F.grid_sample(canv.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False).clamp_min(0)  # (B,1,H,W)
    amf = 1 / torch.cos(torch.deg2rad(sza.clamp(0, 75))) + 1 / torch.cos(torch.deg2rad(vza.clamp(0, 30)))
    sig_r = robust_sigma(log_ratio(x)); sens = tm.sensitivity(sensor, amf).abs()
    noise_ppb = sig_r / sens.clamp_min(1e-9)
    if q_fixed is not None: Q = torch.full((B,), float(q_fixed), device=dev)
    else:
        cut = 3000.0 * noise_ppb / edge_ppb.clamp_min(1e-3)                     # keep the LES downwind cut below the noise
        hi = torch.minimum(torch.full_like(cut, q_max), cut).clamp_min(q_min)
        Q = torch.exp(math.log(q_min) + r(B) * (torch.log(hi) - math.log(q_min)))
    delta = base * (Q / 3000.0)[:, None, None, None]
    r11, r12 = tm.rel(sensor, amf, delta.flatten(1))
    xo = x.clone()
    xo[:, 4] = x[:, 4] * r11.view(B, H, W); xo[:, 5] = x[:, 5] * r12.view(B, H, W)
    floor = floor_sigma * noise_ppb
    label = (delta >= floor[:, None, None, None]).float()
    return xo, label, delta, dict(Q=Q, noise_ppb=noise_ppb, floor=floor, amf=amf, sens=sens, src_i=si, src_j=sj)

def enhancement_stats(x, y):
    """Per chip: mean of (log-ratio, MBMP) inside the mask minus the median outside. Same estimator for real and
    synthetic chips. x (B,8,H,W), y (B,1,H,W) -> dict of (B,) arrays (NaN where the mask is empty)."""
    out = {}
    for k, f in (('ratio', log_ratio), ('mbmp', mbmp)):
        z = f(x); vals = []
        for i in range(len(z)):
            m = y[i, 0] > 0.5
            vals.append(float(z[i, 0][m].mean() - z[i, 0][~m].median()) if m.sum() >= 4 and (~m).sum() > 100 else float('nan'))
        out[k] = np.array(vals)
    out['area_px'] = y.flatten(1).sum(1).cpu().numpy()
    return out

# --------------------------------------------------------------------------------------------- model
class MethaneNetMBMP(mf.MethaneNet):
    """MethaneNet with a 10th in-graph channel: MBMP (target minus reference), fp32, mean-removed, x20.
    Input is still (B,8,H,W). Adds 290 parameters (first conv 32x1x3x3 + 1 BN pair) to the 4.13 M."""
    def __init__(self, encoder='mobilenet_v2', weights=None, decoder=(128, 64, 32, 16, 8)):
        nn.Module.__init__(self)
        import segmentation_models_pytorch as smp
        self.norm = nn.BatchNorm2d(mf.IN_CH + 2)
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=mf.IN_CH + 2, classes=1, decoder_channels=tuple(decoder))
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            t11, t12, r11, r12 = x[:, 4:5], x[:, 5:6], x[:, 6:7], x[:, 7:8]
            eps = 1e-3
            ratio = torch.log((t12 + eps) / (t11 + eps)) - torch.log((r12 + eps) / (r11 + eps))
            ratio = (ratio - ratio.mean(dim=(2, 3), keepdim=True)) * 20.0
            def mb(b11, b12):
                c = (b11 * b12).sum((2, 3), keepdim=True) / ((b12 * b12).sum((2, 3), keepdim=True) + 1e-6)
                return (c * b12 - b11) / (b11 + 1e-3)
            m = mb(t11, t12) - mb(r11, r12)
            m = (m - m.mean(dim=(2, 3), keepdim=True)) * 20.0
            z = self.norm(torch.cat([x, ratio, m], 1))
        H, W = int(x.shape[2]), int(x.shape[3])
        ph, pw = (-H) % 32, (-W) % 32; pt, pl = ph // 2, pw // 2
        if ph or pw: z = F.pad(z, (pl, pw - pl, pt, ph - pt), mode='reflect')
        return self.net(z)[:, :, pt:pt + H, pl:pl + W]

def make_model(kind, encoder='mobilenet_v2'):
    return MethaneNetMBMP(encoder) if kind == 'mbmp' else mf.MethaneNet(encoder, weights=None)

def load_expanded(model, sd):
    """Load a 9-input-channel state dict into a model that may have more input channels: the overlapping slice is
    copied; new conv input weights start at 0 (so the extra channel starts switched off), new BN entries at identity."""
    own = model.state_dict(); new = {}; grown = []
    for k, v in own.items():
        s = sd[k]
        if s.shape == v.shape: new[k] = s
        else:
            t = v.clone()
            if k.endswith('running_var') or (k.startswith('norm.') and k.endswith('weight')): t.fill_(1.0)
            else: t.zero_()
            sl = tuple(slice(0, n) for n in s.shape); t[sl] = s; new[k] = t; grown.append(k)
    model.load_state_dict(new); return grown

def n_params(m): return sum(p.numel() for p in m.parameters())

# --------------------------------------------------------------------------------------------- loss
def enh_weight(E, y, lo=0.5, hi=2.0):
    """Per-chip enhancement weight in [lo, hi] on mask pixels: clip((E - q05)/(q95 - q05), 0, 1) over that chip's mask
    pixels. Units cancel (MBMP leaf or ppb). Chips with < 4 mask pixels or no E get weight 1."""
    w = torch.ones_like(y)
    for i in range(len(y)):
        m = y[i, 0] > 0.5
        if E is None or m.sum() < 4: continue
        v = E[i, 0][m].float()
        if not torch.isfinite(v).all(): v = torch.nan_to_num(v)
        q = torch.quantile(v, torch.tensor([0.05, 0.95], device=v.device))
        z = ((E[i, 0].float() - q[0]) / (q[1] - q[0]).clamp_min(1e-6)).clamp(0, 1)
        w[i, 0] = torch.where(m, lo + (hi - lo) * z, torch.ones_like(z))
    return w

def weighted_dice_bce(logits, y, w, pos_weight=3.0):
    logits = logits.float()
    pw = torch.where(y > 0.5, pos_weight * w, torch.ones_like(w))
    bce = F.binary_cross_entropy_with_logits(logits, y, weight=pw)
    p = torch.sigmoid(logits); inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()

# ------------------------------------------------------------------------------------------- data
class SynScenes(torch.utils.data.Dataset):
    """Plume-free pretraining pairs + one of the scene's k assigned bank canvases (random each time)."""
    def __init__(self, XP, idx, assign, canvas, meta_arr, seed=0):
        self.XP, self.idx, self.assign, self.canvas, self.meta = XP, np.asarray(idx), assign, canvas, meta_arr
        self.seed = seed; self.rng = None
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        if self.rng is None:
            wi = torch.utils.data.get_worker_info(); self.rng = np.random.default_rng(self.seed + (wi.id + 1 if wi else 0) * 7919 + int(time.time()))
        j = int(self.idx[i]); b = int(self.assign[j, self.rng.integers(self.assign.shape[1])])
        return (torch.from_numpy(np.array(self.XP[j])), torch.from_numpy(np.array(self.canvas[b]))[None],
                torch.from_numpy(self.meta[j]), b)

def scene_meta(rows):
    """Per pretraining pair: sza, vza, theta (wind direction, rad ccw from east; NaN if calm), sensor index."""
    u, v = rows['meteo:wind_u'].values, rows['meteo:wind_v'].values
    th = np.arctan2(v, u); th[np.hypot(u, v) < 0.1] = np.nan
    sen = rows['target:sensor'].map({s: i for i, s in enumerate(SENSORS)}).fillna(0).values
    return np.stack([rows['target:sza'].values, rows['target:vza'].values, th, sen], 1).astype(np.float32)

def joint_aug(x, y, E, gen):
    """Per-sample rot90 + horizontal flip applied identically to input, label and enhancement map."""
    B = x.shape[0]; dev = x.device
    k = torch.randint(0, 4, (B,), generator=gen, device=dev).tolist(); f = (torch.rand(B, generator=gen, device=dev) < 0.5).tolist()
    xs, ys, es = [], [], []
    for i in range(B):
        a, b, e = torch.rot90(x[i], k[i], (1, 2)), torch.rot90(y[i], k[i], (1, 2)), torch.rot90(E[i], k[i], (1, 2))
        if f[i]: a, b, e = a.flip(-1), b.flip(-1), e.flip(-1)
        xs.append(a); ys.append(b); es.append(e)
    return torch.stack(xs), torch.stack(ys), torch.stack(es)

# ------------------------------------------------------------------------------------- scene scores
@torch.no_grad()
def scene_scores(model, X, idx, dev, amp_dtype=None, bs=32, kind='pairs', workers=2):
    """Scene-level scores at the 20 m grid (2x2 mean-pooled probabilities).
    kind='pairs': X is an (N,8,200,200) pair cache (real plume-free pairs) -> score (max), area (px >= 0.5).
    kind='plumes': X is the 31-channel fine-tune cache; target vs bg0 plus the plume mask ->
                   score, area, score_in (max inside the 20 m mask), mask_px."""
    model.eval(); out = {k: [] for k in ('score', 'area', 'score_in', 'mask_px')}
    idx = np.asarray(idx)
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s + bs])
        if kind == 'pairs': x = torch.from_numpy(np.array(X[j])).to(dev).float(); y = None
        else:
            a = torch.from_numpy(np.array(X[j, :mf.NCH_BASIC])).to(dev); x, y, _ = mf.eval_batch(a)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            p = torch.sigmoid(model(x).float())
        pp = F.avg_pool2d(p, 2); out['score'].append(pp.flatten(1).amax(1).cpu()); out['area'].append((pp >= 0.5).flatten(1).sum(1).cpu())
        if y is not None:
            yy = F.avg_pool2d(y, 2) >= 0.5
            out['score_in'].append(torch.where(yy, pp, torch.zeros_like(pp)).flatten(1).amax(1).cpu()); out['mask_px'].append(yy.flatten(1).sum(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}

def average_precision(pos, neg):
    """AP for 'plume chip' vs 'plume-free pair' from scene scores (step-wise, like sklearn)."""
    s = np.concatenate([pos, neg]); t = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s, kind='mergesort'); s, t = s[o], t[o]
    tp = np.cumsum(t); fp = np.cumsum(1 - t); last = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    P = tp[last] / (tp[last] + fp[last]); R = tp[last] / max(len(pos), 1)
    return float(np.sum(np.diff(np.r_[0, R]) * P)), (P, R)

def thr_at_fa(neg, fa):
    """Smallest threshold with at most floor(fa * n) negatives scoring >= it."""
    n = len(neg); allowed = int(np.floor(fa * n)); s = np.sort(neg)[::-1]
    return float(np.nextafter(s[allowed], np.inf)) if allowed < n else 0.0

def operating_points(pos, neg, fas=(0.10, 0.05, 0.01)):
    """pos: dict from scene_scores(kind='plumes'); neg: score array. Detection = score >= thr (any flag) and
    hit = score_in >= thr (a flagged 20 m pixel inside the plume mask)."""
    rows = {}
    for fa in fas:
        t = thr_at_fa(neg, fa)
        rows[f'{int(round(fa * 100))}%'] = dict(threshold=t, real_fa=float((neg >= t).mean()) if len(neg) else float('nan'),
                                              detect=float((pos['score'] >= t).mean()) if len(pos['score']) else float('nan'),
                                              hit=float((pos['score_in'] >= t).mean()) if len(pos['score_in']) else float('nan'))
    return rows

# ------------------------------------------------------------------------------------------ training
def real_batch(a, rows, gen, cfg, meth, ch4, dev):
    """Real fine-tune chips: fr_neg's recipe (target vs bg0, a fraction neg_p swapped for bg1-vs-bg0 plume-free pairs),
    no augmentation inside mf.train_batch so that the label, input and ch4 map are augmented together afterwards."""
    x, y = mf.train_batch(a, rows, gen, dict(cfg, aug=0), meth)
    if ch4 is not None:
        E = torch.from_numpy(np.nan_to_num(np.array(ch4[np.sort(rows.cpu().numpy())], dtype=np.float32)))
        order = np.argsort(np.argsort(rows.cpu().numpy()))         # back to batch order
        E = E[torch.as_tensor(order)][:, None].to(dev)
    else: E = torch.zeros_like(y)
    return x, y, E

def syn_batch(xp, canv, pm, bidx, can_src, edge, tm, gen, inj_p, **kw):
    """Plume-free pairs: a fraction inj_p get an injected plume (label from the detectability floor), the rest stay
    clean (label 0)."""
    x = xp.float(); B = x.shape[0]; dev = x.device
    inj = torch.rand(B, generator=gen, device=dev) < inj_p
    y = torch.zeros(B, 1, *x.shape[2:], device=dev); E = torch.zeros_like(y); info = None
    if inj.any():
        k = inj.nonzero()[:, 0]
        xi, yi, di, info = inject(x[k], canv[k], can_src[bidx[k]], pm[k, 2], pm[k, 0], pm[k, 1], pm[k, 3].long(),
                                  edge[bidx[k]], tm, gen, **kw)
        x = x.clone(); x[k] = xi; y[k] = yi; E[k] = di
    return x, y, E, inj, info

def train_syn(name, kind, Xft, meta_ft, ch4, XP, tr_pairs, assign, canvas, can_src, edge, pmeta, tm, dev, out_root, *,
              init_sd, XV=None, val_neg=None, epochs=12, lr=3e-4, bs_real=8, bs_syn=8, inj_p=0.7, neg_p=0.3,
              q_min=500., q_max=20000., floor_sigma=0.5, enh_weighting=True, amp=True, workers=2, seed=0,
              project='methane-se2-syn', skip_done=True, log=print, channels_last=True, max_steps=None,
              preview_every=3):
    """Mixed batches: bs_real real fine-tune chips + bs_syn plume-free pairs (inj_p of them with an injected plume).
    With the defaults (8 + 8, inj_p 0.7) that is 50 % real, 35 % injected, 15 % untouched, as in the plan.
    One epoch = one pass over the real train chips. Keeps final.pt and best.pt (best val hit rate at 5 % val FA)."""
    import trackio
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = f'{out_root}/{name}'; os.makedirs(out, exist_ok=True)
    if skip_done and os.path.exists(f'{out}/final.pt') and os.path.exists(f'{out}/best.pt'):
        log(f'[{name}] final.pt exists, skipping (set SKIP_DONE=False to retrain)'); return out
    cfg = dict(name=name, kind=kind, epochs=epochs, lr=lr, bs_real=bs_real, bs_syn=bs_syn, inj_p=inj_p, neg_p=neg_p,
               q_min=q_min, q_max=q_max, floor_sigma=floor_sigma, enh_weighting=enh_weighting, seed=seed, amp=amp,
               n_pretrain_pairs=len(tr_pairs), resolution='10m', chip=200)
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    tr = np.where(meta_ft.split == 'train')[0]; va = np.where(meta_ft.split == 'validation')[0]
    meth = torch.tensor(meta_ft[['bg:methane1', 'bg:methane2', 'bg:methane3']].fillna(1.0).values, device=dev, dtype=torch.float32)
    model = make_model(kind); grown = load_expanded(model, init_sd); model.to(dev)
    if grown: log(f'[{name}] expanded from the 9-channel init: {grown}')
    if channels_last and dev.type == 'cuda': model = model.to(memory_format=torch.channels_last)
    amp_dtype = mf.amp_dtype_for(dev) if amp else None
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl_r = mf.loader(Xft, tr, bs_real, True, workers, seed, drop_last=True)
    g = torch.Generator(); g.manual_seed(seed)
    ds = SynScenes(XP, tr_pairs, assign, canvas, pmeta, seed)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=10 ** 8, generator=g)
    dl_s = torch.utils.data.DataLoader(ds, batch_size=bs_syn, sampler=sampler, num_workers=workers,
                                       pin_memory=dev.type == 'cuda', persistent_workers=workers > 0, drop_last=True)
    it_s = iter(dl_s)
    can_src_t = torch.tensor(np.asarray(can_src), device=dev, dtype=torch.float32); edge_t = torch.tensor(np.asarray(edge), device=dev, dtype=torch.float32)
    spe = max(len(dl_r), 1) if not max_steps else min(len(dl_r), max_steps); steps = epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps)))))
    dv = mf.loader(Xft, va, 32, False, workers)
    trackio.init(project=project, name=name, config=cfg)
    kw = dict(q_min=q_min, q_max=q_max, floor_sigma=floor_sigma)
    hist = []; best = -1.0
    def preview(xb, yb, ep):
        n = min(6, len(xb)); f, ax = plt.subplots(3, n, figsize=(2.2 * n, 6.6), squeeze=False)
        with torch.no_grad(): p = torch.sigmoid(model(xb[:n]).float()).cpu()
        lr_ = log_ratio(xb[:n]).cpu()
        for i in range(n):
            ax[0, i].imshow(lr_[i, 0], cmap='RdBu', vmin=-0.05, vmax=0.05); ax[1, i].imshow(yb[i, 0].cpu(), cmap='gray')
            ax[2, i].imshow(p[i, 0], cmap='magma', vmin=0, vmax=1)
        for r_ in ax.ravel(): r_.axis('off')
        f.suptitle(f'{name} ep {ep}: synthetic chips. ln(B12/B11) target-ref / label / prediction', fontsize=9); f.tight_layout()
        fn = f'{out}/syn_preview_ep{ep:03d}.png'; f.savefig(fn, dpi=70); plt.close(f); return fn
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); ls = []; qs = []; noise = []; lab_px = []; xb_last = yb_last = None
        for step, (a, rows) in enumerate(dl_r):
            if step >= spe: break
            a = a.to(dev, non_blocking=True); rows = rows.to(dev)
            xr, yr, Er = real_batch(a, rows, gen, dict(neg_p=neg_p), meth, ch4 if enh_weighting else None, dev)
            xp_, cv, pm, bidx = next(it_s)
            xs, ys, Es, inj, info = syn_batch(xp_.to(dev, non_blocking=True), cv.to(dev, non_blocking=True), pm.to(dev),
                                              bidx.to(dev), can_src_t, edge_t, tm, gen, inj_p, **kw)
            if info is not None:
                qs += info['Q'].tolist(); noise += info['noise_ppb'].tolist(); lab_px += ys[inj].flatten(1).sum(1).tolist()
            x = torch.cat([xr, xs]); y = torch.cat([yr, ys]); E = torch.cat([Er, Es])
            x, y, E = joint_aug(x, y, E, gen)
            w = enh_weight(E, y) if enh_weighting else torch.ones_like(y)
            if channels_last and dev.type == 'cuda': x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = weighted_dice_bce(logits, y, w)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ls.append(loss.item()); xb_last, yb_last = xs.detach(), ys.detach()
        rec = dict(epoch=ep, train_loss=float(np.mean(ls)) if ls else float('nan'), lr=opt.param_groups[0]['lr'], epoch_s=time.time() - t0,
                   syn_Q_median_kgh=float(np.median(qs)) if qs else float('nan'),
                   syn_noise_ppb_median=float(np.median(noise)) if noise else float('nan'),
                   syn_label_px_median=float(np.median(lab_px)) if lab_px else float('nan'),
                   syn_label_empty_frac=float(np.mean(np.array(lab_px) == 0)) if lab_px else float('nan'))
        c = mf.predict_counts(model, dv, dev, amp_dtype)
        for gname in ('10m', '20m'):
            rec.update({f'val{"" if gname == "10m" else "20"}/{k}': v for k, v in mf.at(c, 0.5, gname).items()})
        if XV is not None and val_neg is not None and len(val_neg):
            pos = scene_scores(model, Xft, va, dev, amp_dtype, kind='plumes'); neg = scene_scores(model, XV, val_neg, dev, amp_dtype)['score']
            op = operating_points(pos, neg, (0.05,))['5%']
            rec.update({'val/hit@fa5': op['hit'], 'val/detect@fa5': op['detect'], 'val/thr@fa5': op['threshold'],
                        'val/AP': average_precision(pos['score'], neg)[0], 'val/real_fa@0.5': float((neg >= 0.5).mean())})
        sel = rec.get('val/hit@fa5', rec['val/iou'])
        lg = dict(rec)
        if xb_last is not None and (ep % preview_every == 0 or ep in (1, epochs)): lg['syn/preview'] = trackio.Image(preview(xb_last, yb_last, ep))
        trackio.log(lg); hist.append(rec)
        sd = {k: v.contiguous() for k, v in model.state_dict().items()}
        if sel > best:
            best = sel; torch.save(sd, f'{out}/best.pt.tmp'); os.replace(f'{out}/best.pt.tmp', f'{out}/best.pt'); rec['best'] = True
        log(f'[{name}] ep {ep}/{epochs} loss {rec["train_loss"]:.4f} val IoU {rec["val/iou"]:.3f} '
            f'hit@5%FA {rec.get("val/hit@fa5", float("nan")):.3f} AP {rec.get("val/AP", float("nan")):.3f} '
            f'realFA@0.5 {rec.get("val/real_fa@0.5", float("nan")):.3f} | Q~{rec["syn_Q_median_kgh"]:.0f} kg/h noise~{rec["syn_noise_ppb_median"]:.0f} ppb '
            f'label~{rec["syn_label_px_median"]:.0f} px ({rec["epoch_s"]:.0f}s)')
    torch.save(sd, f'{out}/final.pt.tmp'); os.replace(f'{out}/final.pt.tmp', f'{out}/final.pt')
    json.dump(dict(config=cfg, history=hist, best_val_selection=best), open(f'{out}/history.json', 'w'), indent=1)
    trackio.finish()
    return out

# ------------------------------------------------------------------------------------------ evaluation
def load_model(kind, sd, dev):
    m = make_model(kind); m.load_state_dict(sd); return m.to(dev).eval()

def evaluate(name, model, Xft, subsets, XF, fa_sets, XV, val_neg, va_idx, dev, amp_dtype=None, workers=2):
    """subsets: {'all'|'seen'|'unseen': fine-tune test indices}; fa_sets: {name: indices into XF}.
    Returns (summary dict, raw score arrays for plots)."""
    res = dict(model=name, pixel={}, scene={}, val={}); raw = dict(pos={}, neg={})
    for sub, idx in subsets.items():
        if len(idx) == 0: continue
        c = mf.predict_counts(model, mf.loader(Xft, idx, 32, False, workers), dev, amp_dtype)
        res['pixel'][sub] = {g: mf.at(c, 0.5, g) for g in ('10m', '20m')}; res['pixel'][sub]['n'] = int(len(idx))
        raw['pos'][sub] = scene_scores(model, Xft, idx, dev, amp_dtype, kind='plumes')
    for fs, idx in fa_sets.items():
        raw['neg'][fs] = scene_scores(model, XF, idx, dev, amp_dtype)['score'] if len(idx) else np.zeros(0)
    for sub, pos in raw['pos'].items():
        for fs, neg in raw['neg'].items():
            if not len(neg): continue
            ap, _ = average_precision(pos['score'], neg)
            res['scene'][f'{sub}|{fs}'] = dict(n_pos=int(len(pos['score'])), n_neg=int(len(neg)), AP=ap,
                                               real_fa_at_0p5=float((neg >= 0.5).mean()),
                                               detect_at_0p5=float((pos['score'] >= 0.5).mean()),
                                               hit_at_0p5=float((pos['score_in'] >= 0.5).mean()),
                                               ops=operating_points(pos, neg))
    if XV is not None and val_neg is not None and len(val_neg) and len(va_idx):
        pos = scene_scores(model, Xft, va_idx, dev, amp_dtype, kind='plumes'); neg = scene_scores(model, XV, val_neg, dev, amp_dtype)['score']
        c = mf.predict_counts(model, mf.loader(Xft, va_idx, 32, False, workers), dev, amp_dtype)
        res['val'] = dict(AP=average_precision(pos['score'], neg)[0], ops=operating_points(pos, neg), n_pos=int(len(va_idx)),
                          n_neg=int(len(neg)), pixel={g: mf.at(c, 0.5, g) for g in ('10m', '20m')})
        raw['val'] = dict(pos=pos, neg=neg)
    return res, raw

def go_no_go(cand, base, lat_ratio, contract_ok, params_cand, params_base, fa_key='8740', sub_unseen='unseen'):
    """The plan's release rule, evaluated on the val-selected candidate only."""
    def g(r, sub, fs, op, k): return r['scene'].get(f'{sub}|{fs}', {}).get('ops', {}).get(op, {}).get(k, float('nan'))
    d_hit = g(cand, sub_unseen, fa_key, '5%', 'hit') - g(base, sub_unseen, fa_key, '5%', 'hit')
    d_iou = cand['pixel'].get(sub_unseen, {}).get('20m', {}).get('iou', float('nan')) - base['pixel'].get(sub_unseen, {}).get('20m', {}).get('iou', float('nan'))
    d_val = cand['val'].get('ops', {}).get('5%', {}).get('hit', float('nan')) - base['val'].get('ops', {}).get('5%', {}).get('hit', float('nan'))
    d_all = g(cand, 'all', fa_key, '5%', 'hit') - g(base, 'all', fa_key, '5%', 'hit')
    rel_p = abs(params_cand - params_base) / params_base
    crit = {
        '1 unseen-site hit rate at real FA 5% improves by >= 0.10': (d_hit, bool(d_hit >= 0.10)),
        '2 unseen-site pixel IoU at 0.5 (20 m grid) drops by <= 0.01': (d_iou, bool(d_iou >= -0.01)),
        '3 2021 val shows the same direction (val hit rate at 5% val FA)': (d_val, bool(d_val > 0)),
        '4a ONNX input contract 1x8x200x200 unchanged': (contract_ok, bool(contract_ok)),
        '4b parameter count unchanged (within 0.01 %)': (rel_p, bool(rel_p <= 1e-4)),
        '4c CPU latency within 10 %': (lat_ratio, bool(np.isfinite(lat_ratio) and lat_ratio <= 1.10)),
        '5 no worse at matched detection on all test plumes (hit rate at real FA 5% not lower)': (d_all, bool(d_all >= 0)),
    }
    return all(v[1] for v in crit.values()), {k: dict(value=(float(v[0]) if not isinstance(v[0], bool) else v[0]), ok=v[1]) for k, v in crit.items()}

# ------------------------------------------------------------------------------------------ sanity gate
def sanity_gate(XP, gate_idx, assign, canvas, can_src, edge, pmeta, tm, Xft, meta_ft, real_idx, dev, out_png, *,
                q=3000.0, flux_band=(2000.0, 4500.0), floor_sigma='auto', floor_grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
                ratio_tol=3.0, area_tol=4.0, seed=0, real_max=160):
    """Stage 1. Inject Q = 3000 kg/h into the gate scenes and compare with real MethaneSET plumes whose MARS flux is
    in flux_band, using the same estimator (mean in mask minus median outside) for ln(B12/B11) and MBMP.
    Checks: table units, sign of the dip, enhancement magnitude within x ratio_tol, label area within x area_tol.
    floor_sigma='auto' picks the label floor (x scene noise) whose median label area is closest to the real one."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rep = {}; ok = True; why = []
    worst, units = tm.check_units(); rep['units'] = dict(max_rel_err_T1800_vs_bg=worst, with_ltoa_correction=tm.with_ltoa)
    if worst > 0.005: ok = False; why.append(f'transmittance table: T(1800) differs from the tabulated background by {worst:.2%}')
    amf0 = torch.tensor([2.3]); s0 = torch.tensor([0])
    r11, r12 = tm.rel(s0, amf0, torch.tensor([[1000.0]]))
    rep['units'].update(b12_rel_change_per_1000ppb_amf2p3=float(r12 - 1), b11_rel_change_per_1000ppb_amf2p3=float(r11 - 1),
                        b12_table_scale=float(tm.Tbg['b12'][0, 0]), note='B12 table ~64 scale cancels in T(1800+d)/T(1800)')
    # real plumes
    fl = meta_ft['detection:ch4_fluxrate'].values[real_idx]
    sel = np.asarray(real_idx)[(fl >= flux_band[0]) & (fl <= flux_band[1])]
    if len(sel) < 8: sel = np.asarray(real_idx); rep['real_note'] = f'fewer than 8 plumes in {flux_band} kg/h: using all {len(sel)}'
    if len(sel) > real_max: sel = np.random.default_rng(seed).choice(sel, real_max, replace=False)
    a = torch.from_numpy(np.array(X_sorted(Xft, sel, mf.NCH_BASIC))).to(dev); xr, yr, _ = mf.eval_batch(a)
    real = enhancement_stats(xr, yr); rep['real'] = dict(n=int(len(sel)), flux_kgh_median=float(np.median(meta_ft['detection:ch4_fluxrate'].values[np.sort(sel)])))
    # synthetic
    gidx = np.sort(np.asarray(gate_idx)); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    xp = torch.from_numpy(np.array(XP[gidx])).to(dev).float(); b = torch.as_tensor(assign[gidx, 0], device=dev)
    cv = torch.from_numpy(np.array(canvas[np.sort(np.unique(assign[gidx, 0]))])).to(dev)
    lut = {int(v): i for i, v in enumerate(np.sort(np.unique(assign[gidx, 0])))}
    cvb = cv[torch.as_tensor([lut[int(v)] for v in assign[gidx, 0]], device=dev)][:, None]
    pm = torch.from_numpy(pmeta[gidx]).to(dev)
    src_t = torch.tensor(np.asarray(can_src), device=dev, dtype=torch.float32)[b]; edge_t = torch.tensor(np.asarray(edge), device=dev, dtype=torch.float32)[b]
    gst = gen.get_state()
    areas = {}
    for fs in floor_grid:
        gen.set_state(gst)
        _, yl, _, _ = inject(xp, cvb, src_t, pm[:, 2], pm[:, 0], pm[:, 1], pm[:, 3].long(), edge_t, tm, gen, q_fixed=q, floor_sigma=fs)
        areas[fs] = float(np.median(yl.flatten(1).sum(1).cpu().numpy()))
    real_area = float(np.median(real['area_px']))
    fsig = min(floor_grid, key=lambda f: abs(math.log((areas[f] + 1) / (real_area + 1)))) if floor_sigma == 'auto' else float(floor_sigma)
    gen.set_state(gst)
    xi, yi, di, info = inject(xp, cvb, src_t, pm[:, 2], pm[:, 0], pm[:, 1], pm[:, 3].long(), edge_t, tm, gen, q_fixed=q, floor_sigma=fsig)
    syn = enhancement_stats(xi, yi)
    true_dip = []  # exact injected signal (injected minus clean) inside the label, no noise
    lr_i, lr_c = log_ratio(xi), log_ratio(xp)
    for i in range(len(xi)):
        m = yi[i, 0] > 0.5; true_dip.append(float((lr_i[i, 0] - lr_c[i, 0])[m].mean()) if m.sum() else float('nan'))
    med = lambda v: float(np.nanmedian(v)) if np.isfinite(v).any() else float('nan')
    rep['label_floor'] = dict(chosen_sigma=fsig, mode=floor_sigma, median_area_px_by_sigma=areas, real_median_area_px=real_area)
    rep['synthetic'] = dict(n=int(len(gidx)), Q_kgh=q, noise_ppb_median=float(info['noise_ppb'].median()),
                            noise_ppb_iqr=[float(info['noise_ppb'].quantile(0.25)), float(info['noise_ppb'].quantile(0.75))],
                            floor_ppb_median=float(info['floor'].median()), peak_delta_ppb_median=float(di.flatten(1).amax(1).median()),
                            empty_label_frac=float(np.mean(syn['area_px'] == 0)), true_dip_ratio_median=med(np.array(true_dip)))
    for k in ('ratio', 'mbmp'):
        rep[f'{k}_enhancement'] = dict(real_median=med(real[k]), real_iqr=[float(np.nanpercentile(real[k], 25)), float(np.nanpercentile(real[k], 75))] if np.isfinite(real[k]).any() else None,
                                       syn_median=med(syn[k]), syn_iqr=[float(np.nanpercentile(syn[k], 25)), float(np.nanpercentile(syn[k], 75))] if np.isfinite(syn[k]).any() else None,
                                       syn_over_real=med(syn[k]) / med(real[k]) if med(real[k]) else float('nan'),
                                       syn_negative_frac=float(np.nanmean(syn[k] < 0)) if np.isfinite(syn[k]).any() else float('nan'),
                                       real_negative_frac=float(np.nanmean(real[k] < 0)) if np.isfinite(real[k]).any() else float('nan'))
    rep['area'] = dict(real_median_px=real_area, syn_median_px=float(np.median(syn['area_px'])),
                       syn_over_real=(float(np.median(syn['area_px'])) + 1) / (real_area + 1))
    # decisions
    rr = rep['ratio_enhancement']
    if not (rr['syn_negative_frac'] >= 0.8): ok = False; why.append(f"only {rr['syn_negative_frac']:.0%} of injected chips show a B12/B11 dip")
    for k in ('ratio', 'mbmp'):
        x = rep[f'{k}_enhancement']['syn_over_real']
        if not (np.isfinite(x) and 1 / ratio_tol <= x <= ratio_tol): ok = False; why.append(f'{k} enhancement synthetic/real = {x:.2f}, outside [1/{ratio_tol:g}, {ratio_tol:g}]')
    x = rep['area']['syn_over_real']
    if not (1 / area_tol <= x <= area_tol): ok = False; why.append(f'label area synthetic/real = {x:.2f}, outside [1/{area_tol:g}, {area_tol:g}]')
    if rep['synthetic']['empty_label_frac'] > 0.5: ok = False; why.append(f"{rep['synthetic']['empty_label_frac']:.0%} of 3 t/h injections leave an empty label")
    rep['ok'] = ok; rep['why'] = why
    # figure
    n = min(5, len(sel), len(gidx)); pick_r = np.linspace(0, len(sel) - 1, n).astype(int); pick_s = np.linspace(0, len(gidx) - 1, n).astype(int)
    f, ax = plt.subplots(3, 2 * n, figsize=(2.1 * 2 * n, 7.2), squeeze=False)
    lr_r = log_ratio(xr).cpu(); lr_s = lr_i.cpu(); xr_c, xi_c = xr.cpu(), xi.cpu()
    fl_s = meta_ft['detection:ch4_fluxrate'].values[np.sort(sel)]
    for i in range(n):
        for j, (L, X_, Y, t, k_) in enumerate(((lr_r, xr_c, yr.cpu(), f'real {fl_s[pick_r[i]] / 1000:.1f} t/h', pick_r[i]),
                                               (lr_s, xi_c, yi.cpu(), f'synthetic {q / 1000:.0f} t/h', pick_s[i]))):
            c_ = 2 * i + j; z = L[k_, 0] - L[k_, 0].median()
            ax[0, c_].imshow(np.clip(X_[k_, [2, 1, 0]].permute(1, 2, 0).numpy() * 3.5, 0, 1)); ax[0, c_].set_title(t, fontsize=7)
            ax[1, c_].imshow(z, cmap='RdBu', vmin=-0.04, vmax=0.04); ax[2, c_].imshow(Y[k_, 0], cmap='gray')
    for r_ in ax.ravel(): r_.axis('off')
    f.suptitle(f'Injection sanity: RGB / ln(B12/B11) target-ref (median removed, +-0.04) / label. Gate {"PASS" if ok else "FAIL"}', fontsize=9)
    f.tight_layout(); f.savefig(out_png, dpi=80); plt.close(f)
    f, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for a_, k, t in ((ax[0], 'ratio', 'ln(B12/B11) enhancement'), (ax[1], 'mbmp', 'MBMP enhancement')):
        rv, sv = real[k][np.isfinite(real[k])], syn[k][np.isfinite(syn[k])]; both = np.r_[rv, sv]
        bins = np.linspace(*np.percentile(both, [1, 99]), 40) if len(both) > 1 else 10
        a_.hist(rv, bins, alpha=.6, density=True, label=f'real {flux_band[0] / 1000:g}-{flux_band[1] / 1000:g} t/h (n={len(rv)})')
        a_.hist(sv, bins, alpha=.6, density=True, label=f'synthetic {q / 1000:g} t/h (n={len(sv)})'); a_.set_title(t, fontsize=9); a_.legend(fontsize=7)
    bins = np.linspace(0, 4.2, 40)
    ax[2].hist(np.log10(real['area_px'] + 1), bins, alpha=.6, density=True, label='real'); ax[2].hist(np.log10(syn['area_px'] + 1), bins, alpha=.6, density=True, label='synthetic')
    ax[2].set_title('log10(label area px + 1), 10 m', fontsize=9); ax[2].legend(fontsize=7)
    f.tight_layout(); f.savefig(out_png.replace('.png', '_hist.png'), dpi=80); plt.close(f)
    return ok, rep

def X_sorted(X, idx, nch=None):
    j = np.sort(np.asarray(idx)); return X[j, :nch] if nch else X[j]
