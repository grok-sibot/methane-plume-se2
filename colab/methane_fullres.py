"""Full-resolution (10 m, 200x200) methane plume detector for the Spiral Blue SE-2 (Jetson Orin NX).

Written by the Colab notebook methane_fullres_colab.ipynb. Same input contract as the CPU model
(sighmon/methane-plume-se2) except the spatial size:
    input  `reflectance`  float32 1x8x200x200  L1C DN/10000 at 10 m (2 km x 2 km chip)
           channels: target B2,B3,B4,B8A,B11,B12, then reference B11,B12
    output `plume_logits` float32 1x1x200x200  (sigmoid -> threshold)
The SWIR ratio channel is computed inside the graph, in fp32.

Cache layout (float16, N x 31 x 200 x 200, reflectance = DN/10000):
    0 plume mask, 1-6 target, 7-12 bg0, 13-18 bg1, 19-24 bg2, 25-30 bg3 (each B2,B3,B4,B8A,B11,B12).
    Runs without extra references read only channels 0-18 (mask, target, bg0, bg1): 39 % less I/O.
"""
import os, json, time, math, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

BANDS0 = [1, 2, 3, 8, 11, 12]  # 0-based in the 13-band L1C stack: B2,B3,B4,B8A,B11,B12
LEAVES = ['target', 'bg0', 'bg1', 'bg2', 'bg3']
NCH = 31; MASK = 0; NCH_BASIC = 19  # mask + target + bg0 + bg1
IN_CH = 8
INPUT_BANDS = ['T_B2', 'T_B3', 'T_B4', 'T_B8A', 'T_B11', 'T_B12', 'R_B11', 'R_B12']
META_COLS = ['id', 'split', 'detection:case_study', 'detection:ch4_fluxrate', 'site:country',
             'site:location_name', 'target:cloud', 'bg:cloud0', 'bg:cloud1', 'bg:cloud2', 'bg:cloud3',
             'bg:methane1', 'bg:methane2', 'bg:methane3', 'internal:source_file']

# ----------------------------------------------------------------------------------------- data prep
_P = {}
def _init(l1, zipdir):
    _P['l1'] = l1; _P['zip'] = zipdir
def _path(r):
    return f"/vsisubfile/{r['internal:offset']}_{r['internal:size']},{_P['zip']}/{r['internal:source_file']}"
def _work(args):
    import rasterio
    i, rows = args
    out = np.zeros((NCH, 200, 200), np.float16)
    for j, k in enumerate(LEAVES):
        with rasterio.open(_path(rows[k])) as d:
            out[1 + 6 * j:7 + 6 * j] = d.read([b + 1 for b in BANDS0]).astype(np.float32) / 10000.0
    with rasterio.open(_path(rows['plume'])) as d:
        out[MASK] = d.read(1).astype(np.float32)
    return i, out

def build_cache(l0, l1, zipdir, out_npy, workers=None):
    """Read 5 scenes + mask per sample straight out of the .tacozip files (GDAL /vsisubfile)."""
    from multiprocessing import Pool
    l1 = l1.copy(); l1['_pid'] = l1['internal:relative_path'].str.split('/').str[0]
    idx = {(p, k): j for j, (p, k) in enumerate(zip(l1['_pid'], l1['id']))}
    cols = ['internal:offset', 'internal:size', 'internal:source_file']
    jobs = [(i, {k: l1.loc[idx[(pid, k)], cols].to_dict() for k in LEAVES + ['plume']})
            for i, pid in enumerate(l0['id'])]
    X = np.lib.format.open_memmap(out_npy + '.tmp', 'w+', np.float16, (len(l0), NCH, 200, 200))
    t = time.time()
    with Pool(workers or os.cpu_count(), initializer=_init, initargs=(None, zipdir)) as p:
        for n, (i, a) in enumerate(p.imap_unordered(_work, jobs, chunksize=4)):
            X[i] = a
            if n % 250 == 0: print(f'  prep {n}/{len(jobs)}  {time.time() - t:.0f}s', flush=True)
    X.flush(); del X; os.replace(out_npy + '.tmp', out_npy)
    print(f'  prep done: {len(jobs)} chips in {time.time() - t:.0f}s')

def load_cache(npy, to_ram='auto'):
    X = np.load(npy, mmap_mode='r')
    if to_ram == 'auto':
        try:
            import psutil; to_ram = psutil.virtual_memory().available > 1.4 * X.nbytes
        except Exception: to_ram = False
    if to_ram: X = np.ascontiguousarray(X)
    print(f'cache {X.shape} {X.nbytes / 1e9:.1f} GB, in RAM: {bool(to_ram)}')
    return X

class Chips(torch.utils.data.Dataset):
    def __init__(self, X, idx, nch=NCH_BASIC): self.X, self.idx, self.nch = X, np.asarray(idx), nch
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = int(self.idx[i]); return torch.from_numpy(np.array(self.X[j, :self.nch])), j

def loader(X, idx, bs, shuffle, workers=2, seed=0, drop_last=False, nch=NCH_BASIC):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.utils.data.DataLoader(Chips(X, idx, nch), batch_size=bs, shuffle=shuffle, drop_last=drop_last,
                                       num_workers=workers, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=workers > 0, generator=g)

# ------------------------------------------------------------------------------------- batch building
def pair(a, t_blk, r_blk):
    """a: (B,C,H,W), C = 19 or 31. Block 0 = target, 1..4 = bg0..bg3. Build model input from per-sample
    target-block / reference-block indices (all 6 bands of the target block, B11/B12 of the reference)."""
    B = a.shape[0]; ar = torch.arange(B, device=a.device); nb = (a.shape[1] - 1) // 6
    blocks = a[:, 1:1 + 6 * nb].reshape(B, nb, 6, *a.shape[2:])
    return torch.cat([blocks[ar, t_blk], blocks[ar, r_blk][:, 4:6]], 1)

def eval_batch(a):
    """Fixed eval pairs, same as the CPU model: plume pair target vs bg0; plume-free pair bg1 vs bg0."""
    a = a.float(); B = a.shape[0]
    z = torch.zeros(B, dtype=torch.long, device=a.device)
    x = pair(a, z, z + 1); xn = pair(a, z + 2, z + 1)
    y = (a[:, MASK:MASK + 1] >= 0.5).float()
    return x, y, xn

def train_batch(a, rows, gen, cfg, meth):
    """Training pairs. cfg keys: neg_p, extra_refs, extra_ref_p, neg_max_methane, aug.
    meth: (N,3) tensor bg:methane1..3 on device (fraction of reference pixels with own enhancement)."""
    a = a.float(); B, dev = a.shape[0], a.device
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)
    ri = lambda hi, *s: torch.randint(0, hi, s, generator=gen, device=dev)
    t_blk = torch.zeros(B, dtype=torch.long, device=dev); r_blk = torch.ones_like(t_blk)  # target vs bg0
    y = (a[:, MASK:MASK + 1] >= 0.5).float()
    if cfg.get('extra_refs'):  # swap the reference for bg1..bg3 (block 2..4)
        sw = r(B) < cfg.get('extra_ref_p', 0.5)
        r_blk = torch.where(sw, 2 + ri(3, B), r_blk)
    if cfg.get('neg_p', 0) > 0:  # plume-free pair: one reference used as "target" against another
        neg = r(B) < cfg['neg_p']
        if cfg.get('extra_refs'):
            nt = 2 + ri(3, B)                                   # bg1..bg3 as target
            nr = (nt - 1 + 1 + ri(3, B)) % 4 + 1                # any other of bg0..bg3
        else:
            nt = torch.full_like(t_blk, 2); nr = torch.ones_like(t_blk)  # bg1 vs bg0 (CPU recipe)
        thr = cfg.get('neg_max_methane')
        if thr is not None:  # skip "negatives" whose target reference has its own enhancement
            neg &= meth[rows, (nt - 2)] <= thr
        t_blk = torch.where(neg, nt, t_blk); r_blk = torch.where(neg, nr, r_blk)
        y = torch.where(neg[:, None, None, None], torch.zeros_like(y), y)
    x = pair(a, t_blk, r_blk)
    if cfg.get('aug', 0) >= 1:  # per-sample rot90 + horizontal flip
        k = ri(4, B).tolist(); f = (r(B) < 0.5).tolist()  # one host sync per batch
        xs, ys = [], []
        for i in range(B):
            xi, yi = torch.rot90(x[i], k[i], (1, 2)), torch.rot90(y[i], k[i], (1, 2))
            if f[i]: xi, yi = xi.flip(-1), yi.flip(-1)
            xs.append(xi); ys.append(yi)
        x, y = torch.stack(xs), torch.stack(ys)
    if cfg.get('aug', 0) >= 2:  # radiometric jitter: per-chip gain on all bands (scene brightness)
        x = x * (0.9 + 0.2 * r(B, 1, 1, 1))
    return x.contiguous(), y.contiguous()

# ------------------------------------------------------------------------------------------- model
class MethaneNet(nn.Module):
    """Raw TOA reflectance (B,8,H,W) -> plume logits (B,1,H,W). Reflect-pads to a multiple of 32."""
    def __init__(self, encoder='mobilenet_v2', weights='imagenet', decoder=(128, 64, 32, 16, 8)):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.norm = nn.BatchNorm2d(IN_CH + 1)
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=IN_CH + 1, classes=1,
                            decoder_channels=tuple(decoder))
        if hasattr(self.net.encoder, 'set_swish'):  # efficientnet-pytorch: ONNX-exportable swish
            self.net.encoder.set_swish(memory_efficient=False)
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):  # ratio head in fp32 always
            x = x.float()
            t11, t12, r11, r12 = x[:, 4:5], x[:, 5:6], x[:, 6:7], x[:, 7:8]
            eps = 1e-3
            ratio = torch.log((t12 + eps) / (t11 + eps)) - torch.log((r12 + eps) / (r11 + eps))
            ratio = (ratio - ratio.mean(dim=(2, 3), keepdim=True)) * 20.0
            z = self.norm(torch.cat([x, ratio], 1))
        H, W = int(x.shape[2]), int(x.shape[3])
        ph, pw = (-H) % 32, (-W) % 32
        pt, pl = ph // 2, pw // 2
        if ph or pw: z = F.pad(z, (pl, pw - pl, pt, ph - pt), mode='reflect')
        y = self.net(z)
        return y[:, :, pt:pt + H, pl:pl + W]

def amp_dtype_for(dev):
    """bf16 on Ampere+ (A100, L4); fp16 + GradScaler on T4/V100 (bf16 there is emulated and slow)."""
    if dev.type != 'cuda': return None
    return torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16

def dice_bce(logits, y):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(3.0, device=y.device))
    p = torch.sigmoid(logits); inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()

# ------------------------------------------------------------------------------------------ metrics
THRS = np.round(np.arange(0.05, 0.951, 0.05), 2)

@torch.no_grad()
def predict_counts(model, dl, dev, amp_dtype=None):
    """One pass over a split. For every threshold in THRS and for two grids
    ('10m' native, min 40 px = 0.4 ha; '20m' = 2x2 mean-pooled probabilities vs the CPU model's 20 m
    labels (pooled fraction >= 0.5), min 10 px = 0.4 ha) accumulate pixel tp/fp/fn, chip detections on
    plume pairs and false alarms on plume-free pairs (bg1 vs bg0)."""
    model.eval(); T = torch.tensor(THRS, device=dev, dtype=torch.float32)[:, None, None, None, None]
    acc = {g: dict(tp=torch.zeros(len(THRS), device=dev, dtype=torch.float64), fp=None, fn=None, det=None, fa=None)
           for g in ('10m', '20m')}
    for g in acc:
        for k in ('fp', 'fn', 'det', 'fa'): acc[g][k] = torch.zeros_like(acc[g]['tp'])
    n = 0
    for a, _ in dl:
        a = a.to(dev, non_blocking=True); x, y, xn = eval_batch(a)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            p = torch.sigmoid(model(x).float()); pn = torch.sigmoid(model(xn).float())
        for g, (pp, yy, ppn, mpx) in {
                '10m': (p, y, pn, 40),
                '20m': (F.avg_pool2d(p, 2), (F.avg_pool2d(y, 2) >= 0.5).float(), F.avg_pool2d(pn, 2), 10)}.items():
            b = pp[None] >= T; t = yy[None].bool(); bn = ppn[None] >= T
            A = acc[g]
            A['tp'] += (b & t).sum((1, 2, 3, 4)); A['fp'] += (b & ~t).sum((1, 2, 3, 4)); A['fn'] += (~b & t).sum((1, 2, 3, 4))
            flagged = b.flatten(2).sum(2) >= mpx
            A['det'] += (flagged & (b & t).flatten(2).any(2)).sum(1)
            A['fa'] += (bn.flatten(2).sum(2) >= mpx).sum(1)
        n += a.shape[0]
    out = {}
    for g, A in acc.items():
        A = {k: v.cpu().numpy() for k, v in A.items()}
        P = A['tp'] / np.maximum(A['tp'] + A['fp'], 1); R = A['tp'] / np.maximum(A['tp'] + A['fn'], 1)
        out[g] = dict(iou=A['tp'] / np.maximum(A['tp'] + A['fp'] + A['fn'], 1), f1=2 * P * R / np.maximum(P + R, 1e-9),
                      precision=P, recall=R, det_rate=A['det'] / max(n, 1), false_alarm=A['fa'] / max(n, 1))
    return out

def at(counts, thr, grid='10m'):
    i = int(np.argmin(np.abs(THRS - thr))); return {k: float(v[i]) for k, v in counts[grid].items()}

def pick_threshold(val_counts, metric='f1', grid='10m'):
    return float(THRS[int(np.argmax(val_counts[grid][metric]))])

# ------------------------------------------------------------------------------------------ training
def train_run(name, X, meta, dev, out_root, *, encoder='mobilenet_v2', init=None, epochs=15, lr=1e-3, bs=16,
              seed=0, aug=1, neg_p=0.0, extra_refs=False, extra_ref_p=0.5, neg_max_methane=None, amp=True,
              workers=2, project='methane-se2-fullres', preview_every=3, eval_every=1, log=print,
              skip_done=True, channels_last=True):
    import trackio
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out = f'{out_root}/{name}'; os.makedirs(out, exist_ok=True)
    if skip_done and os.path.exists(f'{out}/final.pt'):
        log(f'[{name}] final.pt exists, skipping (set SKIP_DONE=False to retrain)'); return f'{out}/final.pt'
    cfg = dict(name=name, encoder=encoder, init=init, epochs=epochs, lr=lr, bs=bs, seed=seed, aug=aug, neg_p=neg_p,
               extra_refs=extra_refs, extra_ref_p=extra_ref_p, neg_max_methane=neg_max_methane, amp=amp,
               resolution='10m', chip=200)
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    tr = np.where(meta.split == 'train')[0]; va = np.where(meta.split == 'validation')[0]
    meth = torch.tensor(meta[['bg:methane1', 'bg:methane2', 'bg:methane3']].fillna(1.0).values, device=dev, dtype=torch.float32)
    model = MethaneNet(encoder, weights=None if init else 'imagenet')
    if init: model.load_state_dict(torch.load(init, map_location='cpu'))
    model.to(dev)
    if channels_last and dev.type == 'cuda': model = model.to(memory_format=torch.channels_last)
    amp_dtype = None
    if amp and dev.type == 'cuda':
        amp_dtype = amp_dtype_for(dev)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = loader(X, tr, bs, True, workers, seed, drop_last=True, nch=NCH if extra_refs else NCH_BASIC)
    dv = loader(X, va, 32, False, workers)
    spe = max(len(dl), 1); steps = epochs * spe; warm = 0 if init else spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) if warm and s < warm else
                                              0.5 * (1 + math.cos(math.pi * min(1, (s - warm) / max(1, steps - warm)))))
    trackio.init(project=project, name=name, config=cfg)
    va_vis = va[:4]
    def preview(ep):
        model.eval(); a = torch.from_numpy(np.array(X[np.sort(va_vis), :NCH_BASIC])).to(dev); x, y, _ = eval_batch(a)
        with torch.no_grad(): p = torch.sigmoid(model(x).float()).cpu()
        x = x.cpu(); f, ax = plt.subplots(3, len(va_vis), figsize=(2.2 * len(va_vis), 6.6), squeeze=False)
        for i in range(len(va_vis)):
            ax[0, i].imshow(np.clip(x[i, [2, 1, 0]].permute(1, 2, 0).numpy() * 3.5, 0, 1))
            ax[1, i].imshow(y[i, 0].cpu(), cmap='gray'); ax[2, i].imshow(p[i, 0], cmap='magma', vmin=0, vmax=1)
        for r in ax.ravel(): r.axis('off')
        f.suptitle(f'{name} epoch {ep}: RGB / label / prediction (10 m)', fontsize=9); f.tight_layout()
        fn = f'{out}/preview_ep{ep:03d}.png'; f.savefig(fn, dpi=70); plt.close(f); return fn
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time(); ls = []
        for a, rows in dl:
            a = a.to(dev, non_blocking=True); rows = rows.to(dev)
            x, y = train_batch(a, rows, gen, cfg, meth)
            if channels_last and dev.type == 'cuda': x = x.to(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = dice_bce(model(x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            ls.append(loss.item())
        rec = dict(epoch=ep, train_loss=float(np.mean(ls)) if ls else float('nan'), lr=opt.param_groups[0]['lr'],
                   epoch_s=time.time() - t0)
        if ep % eval_every == 0 or ep == epochs:
            c = predict_counts(model, dv, dev, amp_dtype)
            for g in ('10m', '20m'):
                rec.update({f'val{"" if g == "10m" else "20"}/{k}': v for k, v in at(c, 0.5, g).items()})
            rec['val/best_f1_thr'] = pick_threshold(c)
        lg = dict(rec)
        if ep % preview_every == 0 or ep == epochs: lg['val/preview'] = trackio.Image(preview(ep))
        trackio.log(lg); hist.append(rec)
        log(f'[{name}] ep {ep}/{epochs} loss {rec["train_loss"]:.4f} val IoU {rec.get("val/iou", float("nan")):.3f} '
            f'F1 {rec.get("val/f1", float("nan")):.3f} FA {rec.get("val/false_alarm", float("nan")):.3f} ({rec["epoch_s"]:.0f}s)')
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    torch.save(sd, f'{out}/final.pt.tmp'); os.replace(f'{out}/final.pt.tmp', f'{out}/final.pt')
    json.dump(dict(config=cfg, history=hist), open(f'{out}/history.json', 'w'), indent=1)
    trackio.finish()
    return f'{out}/final.pt'

def average(states):
    out = {}
    for k in states[0]:
        v = [s[k] for s in states]
        out[k] = (sum(t.float() for t in v) / len(v)).to(v[0].dtype) if v[0].is_floating_point() else v[0]
    return out

def evaluate_state(state, encoder, X, idx, dev, workers=2, amp=True):
    m = MethaneNet(encoder, weights=None); m.load_state_dict(state); m.to(dev)
    amp_dtype = amp_dtype_for(dev) if amp else None
    return predict_counts(m, loader(X, idx, 32, False, workers), dev, amp_dtype)
