"""Operating thresholds for the released full-res model (`fr_neg`): a two-stage read-out of the same network output.

Written by fr_neg_thresholds_colab.ipynb next to methane_fullres.py (`mf`) and synplume.py (`sp`). The onboard contract
is unchanged: `reflectance` float32 1x8x200x200 -> `plume_logits` 1x1x200x200. Only the read-out changes:

    p  = sigmoid(plume_logits)                 # 1x1x200x200, 10 m
    pp = avg_pool2d(p, 2)                      # 1x1x100x100, 20 m grid
    scene_score = max(pp)                      # same score as synplume.scene_scores
    plume_present = scene_score >= T_scene     # stage 1: T_scene set on 2021 val plume-free pairs (val FA target)
    mask = plume_present & (pp >= t_px)        # stage 2: t_px picked on 2021 val plume chips (pixel IoU)

The scene score is the MAXIMUM of ~10,000 pooled pixels, so its false-alarm-calibrated thresholds sit near 0.99. Used as a
pixel threshold they shrink every mask to its few most confident pixels (IoU ~0.03 in round 2). Hence two thresholds.

Mask metrics are micro-averaged over all chips of a split, like mf.predict_counts. "end-to-end": a chip that stage 1 does
not flag predicts an empty mask (all its plume pixels count as misses). "flagged only": the chips stage 1 flags.
"""
import numpy as np, torch, torch.nn.functional as F
import methane_fullres as mf

PX_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)
GRIDS = {'20m': 10, '10m': 40}   # grid -> minimum flagged pixels for a chip (0.4 ha), as in mf.predict_counts


@torch.no_grad()
def chip_table(model, Xft, idx, dev, amp_dtype=None, bs=16, workers=2, px_grid=PX_GRID):
    """One pass over fine-tune chips `idx` (plume pair target vs bg0, and the chip's plume-free pair bg1 vs bg0).
    Returns numpy arrays, one row per chip in `idx` order:
      score, score_in, mask_px (20 m, as sp.scene_scores), n_score (scene score of the plume-free pair),
      tp_/fp_/fn_{grid}: (N, K) pixel counts at each px threshold, npx_{grid}: (N, K) flagged pixels on the plume-free pair."""
    model.eval(); thr = torch.tensor(np.asarray(px_grid, np.float32), device=dev)[None, :, None, None]
    out = {k: [] for k in ('idx', 'score', 'score_in', 'mask_px', 'n_score')}
    for g in GRIDS: out.update({f'tp_{g}': [], f'fp_{g}': [], f'fn_{g}': [], f'npx_{g}': []})
    for a, j in mf.loader(Xft, idx, bs, False, workers):
        a = a.to(dev, non_blocking=True); x, y, xn = mf.eval_batch(a)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            p = torch.sigmoid(model(x).float()); pn = torch.sigmoid(model(xn).float())
        pp, ppn = F.avg_pool2d(p, 2), F.avg_pool2d(pn, 2); yy = F.avg_pool2d(y, 2) >= 0.5
        out['idx'].append(j); out['score'].append(pp.flatten(1).amax(1).cpu()); out['n_score'].append(ppn.flatten(1).amax(1).cpu())
        out['score_in'].append(torch.where(yy, pp, torch.zeros_like(pp)).flatten(1).amax(1).cpu()); out['mask_px'].append(yy.flatten(1).sum(1).cpu())
        for g, (q, t, qn) in {'20m': (pp, yy, ppn), '10m': (p, y >= 0.5, pn)}.items():
            b = q >= thr; bn = qn >= thr                       # (B, K, H, W)
            out[f'tp_{g}'].append((b & t).flatten(2).sum(2).cpu()); out[f'fp_{g}'].append((b & ~t).flatten(2).sum(2).cpu())
            out[f'fn_{g}'].append((~b & t).flatten(2).sum(2).cpu()); out[f'npx_{g}'].append(bn.flatten(2).sum(2).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    if 'idx' in res: assert np.array_equal(res['idx'], np.asarray(idx)), 'loader order changed'
    res['px_grid'] = np.asarray(px_grid, np.float64)
    return res


def subset(tab, rows):
    """Rows of a chip table (rows: boolean mask or positions)."""
    return {k: (v if k == 'px_grid' else v[rows]) for k, v in tab.items()}


def _k(tab, t_px):
    k = int(np.argmin(np.abs(tab['px_grid'] - t_px)))
    assert abs(tab['px_grid'][k] - t_px) < 1e-6, f'pixel threshold {t_px} is not on the grid {tab["px_grid"]}'
    return k


def _prf(tp, fp, fn):
    if tp + fp + fn == 0: return dict(iou=np.nan, precision=np.nan, recall=np.nan, f1=np.nan)   # no chips / nothing to score
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(iou=tp / (tp + fp + fn), precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9))


def mask_metrics(tab, T_scene, t_px, grid='20m'):
    """Pixel metrics of the two-stage mask. end_to_end: unflagged chips predict nothing. flagged: only flagged chips.
    chip_fa: plume-free pairs (bg1 vs bg0) of the same chips that stage 1 flags AND that keep >= 0.4 ha of mask."""
    k = _k(tab, t_px); flag = tab['score'] >= T_scene
    tp, fp, fn = (tab[f'{c}_{grid}'][:, k].astype(np.float64) for c in ('tp', 'fp', 'fn'))
    e2e = _prf(tp[flag].sum(), fp[flag].sum(), fn[flag].sum() + (tp + fn)[~flag].sum())
    fl = _prf(tp[flag].sum(), fp[flag].sum(), fn[flag].sum()) if flag.any() else dict(iou=np.nan, precision=np.nan, recall=np.nan, f1=np.nan)
    nflag = tab['n_score'] >= T_scene
    return dict(end_to_end=e2e, flagged=fl, n_flagged=int(flag.sum()),
                chip_fa=float((nflag & (tab[f'npx_{grid}'][:, k] >= GRIDS[grid])).mean()) if len(flag) else np.nan,
                chip_fa_scene_only=float(nflag.mean()) if len(flag) else np.nan)


def scene_rates(tab, T_scene):
    if not len(tab['score']): return dict(detect=np.nan, hit=np.nan, n=0)
    return dict(detect=float((tab['score'] >= T_scene).mean()), hit=float((tab['score_in'] >= T_scene).mean()), n=int(len(tab['score'])))


def pick_px(val_tab, grid='20m', metric='iou'):
    """Val-optimal pixel threshold: best ungated val pixel IoU (stage 2 is judged on its own; 193 val chips, so noisy)."""
    vals = [_prf(*(val_tab[f'{c}_{grid}'][:, k].sum() for c in ('tp', 'fp', 'fn')))[metric] for k in range(len(val_tab['px_grid']))]
    k = int(np.nanargmax(vals)); return float(val_tab['px_grid'][k]), dict(zip(map(float, val_tab['px_grid']), map(float, vals)))


def cluster_boot(groups, stat, n=1000, seed=0, alpha=0.05):
    """Percentile CI of stat(rows) when whole clusters (sites) are resampled with replacement. NaN if < 2 clusters."""
    groups = np.asarray(groups); u, inv = np.unique(groups, return_inverse=True)
    if len(u) < 2 or n <= 0: return [float('nan'), float('nan')]
    members = [np.where(inv == c)[0] for c in range(len(u))]; rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        rows = np.concatenate([members[c] for c in rng.integers(0, len(u), len(u))]); vals.append(stat(rows))
    return [float(np.nanquantile(vals, alpha / 2)), float(np.nanquantile(vals, 1 - alpha / 2))]


def logit(p): p = float(p); return float(np.log(p / (1 - p))) if 0 < p < 1 else float('inf') if p >= 1 else float('-inf')
