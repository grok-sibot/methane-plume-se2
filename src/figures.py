"""Training-curves PNG (all runs) and example-prediction grid PNG (test split)."""
import sys, json, glob, os, numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import *
R = '/workspace/methane/runs'; out = sys.argv[2]; ck = sys.argv[1]
runs = sorted(glob.glob(f'{R}/*/history.json'))
f, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for p in runs:
    h = json.load(open(p)); n = p.split('/')[-2]; e = [r['epoch'] for r in h]
    off = 0 if n == 'base' else json.load(open(f'{R}/base/history.json'))[-1]['epoch']
    e = [x + off for x in e]
    for a, k in zip(ax, ['train_loss', 'val/iou', 'val/f1', 'val/false_alarm']):
        a.plot(e, [r[k] for r in h], label=n, lw=1.5, marker='o', ms=2.5)
for a, k in zip(ax, ['train loss (Dice+BCE)', 'val IoU (pixel)', 'val F1 (pixel)', 'val chip false-alarm rate']):
    a.set_title(k); a.set_xlabel('epoch (fine-tunes continue from base)'); a.grid(alpha=.2)
ax[0].legend(fontsize=8); f.tight_layout(); f.savefig(f'{out}/training_curves.png', dpi=120); plt.close(f)
X, meta = load(); te = np.where(meta.split == 'test')[0]
m = MethaneNet(weights=None); m.load_state_dict(torch.load(ck)); m.eval()
rng = np.random.default_rng(3); pick = np.sort(rng.choice(te, 6, replace=False))
x, y = make_batch(X, pick)
with torch.no_grad(): p = torch.sigmoid(m(x)).numpy()
ratio = (np.log((x[:, 5] + 1e-3) / (x[:, 4] + 1e-3)) - np.log((x[:, 7] + 1e-3) / (x[:, 6] + 1e-3))).numpy()
f, ax = plt.subplots(len(pick), 4, figsize=(10, 2.6 * len(pick)))
for i in range(len(pick)):
    rgb = np.clip(x[i, [2, 1, 0]].permute(1, 2, 0).numpy() * 3.5, 0, 1)
    r = ratio[i] - np.median(ratio[i])
    ax[i, 0].imshow(rgb); ax[i, 1].imshow(r, cmap='RdBu', vmin=-0.08, vmax=0.08)
    ax[i, 2].imshow(y[i, 0], cmap='gray'); ax[i, 3].imshow(p[i, 0] >= 0.5, cmap='gray')
    ax[i, 3].contour(y[i, 0], levels=[0.5], colors='r', linewidths=0.6)
    ax[i, 0].set_ylabel(f"{meta.loc[pick[i], 'detection:case_study']}"[:18], fontsize=8)
for j, t in enumerate(['RGB (B4,B3,B2)', 'input: SWIR ratio vs ref', 'ground truth', 'prediction (red = GT)']):
    ax[0, j].set_title(t, fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
f.tight_layout(); f.savefig(f'{out}/example_predictions.png', dpi=110); plt.close(f)
print('ok', pick.tolist())
