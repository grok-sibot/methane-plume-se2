"""Train / fine-tune MethaneNet on MethaneSET S2 (20 m cache). Logs to Trackio."""
import argparse, os, time, json, numpy as np, torch
os.environ.setdefault('TRACKIO_DIR', '/workspace/methane/trackio')
import trackio
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--init', default=None)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=15)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--bs', type=int, default=16)
ap.add_argument('--aug', type=int, default=1); ap.add_argument('--neg_p', type=float, default=0.0)
ap.add_argument('--threads', type=int, default=8); ap.add_argument('--encoder', default='mobilenet_v2')
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
OUT = f'/workspace/methane/runs/{a.name}'; os.makedirs(OUT, exist_ok=True)
X, meta = load()
tr = np.where(meta.split == 'train')[0]; va = np.where(meta.split == 'validation')[0]
model = MethaneNet(a.encoder, weights=None if a.init else 'imagenet')
if a.init: model.load_state_dict(torch.load(a.init))
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
steps = a.epochs * (len(tr) // a.bs); warm = 0 if a.init else len(tr) // a.bs
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) if warm and s < warm else
                                          0.5 * (1 + np.cos(np.pi * min(1, (s - warm) / max(1, steps - warm)))))
trackio.init(project='methane-se2', name=a.name, config=vars(a))
vis = va[:4]
def preview(ep):
    model.eval()
    with torch.no_grad():
        x, y = make_batch(X, vis); p = torch.sigmoid(model(x))
    f, ax = plt.subplots(3, 4, figsize=(8, 6))
    for i in range(4):
        ax[0, i].imshow(np.clip(x[i, [2, 1, 0]].permute(1, 2, 0).numpy() * 3.5, 0, 1)); ax[1, i].imshow(y[i, 0], cmap='gray')
        ax[2, i].imshow(p[i, 0], cmap='magma', vmin=0, vmax=1)
    for r in ax.ravel(): r.axis('off')
    f.suptitle(f'{a.name} epoch {ep}: RGB / label / prediction'); f.tight_layout()
    fn = f'{OUT}/preview_ep{ep:02d}.png'; f.savefig(fn, dpi=70); plt.close(f); return fn
hist = []
for ep in range(1, a.epochs + 1):
    model.train(); perm = rng.permutation(tr); t0 = time.time(); ls = []
    for s in range(0, len(perm) - a.bs + 1, a.bs):
        x, y = make_batch(X, perm[s:s + a.bs], rng, a.aug, a.neg_p)
        loss = dice_bce(model(x), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); ls.append(loss.item())
    m = evaluate(model, X, va)
    rec = dict(epoch=ep, train_loss=float(np.mean(ls)), lr=opt.param_groups[0]['lr'], epoch_s=time.time() - t0,
               **{f'val/{k}': v for k, v in m.items()})
    log = dict(rec)
    if ep % 3 == 0 or ep == a.epochs: log['val/preview'] = trackio.Image(preview(ep))
    trackio.log(log); hist.append(rec); print(json.dumps(rec), flush=True)
torch.save(model.state_dict(), f'{OUT}/final.pt'); json.dump(hist, open(f'{OUT}/history.json', 'w'))
trackio.finish()
