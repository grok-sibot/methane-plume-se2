"""Unit checks for colab/thresholds.py against the existing metric code (methane_fullres.predict_counts, synplume.scene_scores).
usage: python thresholds_unit.py [CACHE10M.npy] [MODEL.safetensors]
Without arguments: random chips and a randomly initialised model (the checks are identities, so any data works)."""
import sys, os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import methane_fullres as mf, synplume as sp, thresholds as th
torch.manual_seed(0); rng = np.random.default_rng(0); dev = torch.device('cpu')
if len(sys.argv) > 1:
    X = np.load(sys.argv[1], mmap_mode='r'); idx = np.sort(rng.choice(len(X), 48, replace=False))
else:
    X = rng.uniform(0.05, 0.4, (48, 31, 200, 200)).astype(np.float16); idx = np.arange(48)
    for i in range(48):   # blob masks, some chips empty
        if i % 6: r, c, s = rng.integers(20, 180, 2).tolist() + [int(rng.integers(5, 30))]; X[i, 0, max(0, r - s):r + s, max(0, c - s):c + s] = 1
model = mf.MethaneNet('mobilenet_v2', weights=None)
if len(sys.argv) > 2:
    from safetensors.torch import load_file; model.load_state_dict(load_file(sys.argv[2]))
model.eval()
tab = th.chip_table(model, X, idx, dev, bs=8, workers=0)
c = mf.predict_counts(model, mf.loader(X, idx, 8, False, 0), dev, thrs=list(th.PX_GRID))
ss = sp.scene_scores(model, X, idx, dev, kind='plumes', workers=0)
ok = True
def check(name, a, b, tol=1e-6):
    global ok; good = np.allclose(a, b, atol=tol, equal_nan=True); ok &= good
    print(f'{"PASS" if good else "FAIL"} {name}: {np.round(np.atleast_1d(a)[:4], 6)} vs {np.round(np.atleast_1d(b)[:4], 6)}')
check('scene score == sp.scene_scores', tab['score'], ss['score']); check('score_in', tab['score_in'], ss['score_in']); check('mask_px', tab['mask_px'], ss['mask_px'])
for g in ('20m', '10m'):
    for t in (0.3, 0.5, 0.7):
        m = th.mask_metrics(tab, -1.0, t, g)   # T_scene below every score: stage 1 flags all -> plain pixel metrics
        ref = mf.at(c, t, g)
        check(f'{g} t={t} IoU (T_scene off) == predict_counts', m['end_to_end']['iou'], ref['iou'])
        check(f'{g} t={t} F1', m['end_to_end']['f1'], ref['f1'])
        check(f'{g} t={t} chip FA (T_scene off) == predict_counts', m['chip_fa'], ref['false_alarm'])
    # stage 1 at T <= t_px cannot change the 20 m mask (a pooled pixel >= t_px implies the max >= T)
    if g == '20m':
        for t in (0.5, 0.7): check(f'20m gate at T=t={t} changes nothing', th.mask_metrics(tab, t, t, g)['end_to_end']['iou'], mf.at(c, t, g)['iou'])
# gating can only remove predicted pixels: end-to-end recall never rises with T
rs = [th.mask_metrics(tab, T, 0.5)['end_to_end']['recall'] for T in np.quantile(tab['score'], [0, .25, .5, .75, .95])]
good = all(a >= b - 1e-12 for a, b in zip(rs, rs[1:])); ok &= good; print(('PASS' if good else 'FAIL'), 'recall non-increasing in T_scene', np.round(rs, 4))
# operating points agree with sp.operating_points
neg = rng.uniform(0.9, 1.0, 500); ops = sp.operating_points(ss, neg)
for op, o in ops.items(): check(f'scene_rates at {op} == sp.operating_points', [th.scene_rates(tab, o['threshold'])[k] for k in ('detect', 'hit')], [o['detect'], o['hit']])
# bootstrap: one cluster -> NaN; constant stat -> zero-width interval
ci = th.cluster_boot(np.zeros(10), lambda r: 1.0); good = all(np.isnan(ci)); ok &= good; print(('PASS' if good else 'FAIL'), 'one cluster -> NaN CI')
ci = th.cluster_boot(np.arange(10) % 3, lambda r: 0.25, n=50); check('constant stat -> [0.25, 0.25]', ci, [0.25, 0.25])
t_px, curve = th.pick_px(tab); check('pick_px is the argmax of the IoU curve', curve[t_px], max(curve.values()))
print('ALL PASS' if ok else 'SOME CHECKS FAILED'); sys.exit(0 if ok else 1)
