"""Run from the box layout (/workspace/methane). Needs the 10 m val+test cache built by the swap run and the 20 m CPU cache.
   Three-way test scoring: all / seen-site / unseen-site.
Site rule: a test chip is 'seen' if some TRAIN chip has the same site:location_name AND an emission
point (emission:lat/lon) within 1 km (haversine). Otherwise 'unseen'."""
import sys, numpy as np, pandas as pd, torch, json
sys.path.insert(0, '/workspace/methane/fr_eval'); sys.path.insert(0, '/workspace/methane/src')
torch.set_num_threads(8)
D = '/workspace/methane/data'
l0 = pd.read_parquet(f'{D}/.tacocat/level0.parquet').reset_index(drop=True)
tr = l0[l0.split == 'train']; te = l0[l0.split == 'test']
def hav(a, b, c, d):
    p = np.pi / 180; x = np.sin((c - a) * p / 2) ** 2 + np.cos(a * p) * np.cos(c * p) * np.sin((d - b) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))
seen = {}
for _, r in te.iterrows():
    m = tr['site:location_name'].values == r['site:location_name']
    dd = hav(r['emission:lat'], r['emission:lon'], tr['emission:lat'].values, tr['emission:lon'].values)
    seen[r['id']] = bool((m & (dd <= 1.0)).any())
sites = te.assign(seen=te.id.map(seen))
summ = dict(n_test=len(te), n_seen=int(sites.seen.sum()), n_unseen=int((~sites.seen).sum()),
            sites_seen=int(sites[sites.seen]['site:location_name'].nunique()),
            sites_unseen=int(sites[~sites.seen]['site:location_name'].nunique()),
            unseen_by_country=sites[~sites.seen]['site:country'].value_counts().to_dict(),
            unseen_by_sector=sites[~sites.seen]['detection:sector'].value_counts().to_dict())
print(summ, flush=True)
rows = []
K = ['iou', 'f1', 'precision', 'recall', 'det_rate', 'false_alarm']
# --- full-res fr_neg (10 m cache of val+test built by the swap run)
import methane_fullres as mf
X = np.load('/workspace/methane/fr_eval/cache10m_vt.npy', mmap_mode='r'); meta = pd.read_csv('/workspace/methane/fr_eval/meta_vt.csv')
sd = torch.load('/workspace/methane/fullres-runs/fr_neg/final.pt', map_location='cpu')
for sub in ('all', 'seen', 'unseen'):
    idx = np.where((meta.split == 'test') & (meta.id.map(lambda i: True if sub == 'all' else seen.get(i) == (sub == 'seen'))))[0]
    c = mf.evaluate_state(sd, 'mobilenet_v2', X, idx, torch.device('cpu'), workers=0, amp=False)
    for g in ('10m', '20m'):
        r = mf.at(c, 0.5, g); rows.append(dict(model='fr_neg (full-res, released)', subset=sub, n=len(idx), grid=g, threshold=0.5, **{k: r[k] for k in K}))
        print(rows[-1], flush=True)
# --- CPU 20 m greedy soup (released sighmon/methane-plume-se2)
import common as cm
X2, meta2 = cm.load()
m = cm.MethaneNet(weights=None); m.load_state_dict(torch.load('/workspace/methane/runs/greedy_soup.pt', map_location='cpu'))
for sub in ('all', 'seen', 'unseen'):
    idx = np.where((meta2.split == 'test') & (meta2.id.map(lambda i: True if sub == 'all' else seen.get(i) == (sub == 'seen'))))[0]
    r = cm.evaluate(m, X2, idx, thr=0.5, min_px=10)
    rows.append(dict(model='greedy soup (20 m CPU, released)', subset=sub, n=len(idx), grid='20m (native)', threshold=0.5, **{k: r[k] for k in K}))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv('test_seen_unseen.csv', index=False, float_format='%.4f')
json.dump(dict(rule='seen = same site:location_name AND emission point within 1 km of a train chip', **summ), open('split_summary.json', 'w'), indent=1)
print(df.round(3).to_string())
