"""Evaluate fine-tunes, uniform soup and greedy soup (Wortsman et al. 2022) on val and test."""
import sys, json, torch, numpy as np
from common import *
torch.set_num_threads(8)
X, meta = load()
va = np.where(meta.split == 'validation')[0]; te = np.where(meta.split == 'test')[0]
R = '/workspace/methane/runs'
base = ['base']; fts = sys.argv[1:]
sd = {n: torch.load(f'{R}/{n}/final.pt') for n in base + fts}
def avg(names):
    out = {}
    for k in sd[names[0]]:
        v = [sd[n][k] for n in names]
        out[k] = (sum(t.float() for t in v) / len(v)).to(v[0].dtype) if v[0].is_floating_point() else v[0]
    return out
def ev(state, idx):
    m = MethaneNet(weights=None); m.load_state_dict(state); return evaluate(m, X, idx)
res = {}
for n in base + fts:
    res[n] = {'val': ev(sd[n], va)}; print(n, res[n], flush=True)
res['uniform_soup'] = {'val': ev(avg(fts), va), 'ingredients': fts}; print('uniform', res['uniform_soup'], flush=True)
order = sorted(fts, key=lambda n: -res[n]['val']['iou']); greedy = [order[0]]; best = res[order[0]]['val']['iou']
for n in order[1:]:
    s = ev(avg(greedy + [n]), va)['iou']
    if s > best: greedy.append(n); best = s
res['greedy_soup'] = {'val': ev(avg(greedy), va), 'ingredients': greedy}; print('greedy', res['greedy_soup'], flush=True)
torch.save(avg(fts), f'{R}/uniform_soup.pt'); torch.save(avg(greedy), f'{R}/greedy_soup.pt')
states = {**sd, 'uniform_soup': avg(fts), 'greedy_soup': avg(greedy)}
for n in res: res[n]['test'] = ev(states[n], te); print(n, 'test', res[n]['test'], flush=True)
json.dump(res, open(f'{R}/results.json', 'w'), indent=1)
