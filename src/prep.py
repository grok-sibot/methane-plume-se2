"""Build a compact 20 m cache of MethaneSET S2-finetune.
Per sample: target[B2,B3,B4,B8A,B11,B12], bg0[B2,B3,B4,B8A,B11,B12], bg1[same], plume mask (fraction).
10 m chips (200x200) are 2x2 mean-pooled to 100x100 (20 m, the native SWIR resolution).
Reflectance stored as float16 = DN/10000."""
import numpy as np, pandas as pd, rasterio, os, sys
from multiprocessing import Pool
D = '/workspace/methane/data'
BANDS = [1, 2, 3, 8, 11, 12]  # 0-based idx into 13-band L1C stack: B2,B3,B4,B8A,B11,B12
l0 = pd.read_parquet(f'{D}/.tacocat/level0.parquet')
l1 = pd.read_parquet(f'{D}/.tacocat/level1.parquet')
l0 = l0.reset_index(drop=True)
l1['_pid'] = l1['internal:relative_path'].str.split('/').str[0]
IDX = {(p, k): j for j, (p, k) in enumerate(zip(l1['_pid'], l1['id']))}
def path(r):
    return f"/vsisubfile/{r['internal:offset']}_{r['internal:size']},{D}/zips/{r['internal:source_file']}"
def pool2(a):
    c, h, w = a.shape
    return a.reshape(c, h // 2, 2, w // 2, 2).mean((2, 4))
def work(i):
    leaves = {k: l1.loc[IDX[(l0.loc[i, 'id'], k)]] for k in ['target', 'bg0', 'bg1', 'plume']}
    out = []
    for k in ['target', 'bg0', 'bg1']:
        with rasterio.open(path(leaves[k])) as d:
            a = d.read([b + 1 for b in BANDS]).astype(np.float32) / 10000.0
        out.append(pool2(a))
    with rasterio.open(path(leaves['plume'])) as d:
        m = d.read(1).astype(np.float32)[None]
    out.append(pool2(m))
    return i, np.concatenate(out).astype(np.float16)
if __name__ == '__main__':
    N = len(l0)
    X = np.lib.format.open_memmap(f'{D}/cache20m.npy', 'w+', np.float16, (N, 19, 100, 100))
    with Pool(6) as p:
        for n, (i, a) in enumerate(p.imap_unordered(work, range(N), chunksize=8)):
            X[i] = a
            if n % 500 == 0: print(n, flush=True)
    X.flush()
    l0[['id', 'split', 'detection:case_study', 'detection:ch4_fluxrate', 'site:country', 'target:cloud', 'bg:cloud0', 'bg:cloud1', 'bg:methane1']].to_csv(f'{D}/meta.csv', index=False)
    print('done', N)
