"""Shared data, model and metric code for the SE-2 methane plume detector."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp

D = '/workspace/methane/data'
# cache channel layout (see prep.py): 0-5 target, 6-11 bg0, 12-17 bg1, 18 plume fraction
# per-scene bands: B2,B3,B4,B8A,B11,B12 (TOA reflectance, 20 m)
TGT = slice(0, 6); REF_SWIR = [10, 11]; BG1 = slice(12, 18); BG0_SWIR_FOR_NEG = [10, 11]
IN_CH = 8  # model input: target B2,B3,B4,B8A,B11,B12 + reference B11,B12
INPUT_BANDS = ['T_B2', 'T_B3', 'T_B4', 'T_B8A', 'T_B11', 'T_B12', 'R_B11', 'R_B12']

def load():
    X = np.load(f'{D}/cache20m.npy', mmap_mode='r')
    meta = pd.read_csv(f'{D}/meta.csv')
    return X, meta

class MethaneNet(nn.Module):
    """Raw TOA reflectance in (B,8,100,100) -> plume logits out (B,1,100,100).
    Adds an MBMP-style SWIR ratio channel in-graph so the deployed engine needs no preprocessing
    beyond reflectance scaling (DN/10000)."""
    def __init__(self, encoder='mobilenet_v2', weights='imagenet', pad=14):
        super().__init__()
        self.pad = pad
        self.norm = nn.BatchNorm2d(IN_CH + 1)
        self.net = smp.Unet(encoder, encoder_weights=weights, in_channels=IN_CH + 1, classes=1,
                            decoder_channels=(128, 64, 32, 16, 8))
    def forward(self, x):
        t11, t12, r11, r12 = x[:, 4:5], x[:, 5:6], x[:, 6:7], x[:, 7:8]
        eps = 1e-3
        ratio = torch.log((t12 + eps) / (t11 + eps)) - torch.log((r12 + eps) / (r11 + eps))
        ratio = (ratio - ratio.mean(dim=(2, 3), keepdim=True)) * 20.0
        z = self.norm(torch.cat([x, ratio], 1))
        p = self.pad
        z = F.pad(z, (p, p, p, p), mode='reflect')
        y = self.net(z)
        return y[:, :, p:-p, p:-p]

def make_batch(X, idx, rng=None, aug=0, neg_p=0.0):
    a = np.asarray(X[np.sort(idx)]).astype(np.float32)
    x = np.concatenate([a[:, TGT], a[:, REF_SWIR]], 1)
    y = (a[:, 18:19] >= 0.5).astype(np.float32)
    if neg_p > 0 and rng is not None:
        neg = rng.random(len(a)) < neg_p
        if neg.any():  # plume-free pair: bg1 as "target" vs bg0 as reference
            x[neg] = np.concatenate([a[neg][:, BG1], a[neg][:, BG0_SWIR_FOR_NEG]], 1)
            y[neg] = 0
    if aug and rng is not None:
        k = rng.integers(4); x = np.rot90(x, k, (2, 3)); y = np.rot90(y, k, (2, 3))
        if rng.random() < .5: x = x[..., ::-1]; y = y[..., ::-1]
        if aug >= 2:  # radiometric jitter: per-chip gain on all bands (scene brightness)
            g = rng.uniform(0.9, 1.1, (len(x), 1, 1, 1)).astype(np.float32); x = x * g
    return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))

def neg_batch(X, idx):
    a = np.asarray(X[np.sort(idx)]).astype(np.float32)
    return torch.from_numpy(np.concatenate([a[:, BG1], a[:, BG0_SWIR_FOR_NEG]], 1))

@torch.no_grad()
def evaluate(model, X, idx, bs=32, thr=0.5, min_px=10):
    """Pixel metrics over all pixels of the split, plus chip-level detection rate on plume chips
    and false-alarm rate on plume-free pairs (bg1 vs bg0)."""
    model.eval(); tp = fp = fn = 0; det = 0; fa = 0; n = 0
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        x, y = make_batch(X, b)
        p = torch.sigmoid(model(x)) >= thr; t = y.bool()
        tp += (p & t).sum().item(); fp += (p & ~t).sum().item(); fn += (~p & t).sum().item()
        flagged = p.flatten(1).sum(1) >= min_px
        det += (flagged & (p & t).flatten(1).any(1)).sum().item()
        pn = torch.sigmoid(model(neg_batch(X, b))) >= thr
        fa += (pn.flatten(1).sum(1) >= min_px).sum().item(); n += len(b)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(iou=tp / max(tp + fp + fn, 1), f1=2 * P * R / max(P + R, 1e-9), precision=P, recall=R,
                det_rate=det / n, false_alarm=fa / n)

def dice_bce(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(3.0))
    p = torch.sigmoid(logits); inter = (p * y).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return bce + dice.mean()
