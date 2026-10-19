#!/usr/bin/env python3
"""Compare the latent geometry of vision LeWM checkpoints on PushT data.

Example (run from the repository root)::

    python scripts/plan/embedding_diagnostics.py \
        bn=noisy_lewm_res_bn_11308936/weights_epoch_20.pt \
        nobn=noisy_lewm_res_nobn_11308937/weights_epoch_20.pt \
        --dataset overhead_rgb_noisy_replay_100k.lance --output emb_diag.json

Checkpoints are ``label=path`` pairs; paths resolve like ``load_pretrained``
(relative to ``$STABLEWM_HOME/checkpoints``). Every model sees the same random
(frame t, frame t + frameskip, action block) samples, prepared with the
training transforms. Per model it reports:

- covariance spectrum: effective rank (sum λ)² / sum λ², top-eigenvalue
  shares and dimensions for 90% variance;
- the SIGReg statistic on the batch (fixed seed; compare between models only);
- one-step prediction: ||pred - e1||² / ||e1 - e0||², and norm ratio and
  cosine between predicted and true latent changes;
- ridge probes from the embedding to T xy, T yaw (sin, cos) and EEF xy;
- the share of embedding variance explained by a quadratic regression on
  T state, EEF state and both.

Probes and regressions fit on the first half of the samples and report
variance-weighted R² on the second half. Runs on CPU or GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
import stable_worldmodel as swm  # noqa: E402
from lewm import get_img_preprocessor, normalize_pixels  # noqa: E402
from stable_worldmodel.data import column_normalizer  # noqa: E402
from stable_worldmodel.wm.loss import SIGReg  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402


def load_batch(args):
    ds = swm.data.load_dataset(
        args.dataset,
        transform=None,
        cache_dir=args.cache_dir,
        num_steps=2,
        frameskip=args.frameskip,
        keys_to_load=['pixels', 'action', 'obs_extra_obj_pose', 'obs_extra_tcp_pose'],
        keys_to_cache=['action'],
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor('pixels', 'pixels', img_size=args.img_size, uint8=True),
        column_normalizer(ds, 'action', 'action'),
    )
    rng = np.random.default_rng(args.seed)
    items = [ds[i] for i in rng.choice(len(ds), args.num_samples, replace=False).tolist()]
    batch = {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in items[0]}
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    normalize_pixels(batch)
    return batch


def state_features(batch):
    """T features (x, y, sin yaw, cos yaw) and EEF features (x, y) at frame t."""
    obj = batch['obs_extra_obj_pose'][:, 0].numpy()
    tcp = batch['obs_extra_tcp_pose'][:, 0].numpy()
    w, x, y, z = obj[:, 3], obj[:, 4], obj[:, 5], obj[:, 6]  # wxyz quaternion
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    t_feat = np.stack([obj[:, 0], obj[:, 1], np.sin(yaw), np.cos(yaw)], 1)
    return t_feat, tcp[:, :2]


def poly2(f):
    f = (f - f.mean(0)) / (f.std(0) + 1e-8)
    n = f.shape[1]
    return np.concatenate([f] + [f[:, i:i + 1] * f[:, j:j + 1] for i in range(n) for j in range(i, n)], 1)


def ridge_r2(x, y, lam):
    """Fit on the first half, variance-weighted R² on the second half."""
    h = len(x) // 2
    x = np.concatenate([x, np.ones((len(x), 1))], 1)
    w = np.linalg.solve(x[:h].T @ x[:h] + lam * np.eye(x.shape[1]), x[:h].T @ y[:h])
    res = y[h:] - x[h:] @ w
    return float(1 - (res**2).sum() / ((y[h:] - y[h:].mean(0)) ** 2).sum())


@torch.no_grad()
def diagnose(model, batch, t_feat, e_feat, device):
    pixels = batch['pixels'].to(device)
    action = batch['action'].to(device)
    emb, preds = [], []
    for sl in torch.split(torch.arange(len(pixels)), 256):
        info = model.encode({'pixels': pixels[sl], 'action': action[sl]})
        emb.append(info['emb'])
        preds.append(model.predict({'emb': info['emb'][:, :1], 'act_emb': info['act_emb'][:, :1]})['preds'][:, 0])
    emb, pred = torch.cat(emb), torch.cat(preds)
    torch.manual_seed(0)
    sigreg = SIGReg().to(device)(emb[:, :1].transpose(0, 1)).item()

    e0, e1, p = emb[:, 0].cpu().numpy(), emb[:, 1].cpu().numpy(), pred.cpu().numpy()
    ev = np.linalg.eigvalsh(np.cov(e0.T))[::-1]
    true_d, pred_d = e1 - e0, p - e0
    true_n, pred_n = np.linalg.norm(true_d, axis=1), np.linalg.norm(pred_d, axis=1)
    return {
        'mean_dim_var': float(ev.mean()),
        'mean_abs_center': float(np.abs(e0.mean(0)).mean()),
        'eff_rank': float(ev.sum() ** 2 / (ev**2).sum()),
        'top1_frac': float(ev[0] / ev.sum()),
        'top10_frac': float(ev[:10].sum() / ev.sum()),
        'dims_90': int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1),
        'sigreg': sigreg,
        'onestep_rel_err': float(((p - e1) ** 2).sum(1).mean() / (true_n**2).mean()),
        'delta_norm_ratio': float(pred_n.mean() / true_n.mean()),
        'delta_cos': float((np.sum(pred_d * true_d, 1) / (pred_n * true_n + 1e-8)).mean()),
        'probe_yaw': ridge_r2(e0, t_feat[:, 2:], lam=1.0),
        'probe_Txy': ridge_r2(e0, t_feat[:, :2], lam=1.0),
        'probe_eef': ridge_r2(e0, e_feat, lam=1.0),
        'var_expl_by_T': ridge_r2(poly2(t_feat), e0, lam=1e-2),
        'var_expl_by_eef': ridge_r2(poly2(e_feat), e0, lam=1e-2),
        'var_expl_by_both': ridge_r2(poly2(np.concatenate([t_feat, e_feat], 1)), e0, lam=1e-2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoints', nargs='+', help='label=path pairs, e.g. bn=run/weights_epoch_20.pt')
    parser.add_argument('--dataset', default='overhead_rgb_noisy_replay_100k.lance')
    parser.add_argument('--cache-dir', default=None, help='dataset root (holds datasets/); default STABLEWM_HOME')
    parser.add_argument('--num-samples', type=int, default=2048)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default=None, help='write all metrics to this JSON file')
    args = parser.parse_args()

    batch = load_batch(args)
    t_feat, e_feat = state_features(batch)
    results = {}
    for spec in args.checkpoints:
        label, _, path = spec.partition('=')
        if not path:
            label, path = Path(spec).parent.name, spec
        model = load_pretrained(path).to(args.device).eval()
        results[label] = diagnose(model, batch, t_feat, e_feat, args.device)
        print(label, json.dumps({k: round(v, 4) for k, v in results[label].items()}), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
