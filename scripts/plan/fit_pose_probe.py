#!/usr/bin/env python3
"""Fit the pose probes used by the oracle planning cost (scripts/plan/pose_probe.py).

For every checkpoint, on rotation-balanced dataset clips (same sampling as rotation_probe.py),
fits an MLP from the embedding to T x, y, yaw and EEF x, y, two ways:

  <label>_real.pt  fit on real frame embeddings only
  <label>_both.pt  fit on real frame embeddings and on the predictor's rollouts with the clip's
                   true action blocks (steps 1..horizon), so it also reads predictions that
                   have drifted away from the encoder's space

Probes are fit on half of the episode groups; summary.md reports the held-out error on real
frames and on predictions per rollout step, i.e. how well the oracle cost can read a plan.

  python scripts/plan/fit_pose_probe.py --out <dir> \\
      base_s1=noisy_lewm_res_nobn_w0.009_s1_27001817/weights_epoch_20.pt ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rotation_probe import (
    MLP,
    choose_clips,
    embed,
    load_model,
    open_dataset,
    pose_features,
    probe_errors,
    resolve,
    saved_frameskip,
    targets,
)


def save_probe(mlp, path, meta):
    torch.save(
        {
            'state_dict': {k: v.detach().cpu() for k, v in mlp.net.state_dict().items()},
            'mu': torch.as_tensor(mlp.mu, dtype=torch.float32),
            'sd': torch.as_tensor(mlp.sd, dtype=torch.float32),
            'in_dim': int(len(mlp.mu)),
            'hidden': int(mlp.hidden),
            'out_dim': 6,
            'meta': meta,
        },
        path,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+', help='label=path to weights_*.pt')
    p.add_argument('--dataset', default='overhead_rgb_noisy_replay_100k.lance')
    p.add_argument('--cache-dir', default=None, help='dataset root (holds datasets/); default STABLEWM_HOME')
    p.add_argument('--horizon', type=int, default=5, help='rollout steps the probe is fit on (>= the planning horizon)')
    p.add_argument('--frameskip', type=int, default=5)
    p.add_argument('--num-clips', type=int, default=8000)
    p.add_argument('--candidates', type=int, default=200000)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--mlp-epochs', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models, paths, H, fs = {}, {}, None, None
    for item in args.checkpoints:
        name, path = item.split('=', 1)
        pt = resolve(path)
        model = load_model(pt).to(args.device).eval()
        h, f = int(getattr(model.predictor, 'num_frames', 1)), saved_frameskip(pt) or args.frameskip
        if H is not None and (h, f) != (H, fs):
            raise SystemExit(f'{name}: history {h} / frameskip {f} differ from the first checkpoint ({H} / {fs})')
        H, fs = h, f
        models[name], paths[name] = model, str(pt)
        print(f'{name}: {pt} (history {h}, frameskip {f})')
    T = args.horizon

    ds = open_dataset(args, H + T, fs)
    idx, group = choose_clips(ds, args, H + T, fs)
    pose = pose_features(ds, idx, H + T, fs)
    rng = np.random.default_rng(args.seed)
    groups = np.unique(group)
    test_groups = set(rng.choice(groups, len(groups) // 2, replace=False).tolist())
    test = np.array([g in test_groups for g in group])
    train = ~test
    print(f'{len(idx)} clips; train {train.sum()} / test {test.sum()} (split by episode group)')

    real, pred = embed(models, ds, idx, H, T, args)
    y = targets(pose)  # (N, H+T, 6)
    results = {'config': {'checkpoints': paths, 'dataset': args.dataset, 'history': H, 'frameskip': fs,
                          'horizon': T, 'clips': int(len(idx)), 'train': int(train.sum()), 'test': int(test.sum())},
               'models': {}}
    for name in models:
        z, zp = real[name], pred[name]  # (N, H+T, D), (N, T, D)
        D = z.shape[-1]
        x_real, y_real = z[train].reshape(-1, D), y[train].reshape(-1, 6)
        x_pred, y_pred = zp[train].reshape(-1, D), y[train][:, H:].reshape(-1, 6)
        fits = {
            'real': (x_real, y_real),
            'both': (np.concatenate([x_real, x_pred]), np.concatenate([y_real, y_pred])),
        }
        results['models'][name] = {}
        for kind, (xt, yt) in fits.items():
            mlp = MLP(args.device, epochs=args.mlp_epochs).fit(xt, yt)
            ev = {'real': probe_errors(mlp.predict(z[test].reshape(-1, D)), y[test].reshape(-1, 6)),
                  'pred_per_step': [probe_errors(mlp.predict(zp[test][:, k]), y[test][:, H + k]) for k in range(T)]}
            path = out / f'{name}_{kind}.pt'
            save_probe(mlp, path, {'checkpoint': paths[name], 'fit_on': kind, 'history': H, 'frameskip': fs, 'test': ev})
            results['models'][name][kind] = {'path': str(path), **ev}
            steps = ' '.join(f"{e['yaw_err_deg']:.1f}" for e in ev['pred_per_step'])
            print(f"{name} [{kind}]: real yaw {ev['real']['yaw_err_deg']:.1f} deg; predicted yaw per step {steps} -> {path}")
    json.dump(results, open(out / 'metrics.json', 'w'), indent=1)
    write_summary(results, out / 'summary.md')
    print(f'wrote {out}')


def write_summary(results, path):
    c = results['config']
    L = ['# Pose probes for the oracle planning cost', '',
         f"- dataset `{c['dataset']}`, {c['clips']} rotation-balanced clips; fit on {c['train']}, tested on {c['test']} (split by episode group)",
         '- held-out median errors; predicted = rollout with the clip\'s true action blocks', '',
         '| model | probe | real yaw (°) / T xy / EEF (mm) | predicted yaw (°) per step | predicted T xy (mm) per step | predicted EEF (mm) per step |',
         '|---|---|---|---|---|---|']
    for name, kinds in results['models'].items():
        for kind, ev in kinds.items():
            r, ps = ev['real'], ev['pred_per_step']
            L.append(f"| {name} | {kind} | {r['yaw_err_deg']:.1f} / {r['txy_err_mm']:.1f} / {r['eef_err_mm']:.1f} | "
                     + ' '.join(f"{e['yaw_err_deg']:.1f}" for e in ps) + ' | '
                     + ' '.join(f"{e['txy_err_mm']:.1f}" for e in ps) + ' | '
                     + ' '.join(f"{e['eef_err_mm']:.1f}" for e in ps) + ' |')
    Path(path).write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
