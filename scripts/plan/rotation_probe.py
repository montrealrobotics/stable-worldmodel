#!/usr/bin/env python3
"""Where do vision LeWM checkpoints lose the T's rotation?

Three tests on the same dataset clips (1 context frame + `--horizon` action
blocks, oversampled for large rotations), for every checkpoint:

1. representation: probes from the embedding of a real frame to T yaw
   (sin, cos), T xy and EEF xy. Linear (ridge) and a small MLP; reported as
   median angular error (deg) and position error (mm) on held-out episode
   groups (an expert episode and its replays never straddle the split).
2. planning cost: the planner ranks plans by squared embedding distance to
   the goal. For frame pairs inside a clip, mean distance vs T rotation (with
   little T motion) and vs T displacement (with little rotation), and a
   non-negative fit  d = a*dxy^2 + b*chord(dyaw)^2 + c*deef^2 + e  that gives
   "how many mm of T displacement cost as much as 45 deg of rotation".
3. prediction: roll the predictor out with the clip's true action blocks
   and apply the probes (fit on real embeddings) to the predicted ones: yaw
   error per step, and how much of the true rotation the prediction shows
   (slope of probed vs true rotation), by rotation size.

  python scripts/plan/rotation_probe.py \\
      inbatch=noisy_lewm_res_nobn_scratch_search_w0.01_inbatch40_s3072_11811959/weights_epoch_20.pt \\
      cem_s1=noisy_lewm_res_nobn_scratch_search_w0.01_s1_27001812/weights_epoch_20.pt \\
      base_s1=noisy_lewm_res_nobn_w0.009_s1_27001817/weights_epoch_20.pt --out reports/rotation_probe

Checkpoint paths resolve like load_pretrained (relative to
$STABLEWM_HOME/checkpoints). Writes metrics.json and summary.md; plot with
scripts/plan/plot_rotation_probe.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
import stable_worldmodel as swm  # noqa: E402
from lewm import get_img_preprocessor, normalize_pixels  # noqa: E402
from stable_worldmodel.data import column_normalizer  # noqa: E402
from stable_worldmodel.wm.utils import load_pretrained  # noqa: E402

ROT_BINS = [0.0, 0.2, 0.5, 0.8, 1.2, np.inf]  # rad, |rotation| from the context frame to the clip's last frame
POSE_KEYS = ['obs_extra_obj_pose', 'obs_extra_tcp_pose']
SMALL_MOVE_MM, SMALL_ROT_DEG = 30, 10  # 'T only rotates' / 'T only moves' pairs in the cost test


# -- checkpoints ---------------------------------------------------------------

def resolve(path):
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = Path(os.environ.get('STABLEWM_HOME', '.')) / 'checkpoints' / p
    return p


def load_model(pt):
    """load_pretrained, or, for weights saved with the legacy HF ViT key names
    (transformers < 5.12), the same model with the keys renamed in memory
    (scripts/train/convert_vit_keys.py; the file is not changed)."""
    try:
        return load_pretrained(str(pt))
    except RuntimeError as exc:
        if 'encoder.' not in str(exc):
            raise
        from convert_vit_keys import match_hf_vit_keys
        from hydra.utils import instantiate

        model = instantiate(json.load(open(pt.parent / 'config.json')))
        sd = match_hf_vit_keys(torch.load(pt, map_location='cpu'), model.state_dict().keys())
        model.load_state_dict(sd)  # strict: fails if the rename did not match
        print(f'  {pt.name}: legacy ViT key names, renamed in memory')
        return model


def saved_frameskip(pt):
    for cand in (pt.parent / 'config.yaml', pt.parent / 'hydra' / 'config.yaml', pt.parent / '.hydra' / 'config.yaml'):
        if cand.exists():
            return int(yaml.safe_load(open(cand))['data']['dataset']['frameskip'])
    return None


# -- data ----------------------------------------------------------------------

def quat_yaw(q):
    w, x, y, z = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def open_dataset(args, num_steps, frameskip):
    ds = swm.data.load_dataset(
        args.dataset,
        transform=None,
        cache_dir=args.cache_dir,
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=['pixels', 'action', *POSE_KEYS],
        keys_to_cache=['action', *POSE_KEYS],
    )
    ds.transform = spt.data.transforms.Compose(
        get_img_preprocessor('pixels', 'pixels', img_size=args.img_size, uint8=True),
        column_normalizer(ds, 'action', 'action'),
    )
    return ds


def choose_clips(ds, args, num_steps, frameskip):
    """Clip indices, oversampling large rotations: equal counts per ROT_BINS bin."""
    obj = ds.get_col_data('obs_extra_obj_pose')
    rng = np.random.default_rng(args.seed)
    cand = rng.choice(len(ds), min(len(ds), args.candidates), replace=False)
    clips = np.array([ds.clip_indices[i] for i in cand])
    rows0 = np.asarray(ds.offsets)[clips[:, 0]] + clips[:, 1]
    yaw0 = quat_yaw(obj[rows0])
    yaw_last = quat_yaw(obj[rows0 + (num_steps - 1) * frameskip])
    rot = np.abs(wrap(yaw_last - yaw0))
    per_bin = args.num_clips // (len(ROT_BINS) - 1)
    chosen = []
    for lo, hi in zip(ROT_BINS[:-1], ROT_BINS[1:]):
        idx = np.nonzero((rot >= lo) & (rot < hi))[0]
        take = rng.choice(idx, min(len(idx), per_bin), replace=False)
        print(f'  rotation {lo:.1f}-{hi:.1f} rad: {len(idx)} candidates, using {len(take)}')
        chosen.append(take)
    chosen = np.concatenate(chosen)
    eps = clips[chosen, 0]
    group = eps.copy()
    try:  # expert + its replays share a group; split by group, not episode
        g = ds.get_episode_data(np.unique(eps))
        if 'group_idx' in g:
            lut = dict(zip(np.unique(eps).tolist(), g['group_idx']))
            group = np.array([lut[e] for e in eps])
    except Exception as exc:  # noqa: BLE001
        print(f'  no episode groups ({exc}); splitting by episode')
    return cand[chosen], group


def pose_features(ds, idx, num_steps, frameskip):
    """Per clip and frame: T x, y (mm), yaw (rad), EEF x, y (mm)."""
    obj, tcp = ds.get_col_data('obs_extra_obj_pose'), ds.get_col_data('obs_extra_tcp_pose')
    clips = np.array([ds.clip_indices[i] for i in idx])
    rows = (np.asarray(ds.offsets)[clips[:, 0]] + clips[:, 1])[:, None] + frameskip * np.arange(num_steps)[None]
    o, t = obj[rows], tcp[rows]  # (N, F, 7)
    return {'txy': 1e3 * o[..., :2], 'yaw': quat_yaw(o), 'exy': 1e3 * t[..., :2]}


# -- embeddings ----------------------------------------------------------------

@torch.no_grad()
def embed(models, ds, idx, H, T, args):
    """Real embeddings (N, H+T, D) and rollout predictions (N, T, D) per model."""
    real = {k: [] for k in models}
    pred = {k: [] for k in models}
    t0 = time.time()
    for s in range(0, len(idx), args.batch_size):
        items = ds.__getitems__(idx[s : s + args.batch_size].tolist())
        batch = {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in ('pixels', 'action')}
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)
        batch = {k: v.to(args.device, non_blocking=True) for k, v in batch.items()}
        normalize_pixels(batch)
        for name, model in models.items():
            emb = model.encode({'pixels': batch['pixels']})['emb']  # (B, H+T, D)
            info = {
                'pixels': batch['pixels'][:, None, :H],
                'emb': emb[:, None, :H],
                'action_history': batch['action'][:, None, : H - 1],
            }
            plans = batch['action'][:, None, H - 1 : H - 1 + T]
            out = model.rollout(info, plans, H)['predicted_emb'][:, 0, H:]  # (B, T, D)
            real[name].append(emb.float().cpu())
            pred[name].append(out.float().cpu())
        if s == 0 or (s // args.batch_size) % 10 == 0:
            print(f'  embedded {min(s + args.batch_size, len(idx))}/{len(idx)} clips ({time.time() - t0:.0f} s)')
    return {k: torch.cat(v).numpy() for k, v in real.items()}, {k: torch.cat(v).numpy() for k, v in pred.items()}


# -- probes --------------------------------------------------------------------

def targets(pose):
    """(…, 6): T x, y (/100 mm), sin, cos yaw, EEF x, y (/100 mm)."""
    return np.concatenate(
        [pose['txy'] / 100, np.sin(pose['yaw'])[..., None], np.cos(pose['yaw'])[..., None], pose['exy'] / 100], -1
    )


class Ridge:
    def fit(self, x, y, lams=(1e-2, 1e-1, 1, 10, 100)):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        z = (x - self.mu) / self.sd
        self.ym = y.mean(0)
        n = len(z)
        perm = np.random.default_rng(0).permutation(n)
        va, tr = perm[: n // 5], perm[n // 5 :]
        best = None
        for lam in lams:  # pick lambda on a validation split of the training set
            w = self._solve(z[tr], y[tr] - self.ym, lam)
            err = ((z[va] @ w + self.ym - y[va]) ** 2).mean()
            if best is None or err < best[0]:
                best = (err, lam)
        self.lam = best[1]
        self.w = self._solve(z, y - self.ym, self.lam)
        return self

    @staticmethod
    def _solve(z, y, lam):
        return np.linalg.solve(z.T @ z + lam * len(z) * np.eye(z.shape[1]), z.T @ y)

    def predict(self, x):
        return ((x - self.mu) / self.sd) @ self.w + self.ym


class MLP:
    def __init__(self, device, hidden=256, epochs=300, lr=1e-3, wd=1e-4):
        self.device, self.hidden, self.epochs, self.lr, self.wd = device, hidden, epochs, lr, wd

    def fit(self, x, y):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        torch.manual_seed(0)
        X = torch.as_tensor((x - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], self.hidden), torch.nn.GELU(),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.GELU(),
            torch.nn.Linear(self.hidden, Y.shape[1]),
        ).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        for _ in range(self.epochs):
            for b in torch.randperm(len(X), device=self.device).split(1024):
                opt.zero_grad()
                torch.nn.functional.mse_loss(self.net(X[b]), Y[b]).backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def predict(self, x):
        X = torch.as_tensor((x - self.mu) / self.sd, dtype=torch.float32, device=self.device)
        return self.net(X).cpu().numpy()


def probe_errors(p, y):
    """Median angular error (deg), T xy and EEF xy error (mm)."""
    yaw_p, yaw_t = np.arctan2(p[:, 2], p[:, 3]), np.arctan2(y[:, 2], y[:, 3])
    return {
        'yaw_err_deg': float(np.median(np.degrees(np.abs(wrap(yaw_p - yaw_t))))),
        'txy_err_mm': float(np.median(100 * np.linalg.norm(p[:, :2] - y[:, :2], axis=1))),
        'eef_err_mm': float(np.median(100 * np.linalg.norm(p[:, 4:] - y[:, 4:], axis=1))),
    }


# -- analysis ------------------------------------------------------------------

def cost_sensitivity(real, pose, test):
    """Embedding distance vs pose change for frame pairs (context, frame k) in test clips."""
    z = real[test]
    d, dxy, dyaw, deef = [], [], [], []
    for k in range(1, z.shape[1]):
        d.append(((z[:, k] - z[:, 0]) ** 2).sum(-1))
        dxy.append(np.linalg.norm(pose['txy'][test, k] - pose['txy'][test, 0], axis=1))
        dyaw.append(np.abs(wrap(pose['yaw'][test, k] - pose['yaw'][test, 0])))
        deef.append(np.linalg.norm(pose['exy'][test, k] - pose['exy'][test, 0], axis=1))
    d, dxy, dyaw, deef = map(np.concatenate, (d, dxy, dyaw, deef))
    scale = float(np.median(d))  # distances in units of the median context->frame distance
    dn = d / scale
    out = {'median_distance': scale}
    rot_bins = [0, 5, 15, 30, 45, 60, 90, 180]
    small_move = dxy < SMALL_MOVE_MM
    out['by_rotation'] = [
        {'deg': [a, b], 'n': int(m.sum()), 'mean_d': float(dn[m].mean()) if m.any() else None}
        for a, b in zip(rot_bins[:-1], rot_bins[1:])
        for m in [small_move & (np.degrees(dyaw) >= a) & (np.degrees(dyaw) < b)]
    ]
    xy_bins = [0, 5, 15, 30, 60, 100, 200]
    small_rot = np.degrees(dyaw) < SMALL_ROT_DEG
    out['by_displacement'] = [
        {'mm': [a, b], 'n': int(m.sum()), 'mean_d': float(dn[m].mean()) if m.any() else None}
        for a, b in zip(xy_bins[:-1], xy_bins[1:])
        for m in [small_rot & (dxy >= a) & (dxy < b)]
    ]
    # Non-negative fit d = a dxy^2 + b chord^2 + c deef^2 + e  (chord = 2 sin(dyaw / 2), ~dyaw for small angles)
    from scipy.optimize import nnls

    chord = 2 * np.sin(dyaw / 2)
    A = np.stack([dxy**2, chord**2, deef**2, np.ones_like(d)], 1)
    coef, _ = nnls(A, dn)
    resid = dn - A @ coef
    r2 = 1 - resid.var() / dn.var()
    a, b, c, e = coef
    c45 = (2 * np.sin(np.radians(45) / 2)) ** 2
    out['fit'] = {
        'a_txy': float(a), 'b_rot': float(b), 'c_eef': float(c), 'intercept': float(e), 'r2': float(r2),
        'mm_equal_to_45deg': float(np.sqrt(b * c45 / a)) if a > 0 else None,
        'share_txy': float(np.var(a * dxy**2) / dn.var()),
        'share_rot': float(np.var(b * chord**2) / dn.var()),
        'share_eef': float(np.var(c * deef**2) / dn.var()),
    }
    return out


def prediction_tracking(pred, real, pose, test, lin, mlp):
    """Probe the predicted embeddings: yaw error per step and fraction of rotation shown."""
    out = {'per_step': [], 'by_rotation': []}
    yaw_true = pose['yaw'][test]
    probes = {'linear': lin, 'mlp': mlp}
    base = {k: p.predict(real[test, 0]) for k, p in probes.items()}  # probed context frame
    T = pred.shape[1]
    for k in range(T):
        row = {'step': k + 1}
        for kind, p in probes.items():
            pp, pr = p.predict(pred[test, k]), p.predict(real[test, k + 1])
            yaw_pp, yaw_pr = np.arctan2(pp[:, 2], pp[:, 3]), np.arctan2(pr[:, 2], pr[:, 3])
            yt = yaw_true[:, k + 1]
            row[f'{kind}_pred_yaw_err'] = float(np.median(np.degrees(np.abs(wrap(yaw_pp - yt)))))
            row[f'{kind}_real_yaw_err'] = float(np.median(np.degrees(np.abs(wrap(yaw_pr - yt)))))
            row[f'{kind}_pred_txy_err'] = float(np.median(100 * np.linalg.norm(pp[:, :2] - targets_t(pose, test, k + 1)[:, :2], axis=1)))
            row[f'{kind}_real_txy_err'] = float(np.median(100 * np.linalg.norm(pr[:, :2] - targets_t(pose, test, k + 1)[:, :2], axis=1)))
        row['no_change_yaw_err'] = float(np.median(np.degrees(np.abs(wrap(yaw_true[:, 0] - yaw_true[:, k + 1])))))
        out['per_step'].append(row)
    # Fraction of the true rotation (context -> last frame) visible in the probes, by rotation size.
    true_rot = wrap(yaw_true[:, -1] - yaw_true[:, 0])
    for lo, hi in zip(ROT_BINS[:-1], ROT_BINS[1:]):
        m = (np.abs(true_rot) >= lo) & (np.abs(true_rot) < hi)
        row = {'rad': [lo, None if np.isinf(hi) else hi], 'n': int(m.sum())}
        for kind, p in probes.items():
            b0 = np.arctan2(base[kind][m, 2], base[kind][m, 3])
            pp, pr = p.predict(pred[test][m, -1]), p.predict(real[test][m, -1])
            rot_pred = wrap(np.arctan2(pp[:, 2], pp[:, 3]) - b0)
            rot_real = wrap(np.arctan2(pr[:, 2], pr[:, 3]) - b0)
            # least-squares slope of probed vs true rotation: 1 = the full rotation is shown, 0 = none
            tr = true_rot[m]
            den = max(float((tr**2).sum()), 1e-9)
            row[f'{kind}_pred_fraction'] = float((rot_pred * tr).sum() / den)
            row[f'{kind}_real_fraction'] = float((rot_real * tr).sum() / den)
            row[f'{kind}_pred_yaw_err'] = float(np.median(np.degrees(np.abs(wrap(np.arctan2(pp[:, 2], pp[:, 3]) - yaw_true[m, -1])))))
        out['by_rotation'].append(row)
    return out


def targets_t(pose, test, k):
    return targets({key: v[test, k] for key, v in pose.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoints', nargs='+', help='label=path to weights_*.pt')
    p.add_argument('--dataset', default='overhead_rgb_noisy_replay_100k.lance')
    p.add_argument('--cache-dir', default=None, help='dataset root (holds datasets/); default STABLEWM_HOME')
    p.add_argument('--horizon', type=int, default=5, help='predicted action blocks per clip')
    p.add_argument('--frameskip', type=int, default=5, help='used when no saved config.yaml says otherwise')
    p.add_argument('--num-clips', type=int, default=8000)
    p.add_argument('--candidates', type=int, default=200000, help='random clips to pick the rotation-balanced set from')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--mlp-epochs', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='reports/rotation_probe')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    models, paths, H, fs = {}, {}, None, None
    for item in args.checkpoints:
        name, path = item.split('=', 1)
        pt = resolve(path)
        model = load_model(pt).to(args.device).eval()
        h = int(getattr(model.predictor, 'num_frames', 1))
        f = saved_frameskip(pt) or args.frameskip
        if H is not None and (h, f) != (H, fs):
            raise SystemExit(f'{name}: history {h} / frameskip {f} differ from the first checkpoint ({H} / {fs})')
        H, fs = h, f
        models[name], paths[name] = model, str(pt)
        print(f'{name}: {pt} (history {h}, frameskip {f})')
    T = args.horizon
    num_steps = H + T

    ds = open_dataset(args, num_steps, fs)
    print(f'dataset: {len(ds):,} clips of {num_steps} frames (frameskip {fs})')
    idx, group = choose_clips(ds, args, num_steps, fs)
    pose = pose_features(ds, idx, num_steps, fs)
    rng = np.random.default_rng(args.seed)
    groups = np.unique(group)
    test_groups = set(rng.choice(groups, len(groups) // 2, replace=False).tolist())
    test = np.array([g in test_groups for g in group])
    train = ~test
    print(f'{len(idx)} clips; train {train.sum()} / test {test.sum()} (split by episode group)')

    real, pred = embed(models, ds, idx, H, T, args)
    y_all = targets(pose)  # (N, F, 6)
    metrics = {
        'config': {
            'checkpoints': paths, 'dataset': args.dataset, 'history': H, 'frameskip': fs, 'horizon': T,
            'clips': int(len(idx)), 'train': int(train.sum()), 'test': int(test.sum()), 'rot_bins': [b if np.isfinite(b) else None for b in ROT_BINS],
        },
        'models': {},
    }
    for name in models:
        z = real[name]
        x_tr, y_tr = z[train].reshape(-1, z.shape[-1]), y_all[train].reshape(-1, 6)
        x_te, y_te = z[test].reshape(-1, z.shape[-1]), y_all[test].reshape(-1, 6)
        lin = Ridge().fit(x_tr, y_tr)
        mlp = MLP(args.device, epochs=args.mlp_epochs).fit(x_tr, y_tr)
        m = {
            'probe': {'linear': {**probe_errors(lin.predict(x_te), y_te), 'lambda': lin.lam}, 'mlp': probe_errors(mlp.predict(x_te), y_te)},
            'cost': cost_sensitivity(z, pose, test),
            'prediction': prediction_tracking(pred[name], z, pose, test, lin, mlp),
        }
        metrics['models'][name] = m
        pr, c = m['probe'], m['cost']['fit']
        print(
            f"{name}: yaw err linear {pr['linear']['yaw_err_deg']:.1f} / mlp {pr['mlp']['yaw_err_deg']:.1f} deg; "
            f"T xy {pr['linear']['txy_err_mm']:.1f} / {pr['mlp']['txy_err_mm']:.1f} mm; "
            f"45 deg costs like {c['mm_equal_to_45deg'] or float('nan'):.0f} mm of T motion (fit R2 {c['r2']:.2f})"
        )
    json.dump(metrics, open(out / 'metrics.json', 'w'), indent=1)
    write_summary(metrics, out / 'summary.md')
    print(f'wrote {out}')


def write_summary(metrics, path):
    cfg = metrics['config']
    L = [
        '# Rotation probe',
        '',
        f"- dataset `{cfg['dataset']}`, {cfg['clips']} clips (history {cfg['history']} + {cfg['horizon']} blocks, frameskip {cfg['frameskip']}), "
        f"rotation-balanced; probes fit on {cfg['train']} clips, tested on {cfg['test']} (split by episode group)",
        '',
        '## 1. Representation: probe from a real frame embedding (test median error)',
        '',
        '| model | yaw linear (deg) | yaw MLP (deg) | T xy linear (mm) | T xy MLP (mm) | EEF xy linear (mm) | EEF xy MLP (mm) |',
        '|---|---|---|---|---|---|---|',
    ]
    for n, m in metrics['models'].items():
        a, b = m['probe']['linear'], m['probe']['mlp']
        L.append(f"| {n} | {a['yaw_err_deg']:.1f} | {b['yaw_err_deg']:.1f} | {a['txy_err_mm']:.1f} | {b['txy_err_mm']:.1f} | {a['eef_err_mm']:.1f} | {b['eef_err_mm']:.1f} |")
    L += ['', '## 2. Planning cost: embedding distance (units of the median) vs pose change', '',
          '| model | 45° costs like (mm of T motion) | share of distance variance: T xy / rotation / EEF | fit R² |', '|---|---|---|---|']
    for n, m in metrics['models'].items():
        f = m['cost']['fit']
        mm = f'{f["mm_equal_to_45deg"]:.0f}' if f['mm_equal_to_45deg'] else '–'
        L.append(f"| {n} | {mm} | {f['share_txy']:.2f} / {f['share_rot']:.2f} / {f['share_eef']:.2f} | {f['r2']:.2f} |")
    L += ['', f'Mean distance for pairs with T motion < {SMALL_MOVE_MM} mm, by rotation (deg):', '']
    names = list(metrics['models'])
    rows = metrics['models'][names[0]]['cost']['by_rotation']
    L += ['| rotation | n | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for i, r in enumerate(rows):
        vals = [metrics['models'][n]['cost']['by_rotation'][i]['mean_d'] for n in names]
        L.append(f"| {r['deg'][0]}–{r['deg'][1]} | {r['n']} | " + ' | '.join('–' if v is None else f'{v:.2f}' for v in vals) + ' |')
    L += ['', f'Mean distance for pairs with rotation < {SMALL_ROT_DEG}°, by T displacement (mm):', '']
    rows = metrics['models'][names[0]]['cost']['by_displacement']
    L += ['| displacement | n | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for i, r in enumerate(rows):
        vals = [metrics['models'][n]['cost']['by_displacement'][i]['mean_d'] for n in names]
        L.append(f"| {r['mm'][0]}–{r['mm'][1]} | {r['n']} | " + ' | '.join('–' if v is None else f'{v:.2f}' for v in vals) + ' |')
    L += ['', '## 3. Prediction: MLP probe on predicted vs real embeddings, yaw error (deg) per step', '',
          '| model | ' + ' | '.join(f'step {s["step"]} pred / real' for s in metrics['models'][names[0]]['prediction']['per_step']) + ' |',
          '|---|' + '---|' * len(metrics['models'][names[0]]['prediction']['per_step'])]
    for n in names:
        L.append(f'| {n} | ' + ' | '.join(f"{s['mlp_pred_yaw_err']:.1f} / {s['mlp_real_yaw_err']:.1f}" for s in metrics['models'][n]['prediction']['per_step']) + ' |')
    nc = ' | '.join(f"{s['no_change_yaw_err']:.1f}" for s in metrics['models'][names[0]]['prediction']['per_step'])
    L += [f'| (no change: context yaw) | {nc} |', '',
          'Rotation shown by the MLP probe (slope of probed vs true rotation, context -> last frame; 1 = all), predicted / real:', '',
          '| rotation (rad) | n | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for i, r in enumerate(metrics['models'][names[0]]['prediction']['by_rotation']):
        hi = '∞' if r['rad'][1] is None else f"{r['rad'][1]:g}"
        vals = [metrics['models'][n]['prediction']['by_rotation'][i] for n in names]
        L.append(f"| {r['rad'][0]:g}–{hi} | {r['n']} | " + ' | '.join(f"{v['mlp_pred_fraction']:.2f} / {v['mlp_real_fraction']:.2f}" for v in vals) + ' |')
    Path(path).write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
