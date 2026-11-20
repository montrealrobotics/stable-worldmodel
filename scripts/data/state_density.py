"""State-density / coverage analysis of ManiSkill PushT Lance datasets.

In the spirit of the mmbench2 coverage figures (state density of the key
agent/object positions per collection policy), but quantitative: every dataset
(or split of one, e.g. expert vs noisy replays) is binned on the SAME edges, so
densities and coverage numbers are directly comparable.

Only low-dimensional columns are read (no pixel decoding). Per frame:

  eef        TCP xy (mm), TCP z                     obs_extra_tcp_pose
  tee        T xy (mm), T yaw (deg)                 obs_extra_obj_pose (xyz + wxyz)
  goal       goal xy (mm), goal yaw (deg)           obs_extra_goal_pos / _goal_orientation
  task       T pose in the goal frame: offset xy, distance, yaw error
  contact    TCP position in the T body frame (where the pusher is w.r.t. the T)
  start      T / TCP pose at the first frame of each episode (reset distribution)
  dynamics   T speed, T yaw rate, TCP speed per step (finite differences)
  action     action dims (2D density of the first two)
  qpos       per-joint marginals                    obs_agent_qpos

Coverage metrics per 2D space (support = bins any group visits):
  coverage     fraction of support bins the group visits (>=1 frame)
  coverage_k   fraction with >= --min-count frames (robust to single strays)
  H_norm       entropy / log(|support|), 1 = uniform over the support
  eff_bins     exp(entropy), the "number of equally visited bins"
  coverage @ N coverage with every group subsampled to the same N frames
               (raw coverage grows with dataset size; this one does not)

Usage (one group per dataset; --split-by turns one dataset into several):
  python scripts/data/state_density.py \
      expert=/path/expert_100k.lance noisy=/path/overhead_rgb_noisy_replay_100k \
      --split-by replay_idx --out reports/state_density
"""

import argparse
import json
import math
from pathlib import Path

import lance
import matplotlib
import numpy as np
import pyarrow as pa

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

# Categorical slots (fixed order, never cycled) and one sequential hue.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
SEQ_CMAP = 'Blues'
INK, MUTED = '#0b0b0b', '#8a8984'

LIGHT = [
    'obs_extra_tcp_pose',
    'obs_extra_obj_pose',
    'obs_extra_goal_pos',
    'obs_extra_goal_orientation',
    'obs_agent_qpos',
    'action',
]

# 2D spaces: name -> (x feature, y feature, title).
SPACES = {
    'eef_xy': ('eef_x', 'eef_y', 'End effector xy (mm)'),
    'tee_xy': ('tee_x', 'tee_y', 'T position xy (mm)'),
    'tee_x_yaw': ('tee_x', 'tee_yaw', 'T x (mm) vs yaw (deg)'),
    'tee_y_yaw': ('tee_y', 'tee_yaw', 'T y (mm) vs yaw (deg)'),
    'goal_xy': ('goal_x', 'goal_y', 'Goal xy (mm)'),
    'goal_yaw_vs_tee_yaw': ('goal_yaw', 'tee_yaw', 'Goal yaw vs T yaw (deg)'),
    'task_offset': ('task_dx', 'task_dy', 'T offset in goal frame (mm)'),
    'task_dist_yawerr': ('task_dist', 'task_yaw_err', 'T-goal distance (mm) vs yaw error (deg)'),
    'contact_xy': ('contact_x', 'contact_y', 'TCP in T frame (mm)'),
    'eef_minus_tee': ('rel_x', 'rel_y', 'TCP - T, world frame (mm)'),
    'start_tee_xy': ('start_tee_x', 'start_tee_y', 'Start T xy (mm)'),
    'start_eef_xy': ('start_eef_x', 'start_eef_y', 'Start TCP xy (mm)'),
    'action_01': ('action_0', 'action_1', 'Action dims 0 vs 1'),
}

FIGURES = {
    'positions': ['eef_xy', 'tee_xy', 'goal_xy', 'start_tee_xy', 'start_eef_xy'],
    'orientation': ['tee_x_yaw', 'tee_y_yaw', 'goal_yaw_vs_tee_yaw'],
    'relative': ['contact_xy', 'eef_minus_tee', 'task_offset', 'task_dist_yawerr'],
    'actions': ['action_01'],
}

ANGULAR = {'tee_yaw', 'goal_yaw', 'task_yaw_err', 'start_tee_yaw', 'tee_yaw_rate'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('datasets', nargs='+', help='label=path (.lance dir or a dir with one frames table)')
    p.add_argument('--split-by', default=None, help='episode side-table column to split each dataset by (e.g. replay_idx, collection_mode)')
    p.add_argument('--episode-stride', type=int, default=1, help='keep episodes with episode_idx %% stride == 0')
    p.add_argument('--bins', type=int, default=64, help='bins per axis for 2D spaces')
    p.add_argument('--yaw-bins', type=int, default=72, help='bins over 360 deg for angles')
    p.add_argument('--min-count', type=int, default=10, help='visits for coverage_k')
    p.add_argument('--clip-pct', type=float, default=0.05, help='range = [pct, 100-pct] percentiles of the union (non-angular)')
    p.add_argument('--moving-mm', type=float, default=0.5, help='T speed (mm/step) above which the T counts as moving')
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--out', default='reports/state_density')
    return p.parse_args()


def resolve(path):
    p = Path(path)
    if p.suffix == '.lance':
        return p, p.with_name(p.stem + '_episodes.lance')
    tables = sorted(t for t in p.glob('*.lance') if not t.stem.endswith(('_episodes', '_videos')))
    if len(tables) != 1:
        raise SystemExit(f'{path}: expected one frames *.lance table, found {[t.name for t in tables]}')
    return tables[0], tables[0].with_name(tables[0].stem + '_episodes.lance')


def as_matrix(col):
    """List / fixed-size-list / scalar arrow column -> [n, d] float32."""
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    n = len(col)
    if pa.types.is_list(col.type) or pa.types.is_large_list(col.type) or pa.types.is_fixed_size_list(col.type):
        flat = col.flatten()
        while pa.types.is_list(flat.type) or pa.types.is_fixed_size_list(flat.type):
            flat = flat.flatten()
        return np.asarray(flat.to_numpy(zero_copy_only=False), np.float32).reshape(n, -1)
    return np.asarray(col.to_numpy(zero_copy_only=False), np.float32).reshape(n, 1)


def load(path, stride, batch_size):
    frames, episodes = resolve(path)
    ds = lance.dataset(str(frames))
    names = set(ds.schema.names)
    cols = [c for c in LIGHT if c in names]
    missing = [c for c in LIGHT if c not in names]
    print(f'{frames}: {ds.count_rows():,} rows; reading {cols}; missing {missing}')
    parts = {k: [] for k in ['episode_idx', 'step_idx', *cols]}
    for batch in ds.to_batches(columns=['episode_idx', 'step_idx', *cols], batch_size=batch_size):
        ep = batch.column('episode_idx').to_numpy()
        keep = ep % stride == 0 if stride > 1 else slice(None)
        parts['episode_idx'].append(ep[keep])
        parts['step_idx'].append(batch.column('step_idx').to_numpy()[keep])
        for c in cols:
            parts[c].append(as_matrix(batch.column(c))[keep])
    data = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.lexsort((data['step_idx'], data['episode_idx']))
    data = {k: v[order] for k, v in data.items()}
    ep_table = None
    if episodes.exists():
        ep_table = lance.dataset(str(episodes)).to_table().to_pydict()
    return data, ep_table


def quat_yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def features(d):
    """Per-frame physical features (mm / deg) from the raw columns."""
    f = {}
    ep = d['episode_idx']
    first = np.r_[True, ep[1:] != ep[:-1]]
    if 'obs_extra_tcp_pose' in d:
        tcp = d['obs_extra_tcp_pose']
        f['eef_x'], f['eef_y'] = 1e3 * tcp[:, 0], 1e3 * tcp[:, 1]
        if tcp.shape[1] > 2:
            f['eef_z'] = 1e3 * tcp[:, 2]
    if 'obs_extra_obj_pose' in d:
        obj = d['obs_extra_obj_pose']
        yaw = quat_yaw(obj[:, 3:7])
        f['tee_x'], f['tee_y'], f['tee_yaw'] = 1e3 * obj[:, 0], 1e3 * obj[:, 1], np.degrees(yaw)
        if 'eef_x' in f:
            rx, ry = f['eef_x'] - f['tee_x'], f['eef_y'] - f['tee_y']
            c, s = np.cos(yaw), np.sin(yaw)
            f['rel_x'], f['rel_y'] = rx, ry
            f['contact_x'], f['contact_y'] = c * rx + s * ry, -s * rx + c * ry
            f['eef_tee_dist'] = np.hypot(rx, ry)
    if 'obs_extra_goal_pos' in d:
        g = d['obs_extra_goal_pos']
        f['goal_x'], f['goal_y'] = 1e3 * g[:, 0], 1e3 * g[:, 1]
    if 'obs_extra_goal_orientation' in d:
        go = d['obs_extra_goal_orientation']
        gyaw = np.arctan2(go[:, 1], go[:, 0])
        f['goal_yaw'] = np.degrees(gyaw)
        if 'tee_x' in f and 'goal_x' in f:
            dx, dy = f['tee_x'] - f['goal_x'], f['tee_y'] - f['goal_y']
            c, s = np.cos(gyaw), np.sin(gyaw)
            f['task_dx'], f['task_dy'] = c * dx + s * dy, -s * dx + c * dy
            f['task_dist'] = np.hypot(dx, dy)
            f['task_yaw_err'] = np.degrees(wrap(np.radians(f['tee_yaw']) - gyaw))
    if 'action' in d:
        for i in range(d['action'].shape[1]):
            f[f'action_{i}'] = d['action'][:, i]
    if 'obs_agent_qpos' in d:
        for i in range(d['obs_agent_qpos'].shape[1]):
            f[f'qpos_{i}'] = d['obs_agent_qpos'][:, i]

    # Per-step finite differences inside an episode (NaN on the first frame).
    def step_diff(x, angular=False):
        dx = np.r_[np.nan, np.diff(x)]
        if angular:
            dx = np.degrees(wrap(np.radians(dx)))
        dx[first] = np.nan
        return dx

    if 'tee_x' in f:
        f['tee_speed'] = np.hypot(step_diff(f['tee_x']), step_diff(f['tee_y']))
        f['tee_yaw_rate'] = step_diff(f['tee_yaw'], angular=True)
    if 'eef_x' in f:
        f['eef_speed'] = np.hypot(step_diff(f['eef_x']), step_diff(f['eef_y']))

    # Episode-start features, repeated per frame only at the first frame (NaN elsewhere).
    for k in ('tee_x', 'tee_y', 'tee_yaw', 'eef_x', 'eef_y'):
        if k in f:
            v = np.full_like(f[k], np.nan)
            v[first] = f[k][first]
            f[f'start_{k}'] = v
    return f


def split_groups(label, d, ep_table, split_by):
    if not split_by:
        return [(label, np.ones(len(d['episode_idx']), bool))]
    if ep_table is None or split_by not in ep_table:
        raise SystemExit(f'{label}: no episode side-table column {split_by!r}')
    values = dict(zip(ep_table['episode_idx'], ep_table[split_by]))
    raw = np.array([values.get(e) for e in d['episode_idx']], dtype=object)
    if split_by == 'replay_idx':  # -1 is the expert, >= 0 its noisy replays / alternatives
        raw = np.where(raw == -1, 'expert', np.where(raw == None, 'n/a', 'replay'))  # noqa: E711
    raw = raw.astype(str)
    return [(f'{label}/{v}', raw == v) for v in sorted(set(raw))]


def edges_for(name, values, bins, yaw_bins, pct):
    if name in ANGULAR and name != 'tee_yaw_rate':
        return np.linspace(-180, 180, yaw_bins + 1)
    v = np.concatenate([x[np.isfinite(x)] for x in values])
    lo, hi = np.percentile(v, [pct, 100 - pct])
    if hi <= lo:
        lo, hi = lo - 1e-3, hi + 1e-3
    pad = 0.02 * (hi - lo)
    return np.linspace(lo - pad, hi + pad, bins + 1)


def hist2d(x, y, ex, ey):
    ok = np.isfinite(x) & np.isfinite(y)
    return np.histogram2d(x[ok], y[ok], bins=(ex, ey))[0]


def coverage(h, support, k):
    n = h.sum()
    p = h[support] / max(n, 1)
    nz = p[p > 0]
    ent = float(-(nz * np.log(nz)).sum())
    m = int(support.sum())
    return {
        'frames': int(n),
        'coverage': float((h[support] > 0).mean()) if m else 0.0,
        f'coverage_{k}': float((h[support] >= k).mean()) if m else 0.0,
        'H_norm': ent / math.log(m) if m > 1 else 0.0,
        'eff_bins': math.exp(ent),
        'support_bins': m,
    }


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)


def plot_grid(fname, space_names, groups, hists, edges, out):
    names = [s for s in space_names if s in hists]
    if not names:
        return
    nr, nc = len(groups), len(names)
    fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.8 * nr + 0.8), squeeze=False, constrained_layout=True)
    for j, s in enumerate(names):
        dens = [hists[s][g] / max(hists[s][g].sum(), 1) for g in groups]
        pos = np.concatenate([d[d > 0] for d in dens]) if any((d > 0).any() for d in dens) else np.array([1.0])
        norm = LogNorm(vmin=pos.min(), vmax=pos.max())
        ex, ey = edges[s]
        for i, g in enumerate(groups):
            ax = axes[i, j]
            im = ax.pcolormesh(ex, ey, np.ma.masked_equal(dens[i], 0).T, cmap=SEQ_CMAP, norm=norm, rasterized=True)
            ax.set_facecolor('#fcfcfb')
            if SPACES[s][2].endswith('(mm)'):
                ax.set_aspect('equal', adjustable='box')
            style(ax)
            if i == 0:
                ax.set_title(SPACES[s][2], fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(g, fontsize=9, color=INK)
        fig.colorbar(im, ax=axes[:, j], location='bottom', shrink=0.85, aspect=25, label='fraction of frames (log)')
    fig.savefig(out / f'{fname}.png', dpi=130)
    plt.close(fig)


def plot_marginals(feats, groups, edges1d, out):
    names = [n for n in edges1d]
    nc = 4
    nr = math.ceil(len(names) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(3.4 * nc, 2.3 * nr), squeeze=False, constrained_layout=True)
    for ax, n in zip(axes.flat, names):
        e = edges1d[n]
        for gi, g in enumerate(groups):
            x = feats[g].get(n)
            if x is None:
                continue
            x = x[np.isfinite(x)]
            h, _ = np.histogram(x, bins=e)
            ax.stairs(h / max(h.sum(), 1), e, color=SERIES[gi % len(SERIES)] if gi < len(SERIES) else MUTED, lw=1.5, label=g)
        ax.set_title(n, fontsize=8, color=INK)
        style(ax)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=min(len(groups), 6), fontsize=8, frameon=False)
    fig.savefig(out / 'marginals.png', dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.datasets) and '=' not in args.datasets[0]:
        args.datasets = [f'{Path(p).stem}={p}' for p in args.datasets]

    feats, stats = {}, {}
    for spec in args.datasets:
        label, path = spec.split('=', 1)
        data, ep_table = load(path, args.episode_stride, args.batch_size)
        f = features(data)
        for g, mask in split_groups(label, data, ep_table, args.split_by):
            feats[g] = {k: v[mask] for k, v in f.items()}
            ep = data['episode_idx'][mask]
            ts = feats[g].get('tee_speed', np.array([np.nan]))
            stats[g] = {
                'frames': int(mask.sum()),
                'episodes': int(len(np.unique(ep))),
                'mean_ep_len': float(mask.sum() / max(len(np.unique(ep)), 1)),
                'tee_moving_frac': float(np.nanmean(ts > args.moving_mm)) if np.isfinite(ts).any() else None,
                'tee_speed_mean_mm': float(np.nanmean(ts)) if np.isfinite(ts).any() else None,
                'eef_speed_mean_mm': float(np.nanmean(feats[g]['eef_speed'])) if 'eef_speed' in feats[g] else None,
                'eef_z_std_mm': float(np.nanstd(feats[g]['eef_z'])) if 'eef_z' in feats[g] else None,
            }
        del data
    groups = list(feats)
    print('groups:', groups)

    # Shared edges per feature over the union of groups.
    all_names = sorted(set().union(*[feats[g].keys() for g in groups]))
    edges1d = {}
    for n in all_names:
        vals = [feats[g][n] for g in groups if n in feats[g]]
        if any(np.isfinite(v).any() for v in vals):
            edges1d[n] = edges_for(n, vals, args.bins, args.yaw_bins, args.clip_pct)

    hists, edges2d, metrics = {}, {}, {}
    for s, (fx, fy, _) in SPACES.items():
        if fx not in edges1d or fy not in edges1d or not all(fx in feats[g] and fy in feats[g] for g in groups):
            continue
        edges2d[s] = (edges1d[fx], edges1d[fy])
        hists[s] = {g: hist2d(feats[g][fx], feats[g][fy], *edges2d[s]) for g in groups}
        support = sum(hists[s].values()) > 0
        metrics[s] = {g: coverage(hists[s][g], support, args.min_count) for g in groups}
        # Frame-matched coverage: every group subsampled to the smallest group's frame count.
        ok = {g: np.flatnonzero(np.isfinite(feats[g][fx]) & np.isfinite(feats[g][fy])) for g in groups}
        n_min = min(len(v) for v in ok.values())
        rng = np.random.default_rng(0)
        for g in groups:
            idx = rng.choice(ok[g], n_min, replace=False)
            h = hist2d(feats[g][fx][idx], feats[g][fy][idx], *edges2d[s])
            metrics[s][g]['matched_frames'] = int(n_min)
            metrics[s][g]['coverage_matched'] = float((h[support] > 0).mean()) if support.any() else 0.0

    for fname, names in FIGURES.items():
        plot_grid(fname, names, groups, hists, edges2d, out)
    plot_marginals(feats, groups, edges1d, out)

    (out / 'metrics.json').write_text(json.dumps({'args': vars(args), 'stats': stats, 'coverage': metrics}, indent=2))
    k = args.min_count
    lines = ['# State density / coverage', '', '## Per-group stats', '',
             '| group | frames | episodes | mean len | T moving frac | T speed mm/step | TCP speed mm/step |', '|---|---|---|---|---|---|---|']
    fmt = lambda v, p=3: '-' if v is None else f'{v:.{p}f}'  # noqa: E731
    for g, st in stats.items():
        lines.append(f"| {g} | {st['frames']:,} | {st['episodes']:,} | {st['mean_ep_len']:.1f} | {fmt(st['tee_moving_frac'])} | {fmt(st['tee_speed_mean_mm'], 2)} | {fmt(st['eef_speed_mean_mm'], 2)} |")
    lines += ['', f'## Coverage (shared bins; support = bins any group visits; coverage_{k} = >= {k} frames)', '']
    for s, per in metrics.items():
        m = next(iter(per.values()))['support_bins']
        lines += [f'### {s}: {SPACES[s][2]} (support {m} bins)', '', f'| group | coverage | coverage_{k} | coverage @ {next(iter(per.values()))["matched_frames"]:,} frames | H_norm | eff_bins |', '|---|---|---|---|---|---|']
        for g, c in per.items():
            lines.append(f"| {g} | {c['coverage']:.3f} | {c[f'coverage_{k}']:.3f} | {c['coverage_matched']:.3f} | {c['H_norm']:.3f} | {c['eff_bins']:.0f} |")
        lines.append('')
    (out / 'metrics.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'wrote {out}/{{positions,orientation,relative,actions,marginals}}.png, metrics.md, metrics.json')


if __name__ == '__main__':
    main()
