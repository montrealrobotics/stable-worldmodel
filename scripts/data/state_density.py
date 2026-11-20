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
  transition s_t -> s_t+K (aligned with the action chunk a_t..a_t+K-1, K = --chunk,
             default 5 = the training frameskip): T step in the T frame, and
             push maps over the pusher position in the T frame: mean T step
             (arrows) and how many net pusher-motion directions the data holds
             per position (action diversity per state)

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
    # Transitions (s_t -> s_t+1): the T's step in its own frame at t.
    'tee_step': ('tee_step_x', 'tee_step_y', 'T step s_t->s_t+1, T frame (mm)'),
    'tee_step_mag_yaw': ('tee_step_mag', 'tee_step_yaw', 'T step |dxy| (mm) vs dyaw (deg)'),
}

FIGURES = {
    'positions': ['eef_xy', 'tee_xy', 'goal_xy', 'start_tee_xy', 'start_eef_xy'],
    'orientation': ['tee_x_yaw', 'tee_y_yaw', 'goal_yaw_vs_tee_yaw'],
    'relative': ['contact_xy', 'eef_minus_tee', 'task_offset', 'task_dist_yawerr'],
    'actions': ['action_01'],
    'transitions': ['tee_step', 'tee_step_mag_yaw'],
}

# Push maps: pusher position in the T frame on a zoomed grid around the T.
PUSH_HALF_MM, PUSH_CELLS, PUSH_DIRS = 150.0, 30, 16
PUSH_MOVE_MM = 1.0  # pusher moves at least this much per step to count a direction

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
    p.add_argument('--chunk', type=int, default=5, help='transition length K: tuples (s_t, a_t..a_t+K-1, s_t+K); 5 = training frameskip')
    p.add_argument('--moving-mm', type=float, default=0.5, help='T speed (mm/step) above which the T counts as moving')
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--out', default='reports/state_density')
    return p.parse_args()


def is_table(p):
    return p.is_dir() and (p / '_versions').is_dir()


def resolve(path):
    """Frames table + its <table>_episodes side table.

    Accepts the table itself, or a directory holding it at most two levels
    down (e.g. where a dataset archive was extracted).
    """
    p = Path(path)
    if is_table(p):
        return p, p.with_name(p.stem + '_episodes.lance')
    tables = sorted(
        t
        for t in [*p.glob('*.lance'), *p.glob('*/*.lance')]
        if is_table(t) and not t.stem.endswith(('_episodes', '_videos'))
    )
    if len(tables) != 1:
        raise SystemExit(f'{path}: expected one frames *.lance table, found {[str(t) for t in tables]}')
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


def features(d, chunk=1):
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

    # Chunk transitions s_t+K - s_t, aligned with the action chunk a_t..a_t+K-1
    # (the model's frameskip); NaN where t+K leaves the episode.
    n = len(ep)
    j = np.arange(n) + chunk
    valid = j < n
    valid[valid] = ep[j[valid]] == ep[valid]

    def next_diff(x, angular=False):
        dx = np.full_like(x, np.nan)
        dx[valid] = x[j[valid]] - x[valid]
        if angular:
            dx = np.degrees(wrap(np.radians(dx)))
        return dx

    if 'tee_x' in f:
        yaw = np.radians(f['tee_yaw'])
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = next_diff(f['tee_x']), next_diff(f['tee_y'])
        f['tee_step_x'], f['tee_step_y'] = c * dx + s * dy, -s * dx + c * dy
        f['tee_step_mag'] = np.hypot(dx, dy)
        f['tee_step_yaw'] = next_diff(f['tee_yaw'], angular=True)
        if 'eef_x' in f:
            ex, ey = next_diff(f['eef_x']), next_diff(f['eef_y'])
            f['push_x'], f['push_y'] = c * ex + s * ey, -s * ex + c * ey

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


def push_maps(f, k, chunk=1):
    """Per pusher-in-T-frame cell: frames, mean T step (T frame), and pusher-direction diversity.

    Directions are the pusher's own motion s_t -> s_t+1 in the T frame, binned
    into PUSH_DIRS sectors; a direction counts when >= k frames move that way.
    """
    e = np.linspace(-PUSH_HALF_MM, PUSH_HALF_MM, PUSH_CELLS + 1)
    need = ('contact_x', 'contact_y', 'tee_step_x', 'tee_step_y', 'tee_step_mag', 'push_x', 'push_y')
    if not all(n in f for n in need):
        return None
    x, y = f['contact_x'], f['contact_y']
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f['tee_step_x']) & np.isfinite(f['push_x'])
    x, y = x[ok], y[ok]
    count = np.histogram2d(x, y, bins=(e, e))[0]
    sums = {n: np.histogram2d(x, y, bins=(e, e), weights=f[n][ok])[0] for n in ('tee_step_x', 'tee_step_y', 'tee_step_mag')}
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = {n: v / count for n, v in sums.items()}
    moving = np.hypot(f['push_x'][ok], f['push_y'][ok]) > PUSH_MOVE_MM * chunk
    ang = np.arctan2(f['push_y'][ok], f['push_x'][ok])
    ea = np.linspace(-np.pi, np.pi, PUSH_DIRS + 1)
    cube = np.histogramdd(np.stack([x[moving], y[moving], ang[moving]], 1), bins=(e, e, ea))[0]
    dirs = (cube >= k).sum(-1).astype(float)
    p = cube / np.maximum(cube.sum(-1, keepdims=True), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        dir_ent = -(np.where(p > 0, p * np.log(p), 0)).sum(-1) / math.log(PUSH_DIRS)
    return {'edges': e, 'count': count, 'mean': mean, 'dirs': dirs, 'dir_entropy': dir_ent, 'moving_count': cube.sum(-1)}


def push_metrics(m, k, chunk=1):
    """Summaries over cells where the pusher is near the T (>= k frames) and over contact cells (T moves)."""
    cells = m['count'] >= k
    contact = cells & (np.nan_to_num(m['mean']['tee_step_mag']) > 1.0 * chunk)
    w = m['moving_count']
    out = {}
    for name, mask in (('near', cells), ('contact', contact)):
        ww = w[mask]
        out[f'{name}_cells'] = int(mask.sum())
        out[f'{name}_dirs_mean'] = float((m['dirs'][mask] * ww).sum() / max(ww.sum(), 1))
        out[f'{name}_dir_entropy'] = float((np.nan_to_num(m['dir_entropy'][mask]) * ww).sum() / max(ww.sum(), 1))
        out[f'{name}_frac_cells_ge8dirs'] = float((m['dirs'][mask] >= 8).mean()) if mask.any() else 0.0
    return out


def plot_push(groups, maps, out, chunk=1):
    groups = [g for g in groups if maps.get(g) is not None]
    if not groups:
        return
    nr = len(groups)
    fig, axes = plt.subplots(nr, 2, figsize=(7.2, 3.3 * nr + 0.8), squeeze=False, constrained_layout=True)
    vals = np.concatenate([np.where(maps[g]['count'] >= 20, maps[g]['mean']['tee_step_mag'], np.nan).ravel() for g in groups])
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0
    for i, g in enumerate(groups):
        m = maps[g]
        e = m['edges']
        c = 0.5 * (e[1:] + e[:-1])
        mag = np.where(m['count'] >= 20, m['mean']['tee_step_mag'], np.nan)
        ax = axes[i, 0]
        im0 = ax.pcolormesh(e, e, mag.T, cmap=SEQ_CMAP, vmin=0, vmax=vmax, rasterized=True)
        u = np.where(m['count'] >= 20, m['mean']['tee_step_x'], np.nan)
        v = np.where(m['count'] >= 20, m['mean']['tee_step_y'], np.nan)
        X, Y = np.meshgrid(c, c, indexing='ij')
        ax.quiver(X, Y, u, v, color=INK, angles='xy', scale_units='xy', scale=0.25 * chunk, width=0.004)
        ax2 = axes[i, 1]
        dirs = np.where(m['count'] >= 20, m['dirs'], np.nan)
        im1 = ax2.pcolormesh(e, e, dirs.T, cmap=SEQ_CMAP, vmin=0, vmax=PUSH_DIRS, rasterized=True)
        for a in (ax, ax2):
            a.set_aspect('equal')
            a.set_facecolor('#fcfcfb')
            style(a)
        ax.set_ylabel(g, fontsize=9, color=INK)
        if i == 0:
            ax.set_title(f'Mean T step s_t->s_t+{chunk} by pusher position (T frame)', fontsize=9, color=INK)
            ax2.set_title(f'Pusher chunk directions seen (of {PUSH_DIRS})', fontsize=9, color=INK)
    fig.colorbar(im0, ax=axes[:, 0], location='bottom', shrink=0.85, aspect=25, label=f'mean |T step| over {chunk} steps (mm); arrows = mean step x{4 / chunk:g}')
    fig.colorbar(im1, ax=axes[:, 1], location='bottom', shrink=0.85, aspect=25, label='directions with >= k frames')
    fig.savefig(out / 'push.png', dpi=130)
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
        f = features(data, args.chunk)
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
    SPACES['tee_step'] = ('tee_step_x', 'tee_step_y', f'T step s_t->s_t+{args.chunk}, T frame (mm)')
    SPACES['tee_step_mag_yaw'] = ('tee_step_mag', 'tee_step_yaw', f'T step over {args.chunk}: |dxy| (mm) vs dyaw (deg)')

    # Shared edges per feature over the union of groups.
    all_names = sorted(set().union(*[feats[g].keys() for g in groups]))
    edges1d, constant = {}, {}
    for n in all_names:
        vals = [feats[g][n] for g in groups if n in feats[g]]
        if not any(np.isfinite(v).any() for v in vals):
            continue
        v = np.concatenate([x[np.isfinite(x)] for x in vals])
        if v.max() - v.min() < 1e-2:  # e.g. a fixed goal or the planar TCP height
            constant[n] = float(v.mean())
            continue
        edges1d[n] = edges_for(n, vals, args.bins, args.yaw_bins, args.clip_pct)
    if constant:
        print('constant features (not plotted):', constant)

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
    maps = {g: push_maps(feats[g], args.min_count, args.chunk) for g in groups}
    plot_push(groups, maps, out, args.chunk)
    push = {g: push_metrics(m, args.min_count, args.chunk) for g, m in maps.items() if m is not None}

    (out / 'metrics.json').write_text(json.dumps({'args': vars(args), 'stats': stats, 'constant': constant, 'push': push, 'coverage': metrics}, indent=2))
    k = args.min_count
    lines = ['# State density / coverage', '', '## Per-group stats', '',
             '| group | frames | episodes | mean len | T moving frac | T speed mm/step | TCP speed mm/step |', '|---|---|---|---|---|---|---|']
    fmt = lambda v, p=3: '-' if v is None else f'{v:.{p}f}'  # noqa: E731
    for g, st in stats.items():
        lines.append(f"| {g} | {st['frames']:,} | {st['episodes']:,} | {st['mean_ep_len']:.1f} | {fmt(st['tee_moving_frac'])} | {fmt(st['tee_speed_mean_mm'], 2)} | {fmt(st['eef_speed_mean_mm'], 2)} |")
    if constant:
        lines += ['', '## Constant features (range < 0.01, not plotted)', '']
        lines += [f'- {n} = {v:.3f}' for n, v in constant.items()]
    if push:
        lines += ['', f'## Transitions: pusher-direction diversity per pusher-in-T-frame cell '
                  f'({PUSH_CELLS}x{PUSH_CELLS} cells over +/-{PUSH_HALF_MM:.0f} mm, {PUSH_DIRS} directions, >= {k} frames)', '',
                  f'Transitions are chunks s_t -> s_t+{args.chunk}; directions are the pusher\'s net chunk motion. '
                  f'near = cells with >= k frames; contact = near cells where the mean T step > {args.chunk} mm. '
                  'dirs = directions seen (frame-weighted mean), H = direction entropy / log(dirs total).', '',
                  '| group | near cells | near dirs | near H | contact cells | contact dirs | contact H | contact cells >= 8 dirs |',
                  '|---|---|---|---|---|---|---|---|']
        for g, p in push.items():
            lines.append(f"| {g} | {p['near_cells']} | {p['near_dirs_mean']:.1f} | {p['near_dir_entropy']:.3f} | {p['contact_cells']} | "
                         f"{p['contact_dirs_mean']:.1f} | {p['contact_dir_entropy']:.3f} | {p['contact_frac_cells_ge8dirs']:.3f} |")
    lines += ['', f'## Coverage (shared bins; support = bins any group visits; coverage_{k} = >= {k} frames)', '']
    for s, per in metrics.items():
        m = next(iter(per.values()))['support_bins']
        lines += [f'### {s}: {SPACES[s][2]} (support {m} bins)', '', f'| group | coverage | coverage_{k} | coverage @ {next(iter(per.values()))["matched_frames"]:,} frames | H_norm | eff_bins |', '|---|---|---|---|---|---|']
        for g, c in per.items():
            lines.append(f"| {g} | {c['coverage']:.3f} | {c[f'coverage_{k}']:.3f} | {c['coverage_matched']:.3f} | {c['H_norm']:.3f} | {c['eff_bins']:.0f} |")
        lines.append('')
    (out / 'metrics.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print(f'wrote {out}/{{positions,orientation,relative,actions,transitions,push,marginals}}.png, metrics.md, metrics.json')


if __name__ == '__main__':
    main()
