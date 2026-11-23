"""Figures for dataset_task_coverage.py output (tasks.csv + windows.json).

  python scripts/data/plot_task_coverage.py reports/dataset_task_coverage/<run>

Writes PNGs next to the inputs:
  feature_<name>.png  top: share of training windows vs eval tasks per bin;
                      bottom: planning success per bin (base / CEM / in-batch)
  feature_<name>_by_g.png  the same, one column per goal offset g
  overview.png        the four main features side by side
  map.png             training windows over (T rotation, T displacement), eval
                      tasks on top, marked solved / failed by the in-batch model
  map_by_g.png        the same, one panel per goal offset g
The training-data share is g-matched: rotation and displacement grow with g and
the dataset has fewer windows at large g, so the per-g shares are averaged with
the eval tasks' g weights (pooled shares if the output predates per-g counts).
  similar.png         success vs how many training windows make the same move
Model groups average their seeds: base = base_s*, CEM = cem_s*.
--horizons 2 (or 2,3) plots success for those planning horizons only; the
checkpoint is chosen when dataset_task_coverage.py runs (--epoch / run specs).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

# Reference palette (dataviz skill, light mode): series slots 1-3, chrome, ink.
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
DATA_BAR, EVAL_BAR = '#b5b3ab', '#52514e'
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
MIN_TASKS = 5  # success points need at least this many tasks in the bin
SELECTION = ''  # e.g. 'epoch 20 · planning horizon 2', set by load()
GROUPS = [  # (label, csv run-name prefix, color); fixed order = fixed colors
    ('in-batch negatives', 'inbatch', '#2a78d6'),
    ('CEM negatives', 'cem_', '#eb6834'),
    ('no search loss (base)', 'base_', '#1baf7a'),
]
FEATURES = {  # name -> (title, x label, tick formatter for bin edges)
    'dyaw': ('T rotation from start to goal', 'rotation (degrees)', lambda v: f'{np.degrees(v):.0f}'),
    'dxy': ('T displacement from start to goal', 'distance (mm)', lambda v: f'{v:g}'),
    'eef_T_goal': ('Pusher distance from the T at the goal', 'distance (mm)', lambda v: f'{v:g}'),
    'goal_step': ('How late in the expert episode the goal is', 'goal step', lambda v: f'{v:g}'),
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(folder, horizons=None):
    rows = list(csv.DictReader(open(folder / 'tasks.csv')))
    meta = json.load(open(folder / 'windows.json'))
    runs = list(meta['runs'])
    if horizons:
        have = set(meta.get('horizons', []))
        if not set(horizons) <= have:
            raise SystemExit(f'horizons {horizons} not in this output (has {sorted(have)}); rerun dataset_task_coverage.py')
        cols = {n: [f'success_{n}_h{h}' for h in horizons] for n in runs}
    else:
        cols = {n: [f'success_{n}'] for n in runs}
    tasks = []
    for r in rows:
        t = {k: float(v) for k, v in r.items() if v != '' and not k.startswith(('success_', 'episodes_'))}
        for label, prefix, _ in GROUPS:
            vals = [float(r[c]) for n in runs if n.startswith(prefix) for c in cols[n] if r.get(c, '') != '']
            t[label] = np.mean(vals) if vals else np.nan
        tasks.append(t)
    epochs = sorted({ep for _, ep in meta['runs'].values()})
    hs = horizons or meta.get('horizons')
    global SELECTION
    SELECTION = (f'epoch {"/".join(map(str, epochs))} · planning horizon{"s" if not hs or len(hs) > 1 else ""} '
                 + (', '.join(map(str, hs)) if hs else 'all'))
    groups = [(lab, col, sum(n.startswith(p) for n in runs)) for lab, p, col in GROUPS if any(n.startswith(p) for n in runs)]
    return tasks, meta, groups


def bin_labels(edges, fmt):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f'> {fmt(lo)}' if hi is None else f'{fmt(lo)}–{fmt(hi)}')
    return out


def success_by_bin(tasks, key, edges, label):
    hi_edges = [np.inf if e is None else e for e in edges]
    out = []
    for lo, hi in zip(hi_edges[:-1], hi_edges[1:]):
        v = [t[label] for t in tasks if lo <= t[key] < hi and np.isfinite(t[label])]
        out.append((100 * np.mean(v) if v else np.nan, len(v)))
    return out


def shares(meta, key, g=None):
    """(training-data %, eval-task %, data label) per bin: at one g, or g-matched over all g."""
    norm = lambda v: 100 * np.asarray(v, float) / max(1, np.sum(v))  # noqa: E731
    if g is not None:
        return norm(meta['hist_by_g'][key][str(g)]['all']), norm(meta['eval_hist_by_g'][key].get(str(g), [0])), 'training data'
    ev = norm(meta['eval_hist'][key])
    if 'hist_by_g' not in meta:
        return norm(meta['hist'][key]['all']), ev, 'training data (pooled over g)'
    data = sum(w * norm(meta['hist_by_g'][key][gg]['all']) for gg, w in meta['g_weight'].items())
    return data, ev, 'training data (same g mix as eval)'


def feature_panels(ax_top, ax_bot, tasks, meta, groups, key, title_on=True, g=None):
    edges = meta['bins'][key]
    title, xlabel, fmt = FEATURES[key]
    labels = bin_labels(edges, fmt)
    x = np.arange(len(labels))
    if g is not None:
        tasks = [t for t in tasks if int(t['g']) == g]
    data, ev, data_label = shares(meta, key, g)
    w = 0.36
    ax_top.bar(x - w / 2 - 0.01, data, w, color=DATA_BAR, label=data_label)
    ax_top.bar(x + w / 2 + 0.01, ev, w, color=EVAL_BAR, label='eval tasks')
    ax_top.set_ylabel('share (%)', color=INK2, fontsize=9)
    if title_on:
        ax_top.set_title(title, color=INK, fontsize=11, loc='left', pad=18)
        ax_top.text(0, 1.02, SELECTION, transform=ax_top.transAxes, fontsize=8, color=MUTED, va='bottom')
    for i, (dv, evv) in enumerate(zip(data, ev)):
        ax_top.text(i - w / 2, dv + 0.8, f'{dv:.0f}', ha='center', va='bottom', fontsize=7.5, color=INK2)
    style(ax_top)
    ax_top.set_xticks(x, [])
    ax_top.set_ylim(0, max(data.max(), ev.max()) * 1.22)
    n = [c for _, c in success_by_bin(tasks, key, edges, groups[0][0])]
    for label, color, _ in groups:
        sb = success_by_bin(tasks, key, edges, label)
        # Bins with fewer than MIN_TASKS tasks are too noisy to draw (one task = 0 or 100%).
        y = np.array([v if c >= MIN_TASKS else np.nan for v, c in sb])
        ax_bot.plot(x, y, color=color, linewidth=2, marker='o', markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    ax_bot.set_ylabel('planning success (%)', color=INK2, fontsize=9)
    ax_bot.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax_bot.set_xticks(x, [f'{lab}\n{c} task{"s" if c != 1 else ""}' for lab, c in zip(labels, n)])
    for tick, c in zip(ax_bot.get_xticklabels(), n):
        tick.set_color(INK2 if c >= MIN_TASKS else MUTED)
    ax_bot.set_ylim(0, 100)
    style(ax_bot)


def direct_labels(ax, tasks, key, edges, groups):
    """Name each line at its last point, nudged apart."""
    ends = []
    for label, color, _ in groups:
        s = [v for v, c in success_by_bin(tasks, key, edges, label) if c >= MIN_TASKS]
        ends.append([s[-1] if s else 50.0, label])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 6)
    n = len(edges) - 1
    for y, label in ends:
        ax.annotate(label, (n - 1, y), xytext=(10, 0), textcoords='offset points', va='center',
                    fontsize=8, color=INK2, annotation_clip=False)


def legend(fig, ax_top, groups):
    h1, l1 = ax_top.get_legend_handles_labels()
    from matplotlib.lines import Line2D

    h2 = [Line2D([], [], color=c, linewidth=2, marker='o', markersize=6) for _, c, _ in groups]
    l2 = [f'{lab} ({n} seed{"s" if n > 1 else ""})' for lab, _, n in groups]
    fig.legend(h1 + h2, l1 + l2, loc='lower center', ncol=len(h1 + h2), frameon=False, fontsize=8,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))


def plot_feature(tasks, meta, groups, key, out):
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    feature_panels(a, b, tasks, meta, groups, key)
    direct_labels(b, tasks, key, meta['bins'][key], groups)
    a.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=INK2)
    b.legend([f'{lab} ({n} seed{"s" if n > 1 else ""})' for lab, _, n in groups], loc='upper right',
             bbox_to_anchor=(1, 0.93), frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_feature_by_g(tasks, meta, groups, key, out):
    offsets = meta['offsets']
    fig, axes = plt.subplots(2, len(offsets), figsize=(3.6 * len(offsets), 5.8), sharey='row',
                             gridspec_kw={'height_ratios': [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    top = 0
    for j, g in enumerate(offsets):
        feature_panels(axes[0, j], axes[1, j], tasks, meta, groups, key, title_on=False, g=g)
        axes[0, j].set_title(f'goal offset g = {g}', color=INK, fontsize=10, loc='left')
        top = max(top, axes[0, j].get_ylim()[1])
        if j:
            axes[0, j].set_ylabel('')
            axes[1, j].set_ylabel('')
            axes[0, j].get_legend() and axes[0, j].get_legend().remove()
    for j in range(len(offsets)):
        axes[0, j].set_ylim(0, top)
    title = FEATURES[key][0]
    fig.suptitle(f'{title}, per goal offset', x=0.01, ha='left', color=INK, fontsize=11)
    fig.text(0.01, 0.935, f'{SELECTION} · success not drawn for bins with fewer than {MIN_TASKS} tasks',
             fontsize=8, color=MUTED)
    legend(fig, axes[0, 0], groups)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_map_by_g(tasks, meta, out, solver):
    edges = meta['map']['edges']
    ex, ey = np.degrees(edges['dyaw']), np.array(edges['dxy'])
    offsets = meta['offsets']
    shares_g = {g: 100 * np.array(meta['map']['by_g'][str(g)]) / max(1, np.sum(meta['map']['by_g'][str(g)])) for g in offsets}
    pos = np.concatenate([v[v > 0] for v in shares_g.values()])
    norm = LogNorm(vmin=max(pos.min(), 1e-4), vmax=pos.max())
    cmap = LinearSegmentedColormap.from_list('seq', SEQ)
    cmap.set_bad(SURFACE)
    fig, axes = plt.subplots(1, len(offsets), figsize=(3.9 * len(offsets), 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, g in zip(axes, offsets):
        mesh = ax.pcolormesh(ex, ey, np.ma.masked_equal(shares_g[g], 0).T, cmap=cmap, norm=norm, shading='flat')
        tg = [t for t in tasks if int(t['g']) == g]
        s = np.array([t[solver] for t in tg])
        x = np.degrees([t['dyaw'] for t in tg])
        y = np.minimum([t['dxy'] for t in tg], ey[-1] - 2)
        good = s >= 0.5
        ax.scatter(x[good], y[good], s=26, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=3,
                   label=f'{solver} solves >= 50%')
        ax.scatter(x[~good], y[~good], s=24, marker='x', color=INK, linewidth=1.3, zorder=3,
                   label=f'{solver} solves < 50%')
        ax.set_title(f'goal offset g = {g}  ({good.sum()} solved / {(~good).sum()} failed)', color=INK, fontsize=9.5, loc='left')
        ax.set_xlabel('T rotation (degrees)', color=INK2, fontsize=9)
        style(ax)
        ax.grid(False)
        ax.set_ylim(0, ey[-1])
    axes[0].set_ylabel('T displacement (mm)', color=INK2, fontsize=9)
    axes[0].legend(loc='upper right', frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=7.5, labelcolor=INK2)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.8, bottom=0.14, wspace=0.08)
    cb = fig.colorbar(mesh, cax=fig.add_axes([0.915, 0.14, 0.01, 0.66]))
    cb.set_label('training windows at this g (% per cell, log)', color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cb.outline.set_visible(False)
    fig.suptitle('Where the eval tasks sit in the training data, per goal offset', x=0.01, ha='left', color=INK, fontsize=11)
    fig.text(0.01, 0.91, SELECTION, fontsize=8, color=MUTED)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_overview(tasks, meta, groups, out):
    keys = list(FEATURES)
    fig, axes = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 5.8), gridspec_kw={'height_ratios': [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    for j, k in enumerate(keys):
        feature_panels(axes[0, j], axes[1, j], tasks, meta, groups, k)
        if j:
            axes[0, j].set_ylabel('')
            axes[1, j].set_ylabel('')
    legend(fig, axes[0, 0], groups)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_map(tasks, meta, out, solver):
    edges, counts = meta['map']['edges'], np.array(meta['map']['counts'])
    ex, ey = np.degrees(edges['dyaw']), np.array(edges['dxy'])
    if 'by_g' in meta['map']:  # g-matched: per-g shares weighted like the eval tasks' g mix
        share = sum(w * 100 * np.array(meta['map']['by_g'][g]) / max(1, np.sum(meta['map']['by_g'][g]))
                    for g, w in meta['g_weight'].items())
        cb_label = 'training windows, same g mix as eval (% per cell, log)'
    else:
        share = 100 * counts / counts.sum()
        cb_label = 'training windows (% per cell, log)'
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    fig.patch.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list('seq', SEQ)
    cmap.set_bad(SURFACE)
    pos = share[share > 0]
    mesh = ax.pcolormesh(ex, ey, np.ma.masked_equal(share, 0).T, cmap=cmap,
                         norm=LogNorm(vmin=max(pos.min(), 1e-4), vmax=pos.max()), shading='flat')
    cb = fig.colorbar(mesh, ax=ax, pad=0.02)
    cb.set_label(cb_label, color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
    cb.outline.set_visible(False)
    s = np.array([t[solver] for t in tasks])
    x = np.degrees([t['dyaw'] for t in tasks])
    y = np.minimum([t['dxy'] for t in tasks], ey[-1] - 2)
    good = s >= 0.5
    ax.scatter(x[good], y[good], s=34, facecolor=SURFACE, edgecolor=INK, linewidth=1.3,
               label=f'eval task, {solver} solves >= 50% ({good.sum()})', zorder=3)
    ax.scatter(x[~good], y[~good], s=30, marker='x', color=INK, linewidth=1.4,
               label=f'eval task, {solver} solves < 50% ({(~good).sum()})', zorder=3)
    ax.set_xlabel('T rotation from start to goal (degrees)', color=INK2, fontsize=9)
    ax.set_ylabel('T displacement from start to goal (mm)', color=INK2, fontsize=9)
    ax.set_title('Where the eval tasks sit in the training data', color=INK, fontsize=11, loc='left', pad=18)
    ax.text(0, 1.02, SELECTION, transform=ax.transAxes, fontsize=8, color=MUTED, va='bottom')
    ax.set_ylim(0, ey[-1])
    n_off = int((np.array([t['dxy'] for t in tasks]) >= ey[-1] - 2).sum())
    ax.text(0.0, -0.13, f'Tasks moving the T more than {ey[-1]:.0f} mm ({n_off}) are drawn at the top edge; '
            'training windows beyond it are not shown.', transform=ax.transAxes, fontsize=7.5, color=MUTED)
    style(ax)
    ax.grid(False)
    ax.legend(loc='upper right', frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_similar(tasks, groups, out, key='similar_windows'):
    edges = [0, 1, 10, 100, 1000, np.inf]
    labels = ['0', '1–9', '10–99', '100–999', '1000+']
    v = np.array([t[key] for t in tasks])
    rot = np.array([t['dyaw'] for t in tasks])
    b = np.digitize(v, edges[1:-1])
    x = np.arange(len(labels))
    fig, (a, c) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True, gridspec_kw={'height_ratios': [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    low = np.array([((b == i) & (rot < 0.8)).sum() for i in x])
    high = np.array([((b == i) & (rot >= 0.8)).sum() for i in x])
    a.bar(x, low, 0.5, color=SEQ[2], label='rotation < 46°')
    a.bar(x, high, 0.5, bottom=low + 0.15 * max(1, (low + high).max()) / 50, color=SEQ[5], label='rotation ≥ 46°')
    for i in x:
        a.text(i, low[i] + high[i] + 0.4, f'{low[i] + high[i]}', ha='center', va='bottom', fontsize=8, color=INK2)
    a.set_ylabel('eval tasks', color=INK2, fontsize=9)
    a.set_title('Success vs how often the training data makes the same move', color=INK, fontsize=11, loc='left', pad=18)
    a.text(0, 1.02, SELECTION, transform=a.transAxes, fontsize=8, color=MUTED, va='bottom')
    style(a)
    a.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=INK2)
    for label, color, _ in groups:
        s = [100 * np.nanmean([t[label] for t, bb in zip(tasks, b) if bb == i]) if (b == i).any() else np.nan for i in x]
        c.plot(x, s, color=color, linewidth=2, marker='o', markersize=6, markeredgecolor=SURFACE,
               markeredgewidth=1.5, label=label)
    c.set_ylim(0, 100)
    c.set_ylabel('planning success (%)', color=INK2, fontsize=9)
    tol = '2x the success tolerance' if key.endswith('_2x') else 'the success tolerance: 20 mm, 9°'
    c.set_xlabel(f'training windows that start and end where the task does (T pose within {tol}, same g)',
                 color=INK2, fontsize=8.5)
    c.set_xticks(x, labels)
    style(c)
    c.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path)
    p.add_argument('--similar', default='similar_windows', choices=['similar_windows', 'similar_windows_2x'])
    p.add_argument('--horizons', type=lambda t: [int(x) for x in t.split(',') if x], default=None,
                   help='planning horizons to plot, e.g. 2 or 2,3 (default: all)')
    p.add_argument('--suffix', default=None, help='file name suffix (default: _h<horizons> when --horizons is set)')
    args = p.parse_args()
    tasks, meta, groups = load(args.folder, args.horizons)
    sfx = args.suffix if args.suffix is not None else ('_h' + ''.join(map(str, args.horizons)) if args.horizons else '')
    plt.rcParams['font.family'] = ['DejaVu Sans']
    for k in FEATURES:
        plot_feature(tasks, meta, groups, k, args.folder / f'feature_{k}{sfx}.png')
        if 'hist_by_g' in meta:
            plot_feature_by_g(tasks, meta, groups, k, args.folder / f'feature_{k}_by_g{sfx}.png')
    plot_overview(tasks, meta, groups, args.folder / f'overview{sfx}.png')
    plot_map(tasks, meta, args.folder / f'map{sfx}.png', groups[0][0])
    if 'by_g' in meta['map']:
        plot_map_by_g(tasks, meta, args.folder / f'map_by_g{sfx}.png', groups[0][0])
    plot_similar(tasks, groups, args.folder / f'similar{sfx}.png', args.similar)
    print(f'wrote figures to {args.folder}')


if __name__ == '__main__':
    main()
