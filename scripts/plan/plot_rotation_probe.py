"""Figures for rotation_probe.py output (metrics.json).

  python scripts/plan/plot_rotation_probe.py reports/rotation_probe/<run>

Writes next to metrics.json:
  representation.png  probe error from a real frame embedding: T yaw (deg), T xy (mm)
  cost.png            embedding distance (planning cost) vs T rotation and vs T displacement
  prediction.png      yaw error of probed predictions per rollout step, and the share of
                      the true rotation the prediction shows, by rotation size
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Reference palette (dataviz skill, light mode) and the model colors used in the coverage figures.
SURFACE, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#898781'
GRID, AXIS = '#e1e0d9', '#c3c2b7'
SLOTS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
PREFIX_SLOT = [('inbatch', 0), ('cem', 1), ('base', 2)]
MIN_PAIRS = 20  # cost curve points need this many frame pairs
PRETTY = [('inbatch', 'in-batch negatives'), ('cem', 'CEM negatives'), ('base', 'no search loss (base)')]


def color_of(names):
    used, out = set(), {}
    for n in names:
        slot = next((s for p, s in PREFIX_SLOT if n.startswith(p) and s not in used), None)
        if slot is None:
            slot = next(s for s in range(len(SLOTS)) if s not in used and s not in {s for _, s in PREFIX_SLOT})
        used.add(slot)
        out[n] = SLOTS[slot]
    return out


def pretty(n):
    for p, lab in PRETTY:
        if n.startswith(p):
            rest = n[len(p) :].strip('_')
            return f'{lab} {rest}'.strip()
    return n


def style(ax, grid='y'):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def header(fig, title, sub):
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha='left', va='top', color=INK, fontsize=11.5)
    fig.text(0.01, 1 - 0.36 / h, sub, fontsize=8, color=MUTED, va='top')
    return 1 - 0.7 / h  # top of the plotting area


def short(n):
    for p, lab in (('inbatch', 'in-batch'), ('cem', 'CEM'), ('base', 'base')):
        if n.startswith(p):
            return f'{lab} {n[len(p):].strip("_")}'.strip()
    return n


def plot_representation(M, colors, out, sub):
    names = list(M['models'])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    fig.patch.set_facecolor(SURFACE)
    panels = [('yaw_err_deg', 'T rotation (yaw)', 'median error (degrees)'),
              ('txy_err_mm', 'T position', 'median error (mm)'),
              ('eef_err_mm', 'pusher (EEF) position', 'median error (mm)')]
    x = np.arange(len(names))
    w = 0.36
    for ax, (key, title, ylab) in zip(axes, panels):
        lin = [M['models'][n]['probe']['linear'][key] for n in names]
        mlp = [M['models'][n]['probe']['mlp'][key] for n in names]
        ax.bar(x - w / 2 - 0.01, lin, w, color=[colors[n] for n in names], alpha=0.45, hatch='////', edgecolor=SURFACE)
        ax.bar(x + w / 2 + 0.01, mlp, w, color=[colors[n] for n in names])
        for i, (a, b) in enumerate(zip(lin, mlp)):
            ax.text(i - w / 2, a, f'{a:.1f}', ha='center', va='bottom', fontsize=7.5, color=INK2)
            ax.text(i + w / 2, b, f'{b:.1f}', ha='center', va='bottom', fontsize=7.5, color=INK2)
        ax.set_xticks(x, [short(n) for n in names], fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        ax.set_ylabel(ylab, color=INK2, fontsize=9)
        style(ax)
    fig.legend([Patch(facecolor=MUTED, alpha=0.45, hatch='////', edgecolor=SURFACE), Patch(facecolor=MUTED)],
               ['linear probe', 'MLP probe'], loc='upper right', ncol=2, frameon=False, fontsize=8, labelcolor=INK2,
               bbox_to_anchor=(0.99, 1.0))
    top = header(fig, 'Is the T\'s rotation in the embedding of a real frame?', sub + ' · lower is better')
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_cost(M, colors, out, sub):
    names = list(M['models'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for n in names:
        c = M['models'][n]['cost']
        rb = [r for r in c['by_rotation'] if r['mean_d'] is not None and r['n'] >= MIN_PAIRS]
        xb = [r for r in c['by_displacement'] if r['mean_d'] is not None and r['n'] >= MIN_PAIRS]
        mm = c['fit']['mm_equal_to_45deg']
        lab = f'{pretty(n)}: 45° ≈ {mm:.0f} mm' if mm else pretty(n)
        a.plot([np.mean(r['deg']) for r in rb], [r['mean_d'] for r in rb], color=colors[n], linewidth=2, marker='o',
               markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
        b.plot([np.mean(r['mm']) for r in xb], [r['mean_d'] for r in xb], color=colors[n], linewidth=2, marker='o',
               markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=pretty(n))
    a.set_title('T mostly rotates (moves < 30 mm)', color=INK, fontsize=10, loc='left')
    a.set_xlabel('T rotation between the two frames (degrees)', color=INK2, fontsize=9)
    a.set_ylabel('embedding distance (x median)', color=INK2, fontsize=9)
    b.set_title('T mostly moves (rotates < 10°)', color=INK, fontsize=10, loc='left')
    b.set_xlabel('T displacement between the two frames (mm)', color=INK2, fontsize=9)
    for ax in (a, b):
        style(ax)
        ax.set_ylim(bottom=0)
    a.legend(loc='upper left', frameon=False, fontsize=8, labelcolor=INK2,
             title='45° of rotation costs as much as … of T motion', title_fontsize=7.5)
    top = header(fig, 'Does the planning cost (embedding distance) notice rotation?',
                 sub + ' · squared embedding distance between two real frames of the same clip, in units of the median')
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_prediction(M, colors, out, sub):
    names = list(M['models'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)
    steps = [s['step'] for s in M['models'][names[0]]['prediction']['per_step']]
    for n in names:
        ps = M['models'][n]['prediction']['per_step']
        a.plot(steps, [s['mlp_pred_yaw_err'] for s in ps], color=colors[n], linewidth=2, marker='o', markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
        a.plot(steps, [s['mlp_real_yaw_err'] for s in ps], color=colors[n], linewidth=1.2, linestyle=(0, (4, 3)))
    a.plot(steps, [s['no_change_yaw_err'] for s in M['models'][names[0]]['prediction']['per_step']], color=MUTED,
           linewidth=1.5, linestyle=(0, (1, 2)))
    a.set_xticks(steps)
    a.set_xlabel('rollout step (action blocks of 5 env steps)', color=INK2, fontsize=9)
    a.set_ylabel('median yaw error (degrees)', color=INK2, fontsize=9)
    a.set_title('Yaw read from predicted embeddings', color=INK, fontsize=10, loc='left')
    a.set_ylim(bottom=0)
    style(a)
    a.legend([Line2D([], [], color=INK2, linewidth=2, marker='o'), Line2D([], [], color=INK2, linewidth=1.2, linestyle=(0, (4, 3))),
              Line2D([], [], color=MUTED, linewidth=1.5, linestyle=(0, (1, 2)))],
             ['predicted frame', 'real frame (probe floor)', 'no change (context yaw)'],
             loc='upper left', frameon=False, fontsize=8, labelcolor=INK2)
    keep = [i for i, r in enumerate(M['models'][names[0]]['prediction']['by_rotation']) if r['rad'][0] >= 0.2]
    rows = [M['models'][names[0]]['prediction']['by_rotation'][i] for i in keep]
    labels = [f"{np.degrees(r['rad'][0]):.0f}–{np.degrees(r['rad'][1]):.0f}" if r['rad'][1] else f"> {np.degrees(r['rad'][0]):.0f}" for r in rows]
    x = np.arange(len(rows))
    for n in names:
        br = [M['models'][n]['prediction']['by_rotation'][i] for i in keep]
        b.plot(x, [100 * r['mlp_pred_fraction'] for r in br], color=colors[n], linewidth=2, marker='o', markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=pretty(n))
        b.plot(x, [100 * r['mlp_real_fraction'] for r in br], color=colors[n], linewidth=1.2, linestyle=(0, (4, 3)))
    b.axhline(100, color=AXIS, linewidth=1)
    b.set_xticks(x, [f'{lab}\n{r["n"]} clips' for lab, r in zip(labels, rows)])
    b.set_xlabel('true T rotation over the rollout (degrees; rotations < 11° left out)', color=INK2, fontsize=9)
    b.set_ylabel('rotation shown (% of true, LS slope)', color=INK2, fontsize=9)
    b.set_title('How much of the rotation the prediction shows', color=INK, fontsize=10, loc='left')
    b.set_ylim(bottom=min(0, b.get_ylim()[0]))
    style(b)
    b.legend([Line2D([], [], color=colors[n], linewidth=2, marker='o', markersize=6) for n in names]
             + [Line2D([], [], color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))],
             [pretty(n) for n in names] + ['real frame (probe ceiling)'],
             loc='best', frameon=False, fontsize=8, labelcolor=INK2)
    top = header(fig, 'Does the predictor lose the rotation?',
                 sub + ' · MLP probe fit on real embeddings, applied to rollouts with the true action blocks')
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folder', type=Path)
    args = p.parse_args()
    M = json.load(open(args.folder / 'metrics.json'))
    plt.rcParams['font.family'] = ['DejaVu Sans']
    colors = color_of(list(M['models']))
    epochs = sorted({Path(v).stem.split('_')[-1] for v in M['config']['checkpoints'].values()})
    c = M['config']
    sub = f"epoch {'/'.join(epochs)} · {c['test']} held-out clips of {c['clips']} (rotation-balanced)"
    plot_representation(M, colors, args.folder / 'representation.png', sub)
    plot_cost(M, colors, args.folder / 'cost.png', sub)
    plot_prediction(M, colors, args.folder / 'prediction.png', sub)
    print(f'wrote figures to {args.folder}')


if __name__ == '__main__':
    main()
