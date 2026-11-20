"""Do planning failures sit where the training data is sparse? (mmbench2-style test)

Joins three things that already exist:
  * eval results  (random_goal_pusht_results.txt): per-episode success + reset seed,
    goal offset g (eval.goal_offset_steps) and horizon h (plan_config.horizon);
  * the expert goal cache (expert_<seed>_<digest>.pkl): for every (seed, g) the
    expert rollout that defined the task -> start pose, goal pose, expert path;
  * a training dataset (Lance): reference T poses / TCP positions.

Density is measured in units of the eval's success tolerance (T xy 20 mm,
T yaw 0.15 rad): a dataset frame "matches" a pose when it is inside the
success region around it, d = sqrt((dxy / 20 mm)^2 + (dyaw / 0.15 rad)^2) <= 1.

  goal_matches   dataset frames whose T pose would count as reaching the goal
  start_matches  dataset frames inside the tolerance of the start T pose, with the
                 TCP within 20 mm of the start TCP (full start state)
  path_min       min over the expert's start->goal path of the T-pose matches:
                 the sparsest point the plan has to pass through

Per eval episode (seed, g) the "success rate" pools every selected run
(checkpoints, horizons). Reported: Spearman correlations of success with the
densities and with task size (T displacement, yaw change), a joint linear fit,
per-run AUROC of density for predicting failure, and figures.

Usage (on Mila, where results, cache and datasets live):
  python scripts/data/eval_goal_density.py \
      --results-glob '/network/scratch/o/ozgur.aslan/le-wm/evals/sched_n100/*/random_goal_pusht_results.txt' \
      --cache-dir /network/scratch/o/ozgur.aslan/le-wm/expert_goal_cache \
      --dataset /network/scratch/o/ozgur.aslan/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance \
      --out reports/eval_goal_density
"""

import argparse
import csv
import glob
import json
import math
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_density import INK, MUTED, SEQ_CMAP, SERIES, load, quat_yaw, style, wrap  # noqa: E402

XY_TOL_MM, YAW_TOL_RAD = 20.0, 0.15


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-glob', required=True)
    p.add_argument('--run-filter', default=None, help='regex on the run directory name')
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--dataset', required=True, help='training dataset (.lance table or a dir holding one)')
    p.add_argument('--max-ref-frames', type=int, default=3_000_000, help='subsample the dataset to this many frames')
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--out', default='reports/eval_goal_density')
    return p.parse_args()


# -- eval results -----------------------------------------------------------

ARRAY = r"'{key}':\s*array\(\[(.*?)\]"


def parse_array(text, key):
    m = re.search(ARRAY.format(key=key), text, re.S)
    if not m:
        return None
    return np.array([float(v) for v in m.group(1).replace('\n', ' ').split(',') if v.strip()])


def parse_results(path):
    """Last run in a results file: dict(run, g, h, seeds, success) or None."""
    text = Path(path).read_text()
    blocks = text.split('==== CONFIG ====')[1:]
    if not blocks:
        return None
    block = blocks[-1]
    if '==== RESULTS ====' not in block:
        return None
    cfg_text, res_text = block.split('==== RESULTS ====', 1)
    try:
        cfg = yaml.safe_load(cfg_text)
    except yaml.YAMLError:
        return None
    succ, seeds = parse_array(res_text, 'episode_successes'), parse_array(res_text, 'seeds')
    if succ is None or seeds is None or len(succ) != len(seeds):
        return None
    return {
        'run': Path(path).parent.name,
        'g': int(cfg['eval']['goal_offset_steps']),
        'h': int(cfg['plan_config']['horizon']),
        'seeds': seeds.astype(int),
        'success': succ.astype(bool),
    }


# -- expert goal cache ------------------------------------------------------

def load_cache(cache_dir, wanted):
    """(seed, g) -> start/goal/path poses from the expert cache; one file per key suffices."""
    tasks = {}
    wanted_seeds = {s for s, _ in wanted}
    for f in sorted(Path(cache_dir).glob('expert_*.pkl')):
        seed = int(f.name.split('_')[1])
        if seed not in wanted_seeds:
            continue
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        s, i = int(r['start_index']), int(r['index'])
        # Files are named by the requested seed; a retried task has a different
        # accepted seed (r['seed']). Results may record either, so index both.
        keys = [k for k in ((seed, i - s), (int(r['seed']), i - s)) if k in wanted and k not in tasks]
        if not keys:
            continue
        poses = np.asarray(r['poses'], np.float64)
        eef = np.asarray(r['eef_positions'], np.float64)
        task = {
            'start_pose': poses[s], 'goal_pose': poses[i], 'start_eef': eef[s], 'goal_eef': eef[i],
            'path': poses[s:i + 1], 'expert_seed': int(r['seed']), 'attempt': int(r['attempt']),
        }
        for key in keys:
            tasks[key] = task
    return tasks


# -- geometry ---------------------------------------------------------------

def tee_embed(x_mm, y_mm, yaw):
    """T pose -> tolerance units; the yaw chord ~= |dyaw| / YAW_TOL for small angles."""
    r = 1.0 / YAW_TOL_RAD
    return np.stack([x_mm / XY_TOL_MM, y_mm / XY_TOL_MM, r * np.cos(yaw), r * np.sin(yaw)], -1)


def pose_parts(pose):
    pose = np.atleast_2d(pose)
    return 1e3 * pose[:, 0], 1e3 * pose[:, 1], quat_yaw(pose[:, 3:7])


def auroc(score, label):
    """P(score of a positive > score of a negative), ties count half."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.r_[pos, neg], kind='stable')
    ranks = np.empty(len(order))
    vals = np.r_[pos, neg][order]
    i = 0
    while i < len(vals):  # average ranks over ties
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. eval runs
    runs = []
    for f in sorted(glob.glob(args.results_glob)):
        if args.run_filter and not re.search(args.run_filter, Path(f).parent.name):
            continue
        r = parse_results(f)
        if r is not None:
            runs.append(r)
    if not runs:
        raise SystemExit('no parsable results')
    wanted = {(int(s), r['g']) for r in runs for s in r['seeds']}
    print(f'{len(runs)} runs, {len(wanted)} (seed, g) tasks')

    # 2. tasks from the expert cache
    tasks = load_cache(args.cache_dir, wanted)
    missing = wanted - set(tasks)
    print(f'{len(tasks)} tasks found in the cache; {len(missing)} missing')

    # 3. reference dataset (only light columns)
    data, _ = load(args.dataset, 1, args.batch_size)
    obj, tcp = data['obs_extra_obj_pose'], data['obs_extra_tcp_pose']
    n = len(obj)
    keep = np.random.default_rng(0).choice(n, min(n, args.max_ref_frames), replace=False)
    ref_tee = tee_embed(1e3 * obj[keep, 0], 1e3 * obj[keep, 1], quat_yaw(obj[keep, 3:7]))
    ref_tcp = 1e3 * tcp[keep, :2] / XY_TOL_MM
    tree_tee = cKDTree(ref_tee)
    tree_state = cKDTree(np.concatenate([ref_tee, ref_tcp], 1))
    print(f'reference: {len(keep):,} of {n:,} frames')
    del data, obj, tcp

    # 4. per-task densities
    keys = sorted(tasks)
    rows = []
    for key in keys:
        t = tasks[key]
        gx, gy, gyaw = pose_parts(t['goal_pose'])
        sx, sy, syaw = pose_parts(t['start_pose'])
        px, py, pyaw = pose_parts(t['path'])
        goal_m = len(tree_tee.query_ball_point(tee_embed(gx, gy, gyaw)[0], 1.0))
        start_vec = np.r_[tee_embed(sx, sy, syaw)[0], 1e3 * t['start_eef'][:2] / XY_TOL_MM]
        # full-state ball: T within tolerance AND TCP within 20 mm -> radius sqrt(2) then filter
        cand = tree_state.query_ball_point(start_vec, math.sqrt(2.0))
        if cand:
            d = tree_state.data[cand] - start_vec
            start_m = int(((np.sum(d[:, :4] ** 2, 1) <= 1) & (np.sum(d[:, 4:] ** 2, 1) <= 1)).sum())
        else:
            start_m = 0
        path_counts = [len(c) for c in tree_tee.query_ball_point(tee_embed(px, py, pyaw), 1.0)]
        rows.append({
            'seed': key[0], 'g': key[1],
            'goal_x_mm': float(gx[0]), 'goal_y_mm': float(gy[0]), 'goal_yaw_deg': float(np.degrees(gyaw[0])),
            'start_x_mm': float(sx[0]), 'start_y_mm': float(sy[0]), 'start_yaw_deg': float(np.degrees(syaw[0])),
            'disp_mm': float(np.hypot(gx[0] - sx[0], gy[0] - sy[0])),
            'dyaw_deg': float(np.degrees(abs(wrap(gyaw[0] - syaw[0])))),
            'goal_matches': goal_m, 'start_matches': start_m, 'path_min': int(min(path_counts)),
        })
    index = {(r['seed'], r['g']): i for i, r in enumerate(rows)}

    # 5. success per task, pooled over runs; per-run AUROC
    succ_sum, succ_n = np.zeros(len(rows)), np.zeros(len(rows))
    per_run = []
    for r in runs:
        idx = np.array([index.get((int(s), r['g']), -1) for s in r['seeds']])
        ok = idx >= 0
        np.add.at(succ_sum, idx[ok], r['success'][ok])
        np.add.at(succ_n, idx[ok], 1)
        dens = np.array([rows[i]['goal_matches'] for i in idx[ok]], float)
        pmin = np.array([rows[i]['path_min'] for i in idx[ok]], float)
        fail = ~r['success'][ok]
        per_run.append({
            'run': r['run'], 'g': r['g'], 'h': r['h'], 'success_rate': float(r['success'][ok].mean()),
            # AUROC that LOW density predicts failure (0.5 = no relation)
            'auroc_goal': auroc(-np.log1p(dens), fail), 'auroc_path': auroc(-np.log1p(pmin), fail),
        })
    for i, r in enumerate(rows):
        r['n_runs'] = int(succ_n[i])
        r['success_rate'] = float(succ_sum[i] / succ_n[i]) if succ_n[i] else float('nan')

    with open(out / 'episodes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / 'runs.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows(per_run)

    # 6. statistics
    sr = np.array([r['success_rate'] for r in rows])
    cols = {
        'log1p goal_matches': np.log1p([r['goal_matches'] for r in rows]),
        'log1p path_min': np.log1p([r['path_min'] for r in rows]),
        'log1p start_matches': np.log1p([r['start_matches'] for r in rows]),
        'T displacement mm': np.array([r['disp_mm'] for r in rows]),
        'yaw change deg': np.array([r['dyaw_deg'] for r in rows]),
        'goal offset g': np.array([r['g'] for r in rows], float),
    }
    ok = np.isfinite(sr)
    spear = {k: spearmanr(v[ok], sr[ok]) for k, v in cols.items()}
    spear_g = {}
    for g in sorted({r['g'] for r in rows}):
        m = ok & (cols['goal offset g'] == g)
        spear_g[g] = {k: spearmanr(cols[k][m], sr[m])[0] for k in ('log1p goal_matches', 'log1p path_min', 'T displacement mm', 'yaw change deg')}
    X = np.stack([(v[ok] - v[ok].mean()) / (v[ok].std() + 1e-12) for v in cols.values()], 1)
    X = np.c_[np.ones(ok.sum()), X]
    beta, *_ = np.linalg.lstsq(X, sr[ok], rcond=None)
    resid = sr[ok] - X @ beta
    r2 = 1 - resid.var() / sr[ok].var()

    au_goal = np.array([p['auroc_goal'] for p in per_run])
    au_path = np.array([p['auroc_path'] for p in per_run])
    lines = [
        '# Eval goals vs training-data density', '',
        f'- runs: {len(runs)}; tasks (seed, g): {len(rows)} (missing from cache: {len(missing)})',
        f'- reference dataset: {args.dataset} ({len(keep):,} frames)',
        f'- density unit: dataset frames inside the success tolerance (T xy {XY_TOL_MM:.0f} mm, yaw {YAW_TOL_RAD} rad)',
        f'- mean success over tasks: {np.nanmean(sr):.3f}', '',
        '## Task densities', '',
        '| quantity | median | 10th pct | min | share of tasks with 0 |', '|---|---|---|---|---|',
    ]
    for k in ('goal_matches', 'path_min', 'start_matches'):
        v = np.array([r[k] for r in rows])
        lines.append(f'| {k} | {np.median(v):.0f} | {np.percentile(v, 10):.0f} | {v.min()} | {(v == 0).mean():.3f} |')
    lines += ['', '## Spearman correlation with per-task success rate (pooled over runs)', '',
              '| variable | rho | p |', '|---|---|---|']
    for k, s in spear.items():
        lines.append(f'| {k} | {s[0]:+.3f} | {s[1]:.2g} |')
    lines += ['', '### Within each goal offset g', '', '| g | goal density | path min density | T displacement | yaw change |', '|---|---|---|---|---|']
    for g, d in spear_g.items():
        lines.append(f"| {g} | {d['log1p goal_matches']:+.3f} | {d['log1p path_min']:+.3f} | {d['T displacement mm']:+.3f} | {d['yaw change deg']:+.3f} |")
    lines += ['', f'## Joint linear fit of success rate on standardized variables (R^2 = {r2:.3f})', '',
              '| variable | coefficient (success per 1 sd) |', '|---|---|']
    for k, b in zip(cols, beta[1:]):
        lines.append(f'| {k} | {b:+.3f} |')
    lines += ['', '## Per-run AUROC: low density predicts failure (0.5 = no relation)', '',
              f'- goal density: mean {np.nanmean(au_goal):.3f}, median {np.nanmedian(au_goal):.3f} over {np.isfinite(au_goal).sum()} runs',
              f'- path min density: mean {np.nanmean(au_path):.3f}, median {np.nanmedian(au_path):.3f}']
    (out / 'summary.md').write_text('\n'.join(lines))
    (out / 'summary.json').write_text(json.dumps({
        'spearman': {k: [float(s[0]), float(s[1])] for k, s in spear.items()},
        'spearman_by_g': {str(g): d for g, d in spear_g.items()},
        'linear_fit': {'r2': float(r2), 'coef': dict(zip(cols, map(float, beta[1:])))},
        'auroc_goal_mean': float(np.nanmean(au_goal)), 'auroc_path_mean': float(np.nanmean(au_path)),
    }, indent=2))
    print('\n'.join(lines))

    # 7. figures
    gs = sorted({r['g'] for r in rows})
    fig, axes = plt.subplots(1, len(gs), figsize=(3.4 * len(gs), 3.6), squeeze=False, constrained_layout=True)
    H, ex, ey = np.histogram2d(ref_tee[:, 0] * XY_TOL_MM, ref_tee[:, 1] * XY_TOL_MM, bins=80)
    for ax, g in zip(axes[0], gs):
        ax.pcolormesh(ex, ey, np.ma.masked_equal(H, 0).T, cmap='Greys', norm=matplotlib.colors.LogNorm(), rasterized=True)
        sel = [r for r in rows if r['g'] == g]
        sc = ax.scatter([r['goal_x_mm'] for r in sel], [r['goal_y_mm'] for r in sel], c=[r['success_rate'] for r in sel],
                        cmap=SEQ_CMAP, vmin=0, vmax=1, s=22, edgecolors=INK, linewidths=0.4)
        ax.set_title(f'g = {g}: goal T position', fontsize=9, color=INK)
        ax.set_aspect('equal')
        style(ax)
    fig.colorbar(sc, ax=axes[0], location='bottom', shrink=0.5, aspect=30, label='success rate over runs (grey = dataset T density)')
    fig.savefig(out / 'goals_on_density.png', dpi=130)
    plt.close(fig)

    panels = [('log1p goal_matches', 'goal density: log(1 + frames in tolerance)'),
              ('log1p path_min', 'path min density: log(1 + frames)'),
              ('T displacement mm', 'T displacement start->goal (mm)'),
              ('yaw change deg', 'yaw change start->goal (deg)')]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), constrained_layout=True)
    for ax, (k, label) in zip(axes, panels):
        for gi, g in enumerate(gs):
            m = ok & (cols['goal offset g'] == g)
            ax.scatter(cols[k][m], sr[m], s=12, color=SERIES[gi % len(SERIES)], alpha=0.7, label=f'g={g}', edgecolors='none')
        ax.set_xlabel(label, fontsize=8, color=INK)
        ax.set_title(f"rho = {spear[k][0]:+.2f}", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel('success rate over runs', fontsize=8, color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    fig.savefig(out / 'success_vs_density.png', dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 3.2), constrained_layout=True)
    ax.hist(au_goal[np.isfinite(au_goal)], bins=30, range=(0, 1), color=SERIES[0], alpha=0.8, label='goal density')
    ax.hist(au_path[np.isfinite(au_path)], bins=30, range=(0, 1), histtype='step', color=SERIES[1], lw=1.5, label='path min density')
    ax.axvline(0.5, color=MUTED, lw=1, ls='--')
    ax.set_xlabel('per-run AUROC (low density -> failure)', fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False)
    style(ax)
    fig.savefig(out / 'auroc_hist.png', dpi=130)
    plt.close(fig)
    print(f'wrote {out}/{{episodes.csv,runs.csv,summary.md,summary.json,goals_on_density.png,success_vs_density.png,auroc_hist.png}}')


if __name__ == '__main__':
    main()
