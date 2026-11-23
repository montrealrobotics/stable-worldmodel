"""How much training data looks like the eval tasks, and the failed ones in particular.

A dataset *window* is a frame t and the frame t + g of the same episode, for
the eval goal offsets g (10, 15, 20, 25 raw steps): the same shape as an eval
task (start state -> goal state g expert steps later). For every window the
same task features as eval_episode_tasks.py are computed (T rotation, T
displacement, EEF-to-T distance at the goal, goal step, static T), so the
dataset and the eval tasks are binned on the same edges.

Per eval task (seed, g), from the expert goal cache:
  similar_windows    windows of the same g whose start T pose AND goal T pose
                     are both within the success tolerance of the task's
                     (20 mm, 0.15 rad; T pose only, EEF ignored)
  similar_windows_2x same with twice the tolerance
  start_frames       dataset frames with the T within tolerance of the start pose
  goal_frames        dataset frames with the T within tolerance of the goal pose
  success_<run>      planning success over the task's eval episodes (all horizons)
Counts are split into expert / replay episodes (episode table replay_idx).

Writes to --out: tasks.csv, windows.json (bin edges + counts), summary.md.
Plot with scripts/data/plot_task_coverage.py. Run on Mila (CPU job:
scripts/cluv/dataset_task_coverage.sbatch).
"""

import argparse
import csv
import json
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree

from eval_episode_tasks import CACHE_DIR, EVAL_ROOT, XY_TOL_MM, YAW_TOL_RAD, load_runs, parse_runs

DATASET = '/network/scratch/o/ozgur.aslan/le-wm/datasets/overhead_rgb_noisy_replay_100k.lance'
DEFAULT_RUNS = [
    'inbatch=11811959:20',
    'cem_s0=26561860:20',
    'cem_s1=27001812:20',
    'cem_s2=27001833:20',
    'base_s1=27001817:20',
    'base_s2=27001843:20',
]
# Fixed, explainable bin edges (same units as eval_episode_tasks: mm / rad / steps).
BINS = {
    'dyaw': [0, 0.2, 0.5, 0.8, 1.2, 1.6, np.inf],
    'dxy': [0, 5, 30, 60, 100, np.inf],
    'eef_T_goal': [0, 50, 100, 150, np.inf],
    'goal_step': [0, 20, 30, 40, 50, np.inf],
    'moved': [0, 0.5, 1.5],  # 0 = static T, 1 = T moves
}
MAP_EDGES = {'dyaw': np.linspace(0, np.pi, 37).tolist(), 'dxy': np.linspace(0, 200, 41).tolist()}


def as_matrix(col):
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    flat = col.flatten()
    return np.asarray(flat.to_numpy(zero_copy_only=False), np.float32).reshape(len(col), -1)


def load_frames(path, max_episodes=None):
    """Frames sorted by (episode, step): episode, T pose, TCP pose; plus expert mask."""
    ds = lance.dataset(path)
    flt = f'episode_idx < {max_episodes}' if max_episodes else None
    parts = {k: [] for k in ('episode_idx', 'step_idx', 'obj', 'tcp')}
    cols = ['episode_idx', 'step_idx', 'obs_extra_obj_pose', 'obs_extra_tcp_pose']
    for b in ds.to_batches(columns=cols, filter=flt, batch_size=262144):
        parts['episode_idx'].append(b.column('episode_idx').to_numpy())
        parts['step_idx'].append(b.column('step_idx').to_numpy())
        parts['obj'].append(as_matrix(b.column('obs_extra_obj_pose')))
        parts['tcp'].append(as_matrix(b.column('obs_extra_tcp_pose')))
    d = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.lexsort((d['step_idx'], d['episode_idx']))
    d = {k: v[order] for k, v in d.items()}
    ep_path = Path(path).with_name(Path(path).stem + '_episodes.lance')
    table = lance.dataset(str(ep_path)).to_table(columns=['episode_idx', 'replay_idx']).to_pydict()
    expert_eps = {e for e, r in zip(table['episode_idx'], table['replay_idx']) if r == -1}
    d['expert'] = np.isin(d['episode_idx'], np.fromiter(expert_eps, np.int64))
    return d


def yaw_of(q):
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tee_embed(pose):
    """(n, 7) T pose (m, quat) -> tolerance units: 1 = the success tolerance."""
    pose = np.atleast_2d(pose)
    yaw = yaw_of(pose)
    r = 1.0 / YAW_TOL_RAD  # chord ~ |dyaw| / tol for small angles
    return np.stack(
        [1e3 * pose[:, 0] / XY_TOL_MM, 1e3 * pose[:, 1] / XY_TOL_MM, r * np.cos(yaw), r * np.sin(yaw)], -1
    )


def window_features(d, g):
    """Indices t with t + g in the same episode, and their task features."""
    ep = d['episode_idx']
    t = np.nonzero(ep[:-g] == ep[g:])[0]
    obj, tcp = d['obj'], d['tcp']
    s_yaw, g_yaw = yaw_of(obj[t]), yaw_of(obj[t + g])
    dyaw = np.abs((g_yaw - s_yaw + np.pi) % (2 * np.pi) - np.pi)
    dxy = 1e3 * np.linalg.norm(obj[t + g, :2] - obj[t, :2], axis=1)
    f = {
        'dyaw': dyaw,
        'dxy': dxy,
        'eef_T_goal': 1e3 * np.linalg.norm(tcp[t + g, :2] - obj[t + g, :2], axis=1),
        'goal_step': d['step_idx'][t + g].astype(np.float64),
        'moved': ((dxy > 5) | (dyaw > 0.05)).astype(np.float64),
    }
    return t, f


def count_split(idx, expert):
    idx = np.asarray(idx, np.int64)
    n_exp = int(expert[idx].sum()) if len(idx) else 0
    return len(idx), n_exp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='*', default=DEFAULT_RUNS, help='name=run_id:epoch (default: the ep20 runs)')
    p.add_argument('--dataset', default=DATASET)
    p.add_argument('--eval-root', default=EVAL_ROOT)
    p.add_argument('--cache-dir', default=CACHE_DIR)
    p.add_argument('--max-episodes', type=int, default=None, help='quick test: first N episodes only')
    p.add_argument('--out', default='reports/dataset_task_coverage')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(args.runs)
    data = load_runs(runs, args.cache_dir, args.eval_root)
    tasks = {}  # (seed, g) -> features + per-run success
    for name, rows in data.items():
        for r in rows:
            t = tasks.setdefault((r['seed'], r['g']), {k: v for k, v in r.items() if k not in ('ok', 'h', 'xy', 'yaw', 'eef', 'start')})
            t.setdefault('_ok', {}).setdefault(name, []).append(r['ok'])
    offsets = sorted({g for _, g in tasks})
    print(f'{len(tasks)} eval tasks, offsets {offsets}')

    d = load_frames(args.dataset, args.max_episodes)
    n_frames, n_eps = len(d['episode_idx']), len(np.unique(d['episode_idx']))
    print(f'{n_frames:,} frames, {n_eps:,} episodes, expert frames {d["expert"].mean():.1%}')

    # Pose-only frame density (any frame), for start and goal poses.
    frame_tree = cKDTree(tee_embed(d['obj']))
    hist = {k: {'all': np.zeros(len(e) - 1), 'expert': np.zeros(len(e) - 1)} for k, e in BINS.items()}
    eval_hist = {k: np.zeros(len(e) - 1) for k, e in BINS.items()}
    map_counts = np.zeros((len(MAP_EDGES['dyaw']) - 1, len(MAP_EDGES['dxy']) - 1))
    n_windows = {'all': 0, 'expert': 0}
    for g in offsets:
        t, f = window_features(d, g)
        exp = d['expert'][t]
        n_windows['all'] += len(t)
        n_windows['expert'] += int(exp.sum())
        for k, edges in BINS.items():
            hist[k]['all'] += np.histogram(f[k], edges)[0]
            hist[k]['expert'] += np.histogram(f[k][exp], edges)[0]
        map_counts += np.histogram2d(f['dyaw'], f['dxy'], [MAP_EDGES['dyaw'], MAP_EDGES['dxy']])[0]  # > 200 mm left out
        # Start + goal T pose of every window, in tolerance units (8D).
        win_tree = cKDTree(np.concatenate([tee_embed(d['obj'][t]), tee_embed(d['obj'][t + g])], 1))
        for (seed, gg), task in tasks.items():
            if gg != g:
                continue
            q_s, q_g = tee_embed(task['start_pose'])[0], tee_embed(task['goal_pose'])[0]
            q = np.r_[q_s, q_g]
            for scale, key in ((1.0, 'similar_windows'), (2.0, 'similar_windows_2x')):
                # Ball of radius sqrt(2) * scale holds every window with both halves within scale; filter exactly.
                cand = np.asarray(win_tree.query_ball_point(q, np.sqrt(2) * scale), np.int64)
                if len(cand):
                    pts = win_tree.data[cand]
                    keep = (np.linalg.norm(pts[:, :4] - q_s, axis=1) <= scale) & (np.linalg.norm(pts[:, 4:] - q_g, axis=1) <= scale)
                    cand = cand[keep]
                n, n_exp = count_split(cand, exp)
                task[key], task[key + '_expert'] = n, n_exp
            for pose_key, key in (('start_pose', 'start_frames'), ('goal_pose', 'goal_frames')):
                idx = frame_tree.query_ball_point(tee_embed(task[pose_key])[0], 1.0)
                task[key], task[key + '_expert'] = count_split(idx, d['expert'])
        print(f'g={g}: {len(t):,} windows')
    for task in tasks.values():
        for k, e in BINS.items():
            eval_hist[k] += np.histogram([task[k]], e)[0]

    # tasks.csv
    feats = ['dyaw', 'dxy', 'path_xy', 'deef', 'eef_T_start', 'eef_T_goal', 'goal_step', 'moved']
    counts = [
        f'{c}{s}' for c in ('similar_windows', 'similar_windows_2x', 'start_frames', 'goal_frames') for s in ('', '_expert')
    ]
    names = list(runs)
    with open(out / 'tasks.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'g', *feats, *counts, *[f'success_{n}' for n in names], *[f'episodes_{n}' for n in names]])
        for (seed, g), t in sorted(tasks.items()):
            ok = t['_ok']
            w.writerow(
                [seed, g, *[float(t[k]) for k in feats], *[t[c] for c in counts]]
                + [np.mean(ok[n]) if n in ok else '' for n in names]
                + [len(ok.get(n, [])) for n in names]
            )
    as_list = lambda e: [x if np.isfinite(x) else None for x in e]  # noqa: E731
    json.dump(
        {
            'dataset': args.dataset,
            'frames': n_frames,
            'episodes': n_eps,
            'offsets': offsets,
            'windows': n_windows,
            'runs': {n: list(v) for n, v in runs.items()},
            'bins': {k: as_list(e) for k, e in BINS.items()},
            'hist': {k: {s: v.tolist() for s, v in h.items()} for k, h in hist.items()},
            'eval_hist': {k: v.tolist() for k, v in eval_hist.items()},
            'map': {'edges': MAP_EDGES, 'counts': map_counts.tolist()},
        },
        open(out / 'windows.json', 'w'),
        indent=1,
    )

    # summary.md: dataset share vs eval share vs success, per feature bin
    lines = [
        '# Dataset coverage of eval tasks',
        '',
        f'- dataset: `{args.dataset}`: {n_frames:,} frames, {n_eps:,} episodes; '
        f'{n_windows["all"]:,} windows for g in {offsets} ({n_windows["expert"] / max(1, n_windows["all"]):.0%} from expert episodes)',
        f'- eval tasks: {len(tasks)}; runs: {", ".join(f"{n} ({r[0]} ep{r[1]})" for n, r in runs.items())}',
        '',
    ]
    for k, edges in BINS.items():
        lines += [f'## {k}', '', '| bin | data % | expert data % | eval tasks % | ' + ' | '.join(names) + ' |',
                  '|---|---|---|---|' + '---|' * len(names)]
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = [t for t in tasks.values() if lo <= t[k] < hi]
            succ = [f'{100 * np.mean([np.mean(t["_ok"][n]) for t in sel if n in t["_ok"]]):.1f}' if sel else '–' for n in names]
            lines.append(
                f'| {lo:g}–{hi:g} | {100 * hist[k]["all"][i] / n_windows["all"]:.1f} | '
                f'{100 * hist[k]["expert"][i] / max(1, n_windows["expert"]):.1f} | '
                f'{100 * eval_hist[k][i] / len(tasks):.1f} | ' + ' | '.join(succ) + ' |'
            )
        lines.append('')
    sim = np.array([t['similar_windows'] for t in tasks.values()])
    lines += [
        '## Similar windows per task (start and goal T pose within tolerance, same g)',
        '',
        f'- median {np.median(sim):.0f}, zero for {np.mean(sim == 0):.0%} of tasks, < 10 for {np.mean(sim < 10):.0%}',
    ]
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
