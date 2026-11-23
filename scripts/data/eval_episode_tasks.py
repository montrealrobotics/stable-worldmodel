"""Shared loading for per-episode eval analyses: eval results x expert goal cache.

Each eval episode (reset seed, goal offset g, horizon h) is joined with the
expert rollout that defined its task in the goal cache, giving start / goal
poses of the T and the EEF. Used by eval_yaw_bins.py and eval_failure_modes.py.
"""

import glob
import pickle
import re
from pathlib import Path

import numpy as np
import yaml

EVAL_ROOT = '/network/scratch/o/ozgur.aslan/le-wm/evals/sched_n100'
CACHE_DIR = '/network/scratch/o/ozgur.aslan/le-wm/expert_goal_cache'
# Success needs all three (eval config: goal_position_tolerance,
# goal_yaw_tolerance, goal_eef_position_tolerance).
XY_TOL_MM, YAW_TOL_RAD, EEF_TOL_MM = 20.0, 0.15, 20.0


def parse_list(text, key):
    """A list or numpy array printed in the results dict, as floats."""
    m = re.search(r"'%s':\s*(?:array\()?\[(.*?)\]" % key, text, re.S)
    if not m:
        return None
    vals = m.group(1).replace('\n', ' ').split(',')
    return np.array(
        [float(v.strip().replace('np.float64(', '').rstrip(')')) for v in vals if v.strip()]
    )


def load_episodes(run_id, epoch, eval_root=EVAL_ROOT):
    """Episodes of every g/h cell of one checkpoint (last result in each file)."""
    rows = []
    pattern = f'{eval_root}/*_{run_id}_ep{epoch}_g*_h*/random_goal_pusht_results.txt'
    for f in sorted(glob.glob(pattern)):
        block = Path(f).read_text().split('==== CONFIG ====')[-1]
        if '==== RESULTS ====' not in block:
            continue
        cfg_text, res = block.split('==== RESULTS ====', 1)
        cfg = yaml.safe_load(cfg_text)
        g, h = int(cfg['eval']['goal_offset_steps']), int(cfg['plan_config']['horizon'])
        seeds, ok = parse_list(res, 'seeds'), parse_list(res, 'episode_successes')
        xy, yaw, eef = (
            parse_list(res, k)
            for k in ('episode_xy_error_m', 'episode_yaw_error_rad', 'episode_eef_position_error_m')
        )
        start = parse_list(res, 'start_steps')
        for i in range(len(seeds)):
            row = {'seed': int(seeds[i]), 'g': g, 'h': h, 'ok': bool(ok[i])}
            if xy is not None:  # older results files may lack per-episode errors
                row.update(xy=1e3 * xy[i], yaw=yaw[i], eef=1e3 * eef[i])
            if start is not None:
                row['start'] = int(start[i])
            rows.append(row)
    return rows


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_tasks(wanted, cache_dir=CACHE_DIR):
    """(seed, g) -> task features from the expert goal cache (mm / rad).

    Files are named by the requested seed; a retried task has a different
    accepted seed, and results may record either, so both are indexed.
    """
    tasks = {}
    wanted_seeds = {s for s, _ in wanted}
    for f in sorted(Path(cache_dir).glob('expert_*.pkl')):
        seed = int(f.name.split('_')[1])
        if seed not in wanted_seeds:
            continue
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        s, i = int(r['start_index']), int(r['index'])
        p, e = np.asarray(r['poses']), np.asarray(r['eef_positions'])
        dyaw = abs(wrap(quat_yaw(p[i, 3:7]) - quat_yaw(p[s, 3:7])))
        dxy = 1e3 * np.linalg.norm(p[i, :2] - p[s, :2])
        task = {
            'dyaw': dyaw,  # T rotation start -> goal
            'dxy': dxy,  # T net displacement
            'path_xy': 1e3 * np.abs(np.diff(p[s : i + 1, :2], axis=0)).sum(),  # T path length (L1)
            'deef': 1e3 * np.linalg.norm(e[i, :2] - e[s, :2]),
            'eef_T_start': 1e3 * np.linalg.norm(e[s, :2] - p[s, :2]),
            'eef_T_goal': 1e3 * np.linalg.norm(e[i, :2] - p[i, :2]),
            'T_r_goal': 1e3 * np.linalg.norm(p[i, :2]),  # goal T distance from the world origin
            'goal_step': i,
            'moved': dxy > 5 or dyaw > 0.05,  # False: the expert only moves the EEF
            # raw poses (m, xyz + wxyz quaternion) for dataset matching
            'start_pose': p[s], 'goal_pose': p[i], 'start_eef': e[s], 'goal_eef': e[i],
        }
        for key in ((seed, i - s), (int(r['seed']), i - s)):
            if key in wanted and key not in tasks:
                tasks[key] = task
    return tasks


def load_runs(runs, cache_dir=CACHE_DIR, eval_root=EVAL_ROOT):
    """{name: (run_id, epoch)} -> {name: episodes joined with their task features}.

    Episodes whose task is not in the cache are dropped (printed).
    """
    episodes = {name: load_episodes(rid, ep, eval_root) for name, (rid, ep) in runs.items()}
    wanted = {(r['seed'], r['g']) for rows in episodes.values() for r in rows}
    tasks = load_tasks(wanted, cache_dir)
    print(f'{len(wanted)} (seed, g) tasks, {len(set(tasks) & wanted)} found in the cache')
    out = {}
    for name, rows in episodes.items():
        out[name] = [{**r, **tasks[(r['seed'], r['g'])]} for r in rows if (r['seed'], r['g']) in tasks]
        print(f'  {name}: {len(rows)} episodes, {len(out[name])} matched')
    return out


def parse_runs(items):
    """['name=run_id:epoch', ...] -> {name: (run_id, epoch)}."""
    runs = {}
    for item in items:
        name, spec = item.split('=', 1)
        rid, ep = spec.split(':')
        runs[name] = (rid, int(ep))
    return runs
