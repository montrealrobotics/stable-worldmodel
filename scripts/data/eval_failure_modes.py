"""What the failed eval episodes look like, apart from large T rotations.

For episodes with |dyaw| below --max-dyaw (start -> goal T rotation):
  1. which success criterion fails: T (xy 20 mm / yaw 0.15 rad), EEF (20 mm),
     or both; final error sizes, near misses vs far misses;
  2. success by tercile of task features (T displacement, path length, EEF
     travel, EEF-to-T distance, goal step, ...), by g, h, and static-T tasks;
  3. static-T tasks (the expert only moves the EEF) separately.

Run on Mila, where results and cache live:

  python scripts/data/eval_failure_modes.py inbatch=11811959:20 cem_s1=27001812:20 base_s1=27001817:20
"""

import argparse

import numpy as np

from eval_episode_tasks import CACHE_DIR, EEF_TOL_MM, EVAL_ROOT, XY_TOL_MM, YAW_TOL_RAD, load_runs, parse_runs

FEATURES = ['dxy', 'path_xy', 'deef', 'eef_T_start', 'eef_T_goal', 'T_r_goal', 'goal_step', 'start', 'dyaw']


def t_off(r):
    return r['xy'] > XY_TOL_MM or r['yaw'] > YAW_TOL_RAD


def criteria(rows, label):
    fails = [r for r in rows if not r['ok']]
    if not fails:
        return f'{label:22s} n={len(rows):4d} no failures'
    tf = np.array([t_off(r) for r in fails])
    ef = np.array([r['eef'] > EEF_TOL_MM for r in fails])
    eef_only = [r['eef'] for r in fails if r['eef'] > EEF_TOL_MM and not t_off(r)]
    return (
        f'{label:22s} n={len(rows):4d} succ {100 * np.mean([r["ok"] for r in rows]):4.1f}%  '
        f'fails: T only {np.mean(tf & ~ef):.0%}, EEF only {np.mean(~tf & ef):.0%}, both {np.mean(tf & ef):.0%}'
        f'  | EEF-only fails: EEF err median {np.median(eef_only) if eef_only else float("nan"):.0f} mm'
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='name=run_id:epoch; the first run sets the tercile edges')
    p.add_argument('--max-dyaw', type=float, default=0.8, help='keep episodes below this rotation (rad)')
    p.add_argument('--eval-root', default=EVAL_ROOT)
    p.add_argument('--cache-dir', default=CACHE_DIR)
    args = p.parse_args()
    data = load_runs(parse_runs(args.runs), args.cache_dir, args.eval_root)
    low = {n: [r for r in rows if r['dyaw'] < args.max_dyaw and 'xy' in r] for n, rows in data.items()}

    # 1. failing criterion and error sizes
    for name, rows in low.items():
        fails = [r for r in rows if not r['ok']]
        bad = sum(r['ok'] and r['eef'] > EEF_TOL_MM for r in rows)
        print(f'\n[{name}] {len(rows)} episodes with |dyaw| < {args.max_dyaw}, {len(fails)} failed')
        if bad:
            print(f'  NOTE: {bad} successes have EEF error > {EEF_TOL_MM} mm (EEF may not be in the criterion)')
        print('  ' + criteria(rows, 'all'))
        print('  ' + criteria([r for r in rows if r['moved']], 'T must move'))
        static = [r for r in rows if not r['moved']]
        print('  ' + criteria(static, 'T static (reach only)'))
        sf = [r for r in static if not r['ok']]
        if sf:
            print(f'  static-T fails where the planner moved the T off its pose: {np.mean([t_off(r) for r in sf]):.0%}')
        if fails:
            xy, yaw = np.array([r['xy'] for r in fails]), np.array([r['yaw'] for r in fails])
            print(
                f'  fail T xy err median {np.median(xy):.0f} mm (IQR {np.percentile(xy, 25):.0f}-{np.percentile(xy, 75):.0f}),'
                f' yaw err median {np.median(yaw):.2f} rad (q75 {np.percentile(yaw, 75):.2f})'
            )
            print(
                f'  near misses (xy < 30 mm and yaw < 0.25): {np.mean((xy < 30) & (yaw < 0.25)):.0%};'
                f'  far (xy > 60 mm or yaw > 0.5): {np.mean((xy > 60) | (yaw > 0.5)):.0%}'
            )

    # 2. success by feature tercile (edges from the first run)
    ref = next(iter(low.values()))
    print('\nsuccess % by tercile (low / mid / high); [tercile edges from the first run]')
    for f in FEATURES:
        if not all(f in r for r in ref):
            continue
        q = np.percentile([r[f] for r in ref], [100 / 3, 200 / 3])
        line = f'{f:12s} [{q[0]:6.1f},{q[1]:6.1f}] '
        for name, rows in low.items():
            v, ok = np.array([r[f] for r in rows]), np.array([r['ok'] for r in rows])
            b = np.digitize(v, q)
            line += f' {name}: ' + '/'.join(f'{100 * ok[b == k].mean():4.0f}' for k in range(3))
        print(line)
    for key in ('g', 'h', 'moved'):
        line = f'{key:12s} '
        for name, rows in low.items():
            vals = sorted({r[key] for r in rows})
            line += f' {name}: ' + ' '.join(
                f'{v}:{100 * np.mean([r["ok"] for r in rows if r[key] == v]):.0f}' for v in vals
            )
        print(line)


if __name__ == '__main__':
    main()
