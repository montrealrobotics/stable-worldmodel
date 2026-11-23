"""Planning success by the T's rotation |dyaw| between the start and goal state.

Pools all g/h cells of each checkpoint; the rotation comes from the expert goal
cache (see eval_episode_tasks.py). Run on Mila, where results and cache live:

  python scripts/data/eval_yaw_bins.py inbatch=11811959:20 cem_s1=27001812:20 base_s1=27001817:20
"""

import argparse

import numpy as np

from eval_episode_tasks import CACHE_DIR, EVAL_ROOT, load_runs, parse_runs

FINE = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, np.inf)]
COARSE = [(0, 0.3), (0.3, 0.8), (0.8, np.inf)]


def rate(rows, cond):
    x = [r['ok'] for r in rows if cond(r['dyaw'])]
    return 100 * np.mean(x) if x else float('nan'), len(x)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='name=run_id:epoch, e.g. inbatch=11811959:20')
    p.add_argument('--cutoff', type=float, default=0.8, help='high-rotation threshold (rad)')
    p.add_argument('--eval-root', default=EVAL_ROOT)
    p.add_argument('--cache-dir', default=CACHE_DIR)
    args = p.parse_args()
    data = load_runs(parse_runs(args.runs), args.cache_dir, args.eval_root)

    for bins in (FINE, COARSE):
        print('\n|dyaw| rad   ' + ' '.join(f'{a:.1f}-{b:<4}'.replace('inf', '') for a, b in bins))
        for name, rows in data.items():
            print(f'{name:12s} ' + ' '.join(f'{rate(rows, lambda d: a <= d < b)[0]:8.1f}' for a, b in bins))
        first = next(iter(data.values()))
        print('n/run        ' + ' '.join(f'{rate(first, lambda d: a <= d < b)[1]:8d}' for a, b in bins))

    c = args.cutoff
    print(f"\n{'run':12s} {'all':>6s} {f'<{c}':>6s} {'<0.3':>6s} {f'>={c}':>6s}   n(all/<{c})")
    for name, rows in data.items():
        print(
            f'{name:12s} {rate(rows, lambda d: True)[0]:6.1f} {rate(rows, lambda d: d < c)[0]:6.1f} '
            f'{rate(rows, lambda d: d < 0.3)[0]:6.1f} {rate(rows, lambda d: d >= c)[0]:6.1f}   '
            f'{len(rows)}/{rate(rows, lambda d: d < c)[1]}'
        )


if __name__ == '__main__':
    main()
