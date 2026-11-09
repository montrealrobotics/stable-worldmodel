"""Side-by-side videos of each expert episode and its alternatives.

Works for --alternatives (whole trajectories) and --correct-chunks datasets.

Example:
    python scripts/visualization/alternative_videos.py \
        --dataset reports/correct_chunks_test/data.lance \
        --output reports/correct_chunks_test/videos
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.plot.video_utils import save_panel_videos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    dataset = swm.data.load_dataset(
        str(Path(args.dataset).resolve()), keys_to_load=['pixels', 'action']
    )
    info = dataset.get_episode_data()
    groups = defaultdict(dict)
    fallbacks = info.get('fallback_chunks') or [None] * len(info['group_idx'])
    for ep, (group, replay, fallback) in enumerate(
        zip(info['group_idx'], info['replay_idx'], fallbacks)
    ):
        groups[int(group)][int(replay)] = (ep, fallback)

    for group, members in sorted(groups.items()):
        panels, actions = {}, {}
        for replay, (ep, fallback) in sorted(members.items()):
            steps = dataset.load_episode(ep)
            frames = np.asarray(steps['pixels'])
            if frames.shape[1] == 3:  # CHW -> HWC
                frames = frames.transpose(0, 2, 3, 1)
            chunks = -(-len(frames) // 5)
            label = 'expert' if replay < 0 else f'alt {replay}'
            if replay >= 0 and fallback is not None:
                label += f' ({chunks - int(fallback)}/{chunks} swapped)'
            label += f' T={len(frames)}'
            panels[label] = [frames.astype(np.uint8)]
            actions[label] = np.asarray(steps['action'])
        expert = next(iter(actions.values()))
        for label, action in actions.items():
            steps = min(len(action), len(expert))
            diff = np.sqrt(((action[:steps] - expert[:steps]) ** 2).mean())
            print(
                f'group {group} {label}: action RMS diff vs expert '
                f'(first {steps} steps) {diff:.3f}'
            )
        save_panel_videos(args.output / f'group_{group}', panels, fps=args.fps)
    print(f'Saved videos under {args.output}')


if __name__ == '__main__':
    main()
