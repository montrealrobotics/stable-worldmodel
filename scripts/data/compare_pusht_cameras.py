"""Re-render collected PushT states in the stock and overhead envs, side by side.

Loads saved simulator states (`state` column) from a dataset collected with
OverheadPushT-v1 and renders them, without stepping physics, in:
  saved     : the dataset's own pixels (JPEG)
  ovh_base  : OverheadPushT-v1 base_camera (should match `saved`)
  stock_base: stock PushT-v1 base_camera (ManiSkill robot, table, lighting)
  stock_hr  : stock PushT-v1 human render camera (what ManiSkill videos use)
  ovh_fov1  : OverheadPushT-v1 base_camera with the human render camera's fov
Writes one PNG per episode plus a JSON of per-view mean pixel differences.

Example (GPU node, Vulkan):
    python scripts/data/compare_pusht_cameras.py \
        --dataset $SCRATCH/le-wm/datasets/basecam_smoke.lance --out-dir camera_compare
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument(
        '--episodes', type=int, nargs='+', default=[0, 1, 2, 5, 40, 90]
    )
    p.add_argument(
        '--size', type=int, default=256, help='Tile size in the PNGs.'
    )
    return p.parse_args()


def make_env(env_id, **kwargs):
    import gymnasium as gym

    return gym.make(
        env_id,
        num_envs=1,
        obs_mode='rgb',
        render_mode='rgb_array',
        sim_backend='physx_cpu',
        **kwargs,
    ).unwrapped


def to_uint8(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x[0] if x.ndim == 4 else x


def render(env, state, qpos, obj_pose, human=False):
    import torch
    from mani_skill.utils.structs.pose import Pose

    state = torch.as_tensor(state, dtype=torch.float32)[None]
    if env.get_state().shape[-1] == state.shape[-1]:
        env.set_state(state)
    else:  # different entity layout: set robot joints and T pose directly
        env.agent.robot.set_qpos(torch.as_tensor(qpos)[None])
        env.tee.set_pose(Pose.create(torch.as_tensor(obj_pose)[None]))
    obs = env.get_obs()
    base = to_uint8(obs['sensor_data']['base_camera']['rgb'])
    if human:
        human = to_uint8(env.render_rgb_array('render_camera'))
    return base, human


def main():
    args = parse_args()
    import lance
    import mani_skill.envs  # noqa: F401
    from PIL import Image

    from stable_worldmodel.envs.maniskill import overhead_pusht  # noqa: F401

    table = (
        lance.dataset(str(args.dataset))
        .to_table(
            columns=[
                'episode_idx',
                'step_idx',
                'state',
                'obs_agent_qpos',
                'obs_extra_obj_pose',
                'pixels',
            ],
            filter='episode_idx IN (%s)' % ','.join(map(str, args.episodes)),
        )
        .to_pydict()
    )
    ep = np.array(table['episode_idx'])
    st = np.array(table['step_idx'])

    stock = make_env('PushT-v1')
    ovh = make_env(
        'OverheadPushT-v1',
        control_mode='pd_ee_delta_xy',
        include_base_camera=True,
    )
    ovh_fov1 = make_env(
        'OverheadPushT-v1',
        control_mode='pd_ee_delta_xy',
        include_base_camera=True,
        sensor_configs={
            'base_camera': {'fov': 1.0, 'width': 256, 'height': 256}
        },
    )
    for e in (stock, ovh, ovh_fov1):
        e.reset(seed=0)
    print(
        'state sizes: stock %d, overhead %d, dataset %d'
        % (
            stock.get_state().shape[-1],
            ovh.get_state().shape[-1],
            len(table['state'][0]),
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ['saved', 'ovh_base', 'stock_base', 'stock_hr', 'ovh_fov1']
    diffs = {n: [] for n in ('ovh_base_vs_saved', 'stock_base_vs_saved')}
    tile = lambda a: np.array(
        Image.fromarray(a).resize((args.size, args.size), Image.BILINEAR)
    )
    for e in args.episodes:
        idx = np.where(ep == e)[0]
        idx = idx[np.argsort(st[idx])]
        if len(idx) == 0:
            continue
        rows = []
        for i in (idx[0], idx[len(idx) // 2], idx[-1]):
            saved = np.array(
                Image.open(io.BytesIO(table['pixels'][i])).convert('RGB')
            )
            s, q, o = (
                table['state'][i],
                table['obs_agent_qpos'][i],
                table['obs_extra_obj_pose'][i],
            )
            ob, _ = render(ovh, s, q, o)
            sb, shr = render(stock, s, q, o, human=True)
            f1, _ = render(ovh_fov1, s, q, o)
            diffs['ovh_base_vs_saved'].append(
                float(np.abs(ob.astype(int) - saved).mean())
            )
            diffs['stock_base_vs_saved'].append(
                float(np.abs(sb.astype(int) - saved).mean())
            )
            rows.append(
                np.concatenate([tile(x) for x in (saved, ob, sb, shr, f1)], 1)
            )
        Image.fromarray(np.concatenate(rows, 0)).save(
            args.out_dir / f'episode_{e:03d}.png'
        )
    summary = {k: float(np.mean(v)) for k, v in diffs.items()}
    summary['columns'] = names
    (args.out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
