"""Pilot: do stochastic expert chunks from the expert's state reach its next state?

Each group of --candidates GPU envs resets with the same seed. Slot 0 runs the
deterministic expert; slot 1 also runs it, as a check that restoring the state
reproduces slot 0 exactly. The other slots sample the expert's Gaussian with
different seeds and std scales (round-robin over --std-scales). After every
chunk of --chunk control steps (one LeWM frameskip action), each slot's
end state is compared with slot 0's, then all slots are reset to slot 0's
state, so every chunk starts from the expert trajectory.

Example:
    python scripts/data/pilot_correct_chunks.py --checkpoint runs/RUN/final_ckpt.pt \
        --output pilot.npz --num-groups 64 --candidates 16
"""

import argparse
import json
from pathlib import Path

from stable_worldmodel.envs.maniskill.chunks import (
    TOLERANCES,
    accepted,
    end_errors,
    restore_from_source,
    snapshot,
    wrap,
)
from stable_worldmodel.envs.maniskill.experts import (
    RGBExpert,
    collection_defaults,
    validate_collection_checkpoint,
)
from stable_worldmodel.envs.maniskill.observations import policy_observation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--output', type=Path, required=True, help='Per-candidate rows (.npz).'
    )
    parser.add_argument(
        '--num-groups', type=int, default=64, help='Expert episodes in total.'
    )
    parser.add_argument(
        '--candidates', type=int, default=16, help='Slots per expert episode.'
    )
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--chunk', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument(
        '--std-scales', type=float, nargs='+', default=(0.5, 1.0, 2.0)
    )
    parser.add_argument('--seed', type=int, default=70000)
    args = parser.parse_args()
    if args.candidates < 3 or args.num_envs % args.candidates:
        parser.error('--candidates must be >= 3 and divide --num-envs')
    return args


def summarize(rows, std_scales):
    import numpy as np

    summary = {
        'determinism_check': {
            key: float(np.abs(rows[key][rows['slot'] == 1]).max())
            for key in ('tee_mm', 'yaw_deg', 'tcp_mm')
        }
    }
    sampled = rows['slot'] >= 2
    for contact in (False, True):
        for scale in std_scales:
            mask = (
                sampled
                & (rows['contact'] == contact)
                & np.isclose(rows['std_scale'], scale)
            )
            if not mask.any():
                continue
            entry = {'chunks': int(mask.sum())}
            for key in ('tee_mm', 'yaw_deg', 'tcp_mm', 'action_rms'):
                entry[key + '_p50_p90'] = [
                    float(v) for v in np.percentile(rows[key][mask], (50, 90))
                ]
            for name, tolerance in TOLERANCES.items():
                ok = mask & accepted(rows, tolerance)
                entry[f'accept_{name}'] = float(ok.sum() / mask.sum())
                entry[f'action_rms_accepted_{name}'] = (
                    float(rows['action_rms'][ok].mean()) if ok.any() else None
                )
            summary[f'{"contact" if contact else "free"}_std{scale:g}'] = entry
    return summary


def main():
    args = parse_args()
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - register environments
    import numpy as np
    import sapien
    import torch
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from stable_worldmodel.envs.maniskill import (
        overhead_pusht,  # noqa: F401
        random_goal_pusht,  # noqa: F401
    )

    device = torch.device('cuda')
    # Settings of overhead_rgb_noisy_replay_*; a training_config.json overrides them.
    defaults = dict(
        env_id='OverheadPushT-v1',
        control_mode='pd_ee_delta_xy',
        reward_mode='normalized_dense',
        encoder='spatial_softmax',
        std_mode='learned',
        initial_std=0.3,
        min_std=0.1,
        max_std=0.5,
    )
    inferred, config = collection_defaults(args.checkpoint)
    if inferred.get('policy_type', 'rgb') != 'rgb':
        raise ValueError('Pilot expects the overhead RGB PPO checkpoint')
    defaults.update(inferred)
    sensor_configs = {'width': 128, 'height': 128}
    if config:
        sensor_configs = {}
        for name, camera in config['environment']['cameras'].items():
            settings = dict(camera)
            pose = settings.pop('pose', None)
            if pose is not None:
                settings['pose'] = sapien.Pose(p=pose['p'], q=pose['q'])
            sensor_configs[name] = settings
    env = gym.make(
        defaults['env_id'],
        num_envs=args.num_envs,
        obs_mode='rgb+state_dict',
        render_mode='rgb_array',
        sim_backend='physx_cuda',
        control_mode=defaults['control_mode'],
        reward_mode=defaults['reward_mode'],
        reconfiguration_freq=0,
        sensor_configs=sensor_configs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=False)
    uw = env.unwrapped
    weights = torch.load(
        args.checkpoint, map_location='cpu', weights_only=True
    )
    validate_collection_checkpoint(
        weights, 'rgb', env.single_action_space.shape
    )
    obs, _ = env.reset(seed=args.seed)
    agent = RGBExpert(
        env,
        sample_obs=policy_observation(obs, device, 'rgb'),
        encoder=defaults['encoder'],
        std_mode=defaults['std_mode'],
        initial_std=defaults['initial_std'],
        min_std=defaults['min_std'],
        max_std=defaults['max_std'],
    ).to(device)
    agent.load_state_dict(weights)
    agent.constrain_std()
    agent.eval()
    low = torch.as_tensor(env.single_action_space.low, device=device)
    high = torch.as_tensor(env.single_action_space.high, device=device)
    half_range = (high - low) / 2

    n, slots = args.candidates, torch.arange(args.num_envs, device=device)
    source = slots // n * n
    deterministic = slots % n < 2
    scale = torch.ones(args.num_envs, device=device)
    scale[~deterministic] = torch.as_tensor(args.std_scales, device=device)[
        (slots[~deterministic] % n - 2) % len(args.std_scales)
    ]
    noise_rng = torch.Generator(device=device).manual_seed(args.seed)
    rows = {
        key: []
        for key in (
            'group',
            'slot',
            'step',
            'std_scale',
            'contact',
            'tee_mm',
            'yaw_deg',
            'tcp_mm',
            'tee_vel',
            'qvel',
            'action_rms',
        )
    }
    groups_per_batch = args.num_envs // n
    with torch.no_grad():
        std = torch.exp(agent.get_logstd())[0]
        print(f'Policy std (action units): {std.cpu().tolist()}', flush=True)
        for first in range(0, args.num_groups, groups_per_batch):
            seeds = [
                args.seed + first + slot // n for slot in range(args.num_envs)
            ]
            obs, _ = env.reset(seed=seeds)
            alive = torch.ones(args.num_envs, dtype=torch.bool, device=device)
            for step in range(0, args.max_steps - args.chunk + 1, args.chunk):
                obs = restore_from_source(uw, source)
                start = snapshot(uw)
                actions, done = (
                    [],
                    torch.zeros(
                        args.num_envs, dtype=torch.bool, device=device
                    ),
                )
                for _ in range(args.chunk):
                    mean = agent.get_action(
                        policy_observation(obs, device, 'rgb'),
                        deterministic=True,
                    )
                    noise = torch.randn(
                        mean.shape, device=device, generator=noise_rng
                    )
                    action = mean + torch.where(
                        deterministic[:, None], 0, noise * std * scale[:, None]
                    )
                    action = torch.clamp(action, min=low, max=high)
                    obs, _, terminated, truncated, _ = env.step(action)
                    done |= terminated.bool() | truncated.bool()
                    actions.append(action)
                end = snapshot(uw)
                keep = alive & ~done[source]
                alive &= ~done[source]
                if not keep.any():
                    break
                actions = torch.stack(actions, 1)
                moved = (
                    (end['tee_xy'] - start['tee_xy']).norm(dim=1) > 1e-3
                ) | (
                    wrap(end['tee_yaw'] - start['tee_yaw']).abs()
                    > np.deg2rad(0.5)
                )
                values = dict(
                    group=first + slots // n,
                    slot=slots % n,
                    step=torch.full_like(slots, step),
                    std_scale=torch.where(deterministic, 0, scale),
                    contact=moved[source],
                    **end_errors(end, source),
                    tee_vel=(end['tee_vel'] - end['tee_vel'][source]).norm(
                        dim=1
                    ),
                    qvel=(end['qvel'] - end['qvel'][source]).norm(dim=1),
                    action_rms=(
                        ((actions - actions[source]) / half_range) ** 2
                    )
                    .mean((1, 2))
                    .sqrt(),
                )
                keep &= slots % n != 0
                for key, value in values.items():
                    rows[key].append(value[keep].cpu().numpy())
            print(
                f'Groups {min(first + groups_per_batch, args.num_groups)}/{args.num_groups}',
                flush=True,
            )
    env.close()
    rows = {key: np.concatenate(value) for key, value in rows.items()}
    rows = {
        key: value[rows['group'] < args.num_groups]
        for key, value in rows.items()
    }
    np.savez(args.output, **rows)
    summary = summarize(rows, args.std_scales)
    summary.update(
        policy_std=std.cpu().tolist(),
        args={k: str(v) for k, v in vars(args).items()},
    )
    args.output.with_suffix('.json').write_text(
        json.dumps(summary, indent=2) + '\n'
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
