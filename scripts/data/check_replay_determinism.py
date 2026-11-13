"""Does the GPU sim reproduce a recorded ManiSkill PushT batch?

Rebuilds the collector's env (collect_maniskill_pusht.py: same num_envs, obs
and reward mode), picks one recorded batch whose 64 expert episodes are all
in the dataset, and replays their recorded actions open loop:

  A1, A2  reset with the batch's seeds, every episode in its original slot,
          replay from step 0 (A2 repeats A1: run-to-run determinism).
  B       same seeds and slots, but restore the recorded state (and
          controller target) at step --restore-step and replay from there:
          what a mid-episode restore loses (solver warm start etc.).
  C1, C2  rebuild the scene (reconfigure reset), then restore as in B; twice.
          C2 == C1 means rebuilt-then-restored rollouts are reproducible.
  D       rebuild, then restore one clip into every slot with its actions:
          do identical rollouts in different slots agree?
  --save / --compare: A1 of two separate processes, each in a fresh sim.
          Bit-exact means the GPU sim is deterministic and A2 vs A1 drift
          comes from solver state carried over from the previous run.

Per 5-step block it prints the T position (mm), T yaw (deg) and TCP (mm)
errors against the dataset, plus the max abs flat-state difference
(0 = bit-exact). Writes nothing.

    python scripts/data/check_replay_determinism.py \
        --dataset overhead_rgb_noisy_replay_100k.lance --cache-dir $SLURM_TMPDIR \
        --collection-json $SCRATCH/le-wm/datasets/overhead_rgb_noisy_replay_100k.collection.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

TARGET_COL = 'obs_agent_controller_arm_target_pose'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--collection-json', required=True, type=Path)
    p.add_argument('--batch', type=int, default=0, help='index among complete batches')
    p.add_argument('--restore-step', type=int, default=10)
    p.add_argument('--save', type=Path, help='save the A1 states (.pt)')
    p.add_argument(
        '--compare',
        type=Path,
        help='A1 states saved by another process: fresh-vs-fresh check',
    )
    return p.parse_args()


def make_env(meta):
    """The collector's env, built the same way (collect_maniskill_pusht.main)."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from stable_worldmodel.envs.maniskill import (  # noqa: F401
        overhead_pusht,
        random_goal_pusht,
    )

    env = gym.make(
        meta['env_id'],
        num_envs=meta['actual_num_envs'],
        obs_mode=meta['obs_mode'],
        render_mode='rgb_array',
        sim_backend='physx_cpu' if meta['cpu'] else 'physx_cuda',
        control_mode=meta['control_mode'],
        reward_mode=meta['reward_mode'],
        reconfiguration_freq=meta['reconfiguration_freq'],
        sensor_configs={'width': meta['image_size'], 'height': meta['image_size']},
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, auto_reset=False, ignore_terminations=False)
    env.reset(seed=meta['seed'])  # the collector's first reset
    return env


def pick_batch(meta, which):
    """Dataset episode indices (by slot) of the which-th complete batch."""
    n = meta['actual_num_envs']
    batches = defaultdict(dict)
    for i, e in enumerate(meta['episodes']):
        if e.get('replay_idx', -1) == -1:
            batches[e['batch_seed']][e['env_slot']] = i
    complete = [b for b in sorted(batches) if len(batches[b]) == n]
    if not complete:
        raise ValueError('no batch has all of its expert episodes')
    seed = complete[which]
    return seed, [batches[seed][s] for s in range(n)]


def replay(env, seeds, actions, lengths, restore=None, reconfigure=False):
    """Recorded actions open loop; returns states (T + 1, N, D), NaN before
    the start. restore = (t0, states, targets) restores the slots at t0.
    reconfigure rebuilds the scene first, i.e. a fresh PhysX GPU system."""
    base = env.unwrapped
    env.reset(seed=seeds, options={'reconfigure': True} if reconfigure else None)
    t0 = 0
    if restore is not None:
        t0, states, targets = restore
        base.set_state(states)
        base.agent.set_controller_state({'arm': {'target_pose': targets}})
    T = actions.size(0)
    out = torch.full((T + 1, *base.get_state().shape), float('nan'), device=actions.device)
    for t in range(t0, T):
        out[t] = base.get_state()
        env.step(actions[t])
    out[T] = base.get_state()
    return out


def decode(base, states):
    """T xy, T yaw, TCP xy for (T + 1, N, D) states, via the env."""
    from stable_worldmodel.envs.maniskill.chunks import snapshot

    # NaN rows (before a restore, after an episode ends) are masked out later;
    # fill them with the nearest recorded state so the sim never sees NaN.
    states = states.clone()
    for t in range(1, len(states)):
        states[t] = torch.where(states[t].isnan(), states[t - 1], states[t])
    for t in range(len(states) - 2, -1, -1):
        states[t] = torch.where(states[t].isnan(), states[t + 1], states[t])
    tee, yaw, tcp = [], [], []
    for s in states:
        base.set_state(s)
        snap = snapshot(base)
        tee.append(snap['tee_xy'])
        yaw.append(snap['tee_yaw'])
        tcp.append(snap['tcp_xy'])
    return torch.stack(tee), torch.stack(yaw), torch.stack(tcp)


def errors(a, b, flat_a, flat_b, mask):
    from stable_worldmodel.envs.maniskill.chunks import wrap

    tee = 1e3 * (a[0] - b[0]).norm(dim=-1)
    yaw = torch.rad2deg(wrap(a[1] - b[1]).abs())
    tcp = 1e3 * (a[2] - b[2]).norm(dim=-1)
    flat = (flat_a - flat_b).abs().amax(-1)
    return [torch.where(mask, e, torch.nan) for e in (tee, yaw, tcp, flat)]


def main():
    args = parse_args()
    import stable_worldmodel as swm

    meta = json.loads(args.collection_json.read_text())
    n, device = meta['actual_num_envs'], torch.device('cuda')
    batch_seed, episodes = pick_batch(meta, args.batch)
    seeds = list(range(batch_seed, batch_seed + n))

    dataset = swm.data.load_dataset(
        args.dataset,
        transform=None,
        cache_dir=args.cache_dir,
        num_steps=1,
        frameskip=1,
        keys_to_load=['action', 'state', TARGET_COL],
    )
    eps = [dataset.load_episode(i) for i in episodes]
    lengths = torch.tensor([len(e['action']) for e in eps], device=device)
    for i, e in zip(episodes, eps):
        if len(e['action']) != meta['episodes'][i]['length']:
            raise ValueError('dataset and collection json are not aligned')
    T = int(lengths.max())
    actions = torch.zeros(T, n, eps[0]['action'].shape[-1], device=device)
    ref = torch.full((T + 1, n, eps[0]['state'].shape[-1]), float('nan'), device=device)
    targets = torch.zeros(n, eps[0][TARGET_COL].shape[-1], device=device)
    t0 = args.restore_step
    for s, e in enumerate(eps):
        L = len(e['action'])
        actions[:L, s] = torch.as_tensor(e['action'], device=device)
        ref[:L, s] = torch.as_tensor(e['state'], device=device)
        targets[s] = torch.as_tensor(e[TARGET_COL][min(t0, L - 1)], device=device)
    start_states = ref[torch.clamp(torch.full((n,), t0, device=device), max=lengths - 1), torch.arange(n, device=device)]

    env = make_env(meta)
    base = env.unwrapped
    runs = {
        'A1': replay(env, seeds, actions, lengths),
        'A2': replay(env, seeds, actions, lengths),
        'B': replay(env, seeds, actions, lengths, (t0, start_states, targets)),
    }
    # Test B: rebuild the scene before restoring (what the collector would do).
    restore = (t0, start_states, targets)
    runs['C1'] = replay(env, seeds, actions, lengths, restore, reconfigure=True)
    runs['C2'] = replay(env, seeds, actions, lengths, restore, reconfigure=True)
    # D: one clip restored into all slots with the same actions (slot effects).
    s0 = int(lengths.argmax())
    same = (
        t0,
        start_states[s0].expand(n, -1).contiguous(),
        targets[s0].expand(n, -1).contiguous(),
    )
    actions_d = actions[:, s0 : s0 + 1].expand(-1, n, -1).contiguous()
    runs['D'] = replay(env, seeds, actions_d, lengths, same, reconfigure=True)
    steps = torch.arange(T + 1, device=device)[:, None]
    valid = steps < lengths[None]  # recorded steps only
    poses = {k: decode(base, v) for k, v in runs.items()}
    ref_pose = decode(base, ref)
    table = {
        'A1 vs dataset': errors(poses['A1'], ref_pose, runs['A1'], ref, valid),
        'A2 vs A1': errors(poses['A2'], poses['A1'], runs['A2'], runs['A1'], valid),
        f'B (restore @{t0}) vs dataset': errors(
            poses['B'], ref_pose, runs['B'], ref, valid & (steps >= t0)
        ),
    }
    table[f'C2 vs C1 (reconfigure + restore @{t0}, twice)'] = errors(
        poses['C2'], poses['C1'], runs['C2'], runs['C1'], valid & (steps >= t0)
    )
    table['C1 vs dataset'] = errors(
        poses['C1'], ref_pose, runs['C1'], ref, valid & (steps >= t0)
    )
    first = [x[:, :1] for x in poses['D']]
    valid_d = ((steps >= t0) & (steps < lengths[s0])).expand(-1, n)
    table[f'D: slot {s0} clip in all {n} slots, each vs slot 0'] = errors(
        poses['D'], first, runs['D'], runs['D'][:, :1], valid_d
    )
    if args.compare is not None:
        other = torch.load(args.compare).to(device)
        table['A1 vs A1 of another fresh process'] = errors(
            poses['A1'], decode(base, other), runs['A1'], other, valid
        )
    if args.save is not None:
        torch.save(runs['A1'].cpu(), args.save)
    print(f'batch_seed {batch_seed}: {n} expert episodes, lengths {int(lengths.min())}-{T}')
    print('T mm / yaw deg / TCP mm as mean (max) over the episodes still running; flat = max |state diff|')
    for name, (tee, yaw, tcp, flat) in table.items():
        print(f'\n{name}\n  step | T mm          | yaw deg       | TCP mm        | flat')
        for t in range(0, T + 1, 5):
            if torch.isnan(tee[t]).all():
                continue
            row = [
                f'{x[t].nanmean().item():6.2f} ({x[t].nan_to_num(-1).max().item():6.2f})'
                for x in (tee, yaw, tcp)
            ]
            print(f'  {t:4d} | ' + ' | '.join(row) + f' | {flat[t].nan_to_num(-1).max().item():.2e}')
    env.close()


if __name__ == '__main__':
    main()
