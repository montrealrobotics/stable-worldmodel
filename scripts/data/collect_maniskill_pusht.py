"""Collect state- or RGB-policy ManiSkill rollouts as a SWM Lance dataset.

Example (run from the repo root):
    python scripts/data/collect_maniskill_pusht.py --checkpoint runs/RUN/final_ckpt.pt \
        --output-path datasets/maniskill_pusht_ppo.lance --num-episodes 1000

The environment runs in rgb+state_dict mode, or state_dict when a state policy
is collected with --no-save-rgb. State PPO receives a flattened copy
of agent/extra state; RGB PPO receives only RGB using the training camera order.
RGB collection supports frame stack 1 and action repeat 1.
The dataset stores ManiSkill's complete, restorable flat
simulator state in `state` and keeps each policy-observation leaf as a named
column (e.g. obs_agent_qpos, obs_extra_goal_pos). Pixels come from the
selected observation sensor unless --no-save-rgb is set. RGB policy inputs
are unchanged by that option. Each row contains observation_t, state_t, and
action_t.
The final post-action observation is omitted; SWM samples prediction windows
within each episode. No reset transition is included in an episode.
ManiSkill terminates on success (or task failure), and the vector wrapper
preserves that signal. max_steps is an upper bound. Collection batches GPU
environments, reuses the scene, and overlaps bounded writes with the next batch.
Changing num_envs changes seeded trajectories; use the same batch layout to replay.
With --noisy-replays K, each successful expert episode is followed by K
open-loop replays of its actions with Gaussian noise from the same initial
state, tagged by group_idx/replay_idx in the episode data.
With --correct-chunks K, each successful expert episode is followed by K
episodes that share its start and chunk-boundary states but replace each
--chunk-step chunk with a sampled expert chunk whose end state matched the
expert's within --chunk-tolerance (fallback_chunks counts chunks that kept the
expert's actions). They use the same group_idx/replay_idx tags, so
replay_filter selects them like noisy replays.
With --alternatives K, each expert episode runs in a group of --candidates
envs reset from the same seed; slot 0 is the deterministic expert and the
others sample the learned policy closed loop (different random draws). Up to K of those that
succeed are saved after the expert with the same group_idx/replay_idx tags.
"""

import argparse
import json
from pathlib import Path

from stable_worldmodel.envs.maniskill.chunks import TOLERANCES
from stable_worldmodel.envs.maniskill.experts import (
    collection_defaults,
    validate_collection_checkpoint,
)
from stable_worldmodel.envs.maniskill.observations import (
    flatten_policy_state,
    named_observation_leaves,
    policy_observation,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--policy-type',
        choices=('state', 'rgb'),
        default=None,
        help='RGB policies use RGB only, one frame and action repeat 1.',
    )
    parser.add_argument(
        '--encoder',
        choices=('spatial_softmax', 'flatten', 'nature'),
        default=None,
    )
    parser.add_argument(
        '--std-mode', choices=('learned', 'bounded', 'fixed'), default=None
    )
    parser.add_argument('--initial-std', type=float, default=None)
    parser.add_argument('--min-std', type=float, default=None)
    parser.add_argument('--max-std', type=float, default=None)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--env-id', default=None)
    parser.add_argument('--control-mode', default=None)
    parser.add_argument('--reward-mode', default=None)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument(
        '--num-envs',
        type=int,
        default=64,
        help='Parallel GPU environments; CPU collection requires 1.',
    )
    parser.add_argument(
        '--reconfiguration-freq',
        type=int,
        default=0,
        help='Scene rebuild interval in batch resets; 0 reuses the scene.',
    )
    parser.add_argument(
        '--sync-writes',
        action='store_true',
        help='Disable overlap of Lance encoding/writes with simulation.',
    )
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--camera-name',
        default=None,
        help='Observation sensor whose RGB frames are saved.',
    )
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument(
        '--no-save-rgb',
        dest='save_rgb',
        action='store_false',
        help='Omit pixels from the dataset; RGB experts still receive camera images.',
    )
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--write-batch-size',
        type=int,
        default=64,
        help='Maximum episodes per Lance write (also limited by --num-envs).',
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Alias for --policy-mode stochastic.',
    )
    parser.add_argument(
        '--policy-mode',
        choices=('deterministic', 'stochastic', 'noisy', 'mixed'),
        default='deterministic',
    )
    parser.add_argument(
        '--mix-weights',
        type=float,
        nargs=3,
        default=(0.5, 0.3, 0.2),
        metavar=('DETERMINISTIC', 'STOCHASTIC', 'NOISY'),
        help='Relative episode counts in mixed mode (default: 50/30/20 percent).',
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=0.1,
        help='Noisy PPO: Gaussian std as a fraction of each action half-range.',
    )
    parser.add_argument(
        '--noise-hold-steps',
        type=int,
        default=5,
        help='Hold each noisy PPO perturbation for this many control steps.',
    )
    parser.add_argument(
        '--noisy-replays',
        type=int,
        default=0,
        help='Per successful expert episode, also save this many open-loop replays '
        'of its actions from the same initial state with --noise-std noise; '
        'failed expert episodes are dropped with their replays.',
    )
    parser.add_argument(
        '--correct-chunks',
        type=int,
        default=0,
        help='Per successful expert episode, also save this many episodes whose '
        '--chunk-step chunks are replaced by sampled expert chunks that reach '
        "the expert's next state (see --chunk-tolerance); failed expert "
        'episodes are dropped with their alternatives.',
    )
    parser.add_argument(
        '--alternatives',
        type=int,
        default=0,
        help='Per successful expert episode, also save up to this many whole '
        'successful episodes from the same initial state, sampled closed loop '
        'from the learned policy in the same --candidates group; failed '
        'expert episodes are dropped with their alternatives.',
    )
    parser.add_argument(
        '--min-path-distance',
        type=float,
        default=20.0,
        help='Alternatives: drop a successful alternative whose gripper XY path '
        '(resampled over normalized time) is on average closer than this many '
        'mm to the expert or to an already kept alternative; 0 disables.',
    )
    parser.add_argument(
        '--candidates',
        type=int,
        default=16,
        help='Correct chunks / alternatives: envs per expert episode '
        '(1 deterministic expert + samples).',
    )
    parser.add_argument(
        '--chunk',
        type=int,
        default=5,
        help='Correct chunks: control steps per chunk (the LeWM frameskip).',
    )
    parser.add_argument(
        '--chunk-std-scales',
        type=float,
        nargs='+',
        default=(1.0,),
        help='Correct chunks: policy-std multipliers, round-robin over samples.',
    )
    parser.add_argument(
        '--chunk-tolerance',
        choices=('tight', 'medium', 'loose'),
        default='medium',
        help='Correct chunks: max end-state error (tee mm, yaw deg, tcp mm): '
        'tight 0.5/0.5/1, medium 1/1/2, loose 2/2/5.',
    )
    # Accept the previous opt-in flag for compatibility; termination is now
    # always enabled for dataset collection.
    parser.add_argument(
        '--stop-on-termination',
        action='store_true',
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU physics and policy inference (still needs rendering).',
    )
    args = parser.parse_args()
    try:
        inferred, _ = collection_defaults(args.checkpoint)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    defaults = dict(
        policy_type='state',
        encoder='spatial_softmax',
        env_id='PushT-v1',
        control_mode='pd_joint_delta_pos',
        reward_mode='normalized_dense',
        camera_name='base_camera',
        image_size=224,
        std_mode='learned',
        initial_std=0.3,
        min_std=0.1,
        max_std=0.5,
    )
    for key, fallback in defaults.items():
        value = getattr(args, key)
        if key in inferred and value is not None and value != inferred[key]:
            parser.error(
                f'--{key.replace("_", "-")} conflicts with checkpoint training_config.json'
            )
        setattr(
            args, key, inferred.get(key, fallback) if value is None else value
        )
    for name in (
        'num_episodes',
        'num_envs',
        'max_steps',
        'image_size',
        'write_batch_size',
        'noise_hold_steps',
    ):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    import math

    if args.cpu and args.num_envs != 1:
        parser.error('--cpu requires --num-envs 1')
    if args.noisy_replays < 0:
        parser.error('--noisy-replays must be nonnegative')
    if args.noisy_replays and args.policy_mode not in (
        'deterministic',
        'stochastic',
    ):
        parser.error(
            '--noisy-replays requires deterministic or stochastic expert episodes'
        )
    if args.correct_chunks < 0:
        parser.error('--correct-chunks must be nonnegative')
    if args.correct_chunks:
        if (
            args.cpu
            or args.noisy_replays
            or args.policy_mode != 'deterministic'
        ):
            parser.error(
                '--correct-chunks needs GPU, deterministic experts and no noisy replays'
            )
        if args.candidates < 2 or args.num_envs % args.candidates:
            parser.error('--candidates must be >= 2 and divide --num-envs')
        if args.chunk < 1 or any(
            not math.isfinite(v) or v < 0 for v in args.chunk_std_scales
        ):
            parser.error('--chunk must be positive and std scales nonnegative')
    if args.alternatives < 0:
        parser.error('--alternatives must be nonnegative')
    if args.alternatives:
        if (
            args.noisy_replays
            or args.correct_chunks
            or args.policy_mode != 'deterministic'
        ):
            parser.error(
                '--alternatives needs deterministic experts, no noisy replays '
                'and no correct chunks'
            )
        if (
            not math.isfinite(args.min_path_distance)
            or args.min_path_distance < 0
        ):
            parser.error('--min-path-distance must be finite and nonnegative')
        if (
            args.candidates <= args.alternatives
            or args.num_envs % args.candidates
        ):
            parser.error(
                '--candidates must exceed --alternatives and divide --num-envs'
            )
    if args.reconfiguration_freq < 0:
        parser.error('--reconfiguration-freq must be nonnegative')
    if not 1 <= args.jpeg_quality <= 100:
        parser.error('--jpeg-quality must be between 1 and 100')
    if not math.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error('--noise-std must be finite and nonnegative')
    if (
        any(not math.isfinite(w) or w < 0 for w in args.mix_weights)
        or not 0 < sum(args.mix_weights) < math.inf
    ):
        parser.error(
            '--mix-weights must be finite nonnegative weights with a positive finite sum'
        )
    if args.stochastic:
        if args.policy_mode not in ('deterministic', 'stochastic'):
            parser.error(
                '--stochastic cannot be combined with noisy or mixed mode'
            )
        args.policy_mode = 'stochastic'
    if args.output_path.suffix != '.lance':
        parser.error('--output-path must end in .lance')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {args.checkpoint}')
    if args.output_path.exists():
        parser.error(f'Output already exists: {args.output_path}')
    return args


def episode_modes(count, mode, weights, seed):
    """Allocate mixture counts by largest remainder, then shuffle reproducibly."""
    import numpy as np

    if mode != 'mixed':
        return [mode] * count
    expected = np.asarray(weights, dtype=float) / sum(weights) * count
    counts = np.floor(expected).astype(int)
    order = np.argsort(-(expected - counts), kind='stable')
    counts[order[: count - int(counts.sum())]] += 1
    modes = [
        name
        for name, number in zip(
            ('deterministic', 'stochastic', 'noisy'), counts
        )
        for _ in range(int(number))
    ]
    np.random.default_rng(seed).shuffle(modes)
    return modes


def collect_episode(
    env,
    agent,
    *,
    seed,
    max_steps,
    camera_name,
    stochastic,
    noise_std=0.0,
    noise_hold_steps=5,
    policy_type='state',
    save_rgb=True,
):
    import numpy as np
    import torch

    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.array(value, copy=True)

    device = next(agent.parameters()).device
    # Make each episode's action randomness independent of earlier lengths.
    torch.manual_seed(seed)
    noise_rng = torch.Generator(device=device).manual_seed(seed)
    if noise_std or policy_type == 'rgb':
        low = torch.as_tensor(env.single_action_space.low, device=device)
        high = torch.as_tensor(env.single_action_space.high, device=device)
        if not (torch.isfinite(low).all() and torch.isfinite(high).all()):
            raise ValueError('RGB/noisy PPO requires finite action bounds')
    obs, _ = env.reset(seed=seed)
    rows = {'state': [], 'action': []}
    if save_rgb:
        rows['pixels'] = []
    observation_keys = tuple(named_observation_leaves(obs))
    rows.update({key: [] for key in observation_keys})
    episode_return = 0.0
    success = False
    for step in range(max_steps):
        if save_rgb:
            sensors = obs['sensor_data']
            if camera_name not in sensors:
                raise ValueError(
                    f'Unknown sensor {camera_name!r}; available: {list(sensors)}'
                )
            pixels = as_numpy(sensors[camera_name]['rgb'])
        policy_input = policy_observation(obs, device, policy_type)
        sim_state = as_numpy(env.unwrapped.get_state())
        leaves = named_observation_leaves(obs)
        if tuple(leaves) != observation_keys:
            raise ValueError('Observation keys changed within an episode')
        if save_rgb:
            if (
                pixels.ndim != 4
                or pixels.shape[0] != 1
                or pixels.shape[-1] != 3
            ):
                raise ValueError(
                    f'Expected camera RGB [1,H,W,3], got {pixels.shape}'
                )
            if pixels.dtype != np.uint8:
                raise ValueError(
                    f'Expected uint8 camera RGB, got {pixels.dtype}'
                )
        if sim_state.ndim != 2 or sim_state.shape[0] != 1:
            raise ValueError(
                f'Expected single-env simulator state [1,D], got {sim_state.shape}'
            )
        if (
            not np.issubdtype(sim_state.dtype, np.number)
            or not np.isfinite(sim_state).all()
        ):
            raise ValueError(
                f'Expected finite numeric simulator state, got {sim_state.dtype}'
            )
        with torch.inference_mode():
            action = agent.get_action(
                policy_input, deterministic=not stochastic
            )
            if noise_std:
                if step % noise_hold_steps == 0:
                    noise = (
                        torch.randn(
                            action.shape, device=device, generator=noise_rng
                        )
                        * noise_std
                        * (high - low)
                        / 2
                    )
                action = torch.clamp(action + noise, min=low, max=high)
            if policy_type == 'rgb':
                if not torch.isfinite(action).all():
                    raise ValueError('Policy produced a non-finite action')
                action = torch.clamp(action, min=low, max=high)
        if not torch.isfinite(action).all():
            raise ValueError('Policy produced a non-finite action')
        if save_rgb:
            rows['pixels'].append(pixels[0])
        rows['state'].append(sim_state[0].astype(np.float32, copy=False))
        for key, (_, value) in leaves.items():
            value = as_numpy(value)
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(
                    f'Expected single-env batched state for {key}, got {value.shape}'
                )
            rows[key].append(value[0])
        # Save exactly what is sent, including perturbation/clipping in noisy mode.
        rows['action'].append(as_numpy(action[0]).astype(np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(as_numpy(reward).sum())
        success |= bool(as_numpy(info.get('success', False)).any())
        if as_numpy(terminated).any() or as_numpy(truncated).any():
            break
    # SWM's image-column detection checks `if not vals`, which requires a
    # Python sequence rather than a stacked NumPy array. Keep per-step arrays.
    return rows, episode_return, success


def collect_episode_batch(
    env,
    agent,
    *,
    seeds,
    modes,
    max_steps,
    camera_name,
    noise_std=0.0,
    noise_hold_steps=5,
    policy_type='state',
    save_rgb=True,
    replay_actions=None,
    replay_lengths=None,
    noise_seed=None,
    sources=None,
):
    """Collect one episode per slot, retaining only pre-terminal transitions.

    Rollout tensors stay on the device until the batch ends. Slots that finish
    early stop recording; the next full reset occurs when all slots finish.
    Reproducibility requires the same seeds AND batch size/layout.
    "replay" slots execute replay_actions[step, slot] open loop plus held
    Gaussian noise for at most replay_lengths[slot] steps. Resetting with the
    same seeds and layout reproduces the recorded episode's initial state.
    """
    import numpy as np
    import torch

    count = len(seeds)
    if count != env.num_envs or len(modes) != count:
        raise ValueError('Seeds and modes must match env.num_envs')
    if any(
        mode not in ('deterministic', 'stochastic', 'noisy', 'replay')
        for mode in modes
    ):
        raise ValueError('Unknown episode policy mode')
    if ('replay' in modes) != (replay_actions is not None) or (
        replay_actions is None
    ) != (replay_lengths is None):
        raise ValueError(
            'Replay mode requires replay_actions and replay_lengths'
        )
    device = next(agent.parameters()).device
    obs, _ = env.reset(seed=list(seeds))
    if sources is not None:
        # GPU resets with equal seeds differ; copy each source slot's state
        # (incl. controller target) so a group starts from one state.
        from stable_worldmodel.envs.maniskill.chunks import restore_from_source

        obs = restore_from_source(
            env.unwrapped, torch.as_tensor(sources, device=device)
        )
    torch.manual_seed(seeds[0])
    noise_rng = torch.Generator(device=device).manual_seed(
        seeds[0] if noise_seed is None else noise_seed
    )
    groups = {
        mode: torch.tensor(
            [i for i, value in enumerate(modes) if value == mode],
            device=device,
            dtype=torch.long,
        )
        for mode in ('deterministic', 'stochastic', 'noisy')
        if mode in modes
    }
    replay = torch.tensor(
        [value == 'replay' for value in modes], device=device
    )
    if replay_actions is not None:
        replay_actions = torch.as_tensor(
            replay_actions, device=device, dtype=torch.float32
        )
        replay_lengths = torch.as_tensor(
            replay_lengths, device=device, dtype=torch.long
        )
    bounded = policy_type == 'rgb' or 'noisy' in groups or replay.any()
    if bounded:
        low = torch.as_tensor(
            env.single_action_space.low, device=device, dtype=torch.float32
        )
        high = torch.as_tensor(
            env.single_action_space.high, device=device, dtype=torch.float32
        )
        if not (torch.isfinite(low).all() and torch.isfinite(high).all()):
            raise ValueError('RGB/noisy PPO requires finite action bounds')
    active = torch.ones(count, dtype=torch.bool, device=device)
    lengths = torch.zeros(count, dtype=torch.long, device=device)
    returns = torch.zeros(count, device=device)
    successes = torch.zeros(count, dtype=torch.bool, device=device)
    buffers = {}
    keys = tuple(named_observation_leaves(obs))
    noise = None
    # Controllers cache tensors during step and modify them during the next
    # reset. Keep those tensors mutable outside this context.
    with torch.no_grad():
        for step in range(max_steps):
            leaves = named_observation_leaves(obs)
            if tuple(leaves) != keys:
                raise ValueError('Observation keys changed within an episode')
            values = {key: value for key, (_, value) in leaves.items()}
            values['state'] = env.unwrapped.get_state()
            if values['state'].ndim != 2:
                raise ValueError('Expected simulator state [N,D]')
            if save_rgb:
                sensors = obs['sensor_data']
                if camera_name not in sensors:
                    raise ValueError(
                        f'Unknown sensor {camera_name!r}; available: {list(sensors)}'
                    )
                values['pixels'] = sensors[camera_name]['rgb']
                pixels = values['pixels']
                if (
                    pixels.ndim != 4
                    or pixels.shape[-1] != 3
                    or pixels.dtype != torch.uint8
                ):
                    raise ValueError('Expected camera RGB uint8 [N,H,W,3]')
            policy_input = policy_observation(obs, device, policy_type)
            action = torch.empty(
                (count, *env.single_action_space.shape), device=device
            )
            for mode, indices in groups.items():
                inputs = (
                    {
                        key: value[indices]
                        for key, value in policy_input.items()
                    }
                    if isinstance(policy_input, dict)
                    else policy_input[indices]
                )
                action[indices] = agent.get_action(
                    inputs, deterministic=mode != 'stochastic'
                )
            if not torch.isfinite(action).all():
                raise ValueError('Policy produced a non-finite action')
            if replay.any():
                action[replay] = replay_actions[step, replay]
            noisy = replay | torch.tensor(
                [value == 'noisy' for value in modes], device=device
            )
            if noisy.any() and noise_std:
                if step % noise_hold_steps == 0:
                    noise = (
                        torch.randn(
                            action.shape, device=device, generator=noise_rng
                        )
                        * noise_std
                        * (high - low)
                        / 2
                    )
                action[noisy] = torch.clamp(
                    action[noisy] + noise[noisy], min=low, max=high
                )
            if policy_type == 'rgb':
                action = torch.clamp(action, min=low, max=high)
            action[~active] = 0
            values['action'] = action
            for key, value in values.items():
                value = torch.as_tensor(value, device=device)
                if value.ndim == 0 or value.shape[0] != count:
                    raise ValueError(
                        f'Expected batched value [N,...] for {key}'
                    )
                if key not in buffers:
                    buffers[key] = torch.empty(
                        (max_steps, *value.shape),
                        dtype=value.dtype,
                        device=device,
                    )
                # Copy before stepping: simulator observations can alias mutable state.
                buffers[key][step].copy_(value)
            obs, reward, terminated, truncated, info = env.step(action)
            returns += torch.where(active, reward.reshape(count), 0)
            successes |= (
                active
                & torch.as_tensor(
                    info.get('success', False), device=device
                ).bool()
            )
            lengths += active.long()
            active &= ~(
                terminated.reshape(count).bool()
                | truncated.reshape(count).bool()
            )
            if replay_lengths is not None:
                active &= ~replay | (step + 1 < replay_lengths)
            if not active.any():
                break
    lengths = lengths.cpu().tolist()
    returns = returns.cpu().tolist()
    successes = successes.cpu().tolist()
    arrays = {
        key: value[: step + 1].cpu().numpy() for key, value in buffers.items()
    }
    episodes = []
    for slot, length in enumerate(lengths):
        episode = {}
        for key, value in arrays.items():
            # Own each episode's storage so queued writes do not retain a whole batch.
            data = value[:length, slot].copy()
            if key in ('state', 'action'):
                data = data.astype(np.float32, copy=False)
                if not np.isfinite(data).all():
                    raise ValueError(f'Non-finite {key} in collected episode')
            episode[key] = list(
                data
            )  # SWM image detection requires a Python sequence.
        episodes.append((episode, returns[slot], successes[slot]))
    return episodes


def assemble_correct_chunk_episodes(
    arrays, lengths, accepted, *, candidates, num_alternatives, chunk, rng
):
    """Build each group's expert episode and its correct-chunk alternatives.

    arrays[key] is [T, N, ...] for N = groups x candidates slots, where slot 0
    of each group is the expert. accepted[c] is a [groups, candidates] bool
    mask of slots whose chunk c ended at the expert's end state. Alternative
    j takes, per chunk, the j-th of a random order of accepted slots, or the
    expert's own rows when fewer than j+1 were accepted or the chunk is the
    expert's last, partial one. Returns per group (episode, slots_per_step,
    fallback_chunks) for the expert followed by each alternative.
    """
    import numpy as np

    groups = []
    for group, length in enumerate(lengths):
        choices = np.zeros((num_alternatives + 1, length), dtype=int)
        fallbacks = [0] * (num_alternatives + 1)
        for c, first in enumerate(range(0, length, chunk)):
            end = first + chunk
            ok = (
                np.flatnonzero(accepted[c][group])
                if end <= length
                else np.array([], dtype=int)
            )
            ok = rng.permutation(ok[ok != 0])
            for j in range(num_alternatives):
                if j < len(ok):
                    choices[j + 1, first:end] = ok[j]
                else:
                    fallbacks[j + 1] += 1
        rollouts = []
        steps = np.arange(length)
        for row, fallback in zip(choices, fallbacks):
            slots = group * candidates + row
            episode = {}
            for key, value in arrays.items():
                data = value[steps, slots]
                if key in ('state', 'action'):
                    data = data.astype(np.float32, copy=False)
                    if not np.isfinite(data).all():
                        raise ValueError(
                            f'Non-finite {key} in collected episode'
                        )
                episode[key] = list(data)
            rollouts.append((episode, slots, fallback))
        groups.append(rollouts)
    return groups


def collect_correct_chunk_batch(
    env,
    agent,
    *,
    seeds,
    candidates,
    num_alternatives,
    chunk,
    max_steps,
    camera_name,
    std_scales,
    tolerance,
    policy_type='state',
    save_rgb=True,
    noise_seed=0,
):
    """Expert episodes plus alternatives built from accepted sampled chunks.

    Each group of `candidates` slots resets with one seed. Slot 0 runs the
    deterministic expert; the others add Gaussian noise of the policy std
    times a per-slot scale. Before every chunk all slots are restored to the
    expert's state, so each candidate chunk starts from the expert trajectory
    and is accepted when it ends within `tolerance` of the expert's end state.
    Returns per group a list of (mode, replay_idx, episode, return, success,
    fallback_chunks); the expert is replay_idx -1.
    """
    import numpy as np
    import torch

    from stable_worldmodel.envs.maniskill.chunks import (
        accepted,
        end_errors,
        restore_from_source,
        snapshot,
    )

    count, n = env.num_envs, candidates
    if count % n or len(seeds) != count // n:
        raise ValueError('num_envs must equal len(seeds) x candidates')
    uw = env.unwrapped
    device = next(agent.parameters()).device
    slots = torch.arange(count, device=device)
    source = slots // n * n
    sampled = slots % n != 0
    scales = torch.as_tensor(std_scales, device=device, dtype=torch.float32)
    scale = torch.zeros(count, device=device)
    scale[sampled] = scales[(slots[sampled] % n - 1) % len(std_scales)]
    logstd = (
        agent.get_logstd()
        if hasattr(agent, 'get_logstd')
        else agent.actor_logstd
    )
    std = torch.exp(logstd).reshape(-1)
    low = torch.as_tensor(
        env.single_action_space.low, device=device, dtype=torch.float32
    )
    high = torch.as_tensor(
        env.single_action_space.high, device=device, dtype=torch.float32
    )
    noise_rng = torch.Generator(device=device).manual_seed(noise_seed)
    obs, _ = env.reset(seed=[seeds[i // n] for i in range(count)])
    keys = tuple(named_observation_leaves(obs))
    groups = count // n
    active = torch.ones(groups, dtype=torch.bool, device=device)
    lengths = torch.zeros(groups, dtype=torch.long, device=device)
    returns = torch.zeros(groups, device=device)
    successes = torch.zeros(groups, dtype=torch.bool, device=device)
    buffers, chunk_ok = {}, []
    with torch.no_grad():
        for first in range(0, max_steps, chunk):
            obs = restore_from_source(uw, source)
            for step in range(first, min(first + chunk, max_steps)):
                leaves = named_observation_leaves(obs)
                if tuple(leaves) != keys:
                    raise ValueError(
                        'Observation keys changed within an episode'
                    )
                values = {key: value for key, (_, value) in leaves.items()}
                values['state'] = uw.get_state()
                if save_rgb:
                    sensors = obs['sensor_data']
                    if camera_name not in sensors:
                        raise ValueError(
                            f'Unknown sensor {camera_name!r}; available: {list(sensors)}'
                        )
                    values['pixels'] = sensors[camera_name]['rgb']
                mean = agent.get_action(
                    policy_observation(obs, device, policy_type),
                    deterministic=True,
                )
                noise = torch.randn(
                    mean.shape, device=device, generator=noise_rng
                )
                action = torch.clamp(
                    mean + noise * std * scale[:, None], min=low, max=high
                )
                if not torch.isfinite(action).all():
                    raise ValueError('Policy produced a non-finite action')
                values['action'] = action
                for key, value in values.items():
                    value = torch.as_tensor(value, device=device)
                    if key not in buffers:
                        buffers[key] = torch.empty(
                            (max_steps, *value.shape),
                            dtype=value.dtype,
                            device=device,
                        )
                    buffers[key][step].copy_(value)
                obs, reward, terminated, truncated, info = env.step(action)
                expert = slice(None, None, n)
                returns += torch.where(
                    active, reward.reshape(count)[expert], 0
                )
                success = info.get('success', torch.zeros(count))
                success = torch.as_tensor(success, device=device).bool()
                successes |= active & success.reshape(count)[expert]
                lengths += active.long()
                done = terminated.reshape(count) | truncated.reshape(count)
                active &= ~done.bool()[expert]
                if not active.any():
                    break
            ok = accepted(end_errors(snapshot(uw), source), tolerance)
            chunk_ok.append(ok.reshape(groups, n).cpu().numpy())
            if not active.any():
                break
    arrays = {
        key: value[: step + 1].cpu().numpy() for key, value in buffers.items()
    }
    lengths = lengths.cpu().tolist()
    returns = returns.cpu().tolist()
    successes = successes.cpu().tolist()
    rng = np.random.default_rng(noise_seed)
    assembled = assemble_correct_chunk_episodes(
        arrays,
        lengths,
        chunk_ok,
        candidates=n,
        num_alternatives=num_alternatives,
        chunk=chunk,
        rng=rng,
    )
    return [
        [
            (
                'deterministic' if index == 0 else 'correct_chunks',
                index - 1,
                episode,
                returns[group],
                successes[group],
                fallback,
            )
            for index, (episode, _, fallback) in enumerate(rollouts)
        ]
        for group, rollouts in enumerate(assembled)
    ]


def path_distance(a, b, points=50):
    """Mean distance between two XY paths resampled over normalized time."""
    import numpy as np

    def resample(path):
        path = np.asarray(path, dtype=np.float64)
        t, u = np.linspace(0, 1, len(path)), np.linspace(0, 1, points)
        return np.stack([np.interp(u, t, path[:, i]) for i in range(2)], 1)

    return float(np.linalg.norm(resample(a) - resample(b), axis=1).mean())


def select_alternatives(
    batch,
    candidates,
    num_alternatives,
    min_distance=0.0,
    path_key='obs_extra_tcp_pose',
):
    """Group expert episodes (slot 0) with distinct successful alternatives.

    batch holds (episode, return, success) per slot, `candidates` consecutive
    slots per group, all starting from the group's expert state. Successful
    slots are kept in order while their gripper XY path (`path_key`) is at
    least `min_distance` (same units, see path_distance) from the expert's and
    every kept alternative's path, up to `num_alternatives`. Returns per group
    (mode, replay_idx, episode, return, success, fallback_chunks) tuples, the
    expert first, and the number of successes rejected as too similar.
    """
    import numpy as np

    groups, too_similar = [], 0
    for first in range(0, len(batch), candidates):
        expert, *others = batch[first : first + candidates]
        start = expert[0]['state'][0]
        rollouts = [('deterministic', -1, *expert, 0)]
        paths = []
        if min_distance:
            paths.append(np.asarray(expert[0][path_key])[:, :2])
        for episode, episode_return, success in others:
            if not np.allclose(episode['state'][0], start):
                raise ValueError(
                    'Alternative did not start from the expert initial state'
                )
            if not success or len(rollouts) > num_alternatives:
                continue
            if min_distance:
                path = np.asarray(episode[path_key])[:, :2]
                if (
                    min(path_distance(path, kept) for kept in paths)
                    < min_distance
                ):
                    too_similar += 1
                    continue
                paths.append(path)
            rollouts.append(
                (
                    'alternative',
                    len(rollouts) - 1,
                    episode,
                    episode_return,
                    success,
                    0,
                )
            )
        groups.append(rollouts)
    return groups, too_similar


def write_episode_batch(writer, episodes, batch_size):
    """Single writer worker; no concurrent Lance mutations."""
    for start in range(0, len(episodes), batch_size):
        writer.write_episodes(episodes[start : start + batch_size])


def main():
    args = parse_args()
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - register environments
    import numpy as np
    import torch
    from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    from stable_worldmodel.data import EPISODE_DATA_KEY, LanceWriter
    from stable_worldmodel.envs.maniskill import (
        overhead_pusht,  # noqa: F401 - register OverheadPushT-v1
        random_goal_pusht,  # noqa: F401 - register RandomGoalPushT-v1
    )

    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is unavailable; run on a GPU node or pass --cpu'
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cpu' if args.cpu else 'cuda')
    weights = torch.load(
        args.checkpoint, map_location='cpu', weights_only=True
    )
    validate_collection_checkpoint(weights, args.policy_type)
    _, training_config = collection_defaults(args.checkpoint)
    sensor_configs = {'width': args.image_size, 'height': args.image_size}
    if training_config and args.policy_type == 'rgb':
        import sapien

        sensor_configs = {}
        for name, camera in training_config['environment']['cameras'].items():
            settings = dict(camera)
            pose = settings.pop('pose', None)
            if pose is not None:
                settings['pose'] = sapien.Pose(p=pose['p'], q=pose['q'])
            sensor_configs[name] = settings
    # Correct chunks and alternatives run groups of candidates per expert
    # episode; spare groups in the last batch are collected and discarded.
    in_groups = bool(args.correct_chunks or args.alternatives)
    num_envs = (
        args.num_envs if in_groups else min(args.num_envs, args.num_episodes)
    )
    per_batch = num_envs // args.candidates if in_groups else num_envs
    obs_mode = (
        'rgb+state_dict'
        if args.save_rgb or args.policy_type == 'rgb'
        else 'state_dict'
    )
    env = gym.make(
        args.env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        render_mode='rgb_array',
        sim_backend='physx_cpu' if args.cpu else 'physx_cuda',
        control_mode=args.control_mode,
        reward_mode=args.reward_mode,
        reconfiguration_freq=args.reconfiguration_freq,
        sensor_configs=sensor_configs,
    )
    try:
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(
            env,
            auto_reset=False,
            ignore_terminations=False,
        )
        validate_collection_checkpoint(
            weights, args.policy_type, env.single_action_space.shape
        )
        # StateExpert only needs the state/action spaces to rebuild the PPO MLP.
        # Use the actual flattened observation to size the checkpoint's MLP.
        initial_obs, _ = env.reset(seed=args.seed)
        if training_config:
            saved_env = training_config['environment']
            if (
                list(env.single_action_space.shape)
                != saved_env['action_shape']
                or not np.array_equal(
                    env.single_action_space.low, saved_env['action_low']
                )
                or not np.array_equal(
                    env.single_action_space.high, saved_env['action_high']
                )
            ):
                raise ValueError(
                    'Environment action space differs from the checkpoint config'
                )
        initial_state = flatten_policy_state(initial_obs, device)
        initial_sim_state = env.unwrapped.get_state()
        observation_fields = {
            key: {
                'path': list(path),
                'shape': list(value.shape[1:]),
                'source_dtype': str(value.dtype),
            }
            for key, (path, value) in named_observation_leaves(
                initial_obs
            ).items()
        }
        policy_spaces = SimpleNamespace(
            single_observation_space=gym.spaces.Box(
                -np.inf,
                np.inf,
                shape=tuple(initial_state.shape[1:]),
                dtype=np.float32,
            ),
            single_action_space=env.single_action_space,
        )
        if args.policy_type == 'rgb':
            from stable_worldmodel.envs.maniskill.experts import RGBExpert

            agent = RGBExpert(
                env,
                sample_obs=policy_observation(initial_obs, device, 'rgb'),
                encoder=args.encoder,
                std_mode=args.std_mode,
                initial_std=args.initial_std,
                min_std=args.min_std,
                max_std=args.max_std,
            ).to(device)
        else:
            from stable_worldmodel.envs.maniskill.experts import StateExpert

            agent = StateExpert(policy_spaces).to(device)
        agent.load_state_dict(weights)
        del weights
        if args.policy_type == 'rgb':
            agent.constrain_std()
        agent.eval()
        total_steps = 0
        successes = 0
        dropped_groups = 0
        too_similar = 0
        modes = episode_modes(
            args.num_episodes, args.policy_mode, args.mix_weights, args.seed
        )
        episode_summaries = []
        started = time.perf_counter()
        print(
            f'Collecting with {num_envs} environments, obs_mode={obs_mode}, '
            f'reconfiguration_freq={args.reconfiguration_freq}',
            flush=True,
        )
        with LanceWriter(
            args.output_path, mode='error', jpeg_quality=args.jpeg_quality
        ) as writer:
            # One running write plus the current rollout bounds host-memory use.
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending_write = None
                for start in range(0, args.num_episodes, per_batch):
                    count = min(per_batch, args.num_episodes - start)
                    batch_seeds = list(
                        range(args.seed + start, args.seed + start + per_batch)
                    )
                    if args.correct_chunks:
                        groups = collect_correct_chunk_batch(
                            env,
                            agent,
                            seeds=batch_seeds,
                            candidates=args.candidates,
                            num_alternatives=args.correct_chunks,
                            chunk=args.chunk,
                            max_steps=args.max_steps,
                            camera_name=args.camera_name,
                            std_scales=args.chunk_std_scales,
                            tolerance=TOLERANCES[args.chunk_tolerance],
                            policy_type=args.policy_type,
                            save_rgb=args.save_rgb,
                            noise_seed=args.seed + start,
                        )[:count]
                    elif args.alternatives:
                        n = args.candidates
                        batch = collect_episode_batch(
                            env,
                            agent,
                            seeds=[
                                batch_seeds[i // n] for i in range(num_envs)
                            ],
                            modes=[
                                'deterministic' if i % n == 0 else 'stochastic'
                                for i in range(num_envs)
                            ],
                            max_steps=args.max_steps,
                            camera_name=args.camera_name,
                            noise_std=args.noise_std,
                            noise_hold_steps=args.noise_hold_steps,
                            policy_type=args.policy_type,
                            save_rgb=args.save_rgb,
                            noise_seed=args.seed + start,
                            sources=[i // n * n for i in range(num_envs)],
                        )
                        groups, similar = select_alternatives(
                            batch,
                            n,
                            args.alternatives,
                            min_distance=args.min_path_distance / 1000,
                        )
                        groups = groups[:count]
                        too_similar += similar
                        del batch
                    else:
                        batch_modes = modes[start : start + count] + [
                            'deterministic'
                        ] * (num_envs - count)
                        collect = {
                            'seeds': batch_seeds,
                            'max_steps': args.max_steps,
                            'camera_name': args.camera_name,
                            'noise_std': args.noise_std,
                            'noise_hold_steps': args.noise_hold_steps,
                            'policy_type': args.policy_type,
                            'save_rgb': args.save_rgb,
                        }
                        batch = collect_episode_batch(
                            env, agent, modes=batch_modes, **collect
                        )[:count]
                        replays = []
                        if args.noisy_replays:
                            # Same seeds and layout reset to the expert's initial states.
                            expert_actions = torch.zeros(
                                (
                                    args.max_steps,
                                    num_envs,
                                    *env.single_action_space.shape,
                                )
                            )
                            expert_lengths = [1] * num_envs
                            for slot, (episode, _, _) in enumerate(batch):
                                expert_lengths[slot] = len(episode['action'])
                                expert_actions[
                                    : expert_lengths[slot], slot
                                ] = torch.as_tensor(
                                    np.stack(episode['action'])
                                )
                            replays = [
                                collect_episode_batch(
                                    env,
                                    agent,
                                    modes=['replay'] * num_envs,
                                    replay_actions=expert_actions,
                                    replay_lengths=expert_lengths,
                                    noise_seed=args.seed
                                    + start
                                    + (index + 1) * (1 << 32),
                                    **collect,
                                )[:count]
                                for index in range(args.noisy_replays)
                            ]
                        groups = [
                            [
                                (batch_modes[slot], -1, *expert, 0),
                                *(
                                    ('noisy_replay', index, *replay[slot], 0)
                                    for index, replay in enumerate(replays)
                                ),
                            ]
                            for slot, expert in enumerate(batch)
                        ]
                        del batch, replays
                    grouped = bool(args.noisy_replays or in_groups)
                    episodes = []
                    for slot, rollouts in enumerate(groups):
                        episode_idx = start + slot
                        success = rollouts[0][4]
                        if grouped and not success:
                            dropped_groups += 1
                            continue
                        for (
                            mode,
                            replay_idx,
                            rollout,
                            rollout_return,
                            rollout_success,
                            fallback_chunks,
                        ) in rollouts:
                            rollout[EPISODE_DATA_KEY] = {
                                'collection_mode': mode
                            }
                            summary = {
                                'episode_idx': episode_idx,
                                'seed': args.seed + episode_idx,
                                'batch_seed': args.seed + start,
                                'env_slot': slot * args.candidates
                                if in_groups
                                else slot,
                                'mode': mode,
                                'length': len(rollout['action']),
                                'success': rollout_success,
                                'episode_return': rollout_return,
                            }
                            if grouped:
                                rollout[EPISODE_DATA_KEY].update(
                                    group_idx=episode_idx,
                                    replay_idx=replay_idx,
                                )
                                summary.update(replay_idx=replay_idx)
                            if args.correct_chunks:
                                rollout[EPISODE_DATA_KEY].update(
                                    fallback_chunks=fallback_chunks
                                )
                                summary.update(fallback_chunks=fallback_chunks)
                            episode_summaries.append(summary)
                            episodes.append(rollout)
                            total_steps += len(rollout['action'])
                        successes += int(success)
                    if pending_write is not None:
                        pending_write.result()
                    if args.sync_writes:
                        write_episode_batch(
                            writer, episodes, args.write_batch_size
                        )
                    else:
                        pending_write = executor.submit(
                            write_episode_batch,
                            writer,
                            episodes,
                            args.write_batch_size,
                        )
                    completed = start + count
                    if (
                        completed == args.num_episodes
                        and pending_write is not None
                    ):
                        pending_write.result()
                    elapsed = time.perf_counter() - started
                    rate = completed / elapsed
                    print(
                        f'Episodes {completed}/{args.num_episodes}: '
                        f'{rate:.2f} episodes/s {total_steps / elapsed:.0f} steps/s '
                        f'success={successes / completed:.3f} saved={len(episode_summaries)} '
                        f'too_similar={too_similar} '
                        f'ETA={(args.num_episodes - completed) / rate / 3600:.2f}h '
                        f'200k_projection={200000 / rate / 3600:.2f}h',
                        flush=True,
                    )
                    del groups, episodes
        metadata = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        metadata.update(
            total_steps=total_steps,
            successful_episodes=successes,
            saved_episodes=len(episode_summaries),
            dropped_failed_groups=dropped_groups,
            alternatives_too_similar=too_similar,
            policy_training_config=training_config,
            stop_on_success=True,
            episodes=episode_summaries,
            obs_mode=obs_mode,
            actual_num_envs=num_envs,
            collection_seconds=time.perf_counter() - started,
            seeding='batch seeds and env slots; reproduce with the same num_envs',
            chunk_tolerance_values=TOLERANCES[args.chunk_tolerance]
            if args.correct_chunks
            else None,
            pixel_source='sensor_data' if args.save_rgb else None,
            observation_fields=observation_fields,
            state_storage_dtype='float32',
            simulator_state_column='state',
            simulator_state_size=int(initial_sim_state.shape[-1]),
            simulator_state_source='ManiSkill BaseEnv.get_state',
            simulator_state_restore='ManiSkill BaseEnv.set_state',
            alignment='observation_t, action_t',
        )
        if args.policy_type == 'rgb':
            metadata.update(
                include_state=False,
                frame_stack=1,
                action_repeat=1,
                policy_cameras=list(initial_obs['sensor_data']),
            )
        metadata_path = args.output_path.with_suffix('.collection.json')
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        print(
            f'Saved {len(episode_summaries)} episodes ({total_steps} rows) to {args.output_path}'
        )
        print(f'Collection settings: {metadata_path}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
