"""Execute the search loss's plans in the simulator (on-policy rollouts).

For each sampled training clip: restore the simulator to the clip's start
state, pick plans exactly as the search InfoNCE does (``search_negatives``:
noisy experts + CEM elites toward the clip's last frame, using the given
checkpoint), execute a subset of them in ManiSkill and record the real
observation after every action block. The rollouts are written as a Lance
dataset with one row per block (``frameskip`` raw actions per row), which
``lewm.py`` mixes into training via ``onpolicy.datasets``
(config ``lewm_onpolicy``).

Usage (GPU node, from the repo root):
    python scripts/train/collect_onpolicy.py \
        --checkpoint $SCRATCH/le-wm/checkpoints/<run>/weights_epoch_20.pt \
        --dataset overhead_rgb_noisy_replay_100k.lance --cache-dir $SLURM_TMPDIR \
        --output $SCRATCH/le-wm/onpolicy/<name>.lance --num-clips 20000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lewm import (  # noqa: E402
    get_img_preprocessor,
    normalize_pixels,
    plan_costs,
    search_negatives,
)

TARGET_COL = 'obs_agent_controller_arm_target_pose'
SEARCH_DEFAULTS = Path(__file__).resolve().parent / 'config/lewm_search.yaml'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', required=True, help='weights .pt; config.yaml beside it')
    p.add_argument('--dataset', required=True, help='source dataset name or path')
    p.add_argument('--cache-dir', default=None, help='dataset cache dir')
    p.add_argument('--output', required=True, type=Path, help='output .lance')
    p.add_argument('--num-clips', type=int, default=20000)
    p.add_argument('--clips-per-batch', type=int, default=64)
    p.add_argument('--num-elites', type=int, default=10, help='of cem_steps * topk')
    p.add_argument('--num-noisy', type=int, default=4, help='of num_perturbed')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--search',
        nargs='*',
        default=[],
        help='loss.search overrides, e.g. horizon=5 cem_steps=5',
    )
    p.add_argument('--env-id', default='OverheadPushT-v1')
    p.add_argument('--camera', default='overhead_camera')
    p.add_argument('--control-mode', default='pd_ee_delta_xy')
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument(
        '--reconfigure',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='rebuild the sim scene before each batch (reproducible rollouts)',
    )
    p.add_argument(
        '--verify',
        action='store_true',
        help='replay the expert plan from one batch of restored states, '
        'compare with the dataset and exit without writing',
    )
    args = p.parse_args()
    if args.output.suffix != '.lance':
        p.error('--output must end in .lance')
    if args.output.exists() and not args.verify:
        p.error(f'{args.output} exists')
    return args


def load_config(args):
    """History/frameskip/img size from the checkpoint run; search settings
    from lewm_search.yaml, then the run's own loss.search, then --search."""
    run_cfg = OmegaConf.load(Path(args.checkpoint).parent / 'config.yaml')
    search = OmegaConf.load(SEARCH_DEFAULTS).loss.search
    run_search = OmegaConf.select(run_cfg, 'loss.search')
    if run_search is not None:
        search = OmegaConf.merge(search, run_search)
    search = OmegaConf.merge(search, OmegaConf.from_dotlist(args.search))
    if search.get('clip_actions'):
        raise ValueError('clip_actions is not supported by the collector')
    if search.get('sampler', 'cem') != 'cem':
        raise ValueError('only the cem sampler picks plans here')
    return OmegaConf.create(
        {
            'wm': {'history_size': int(run_cfg.wm.history_size)},
            'loss': {'search': search},
            'frameskip': int(run_cfg.data.dataset.frameskip),
            'img_size': int(run_cfg.img_size),
        }
    )


def load_source(args, cfg, num_steps):
    import stable_worldmodel as swm
    import stable_pretraining as spt
    from stable_worldmodel.data import column_normalizer

    dataset = swm.data.load_dataset(
        args.dataset,
        transform=None,
        cache_dir=args.cache_dir,
        num_steps=num_steps,
        frameskip=cfg.frameskip,
        keys_to_load=['pixels', 'action', 'state', TARGET_COL],
        keys_to_cache=['action'],
    )
    # Fitted on the full column, as in training, so the stats match.
    normalizer = column_normalizer(dataset, 'action', 'action')
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size, uint8=True
        ),
        normalizer,
    )
    return dataset, normalizer.lambd, dataset.get_dim('action')


def make_env(args, num_envs):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401 - register environments

    from stable_worldmodel.envs.maniskill import overhead_pusht  # noqa: F401

    env = gym.make(
        args.env_id,
        num_envs=num_envs,
        obs_mode='rgb',
        sim_backend='physx_cuda',
        control_mode=args.control_mode,
        reconfiguration_freq=0,
        sensor_configs={'width': 128, 'height': 128},
    )
    env.reset(seed=args.seed)
    return env


def pick(plans, n, start, count, gen):
    """``n`` random plans per clip from plans[:, start : start + count]."""
    if n > count:
        raise ValueError(f'asked for {n} plans, only {count} available')
    B = plans.size(0)
    order = torch.rand(B, count, generator=gen, device=plans.device).argsort(1)
    idx = start + order[:, :n]
    return plans[torch.arange(B, device=plans.device)[:, None], idx], idx


def task_pose(base, flat):
    """T xy, T yaw and TCP xy of flat sim states (loads them into the env)."""
    from stable_worldmodel.envs.maniskill.chunks import snapshot

    base.set_state(flat.to(base.device))
    s = snapshot(base)
    return s['tee_xy'], s['tee_yaw'], s['tcp_xy']


def pose_errors(a, b):
    """Per-env (T mm, T yaw deg, TCP mm) between two task_pose results."""
    from stable_worldmodel.envs.maniskill.chunks import wrap

    return (
        1e3 * (a[0] - b[0]).norm(dim=-1),
        torch.rad2deg(wrap(a[1] - b[1]).abs()),
        1e3 * (a[2] - b[2]).norm(dim=-1),
    )


def verify(base, frames, states, clip_pixels, clip_states, H, M, B):
    """Expert replays from restored states vs the recorded clip, per block.

    dataset gap: each replay vs the dataset (T position mm, T yaw deg, TCP
    mm, pixel MAE). copy spread: the M identical replays of a clip vs the
    first one, i.e. the simulator's own run-to-run non-determinism. A gap
    close to the spread means the drift is physics, not a bug.
    Block 0 tests the restore alone.
    """
    dev = clip_states.device  # the CPU sim keeps its tensors on the CPU
    sim_px = torch.stack(frames, 1).permute(0, 1, 4, 2, 3).float().to(dev)
    sim_st = torch.stack(states, 1).float().to(dev)
    ref_px = clip_pixels[:, H - 1 :].float().repeat_interleave(M, 0)
    ref_st = clip_states[:, H - 1 :].repeat_interleave(M, 0)
    print(
        'block |  dataset gap: T mm (mean/max)  yaw deg (mean/max)  TCP mm  px MAE'
        ' |  copy spread: T mm (mean/max)  yaw deg (mean/max)  TCP mm'
    )
    for k in range(sim_st.size(1)):
        sim = task_pose(base, sim_st[:, k])
        ref = task_pose(base, ref_st[:, k])
        gap = pose_errors(sim, ref)
        first = [v.view(B, M, *v.shape[1:])[:, :1].expand(B, M, *v.shape[1:])
                 .reshape(v.shape) for v in sim]
        spread = [e.view(B, M)[:, 1:] for e in pose_errors(sim, first)]
        px = (sim_px[:, k] - ref_px[:, k]).abs().mean().item()
        print(
            f'{k:5d} |  {gap[0].mean():6.2f} / {gap[0].max():6.2f}'
            f'     {gap[1].mean():6.2f} / {gap[1].max():6.2f}'
            f'      {gap[2].mean():6.2f}  {px:6.2f}'
            f' |  {spread[0].mean():6.2f} / {spread[0].max():6.2f}'
            f'     {spread[1].mean():6.2f} / {spread[1].max():6.2f}'
            f'      {spread[2].mean():6.2f}'
        )
    print(f'{B} clips x {M} identical replays; block 0 checks the restore only')


def main():
    args = parse_args()
    from stable_worldmodel.data import EPISODE_DATA_KEY, LanceWriter
    from stable_worldmodel.wm.utils import load_pretrained

    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    cfg = load_config(args)
    H, search, fs = cfg.wm.history_size, cfg.loss.search, cfg.frameskip
    T = search.horizon
    n_noisy, n_elite = search.num_perturbed, search.cem_steps * search.topk
    M = args.num_noisy + args.num_elites
    B = args.clips_per_batch
    num_clips = args.num_clips - args.num_clips % B

    model = load_pretrained(args.checkpoint).to(device).eval()
    dataset, scaler, adim = load_source(args, cfg, H + T)
    rng = np.random.default_rng(args.seed)
    clips = rng.choice(len(dataset), size=num_clips, replace=False)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    env = make_env(args, B * M)
    base = env.unwrapped
    low = torch.as_tensor(base.single_action_space.low, device=device)
    high = torch.as_tensor(base.single_action_space.high, device=device)

    def pixels(obs):
        return obs['sensor_data'][args.camera]['rgb']  # (N, h, w, 3) uint8

    started, written = time.perf_counter(), 0
    stats = {'pred_cost': [], 'expert_cost': []}
    with LanceWriter(args.output, mode='error', jpeg_quality=args.jpeg_quality) as writer:
        for b0 in range(0, num_clips, B):
            ids = clips[b0 : b0 + B].tolist()
            batch = torch.utils.data.default_collate(dataset.__getitems__(ids))
            batch = {k: v.to(device) for k, v in batch.items()}
            clip_states = batch.pop('state').float()  # (B, H + T, D)
            clip_pixels = batch['pixels'].clone()  # uint8 (B, H + T, 3, h, w)
            start_state = clip_states[:, H - 1]
            target = batch.pop(TARGET_COL)[:, H - 1].float()
            batch['action'] = torch.nan_to_num(batch['action'], 0.0)
            normalize_pixels(batch)

            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                emb = model.encode(dict(batch))['emb']
                plans = search_negatives(model, batch, emb, cfg)
                noisy, noisy_idx = pick(plans, args.num_noisy, 1, n_noisy, gen)
                elite, elite_idx = pick(
                    plans, args.num_elites, 1 + n_noisy, n_elite, gen
                )
                chosen = torch.cat([noisy, elite], 1)  # (B, M, T, fs * adim)
                if args.verify:  # the expert plan must reproduce the clip
                    chosen = plans[:, :1].expand_as(chosen).contiguous()
                pred = plan_costs(model, batch, emb, cfg, chosen).float()
                expert = plan_costs(model, batch, emb, cfg, plans[:, :1]).float()
            slot = torch.cat([noisy_idx, elite_idx], 1)

            # Normalized blocks -> raw per-step actions, clipped like the env.
            raw = scaler.inverse_transform(
                chosen.float().reshape(B * M, T * fs, adim)
            )
            raw = torch.maximum(torch.minimum(raw, high), low)

            # Every plan of a clip starts from that clip's state; set_state
            # overwrites the scene. reconfigure rebuilds it first: reset and
            # set_state keep PhysX solver state (warm starts, contact caches)
            # from earlier steps, which makes restored rollouts depend on the
            # slot's history; after a rebuild they are bit-reproducible
            # (scripts/data/check_replay_determinism.py, runs C and D).
            env.reset(options={'reconfigure': True} if args.reconfigure else None)
            base.set_state(start_state.repeat_interleave(M, 0).to(base.device))
            base.agent.set_controller_state(
                {'arm': {'target_pose': target.repeat_interleave(M, 0).to(base.device)}}
            )
            obs = base.get_obs()
            frames = [pixels(obs).clone()]
            states = [base.get_state().clone()]
            for k in range(T):
                for j in range(fs):
                    obs, *_ = env.step(raw[:, k * fs + j].to(base.device))
                frames.append(pixels(obs).clone())
                states.append(base.get_state().clone())

            if args.verify:
                verify(base, frames, states, clip_pixels, clip_states, H, M, B)
                return
            frames = torch.stack(frames, 1).cpu().numpy()  # (N, T+1, h, w, 3)
            states = torch.stack(states, 1).float().cpu().numpy()
            blocks = raw.reshape(B * M, T, fs * adim).float().cpu().numpy()
            pad = np.full((B * M, 1, fs * adim), np.nan, dtype=np.float32)
            blocks = np.concatenate([blocks, pad], 1)  # last frame: no action
            pred, expert, slot = pred.cpu(), expert.cpu(), slot.cpu()
            stats['pred_cost'].append(pred)
            stats['expert_cost'].append(expert)

            episodes = []
            for n in range(B * M):
                c, m = divmod(n, M)
                ep, t0 = dataset.clip_indices[ids[c]]
                episodes.append(
                    {
                        'pixels': list(frames[n]),
                        'action': list(blocks[n]),
                        'state': list(states[n]),
                        EPISODE_DATA_KEY: {
                            'source_clip': int(ids[c]),
                            'source_episode': int(ep),
                            'source_start': int(t0),
                            'kind': 'noisy' if m < args.num_noisy else 'elite',
                            'plan_slot': int(slot[c, m]),
                            'predicted_cost': float(pred[c, m]),
                            'expert_predicted_cost': float(expert[c, 0]),
                        },
                    }
                )
            writer.write_episodes(episodes)
            written += len(episodes)
            rate = (b0 + B) / (time.perf_counter() - started)
            print(
                f'clips {b0 + B}/{num_clips} rollouts {written} '
                f'{rate:.1f} clips/s ETA {(num_clips - b0 - B) / rate / 60:.1f} min',
                flush=True,
            )

    pred = torch.cat(stats['pred_cost'])
    expert = torch.cat(stats['expert_cost'])
    meta = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'search': OmegaConf.to_container(search),
        'history_size': H,
        'frameskip': fs,
        'num_clips': num_clips,
        'rollouts': written,
        'rows_per_rollout': T + 1,
        'layout': 'one row per action block; action = frameskip raw actions, NaN on the last row',
        'mean_predicted_cost': {
            'noisy': float(pred[:, : args.num_noisy].mean()),
            'elite': float(pred[:, args.num_noisy :].mean()),
            'expert': float(expert.mean()),
        },
        'collection_seconds': time.perf_counter() - started,
    }
    args.output.with_suffix('.collection.json').write_text(
        json.dumps(meta, indent=2) + '\n'
    )
    env.close()


if __name__ == '__main__':
    main()
