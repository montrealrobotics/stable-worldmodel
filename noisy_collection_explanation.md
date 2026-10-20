# Noisy dataset collection in `mani-exps/collect_ppo_dataset.py`

The collector adds explicit action noise in two ways: noisy PPO episodes and noisy open-loop replays of expert episodes.

## 1. Noisy PPO episodes (`--policy-mode noisy` or `mixed`)

See `collect_episode_batch`, lines 343-350.

- The policy outputs its deterministic (mean) action from the current observation. It runs closed loop, so it can react to and correct the noise.
- Gaussian noise is added with `std = noise_std * (high - low) / 2`. `--noise-std` is a fraction of each action dimension's half-range (default 0.1).
- The noise is redrawn only when `step % noise_hold_steps == 0` (default every 5 steps) and reused in between. The noise is therefore piecewise-constant and correlated over time, not white.
- The noisy action is clipped to the action bounds, and the dataset stores the action actually sent: noisy and clipped.
- `mixed` mode splits episodes into deterministic, stochastic and noisy groups using `--mix-weights` (default 50/30/20). The counts use largest-remainder rounding and the order is shuffled with a fixed seed (`episode_modes`).
- `stochastic` mode is different: it samples from the PPO policy's own Gaussian and adds no extra noise.

## 2. Noisy replays (`--noisy-replays K`, with `deterministic` or `stochastic` mode)

See `main`, lines 512-523.

- An expert batch is collected first, and its actions are put into a `[max_steps, num_envs, A]` tensor.
- K more batches then run in `"replay"` mode. Each resets with the same seeds and batch layout, which recreates the expert's initial states.
- Each replay plays the expert's recorded actions open loop, with the same held, clipped Gaussian noise as above. The policy is never queried, so nothing corrects for drift.
- A replay stops at the expert episode's length, or earlier if the environment terminates.
- Replay `i` gets its own noise seed: `seed + start + (i+1)·2³²`, so the K replays get different noise.
- If an expert episode failed, it is dropped together with all its replays. Replays themselves are kept whether or not they succeed.
- Labels in the episode data:
  - Experts: `collection_mode = deterministic` or `stochastic`, `replay_idx = -1`.
  - Replays: `collection_mode = "noisy_replay"`, `replay_idx = 0..K-1`.
  - Both: `group_idx = episode_idx`.

## Example

This toy example uses made-up numbers. It assumes a single action dimension with bounds `[-1, 1]` and the defaults `--noise-std 0.1` and `--noise-hold-steps 5`. The half-range is 1, so the noise std is 0.1. New noise is drawn at steps 0, 5 and 10; suppose the draws are `+0.12`, `-0.05` and `+0.21`.

| step | expert action | noise held | expert + noise | stored replay action |
|---|---|---|---|---|
| 0 | 0.30 | **+0.12** (new draw) | 0.42 | 0.42 |
| 1 | 0.35 | +0.12 | 0.47 | 0.47 |
| 2 | 0.40 | +0.12 | 0.52 | 0.52 |
| 3 | 0.45 | +0.12 | 0.57 | 0.57 |
| 4 | 0.50 | +0.12 | 0.62 | 0.62 |
| 5 | 0.55 | **−0.05** (new draw) | 0.50 | 0.50 |
| 6 | 0.60 | −0.05 | 0.55 | 0.55 |
| 7 | 0.65 | −0.05 | 0.60 | 0.60 |
| 8 | 0.70 | −0.05 | 0.65 | 0.65 |
| 9 | 0.75 | −0.05 | 0.70 | 0.70 |
| 10 | 0.85 | **+0.21** (new draw) | 1.06 | **1.00** (clipped) |
| 11 | 0.95 | +0.21 | 1.16 | **1.00** (clipped) |

- **Step-shaped offset:** the noise stays fixed for 5 steps at a time, so it acts as a time-correlated bias on the expert's action rather than per-step jitter.
- **Open loop:** the replay starts from the expert's initial state, but the actions differ slightly. The states drift apart, and the replay keeps playing the expert's actions without reacting. It may fail where the expert succeeded, and it is kept anyway.
- **Real actions:** they have several dimensions (e.g. one per joint delta). Each dimension gets its own independent draw, and its std scales with that dimension's range.
- **K replays:** with `--noisy-replays 3`, the same expert episode is replayed 3 times, each with a different noise sequence.

## Things to watch for

- One generator draws noise for the whole batch at once. Each environment slot gets its own noise, but every slot redraws on the same steps.
- Changing `--num-envs` changes both the noise and the initial states. Replays only match their expert because they use the same seeds and batch layout.
- Slots that have already finished are sent zero actions and are not recorded.
