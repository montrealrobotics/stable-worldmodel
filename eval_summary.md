# ResLeWM PushT eval summary

Sources:
- `evals/sched_report`: 600 evals, **20 episodes each** (±~10pp per cell), solver batch size 1, 9 models.
- `evals/sched_n100/report`: 400 evals, **100 episodes each** (±~5pp per cell), solver batch size 100, shared expert goal cache, 5 noisy/proprio models.

Shared across both: OverheadPushT-v1, CEM (300 samples, 30 iters, top-30), action_block 5, receding horizon 1, eval budget 100, seed 9.
Sweep axes: checkpoint epoch, goal offset g ∈ {10, 15, 20, 25}, planning horizon h ∈ {1..5}.

Single cells are noisy; the conclusions below use means over many cells. The two reports agree where they overlap (noisy bn: 24.8 vs 25.7 mean SR; noisy nobn: 18.1 vs 20.3; proprio: 11.1 vs 14.1), so the 20-episode report holds up in aggregate.

## Models

All are `ResLeWM` trained 20 epochs. Data is `overhead_rgb_100k` unless noted.

| Model | Change vs `lewm_res_bn_sched` |
| --- | --- |
| `lewm_res_bn_sched` | baseline: BN projector/pred_proj, cosine SIGReg schedule |
| `lewm_res_nobn_sched` | no norm in projector/pred_proj |
| `lewm_res_rmsnorm_sched` | RMSNorm instead of BN |
| `noisy_lewm_res_bn_sched` | `overhead_rgb_noisy_replay_100k` data |
| `noisy_lewm_res_nobn_sched` | noisy data + no norm |
| `proprio_lewm_res_bn_sched` | `LeWMProprioRes` (qpos + TCP pose), state goal |
| `tc_lewm_res_bn_sched` | SIGReg `temporal_window=4` (also sets `num_steps=4`) |
| `tc_lewm_res_bn` | temporal window, no SIGReg schedule |
| `tc_lewm_res_nobn_sched` | temporal window, no norm |

## Results

Mean success rate (%) over epochs 5/10/15 (common to all models), all horizons:

| Model | All | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| noisy_lewm_res_bn_sched (n100) | **24.8** | **34.8** | **26.7** | 21.3 | **16.3** | **61** |
| noisy_lewm_res_bn_sched | 23.5 | 32.3 | 24.7 | **22.3** | 14.7 | 65 |
| noisy_lewm_res_nobn_sched (n100) | 18.9 | 30.0 | 21.9 | 13.5 | 10.4 | 55 |
| noisy_lewm_res_nobn_sched | 17.3 | 29.0 | 18.0 | 15.0 | 7.3 | 50 |
| lewm_res_bn_sched | 14.9 | 25.7 | 16.3 | 9.0 | 5.5 | 40 |
| proprio_lewm_res_bn_sched (n100) | 13.9 | 28.6 | 15.7 | 7.4 | 3.9 | 46 |
| lewm_res_nobn_sched | 11.6 | 21.0 | 10.7 | 9.7 | 2.0 | 45 |
| lewm_res_rmsnorm_sched | 11.5 | 23.7 | 11.7 | 6.0 | 1.5 | 40 |
| proprio_lewm_res_bn_sched | 11.2 | 26.3 | 11.0 | 5.7 | 2.0 | 40 |
| tc_lewm_res_nobn_sched | 9.0 | 18.0 | 9.5 | 3.0 | 2.0 | 30 |
| tc_lewm_res_bn_sched | 6.2 | 16.3 | 6.0 | 2.0 | 0.3 | 45 |
| tc_lewm_res_bn | 4.5 | 11.3 | 5.3 | 1.3 | 0.0 | 35 |

Rows without "(n100)" come from `sched_report`. Its clean-data models are missing a few cells (55 of 60; `tc_lewm_res_nobn_sched` has 35 and only epochs 5/10).

Mean success rate by planning horizon (all epochs and goals):

| Model | h1 | h2 | h3 | h4 | h5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| noisy bn (n100) | 25.6 | **39.1** | 30.5 | 21.0 | 12.5 |
| noisy nobn (n100) | 19.3 | **29.1** | 24.4 | 17.1 | 11.6 |
| proprio bn (n100) | 14.9 | **19.6** | 17.7 | 11.1 | 7.4 |
| clean bn | 9.5 | 18.6 | **19.5** | 14.1 | 12.7 |
| tc bn sched | 6.2 | **11.2** | 5.6 | 3.4 | 4.4 |

## What works

- **Noisy replay data is the biggest win.** BN model: 14.9 → 23.5 mean SR, and at g25 5.5 → 14.7. It mainly fixes end-effector placement: EEF error at g10/h2 drops from 0.137 to 0.060 m. The no-norm model gains as well (11.6 → 17.3).
- **BN in projector/pred_proj.** It's the best norm in every comparison: bn > nobn by about 6pp on noisy data (in both reports) and about 3pp on clean data. The gap widens at far goals, and bn has lower yaw error (0.165 vs 0.226 rad at g10/h2, n100).
- **Planning horizon h=2.** It's best for almost every model. h=3 ties or wins only at g ≥ 20.
- **Training longer on noisy data.** It's still improving at epoch 20 (bn 22.0 → 28.6, nobn 14.6 → 24.4, n100), so these runs are likely undertrained.

## What doesn't

- **Temporal-window SIGReg (`tc_*`).** It's the worst setting in every variant (4.5–9.0 mean SR, near 0 at g25) with very high yaw error (0.46–0.55 rad). Caveat: it also changes the training sequence length to 4 steps, so the two changes are confounded.
- **Proprio fusion (on clean data).** It isn't better than image-only on the same data (11.2 vs 14.9), it stops improving after epoch 10, and it falls apart at far goals (2–4% at g25). T XY error at g10 is the lowest of all models (0.035 m), so the proprio signal helps local precision but not longer-range planning.
- **RMSNorm.** It's no better than no norm (11.5 vs 11.6).
- **Long horizons (h4–5).** Success drops and EEF error grows with horizon (noisy bn g10: 0.051 → 0.148 m from h2 to h5) while T XY error barely moves. The arm drifts in long imagined rollouts.

## Cosine SIGReg schedule (added 2026-10-19)

No-schedule noisy runs `noisy_lewm_res_bn_11308936` / `noisy_lewm_res_nobn_11308937`, evaluated in `sched_n100` (100 episodes, same goals). Paired over all 80 cells (epoch × g × h):

| Norm | No schedule (mean SR) | Schedule (mean SR) | Schedule − none | Schedule wins |
| --- | ---: | ---: | ---: | ---: |
| BN | **26.6** | 25.7 | −0.9 | 31 / 80 (11 ties) |
| none | 14.2 | **20.3** | **+6.1** | 74 / 80 |

- **With BN the schedule doesn't matter.** The two are tied at every epoch (diffs −2.9 to +0.5). The no-schedule BN run has the best single cell so far: **69%** (epoch 10, g10, h2).
- **Without BN the schedule clearly helps**, by about 6pp at every epoch and about 10pp at h=2. It also lowers yaw error (0.281 → 0.226 rad at g10/h2). BN appears to do the same job as decaying the SIGReg weight, so you only need one of them.
- Noisy data + BN, with or without the schedule, remains the best setting.

## Unclear

- **Clean vs noisy across reports.** The clean-data models only appear in the 20-episode report, before the shared goal cache, so their exact numbers aren't comparable with n100.

## In progress

| Job | Run | Question |
| --- | --- | --- |
| 11310208 | noisy proprio ResLeWM bn, sched | does proprio help once the data is noisy? |
| 11310342 | noisy 20k ResLeWM bn, sched, 100 epochs | data diversity: 4k vs 20k unique starts at equal update count |

## Suggested next steps

1. Train the best setting (noisy data, BN, h=2) longer, since it hasn't plateaued.
2. Re-evaluate the clean `lewm_res_*` models with 100 episodes on the shared goal cache to confirm the clean vs noisy gap exactly.
3. If temporal-window SIGReg is still of interest, rerun it with `num_steps` matched to the baseline to separate the two changes.
