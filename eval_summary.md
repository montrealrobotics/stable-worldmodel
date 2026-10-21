# ResLeWM PushT eval summary

Last updated 2026-10-21. Slides with the same results: https://claude.ai/artifact/A9TxduXwTfy9RdpQDMHL5k

## Protocol

All numbers come from `evals/sched_n100`: 23 runs, 100 episodes per eval, identical start and goal states from the shared expert goal cache, L40S, eval budget 100.

- **Task:** OverheadPushT-v1, 128 px overhead RGB. Success: T within 2 cm and 0.15 rad of its goal pose, and the end effector within 2 cm of its goal position.
- **Planner:** CEM, 300 samples, 30 iterations, top-30; action block 5, receding horizon 1; seed 9.
- **Grid:** 4 checkpoints × goal offsets g ∈ {10, 15, 20, 25} expert steps × planning horizons h ∈ 1–5 = 80 cells per run.
- **Checkpoints:** 20-epoch runs at epochs 5/10/15/20; 100-epoch runs (noisy 20k, ratio sweep) at 25/50/75/100. Both are about the same number of updates, so columns are labelled by share of training.
- **Mean** is over all 80 cells; the g columns average over checkpoints and horizons. One seed per run: trust gaps of 3+ points, not 1–2.
- Three cells are missing (transient dataset read errors): `bnproj` ep10/g20/h3, `bnpred` ep20/g20/h5, ratio `k4` ep100/g15/h4.

Full per-eval report without videos: `reports/sched_n100_novideos/results.{md,html}` (git-ignored; rebuild with `scripts/visualization/gather_eval_results.py <evals>/sched_n100 --no-videos`). Mechanism write-up: `batchnorm_explanation.md`.

## Headline

| Factor | Effect on mean success |
| --- | --- |
| Noisy replay data vs expert-only data | **+9** (14.0 → 23–24) |
| Projector BN, or λ lowered 10× to 0.009 | **+12** (14.2 → 26.0–26.6) |
| Residual vs classic dynamics | **+5** (21.3 → 26.6) |
| Proprio input | **−8** (25.7 → 17.5) |
| Planning horizon | h = 2 best for every run |

## 1. Data: noisy/expert ratio

| Training mix (757k clips/epoch, 100 epochs) | 25% | 50% | 75% | 100% | Mean | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert only | 13.0 | 13.3 | 14.3 | 15.2 | **14.0** | 26.8 | 16.1 | 8.3 | 4.7 | 43 |
| Expert + 1 replay each | 20.6 | 21.7 | 25.0 | 25.6 | **23.2** | 34.8 | 26.2 | 17.9 | 13.8 | 58 |
| Expert + 2 replays each | 23.1 | 23.9 | 24.9 | 23.6 | **23.8** | 37.1 | 26.9 | 17.8 | 13.6 | 61 |
| Expert + 4 replays each | 22.1 | 22.1 | 24.5 | 23.4 | **23.0** | 35.9 | 26.7 | 17.1 | 12.6 | 61 |
| Replays only | 20.2 | 22.6 | 25.1 | 25.1 | **23.3** | 35.0 | 26.6 | 17.9 | 13.5 | 60 |

- **Any replay data adds about 9 points; the mix ratio barely matters.** Replays alone match every mixture.
- Expert-only data also has the worst planned yaw error (0.321 rad at g10/h2, vs 0.18–0.23 with replays).
- Arms are budget-matched: 757k clips per epoch for 100 epochs (≈ 20 epochs of the full set), BN ResLeWM, λ 0.09.

## 2. Data: amount and diversity

| Data | 25% | 50% | 75% | 100% | Mean | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Noisy 100k, BN, cosine λ (20 ep) | 22.0 | 24.5 | 27.9 | 28.6 | **25.7** | 35.5 | 28.3 | 22.1 | 17.1 | 61 |
| Noisy 20k, BN, cosine λ (100 ep) | 21.6 | 23.8 | 26.2 | 26.4 | **24.5** | 36.6 | 27.2 | 19.4 | 14.8 | 65 |
| Noisy 100k, BN, λ 0.09 (20 ep) | 23.2 | 27.4 | 27.4 | 28.6 | **26.6** | 39.1 | 30.1 | 20.7 | 16.7 | 69 |

- At matched updates, fewer distinct episodes costs 1.2 points (noisy 20k vs 100k) to 3.3 points (757k-clip ratio subsets, mean 23.3, vs full data 26.6). Much smaller than dropping the replays.
- Clean vs noisy (proprio models): 14.1 → 17.5. The older 20-episode sweep agrees for image-only models (14.9 → 23.5).

## 3. Model choices

| Model (noisy 100k, BN) | 25% | 50% | 75% | 100% | Mean | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResLeWM, λ 0.09 | 23.2 | 27.4 | 27.4 | 28.6 | **26.6** | 39.1 | 30.1 | 20.7 | 16.7 | 69 |
| Classic LeWM, λ 0.09 | 20.6 | 21.1 | 22.6 | 20.7 | **21.3** | 30.9 | 25.8 | 16.3 | 12.1 | 50 |
| ResLeWM, cosine λ | 22.0 | 24.5 | 27.9 | 28.6 | **25.7** | 35.5 | 28.3 | 22.1 | 17.1 | 61 |
| ResLeWM + proprio, cosine λ | 14.7 | 16.8 | 19.2 | 19.2 | **17.5** | 27.9 | 20.7 | 12.1 | 9.2 | 51 |
| ResLeWM + proprio, cosine λ, clean data | 12.8 | 13.9 | 15.1 | 14.8 | **14.1** | 28.9 | 15.6 | 7.8 | 4.3 | 46 |

- **Residual dynamics beat classic LeWM by 5 points.** Classic LeWM still encodes yaw (probe 0.91) but stops improving after 25% of training, so its gap is in the dynamics.
- **Proprio hurts** with clean or noisy data.

## 4. BatchNorm and SIGReg weight

| Norm, SIGReg weight | 25% | 50% | 75% | 100% | Mean | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BN, λ 0.09 | 23.2 | 27.4 | 27.4 | 28.6 | **26.6** | 39.1 | 30.1 | 20.7 | 16.7 | 69 |
| BN, λ 0.03 | 21.3 | 25.4 | 27.3 | 26.2 | **25.1** | 34.3 | 28.0 | 21.6 | 16.4 | 61 |
| BN, λ 0.009 | 24.1 | 26.4 | 28.0 | 26.7 | **26.3** | 36.2 | 27.6 | 22.2 | 19.1 | 59 |
| BN, cosine 0.09 → 0.009 | 22.0 | 24.5 | 27.9 | 28.6 | **25.7** | 35.5 | 28.3 | 22.1 | 17.1 | 61 |
| No norm, λ 0.09 | 9.2 | 13.8 | 16.8 | 17.1 | **14.2** | 25.8 | 14.7 | 10.2 | 6.0 | 45 |
| No norm, λ 0.03 | 18.2 | 25.4 | 26.4 | 26.3 | **24.1** | 37.4 | 26.3 | 17.8 | 14.9 | 62 |
| No norm, λ 0.009 | 22.0 | 27.8 | 27.3 | 26.8 | **26.0** | 37.4 | 28.1 | 22.2 | 16.2 | 63 |
| No norm, cosine 0.09 → 0.009 | 14.6 | 20.2 | 22.0 | 24.4 | **20.3** | 31.4 | 23.7 | 14.9 | 11.2 | 55 |

- **λ explains the BN effect.** No norm gains 12 points from λ 0.09 → 0.009 (better in 76/80 paired cells); BN is insensitive to λ.
- **The cosine schedule is a partial fix** without BN (+6.1), less than a low constant λ.
- Without BN at λ 0.09, the encoder never learns yaw (CLS probe 0.22 vs 0.95). Projector BN cuts SIGReg's gradient at the encoder about 8×. Details: `batchnorm_explanation.md`.

## 5. BN placement and partial fixes

| Head normalization (λ 0.09) | 25% | 50% | 75% | 100% | Mean | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BN in projector and predictor head | 23.2 | 27.4 | 27.4 | 28.6 | **26.6** | 39.1 | 30.1 | 20.7 | 16.7 | 69 |
| BN in projector only | 23.6 | 25.4 | 27.1 | 28.5 | **26.2** | 38.8 | 28.6 | 19.7 | 17.4 | 66 |
| BN in predictor head only | 8.4 | 14.9 | 17.4 | 17.5 | **14.5** | 28.2 | 15.7 | 8.7 | 5.2 | 49 |
| LayerNorm in both heads | 17.9 | 21.6 | 23.2 | 25.4 | **22.0** | 33.0 | 26.5 | 15.1 | 13.4 | 55 |
| No norm, lr 3e-4 | 18.9 | 20.8 | 21.1 | 21.6 | **20.6** | 32.5 | 22.5 | 14.8 | 12.8 | 64 |
| No norm | 9.2 | 13.8 | 16.8 | 17.1 | **14.2** | 25.8 | 14.7 | 10.2 | 6.0 | 45 |

- **Only the projector's BN matters.** Projector-only ties full BN; predictor-head-only ties no norm.
- LayerNorm heads and a higher learning rate each recover about half the gap. LayerNorm is still climbing at the last checkpoint.

## 6. Aux-yaw diagnostic

| Run (λ 0.09) | 25% | 50% | 75% | 100% | Mean | g10 | g15 | g20 | g25 | Best cell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BN | 23.2 | 27.4 | 27.4 | 28.6 | **26.6** | 39.1 | 30.1 | 20.7 | 16.7 | 69 |
| BN + aux-yaw | 22.2 | 25.9 | 28.6 | 27.6 | **26.1** | 37.4 | 29.0 | 21.6 | 16.4 | 70 |
| No norm + aux-yaw | 13.6 | 21.7 | 23.4 | 23.0 | **20.4** | 32.4 | 23.0 | 14.8 | 11.5 | 56 |
| No norm | 9.2 | 13.8 | 16.8 | 17.1 | **14.2** | 25.8 | 14.7 | 10.2 | 6.0 | 45 |

- A linear head on privileged yaw puts yaw into the no-norm latent (probe 0.95, yaw variance share 1.2% vs BN 1.4%) and recovers only half the gap (+6.2). Planned yaw error barely moves (0.276 vs 0.281 rad).
- One-step error is fine (0.040 vs BN 0.048). Open suspect: yaw over multi-step rollouts.
- With BN, aux-yaw changes nothing (−0.6).

## 7. Planning horizon and goal distance

| Run | h1 | h2 | h3 | h4 | h5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BN, λ 0.09 | 25.8 | **41.1** | 32.9 | 20.4 | 13.1 |
| No norm, λ 0.009 | 21.8 | **37.8** | 31.8 | 23.2 | 15.4 |
| No norm, λ 0.09 | 14.1 | **18.7** | 18.5 | 11.5 | 8.2 |
| Expert-only data | 12.9 | **18.5** | 16.4 | 12.8 | 9.2 |

- **h = 2 is best** for every run; h ≥ 3 helps only at g ≥ 20. EEF error grows with horizon (rollout drift).
- **Far goals stay hard.** g25 tops out at 19.1 (BN λ 0.009); best single g25 cell 31%. Yaw error there is 0.6–0.9 rad: orientation is the main failure.
- Best single cell overall: 70% (BN + aux-yaw, g10/h2).

## All runs

| Run | Checkpoints | 25% | 50% | 75% | 100% | Mean | Best cell | Yaw err g10/h2 (rad) | Cells |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noisy_lewm_res_bn_11308936` | 5/10/15/20 | 23.2 | 27.4 | 27.4 | 28.6 | **26.6** | 69 | 0.175 | 80 |
| `noisy_lewm_res_bn_w0.009_11311540` | 5/10/15/20 | 24.1 | 26.4 | 28.0 | 26.7 | **26.3** | 59 | 0.131 | 80 |
| `noisy_lewm_res_bnproj_11312006` | 5/10/15/20 | 23.6 | 25.4 | 27.1 | 28.5 | **26.2** | 66 | 0.200 | 79 |
| `noisy_lewm_res_bn_auxyaw0.1_11312360` | 5/10/15/20 | 22.2 | 25.9 | 28.6 | 27.6 | **26.1** | 70 | 0.185 | 80 |
| `noisy_lewm_res_nobn_w0.009_11311542` | 5/10/15/20 | 22.0 | 27.8 | 27.3 | 26.8 | **26.0** | 63 | 0.135 | 80 |
| `noisy_lewm_res_bn_sched_11302701` | 5/10/15/20 | 22.0 | 24.5 | 27.9 | 28.6 | **25.7** | 61 | 0.165 | 80 |
| `noisy_lewm_res_bn_w0.03_11311539` | 5/10/15/20 | 21.3 | 25.4 | 27.3 | 26.2 | **25.1** | 61 | 0.140 | 80 |
| `noisy20k_lewm_res_bn_sched_11310342` | 25/50/75/100 | 21.6 | 23.8 | 26.2 | 26.4 | **24.5** | 65 | 0.172 | 80 |
| `noisy_lewm_res_nobn_w0.03_11311541` | 5/10/15/20 | 18.2 | 25.4 | 26.4 | 26.3 | **24.1** | 62 | 0.157 | 80 |
| `noisy_ratio_k2_lewm_res_bn_11326940` | 25/50/75/100 | 23.1 | 23.9 | 24.9 | 23.6 | **23.8** | 61 | 0.205 | 80 |
| `noisy_ratio_replays_only_lewm_res_bn_11326937` | 25/50/75/100 | 20.2 | 22.6 | 25.1 | 25.1 | **23.3** | 60 | 0.178 | 80 |
| `noisy_ratio_k1_lewm_res_bn_11326939` | 25/50/75/100 | 20.6 | 21.7 | 25.0 | 25.6 | **23.2** | 58 | 0.232 | 80 |
| `noisy_ratio_k4_lewm_res_bn_11326938` | 25/50/75/100 | 22.1 | 22.1 | 24.5 | 23.4 | **23.0** | 61 | 0.195 | 79 |
| `noisy_lewm_res_ln_11311981` | 5/10/15/20 | 17.9 | 21.6 | 23.2 | 25.4 | **22.0** | 55 | 0.257 | 80 |
| `noisy_lewm_bn_11311636` | 5/10/15/20 | 20.6 | 21.1 | 22.6 | 20.7 | **21.3** | 50 | 0.170 | 80 |
| `noisy_lewm_res_nobn_lr3e-4_11312218` | 5/10/15/20 | 18.9 | 20.8 | 21.1 | 21.6 | **20.6** | 64 | 0.209 | 80 |
| `noisy_lewm_res_nobn_auxyaw0.1_11312361` | 5/10/15/20 | 13.6 | 21.7 | 23.4 | 23.0 | **20.4** | 56 | 0.276 | 80 |
| `noisy_lewm_res_nobn_sched_11302702` | 5/10/15/20 | 14.6 | 20.2 | 22.0 | 24.4 | **20.3** | 55 | 0.226 | 80 |
| `noisy_proprio_lewm_res_bn_sched_11310208` | 5/10/15/20 | 14.7 | 16.8 | 19.2 | 19.2 | **17.5** | 51 | 0.165 | 80 |
| `noisy_lewm_res_bnpred_11312007` | 5/10/15/20 | 8.4 | 14.9 | 17.4 | 17.5 | **14.5** | 49 | 0.252 | 79 |
| `noisy_lewm_res_nobn_11308937` | 5/10/15/20 | 9.2 | 13.8 | 16.8 | 17.1 | **14.2** | 45 | 0.281 | 80 |
| `proprio_lewm_res_bn_sched_11302682` | 5/10/15/20 | 12.8 | 13.9 | 15.1 | 14.8 | **14.1** | 46 | 0.241 | 80 |
| `noisy_ratio_expert_lewm_res_bn_11326936` | 25/50/75/100 | 13.0 | 13.3 | 14.3 | 15.2 | **14.0** | 43 | 0.321 | 80 |

## Older 20-episode sweep (`evals/sched_report`)

Before the shared goal cache; 20 episodes per eval, so compare only means. Settings not rerun at 100 episodes:

- **Temporal-window SIGReg (`tc_*`) is the worst setting** (4.5–9.0 mean success, near 0 at g25, yaw error 0.46–0.55 rad). It also changes the sequence length to 4 steps, so the two are confounded.
- **RMSNorm ≈ no norm** (11.5 vs 11.6, clean data).
- **Clean vs noisy, BN, image-only:** 14.9 → 23.5. Noisy data mainly fixes end-effector placement (EEF error at g10/h2: 0.137 → 0.060 m).

## In progress

| Job | Run | Status |
| --- | --- | --- |
| 11333907 | DINO-WM, frozen DINOv3 ViT-S/16, CLS token, batch 128, lr 5e-4 | training, epoch 9/10 |
| 11333908 | DINO-WM, 64 patch tokens, batch 128, lr 5e-4 | training, epoch 8/10 |
| not submitted | DINO-WM, CLS token, batch 1024, lr 1.4e-3, warmup + cosine | command ready |

Evaluate at epochs 3/5/8/10 on the same grid (patch model may need a smaller `solver.batch_size`).

## Next steps

1. DINO-WM evals once training finishes.
2. SIGReg weight below 0.009 (0.003, 0.001, 0) for BN and no norm, to find where collapse starts.
3. Probe yaw on predicted latents over multi-step rollouts (aux-yaw follow-up).
4. Far-goal planning: orientation error dominates at g ≥ 20.
