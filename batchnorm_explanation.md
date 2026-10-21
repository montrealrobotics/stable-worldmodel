# Why BatchNorm helped ResLeWM plan, and what it really does

## Summary

At the default SIGReg weight (λ = 0.09), ResLeWM with BatchNorm (BN) in its heads planned PushT far better than without it: 26.6% vs 14.2% mean success. Follow-up runs and diagnostics show **BN is not needed for a good model. It protects the encoder from too much SIGReg pressure.**

1. **Lowering SIGReg's weight makes the no-norm model as good as BN.** No-norm at λ = 0.009 reaches 26.0% (vs 26.6% for BN); BN is insensitive to λ (26.6 / 25.1 / 26.3 at λ = 0.09 / 0.03 / 0.009).
2. **The failure is the encoder not learning the T's orientation (yaw).** The no-norm λ = 0.09 model's ViT output barely encodes yaw (linear probe R² 0.22 vs 0.95 for BN), and never learns it: 0.03 at epoch 5, −0.12 at epoch 1. The task requires yaw within 0.15 rad.
3. **Only the BN in the projector matters.** BN in the projector alone reproduces the full effect (yaw 0.94, 26.2% success). BN in the predictor head alone does nothing (yaw 0.14, 14.5%).
4. **Mechanism, confirmed with training logs:** the projector's BN shrinks the SIGReg gradient reaching the encoder by about 8× early in training (norm 54 vs 443). Without it, SIGReg dominates the encoder's update 12.6:1 over the prediction loss, and the weak yaw signal from prediction can't get through. Anything that lowers that ratio (lower λ, projector BN) restores yaw. Partial fixes (LayerNorm heads, a higher learning rate) partially restore it, and recover about half the planning gap (22.0% and 20.6%).
6. **Yaw in the latent is necessary but not sufficient.** Forcing yaw into the no-norm embedding with a supervised head (aux-yaw) recovers only half the gap (20.4%) and barely lowers planned yaw error.
5. **Isotropy is not what makes a latent good for planning.** The best planners at λ = 0.009 have *less* isotropic latents (effective rank 76–80 of 192, higher SIGReg loss) than the λ = 0.09 BN model (rank 147).

## 1. Terms

- **Linear probe R².** Fit a ridge (lightly regularized linear) regression from a representation to a target on half of 2048 sample frames, then score it on the other half: R² = 1 − (prediction error variance / variance of the true values).
  - *Prediction error variance* is the average squared probe error.
  - *Variance of the true values* is the error of always predicting the mean.
  - 1 means the target is perfectly linearly recoverable; 0 means no better than guessing the mean; it can go slightly negative on held-out data.
  - Yaw is encoded as (sin, cos). Linear probes matter because the planning cost is a squared Euclidean distance between latents.
- **CLS:** the ViT's output token, the encoder's representation before the projector. "Before LN" means before the ViT's final LayerNorm.
- **Variance explained by a factor:** R² of predicting the representation from a quadratic function of one state factor (yaw, T xy, EEF xy). It is the share of the latent, and hence of the planning cost, driven by that factor.
- **Effective rank:** (Σλ)² / Σλ² of the embedding covariance eigenvalues. 192 means perfectly isotropic, 1 means collapsed.
- **SIGReg statistic:** the training regularizer on the diagnostic batch with fixed random directions. Lower means closer to N(0, I). Compare it only between models.
- **One-step error:** ‖predicted e_{t+1} − e_{t+1}‖² / ‖e_{t+1} − e_t‖².
- **Encoder gradient ratio:** logged during training as `fit/relative_grad_magnitude` = λ · ‖∇ SIGReg‖ / ‖∇ prediction loss‖, with both gradients taken with respect to the encoder's parameters. Above 1 means SIGReg dominates the encoder's update. `fit/grad_cosine_similarity` is the cosine between the two gradients.

## 2. Setup

```
pixels ─ ViT-tiny (LayerNorm in 12 blocks) ─ final LayerNorm ─ CLS (192) ─ projector ─ emb (192)
                                                                            Linear 192→2048
                                                                            BatchNorm1d   ← BN / LN / nothing
                                                                            GELU
                                                                            Linear 2048→192

emb_t, action ─ predictor ─ pred_proj (same MLP) ─ Δ ;   emb_{t+1} = emb_t + Δ        (ResLeWM)

loss          = MSE(pred, emb_{t+1}) + λ · SIGReg(emb)       (default λ = 0.09)
planning cost = Σ_d (rollout_emb_T − goal_emb)²                (GoalMSE; CEM, 300 samples, 30 iterations)
```

- **SIGReg** (LeJEPA): it projects the batch's embeddings onto 1024 random unit directions, resampled every step, and tests each projection against N(0,1) with the Epps–Pulley characteristic-function statistic. The batch mean is taken over the batch axis (`x_t.cos().mean(-3)` in `stable_worldmodel/wm/loss.py`), so it constrains the batch *distribution*: zero mean, unit variance in every direction, Gaussian shape, full rank. It constrains the shape of the latent, not which information fills it.
- **LeWM's projector:** the paper adds the projector because *"the final ViT layer applies a Layer Normalization, which prevents our anti-collapse objective from being optimized effectively"*. It specifies a *"1-layer MLP with Batch Normalization"* and doesn't ablate the normalization choice.
- **BN at inference** (`model.eval()`) uses running statistics and is a fixed affine map, foldable into the preceding `Linear`. Every effect of BN therefore acts through training.

## 3. Evaluation results

All 100-episode evals in `evals/sched_n100` use noisy-replay data, identical start and goal states, and the mean over 4 checkpoints × 4 goal offsets (g) × 5 planning horizons (h). "Better in" counts the 80 paired settings where the run beat its comparison.

| ResLeWM run | Mean success (%) | Best cell | Yaw error at g10/h2 (rad) | vs reference |
| --- | ---: | ---: | ---: | --- |
| BN, λ = 0.09 (reference) | **26.6** | 69 | 0.175 | |
| BN, λ = 0.009 | 26.3 | 59 | **0.131** | −0.4 (better in 38/80) |
| no norm, λ = 0.009 | 26.0 | 63 | 0.135 | vs no norm λ = 0.09: **+11.8 (76/80)** |
| BN, cosine schedule 0.09 → 0.009 | 25.7 | 61 | 0.165 | |
| BN, λ = 0.03 | 25.1 | 61 | 0.140 | −1.6 (25/80) |
| no norm, λ = 0.03 | 24.1 | 62 | 0.157 | vs no norm λ = 0.09: +9.9 (80/80) |
| no norm, cosine schedule | 20.3 | 55 | 0.226 | |
| no norm, λ = 0.09 | 14.2 | 45 | 0.281 | |

Variants at λ = 0.09 (BN placement, partial fixes, aux-yaw from section 7):

| ResLeWM run, λ = 0.09 | Mean success (%) | Best cell | Yaw error at g10/h2 (rad) | vs reference |
| --- | ---: | ---: | ---: | --- |
| BN in projector only | 26.2 | 66 | 0.200 | vs BN: −0.4 (32/79) |
| BN + aux-yaw | 26.1 | 70 | 0.185 | vs BN: −0.6 (39/80) |
| LayerNorm heads | 22.0 | 55 | 0.257 | vs no norm: **+7.8 (74/80)** |
| no norm, lr 3e-4 | 20.6 | 64 | 0.209 | vs no norm: +6.4 (65/80) |
| no norm + aux-yaw | 20.4 | 56 | 0.276 | vs no norm: +6.2 (73/80); vs BN: −6.2 (11/80) |
| BN in pred_proj only | 14.5 | 49 | 0.252 | vs no norm: +0.2 (37/79) |

- **λ explains the norm effect.** No-norm improves by about 10–12 points just from lowering λ, and the λ = 0.009 runs have the lowest yaw errors of all.
- **BN makes λ irrelevant:** the three BN weights tie within noise.
- **The schedule is a partial fix:** decaying λ late in training helps no-norm (+6.1) but not as much as a low constant λ (+5.7 more).
- **Only projector BN matters, confirmed in planning.** Projector-only BN ties full BN; pred_proj-only BN ties no norm.
- **The partial fixes are partial in planning too.** LayerNorm heads, lr 3e-4 and aux-yaw each recover about half of the 12-point gap (+6 to +8). None reaches BN or low-λ no-norm.

One cell is missing for each BN-placement run (bnproj ep10/g20/h3, bnpred ep20/g20/h5); their comparisons use 79 pairs.

## 4. Where yaw is lost: the encoder

Epoch-20 checkpoints, `scripts/plan/embedding_diagnostics.py`:

| Linear probe R² | BN λ=0.09 | no norm λ=0.09 | no norm λ=0.009 |
| --- | ---: | ---: | ---: |
| Yaw, CLS before the final LN | 0.94 | 0.21 | 0.89 |
| Yaw, CLS after the final LN | 0.95 | 0.22 | 0.89 |
| Yaw, embedding | 0.90 | 0.17 | 0.87 |
| T xy, CLS | 0.96 | 0.93 | 0.97 |
| EEF xy, CLS | 0.96 | 0.96 | 0.98 |
| Share of CLS variance explained by yaw | 6.2% | 0.9% | 9.7% |

- **The ViT itself drops yaw** in the failing model, while still encoding T and arm position well.
- **Not the final LayerNorm:** values are the same before and after it.
- **Not the projector:** CLS and embedding carry about the same yaw information.
- An earlier version of this document blamed the projector amplifying or losing small yaw-tracking units. Hidden-layer measurements contradict that. In the BN model, yaw units are as large as the others (1.01× the median std) and BN's rescaling doesn't change yaw's share. The no-norm projector has no yaw units, because its input has no yaw.

## 5. When it happens: from the first epoch

Probe R² on the CLS token. The early runs train for 5 epochs with checkpoints every epoch; the main runs train for 20.

| Yaw, CLS | ep1 | ep2 | ep3 | ep5 | ep10 | ep15 | ep20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BN λ=0.09, 5-epoch run | 0.37 | 0.57 | 0.67 | 0.72 | | | |
| no norm λ=0.09, 5-epoch run | −0.12 | 0.05 | 0.09 | 0.14 | | | |
| BN λ=0.09, 20-epoch run | | | | 0.86 | 0.94 | 0.95 | 0.95 |
| no norm λ=0.09, 20-epoch run | | | | 0.04 | 0.15 | 0.20 | 0.22 |
| no norm λ=0.009, 20-epoch run | | | | 0.52 | | | 0.89 |

- **BN starts learning yaw in epoch 1.** No-norm at λ = 0.09 doesn't start at all for several epochs, then creeps up with diminishing gains.
- **BN accelerates all of state learning early.** At epoch 1, T xy is 0.86 with BN vs 0.21 without; EEF is 0.82 vs 0.32. No-norm catches up on those strong factors within a few epochs. It never catches up on yaw, the weakest one.
- **Lowering λ alone** gives no-norm 0.52 yaw by epoch 5 and 0.89 by epoch 20.

## 6. Mechanism: the SIGReg gradient overwhelms the encoder

Training logs, read from the runs' local wandb files: the ratio of SIGReg's to the prediction loss's gradient at the encoder, averaged over the first 10% of training (when features form) and the last 20%. Yaw probe from the epoch-20 diagnostics.

| Run | Encoder gradient ratio, first 10% | last 20% | ‖∇ SIGReg‖ at encoder, first 10% | Grad cosine, first 10% | Yaw probe, CLS |
| --- | ---: | ---: | ---: | ---: | ---: |
| BN λ=0.009 | **0.9** | 0.5 | 118 | −0.39 | **0.97** |
| no norm λ=0.009 | 2.6 | 0.9 | 341 | −0.12 | 0.89 |
| BN in projector only | 3.2 | 1.3 | **49** | −0.30 | 0.94 |
| BN λ=0.09 | 3.3 | 1.3 | **54** | −0.31 | 0.95 |
| no norm λ=0.03 | 6.1 | 1.5 | 492 | +0.08 | 0.69 |
| BN in pred_proj only | 9.1 | 2.7 | 376 | +0.16 | 0.14 |
| no norm λ=0.09 | **12.6** | 2.5 | **443** | +0.20 | **0.22** |
| no norm λ=0.09, lr 3e-4 | 14.3 | 2.2 | 494 | +0.21 | 0.78 |
| LayerNorm heads λ=0.09 | 14.5 | 2.9 | 668 | −0.11 | 0.64 |

### 6.1 What the logs show

- **Projector BN absorbs most of SIGReg's gradient.** With BN in the projector, the SIGReg gradient reaching the encoder is about 8× smaller (49–54 vs 376–668). BN in the predictor head doesn't do this (376), and neither does LayerNorm (668).
- **The ratio predicts yaw for every λ and BN-placement variant.** Ratios of 0.9–3.3 give yaw R² 0.89–0.97. Ratios of 6.1 give 0.69, and ratios of 9–13 give 0.14–0.22.
- **λ and projector BN are two routes to the same place.** Lowering λ shrinks the ratio by scaling SIGReg down; BN shrinks it by blocking SIGReg's gradient. Doing both (BN λ = 0.009) gives the lowest ratio and the highest yaw.

### 6.2 Why BN absorbs it

SIGReg asks for zero mean, unit variance and Gaussian shape along every direction. Much of its gradient is about the first two moments, the per-direction mean and scale of the batch. In training mode, BN normalizes each hidden unit by the batch's own mean and std. Its backward pass removes from the gradient exactly the components that would change a unit's batch mean or scale. So the moment-fixing part of SIGReg's gradient is handled inside the projector, by BN's normalization and learned affine parameters, and doesn't propagate into the encoder. What reaches the encoder is mostly the prediction gradient, which carries the weak yaw signal.

Without BN, the encoder itself must reshape its output to satisfy SIGReg. At λ = 0.09 that gradient is 12.6× the prediction gradient and points in a positively correlated direction (cosine +0.2). The encoder becomes a feature factory for whatever spreads the batch into a Gaussian most cheaply: arm position, T position, appearance. Yaw, a small visual change that only the prediction loss asks for, is starved.

### 6.3 Partial fixes

- **A higher learning rate** (no norm, lr 3e-4) doesn't change the ratio (14.3), but larger steps let the prediction gradient's small share still make progress: yaw 0.78. Planning: 20.6% (+6.4 over no norm, 6 below BN).
- **LayerNorm heads** don't block the SIGReg gradient (ratio 14.5, gradient norm 668) yet reach yaw 0.64. Per-sample normalization changes the geometry SIGReg acts through; its gradient is less aligned with prediction (cosine −0.11 vs +0.20). This is less understood. Planning: 22.0% (+7.8 over no norm, 4.6 below BN).
- The yaw probe doesn't rank these two: lr 3e-4 has the higher probe (0.78 vs 0.64) but the lower success (20.6 vs 22.0). The difference is small for one seed.
- **The SIGReg schedule** only lowers λ late in training. But yaw has to be learned early, when the ratio is highest, so the schedule recovers only part of the gap.

## 7. Supervised yaw (aux-yaw diagnostic)

A linear head trained to predict yaw from the embedding (weight 0.1, privileged simulator pose) forces yaw into the latent:

| | BN + aux | no norm + aux | BN | no norm |
| --- | ---: | ---: | ---: | ---: |
| Yaw probe, embedding | 0.99 | 0.95 | 0.90 | 0.17 |
| Yaw probe, CLS | 0.97 | 0.77 | 0.95 | 0.22 |
| Mean success (%) | 26.1 | 20.4 | 26.6 | 14.2 |
| Yaw error at g10/h2 (rad) | 0.185 | 0.276 | 0.175 | 0.281 |

Supervision gets yaw in, even through the no-norm encoder, but **missing yaw doesn't fully explain the gap.**

- **No norm + aux recovers half the gap:** +6.2 over no norm (better in 73/80), still 6.2 below BN (better in only 11/80).
- **Its planned yaw barely improves** (0.276 vs 0.281 rad at g10/h2), even though yaw is now linearly decodable from the embedding (0.95). The success gain must come from elsewhere, for example a better-shaped latent overall.
- **Linearly decodable is not the same as plannable.** Diagnostics at epoch 20 (same 2048 samples) rule out the two simple explanations:

  | Epoch 20 | Yaw share of embedding variance | T xy share | EEF share | One-step error | Yaw probe, CLS |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | BN | 1.4% | 1.5% | 1.4% | 0.048 | 0.95 |
  | BN + aux | 1.5% | 1.5% | 1.4% | 0.046 | 0.97 |
  | no norm + aux | 1.2% | 1.7% | 1.4% | 0.040 | 0.77 |
  | no norm | 0.1% | 2.0% | 1.5% | 0.035 | 0.22 |
  | no norm λ=0.009 | 2.9% | 3.8% | 3.4% | 0.014 | 0.89 |

  - **Cost weighting is not it:** no-norm + aux gives yaw almost the same share of the planning cost as BN (1.2% vs 1.4%).
  - **Average one-step error is not it:** 0.040, better than BN's 0.048.
  - **What differs:** the encoder itself still carries less yaw (CLS probe 0.77 vs 0.95). The aux head builds yaw in the projector from a weaker encoder signal. The open suspect is yaw-specific dynamics: one-step error averages over all factors, so it can hide a predictor that doesn't move yaw correctly. Next check: probe yaw on *predicted* embeddings over multi-step rollouts.
  - The best planner (no norm, λ = 0.009) gives every task factor about twice the variance share (yaw 2.9%, T xy 3.8%) and has a 3× lower one-step error.
- **BN + aux changes nothing** (−0.6): once yaw is present, more of it doesn't help.

## 8. Isotropy is not the goal

| Epoch 20 | Effective rank | SIGReg stat | One-step error | Mean success (%) |
| --- | ---: | ---: | ---: | ---: |
| no norm λ=0.009 | 76 | 6.3 | 0.014 | 26.0 |
| BN λ=0.009 | 80 | 6.1 | 0.015 | 26.3 |
| BN λ=0.09 | 147 | 2.3 | 0.048 | 26.6 |
| no norm λ=0.09 | 134 | 2.8 | 0.035 | 14.2 |
| no norm lr 3e-4 | 170 | 1.8 | 0.027 | 20.6 |
| classic LeWM BN λ=0.09 | 154 | 2.0 | 0.035 | 21.3 |

- **Low-λ latents are far less isotropic yet plan best.** They also predict more accurately (one-step error 0.014 vs 0.048).
- **The highest-rank latents** (no-norm lr 3e-4: 170; classic LeWM: 154) are not the best planners.
- **SIGReg is an anti-collapse device, not a quality target.** It needs only enough weight to prevent collapse. Beyond that, it competes with learning task features.

This is consistent with the LeWM paper's own λ sweep (PushT success above 80% for λ ∈ [0.01, 0.2], peaking near 0.09). Our encoder without projector BN is more sensitive than theirs.

## 9. How this fits with other self-supervised methods

Modern backbones prefer LayerNorm, RMSNorm or GroupNorm because they are per-sample. That matters for variable-length sequences, autoregressive inference, small per-GPU batches and multi-GPU training, none of which apply to LeWM's heads. The ViT backbone keeps LayerNorm; "LayerNorm in the backbone, BatchNorm in the projection head" is the standard self-supervised pattern (SimCLR, BYOL, SimSiam, VICReg, Barlow Twins).

Methods whose learned target can collapse need something that looks across the batch:

| Method | Can the target collapse? | Batch-level mechanism |
| --- | --- | --- |
| MAE | No (target is pixels) | none |
| DINO / DINOv2 / iBOT | Yes | in the loss: teacher centering, Sinkhorn–Knopp, KoLeo; no BN with ViT |
| SimCLR / MoCo | Yes | in the loss: negatives |
| VICReg | Yes | in the loss: per-dimension variance and covariance |
| BYOL / SimSiam (ResNet) | Yes | in the architecture: BN in heads, plus asymmetry |
| I-JEPA | Yes | asymmetry: EMA teacher plus predictor |
| LeWM | Yes | in the loss: SIGReg (projector BN shields the encoder from it) |

The lesson here is specific to regularizer-based methods: a batch-statistics regularizer can push the encoder too hard, and a projector with BN is a buffer that lets the regularizer act on the head while the encoder follows the predictive loss. The same role has been noted for projector heads in contrastive learning, where the projector absorbs the invariances the loss imposes.

## 10. Caveats

- **One seed per configuration.** The large effects (λ: +10–12 points, better in 76–80 of 80 cells) are robust. Differences of 1–2 points are not.
- **Linear probes only.** GoalMSE sees only linear/Euclidean structure.
- **Small diagnostic sample:** 2048 transitions, BN in eval mode.
- **The 5-epoch early runs** use a different learning-rate schedule from the 20-epoch runs; compare trends, not values.
- **The gradient ratio is logged every 100 steps**, averaged over phases.
- **Partial-fix differences** (LayerNorm 22.0, lr 3e-4 20.6, no norm + aux 20.4) are within 2 points of each other and not ranked reliably.

## 11. Practical guidance

- **Use BN in the projector.** It makes training robust to λ. BN in `pred_proj` is optional.
- **If you drop projector BN,** lower λ to about 0.01. The decaying schedule doesn't substitute.
- **Don't judge latents by SIGReg loss or rank.** Probe the task factors (yaw, T xy, EEF), and watch `fit/relative_grad_magnitude`: values well above ~3 early in training predict the yaw failure.
- **Next experiments:**
  1. Probe yaw on predicted embeddings over multi-step rollouts (aux-yaw runs vs BN) to test whether the predictor moves yaw correctly.
  2. Sweep λ lower (0.003, 0.001, 0) for BN and no-norm, to find where collapse starts.
  3. Vary the noisy/expert ratio.
  4. Try InfoNCE as a replacement for SIGReg.

## 12. Reproducing

- **Diagnostics:** `scripts/plan/embedding_diagnostics.py label=<run>/weights_epoch_N.pt ...`, run inside the interactive job with `STABLEWM_HOME=/network/scratch/o/ozgur.aslan/le-wm`.
- **Gradient logs:** wandb keys `fit/relative_grad_magnitude`, `fit/grad_cosine_similarity`, `fit/pred_grad_norm` and `fit/sigreg_grad_norm`, logged every 100 steps by `encoder_gradient_diagnostics` (`stable_worldmodel/wm/diagnostics.py`). They can be read offline from `checkpoints/<run>/wandb/wandb/run-*/run-*.wandb`.
- **Runs:**

  | Run | Checkpoint directory |
  | --- | --- |
  | BN λ=0.09 | `noisy_lewm_res_bn_11308936` |
  | no norm λ=0.09 | `noisy_lewm_res_nobn_11308937` |
  | BN λ=0.03 / 0.009 | `noisy_lewm_res_bn_w0.03_11311539` / `noisy_lewm_res_bn_w0.009_11311540` |
  | no norm λ=0.03 / 0.009 | `noisy_lewm_res_nobn_w0.03_11311541` / `noisy_lewm_res_nobn_w0.009_11311542` |
  | BN schedule / no norm schedule | `noisy_lewm_res_bn_sched_11302701` / `noisy_lewm_res_nobn_sched_11302702` |
  | LayerNorm heads | `noisy_lewm_res_ln_11311981` |
  | BN in projector only / pred_proj only | `noisy_lewm_res_bnproj_11312006` / `noisy_lewm_res_bnpred_11312007` |
  | no norm lr 3e-4 | `noisy_lewm_res_nobn_lr3e-4_11312218` |
  | early runs, BN / no norm | `noisy_lewm_res_bn_early_11312227` / `noisy_lewm_res_nobn_early_11312228` |
  | aux-yaw BN / no norm | `noisy_lewm_res_bn_auxyaw0.1_11312360` / `noisy_lewm_res_nobn_auxyaw0.1_11312361` |
  | classic LeWM BN | `noisy_lewm_bn_11311636` |

## Sources

- Balestriero & LeCun, *LeJEPA*, arXiv:2511.08544 — https://arxiv.org/abs/2511.08544
- Maes et al., *LeWorldModel*, arXiv:2603.19312 — https://arxiv.org/html/2603.19312v2
- *SIGReg: What It Computes, Constrains, and Where It Breaks* — https://rezabyt.github.io/blogposts/sigreg-tutorial.html
- Self-supervised methods in section 9: MAE, DINO, DINOv2, iBOT, SimCLR, MoCo, BYOL, SimSiam, VICReg, Barlow Twins, I-JEPA; Richemond et al. 2020, *BYOL works even without batch statistics*.
