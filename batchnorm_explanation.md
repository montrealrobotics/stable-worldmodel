# Why the BatchNorm ResLeWM models plan best

## Summary

BatchNorm (BN) in the projector and the predictor's output projection (`pred_proj`) makes ResLeWM's latent space **encode the T block's orientation**. Without it, the orientation is mostly lost.

| Embedding diagnostic (noisy data, epoch 20) | BN | no norm | BN + sched | no norm + sched |
| --- | ---: | ---: | ---: | ---: |
| Linear probe R², T yaw (sin, cos) | **0.90** | 0.17 | **0.92** | 0.35 |
| Linear probe R², T xy | 0.94 | 0.91 | 0.95 | 0.92 |
| Linear probe R², EEF xy | 0.75 | 0.69 | 0.78 | 0.70 |

(Diagnostic details are in "Diagnostics" below.)

Task success requires yaw error under 0.15 rad. A planner minimizing latent distance to the goal can't fix an orientation its latent barely represents. Two things explain why BN preserves it:

1. **BN fits SIGReg.** SIGReg is a constraint on the distribution of embeddings across the batch. BN normalizes each hidden unit using batch statistics, so the BN projector reaches a more isotropic, higher-rank latent: lower SIGReg loss and higher effective rank.
2. **BN equalizes feature scales.** BN rescales every hidden unit to unit variance across the batch. Subtle, low-variance visual features such as the T's orientation get amplified to the same scale as dominant ones such as arm and T position, so they survive into the latent instead of being drowned out.

BN has no extra capacity at inference: in eval mode it is a fixed affine map that folds into the adjacent linear layer. The entire benefit comes from how it shapes training.

## 1. What the evals show

From `evals/sched_n100` (100 episodes per eval, identical start and goal states, noisy-replay data):

| ResLeWM | Mean success (%) | Best cell | g10 | g25 | Epoch 5 | Epoch 20 | Yaw error at g10/h2 (rad) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BN, constant SIGReg | **26.6** | **69** | 39.1 | 16.7 | **23.2** | 28.6 | 0.175 |
| BN, cosine schedule | 25.7 | 61 | 35.5 | 17.1 | 22.0 | 28.6 | **0.165** |
| no norm, cosine schedule | 20.3 | 55 | 31.4 | 11.2 | 14.6 | 24.4 | 0.226 |
| no norm, constant SIGReg | 14.2 | 45 | 25.8 | 6.0 | 9.2 | 17.1 | 0.281 |

The BN models differ from the no-norm models in four ways:
- **Higher success** in every comparison, with the gap widening for farther goals: at g25, 16.7 vs 6.0.
- **Lower yaw error**, while T XY error (0.037–0.046 m) and EEF error (0.049–0.053 m) are similar. The models differ mainly in **orientation**.
- **Faster learning:** at epoch 5, BN is already at 23% while no-norm is at 9%.
- **Indifference to the SIGReg schedule.** With BN, constant and decayed weights tie (−0.9 points, scheduled better in 31 of 80 cells). Without BN, the schedule is worth +6.1 points (scheduled better in 74 of 80 cells).

On clean data, in the 20-episode `sched_report`, the ordering also holds: BN 14.9 > no norm 11.6 ≈ RMSNorm 11.5.

## 2. Where BN sits

```
pixels ─ ViT-tiny ─ CLS (192, after the ViT's final LayerNorm) ─ projector ─ emb
                                                                  Linear 192→2048
                                                                  BatchNorm1d     ← BN or nothing
                                                                  GELU
                                                                  Linear 2048→192

emb_t, action ─ predictor ─ pred_proj (same MLP) ─ Δ ;   emb_{t+1} = emb_t + Δ   (ResLeWM)

loss = MSE(pred, emb_{t+1}) + λ · SIGReg(emb)
planning cost = Σ_d (rollout_emb_T − goal_emb)²     (GoalMSE, sum over 192 dims)
```

- **Training:** BN normalizes each of the 2048 hidden units with the batch mean and variance (batch 1024 × frames), then applies a learned scale and shift.
- **Evaluation:** `eval_wm.py` calls `model.eval()`, so BN uses its running statistics and becomes a fixed per-unit affine map. That can be merged into the preceding `Linear`. A BN model at inference is therefore in the same function class as the no-norm model. The difference must come from which solution training finds, not from capacity.

## 3. Background: SIGReg and why LeWM has a projector

**LeJEPA** (Balestriero & LeCun, 2025):
- It argues that the isotropic Gaussian is the embedding distribution that minimizes downstream prediction risk. Anisotropy amplifies both the bias and the variance of probes.
- It introduces **SIGReg** to enforce that distribution. SIGReg projects the batch onto many random unit directions (1024 here, resampled every step). For each direction it compares the empirical characteristic function of the 1-D projection with that of N(0,1) using the Epps–Pulley test.
- By the Cramér–Wold theorem, if every 1-D projection is N(0,1), the embedding is N(0, I). So SIGReg pushes for zero mean, unit variance in every direction, Gaussian shape and full rank, which rules out collapse.
- SIGReg constrains the batch distribution, not individual samples.

**LeWorldModel** (Maes et al., 2026):
- It trains end to end from pixels with only the next-embedding MSE plus SIGReg (λ ≈ 0.1; PushT success stays above 80% for λ ∈ [0.01, 0.2]).
- About the projector, the paper says: *"This step is necessary because the final ViT layer applies a Layer Normalization, which prevents our anti-collapse objective from being optimized effectively."* The encoder and predictor projectors are a *"1-layer MLP with Batch Normalization."*
- The paper doesn't ablate the normalization choice, and the SIGReg tutorial doesn't discuss normalization layers. The explanation below is our own reasoning plus the diagnostics.

## 4. Mechanisms

### 4.1 BN matches what SIGReg measures

LayerNorm and RMSNorm normalize each sample over its own features. Every embedding is forced onto roughly the same shell, which conflicts with N(0, I), where the norms vary across samples. Per-sample normalization also says nothing about how a feature is distributed across the batch, and that distribution is exactly what SIGReg tests.

BN does the opposite: each hidden unit gets zero mean and unit variance across the batch. The output `Linear` then maps standardized, decorrelation-friendly features to the embedding. Producing marginals near N(0,1) in every direction becomes easy for the network, instead of something it must learn through weight scales.

Diagnostics (lower SIGReg and higher effective rank mean closer to isotropic):

| | BN | no norm | BN + sched | no norm + sched |
| --- | ---: | ---: | ---: | ---: |
| SIGReg statistic | **2.27** | 2.81 | 2.90 | 3.36 |
| Effective rank (of 192) | **147** | 134 | 138 | 126 |
| Dims for 90% variance | 134 | 122 | 127 | 118 |
| Mean per-dim variance | 1.01 | 1.01 | 1.04 | 1.04 |
| Mean \|per-dim mean\| | 0.018 | 0.018 | 0.019 | 0.020 |

- **BN reaches a more isotropic latent** at the same λ.
- **Scale and centering are identical across all four.** Every model learns unit variance and zero mean regardless of norm, so BN's benefit isn't just "fixing the scale". It's in how variance is spread over directions.
- **RMSNorm is the control.** It normalizes per sample, like the ViT's final LayerNorm the LeWM paper warns about, and it performs no better than no norm (11.5 vs 11.6). Normalizing with **batch statistics** is what matters, not normalization in general.

### 4.2 Less conflict between SIGReg and prediction, so the schedule doesn't matter with BN

The prediction loss is easiest to minimize with a latent that changes little or predictably. SIGReg demands a spread-out, full-rank latent. If the architecture makes SIGReg cheap to satisfy, the two terms barely compete. If SIGReg is expensive, it pulls against prediction throughout training.

That fits the schedule results:
- **With BN** (SIGReg cheap), keeping λ = 0.09 the whole time costs nothing. The constant-weight BN model is even marginally better.
- **Without BN**, decaying λ from 0.09 to 0.009 relaxes the conflict late in training and gains 6 points.

The diagnostics show the trade directly. The schedule lowers one-step prediction error for both norms (BN 0.048 → 0.032, no norm 0.035 → 0.028) but raises the SIGReg statistic (2.27 → 2.90, 2.81 → 3.36).

### 4.3 BN keeps subtle features in the latent

The yaw probe is the clearest difference between the models: R² 0.90 vs 0.17 at constant λ, and 0.92 vs 0.35 with the schedule. T position and EEF position are decoded about equally well by all models.

- **Why orientation is fragile:** in a 128 px overhead image, rotating the T changes few pixels compared with moving the arm or translating the T. Features that respond to it have low variance across the batch.
- **Without BN:** the hidden layer's activations keep whatever scale they happen to have, and high-variance features dominate both the GELU nonlinearity and the output `Linear`.
- **With BN:** every hidden unit is rescaled to unit batch variance before the GELU, so a unit tracking orientation carries as much signal as one tracking arm position. BN amplifies weak but informative features.

There's a planning consequence too. GoalMSE sums squared differences over all 192 dimensions. If orientation lives in a few weak directions, or isn't linearly present at all, two latents that differ only in T rotation are almost equally "close to the goal". CEM then has no gradient-free signal for rotating the T. That fits the pattern of the no-norm models getting XY and EEF close but failing on yaw.

A related measurement: a quadratic regression from the state (T xy, yaw, EEF xy) to the embedding explains 7.9% of embedding variance for BN versus 5.2% for no norm (11.6% vs 7.7% with the schedule). So BN devotes a larger share of the latent, and therefore of the planning cost, to task state. (Most variance in every model comes from other factors, such as the per-episode goal marker and appearance.)

### 4.4 Lower prediction error doesn't mean better planning

The no-norm models have **lower** one-step prediction error than BN (0.035 vs 0.048, as a fraction of the true latent change). They predict their own latent better, yet plan worse. The prediction loss is measured in a latent the model chooses, so a latent that drops hard-to-predict information (like orientation) is easier to predict.

That's why prediction loss can't be used to choose between architectures here. It's also why the anti-collapse pressure, and how effectively the architecture lets it act, decides what information the latent keeps.

### 4.5 Faster, better-conditioned optimization

BN is also the standard conditioning trick: it keeps pre-activations in the GELU's useful range and makes the projector's gradients insensitive to the encoder's output scale. Here that shows up as the BN models reaching 22–23% success by epoch 5 versus 9–15% for no norm, with no-norm still behind at epoch 20.

## 5. Things that do *not* explain it

- **Capacity:** BN folds into the linear layer at inference (section 2).
- **Embedding scale or centering:** identical across the four models (section 4.1).
- **Residual vs classic:** all four are ResLeWM. The classic-LeWM run `noisy_lewm_bn_11311636` will show whether the effect carries over.
- **T position or arm placement:** probe R² and eval errors are similar across models. The gap is orientation.

## 6. Caveats

- **One seed per configuration.** The bn/nobn gap (6–12 points mean, 74/80 paired cells for the schedule effect) is well above the ±5-point per-cell noise, but run-to-run variance hasn't been measured.
- **Linear probes only.** A nonlinear probe might recover some yaw from the no-norm latent. What matters for GoalMSE, though, is linear/Euclidean structure.
- **Small diagnostic sample:** 2048 random transitions from the noisy 100k dataset, epoch-20 checkpoints, BN in eval mode, half used to fit and half to evaluate. The SIGReg statistic is measured on that batch with fixed random directions, so compare it only between models, not with training logs.
- The mechanisms in 4.1–4.3 are consistent with all the measurements but aren't proven causally. The experiments below would test them.

## 7. Experiments to confirm

1. **LayerNorm projector** (`model.projector.norm_fn._target_=torch.nn.LayerNorm`, same for `pred_proj`). Prediction from 4.1: no better than no norm, like RMSNorm.
2. **BN in only one place:** in `projector` only vs in `pred_proj` only. The yaw story predicts that the projector BN carries most of the effect, since it shapes the latent itself.
3. **Yaw probe across epochs** for bn vs nobn, to check that the gap appears early (matching the epoch-5 success gap).
4. **Planning with a yaw-aware cost** on the no-norm model: if success recovers, the latent is the bottleneck, not the predictor.
5. **The constant low-weight runs** now training (`*_w0.03`, `*_w0.009`) test whether no-norm models are mainly hurt by SIGReg conflict (4.2).

## Diagnostics

Script: [scripts/plan/embedding_diagnostics.py](scripts/plan/embedding_diagnostics.py). Reproduce with:

```bash
STABLEWM_HOME=/network/scratch/o/ozgur.aslan/le-wm python scripts/plan/embedding_diagnostics.py \
  bn=noisy_lewm_res_bn_11308936/weights_epoch_20.pt nobn=noisy_lewm_res_nobn_11308937/weights_epoch_20.pt \
  bn_sched=noisy_lewm_res_bn_sched_11302701/weights_epoch_20.pt nobn_sched=noisy_lewm_res_nobn_sched_11302702/weights_epoch_20.pt
```

It loads each `weights_epoch_20.pt` with `load_pretrained`, encodes 2048 random (frame, frame+5, action) samples with the training transforms, and computes:
- covariance spectrum (effective rank = (Σλ)² / Σλ²);
- the SIGReg statistic;
- one-step error: ‖pred − e₁‖² / ‖e₁ − e₀‖²;
- ridge probes from the embedding to the state;
- a quadratic state-to-embedding regression.

Checkpoints: `noisy_lewm_res_{bn,nobn}_1130893{6,7}` and `noisy_lewm_res_{bn,nobn}_sched_1130270{1,2}`.

Full numbers (one-step error is relative; Δ = predicted vs true latent change):

| | BN | no norm | BN + sched | no norm + sched |
| --- | ---: | ---: | ---: | ---: |
| One-step rel. error | 0.048 | 0.035 | 0.032 | 0.028 |
| ‖Δ_pred‖ / ‖Δ_true‖ | 0.98 | 0.98 | 0.99 | 0.99 |
| cos(Δ_pred, Δ_true) | 0.96 | 0.97 | 0.97 | 0.98 |
| Top eigenvalue share | 1.2% | 1.3% | 1.2% | 1.5% |
| Top-10 eigenvalue share | 10.4% | 11.4% | 11.0% | 12.8% |
| Embedding variance explained by T state | 4.6% | 2.3% | 6.9% | 3.8% |
| ... by EEF | 1.4% | 1.5% | 2.2% | 2.3% |
| ... by both | 7.9% | 5.2% | 11.6% | 7.7% |

## Sources

- Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544 — https://arxiv.org/abs/2511.08544
- Maes, Le Lidec, Scieur, LeCun & Balestriero, *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*, arXiv:2603.19312 — https://arxiv.org/html/2603.19312v2
- *SIGReg: What It Computes, Constrains, and Where It Breaks* (tutorial) — https://rezabyt.github.io/blogposts/sigreg-tutorial.html
- Code: `stable_worldmodel/wm/lewm/lewm.py`, `lewm_res.py`, `module.py` (MLP), `stable_worldmodel/wm/loss.py` (SIGReg), `scripts/train/lewm.py` (loss and schedule), `stable_worldmodel/planning/objective.py` (GoalMSE).
