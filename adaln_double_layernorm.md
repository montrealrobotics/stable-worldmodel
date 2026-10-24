# Double LayerNorm in the AdaLN predictor blocks

## Summary

The AdaLN-conditioned transformer blocks in LeWM's predictor (and in the fork's AdaLN DINO-WM predictor) normalize each token twice before attention and before the MLP. The second LayerNorm sits *after* the action modulation, so it partly cancels the action's scale and shift.

- It is not a crash or a wrong-result bug. Conditioning still reaches the model through per-feature patterns and through the gates.
- It is present in upstream `galilai-group/stable-worldmodel` (LeWM) and inherited by this fork.
- Its effect on planning is untested.

## Where

`ConditionalBlock` in [`stable_worldmodel/wm/lewm/module.py`](stable_worldmodel/wm/lewm/module.py#L77-L107):

```python
x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))   # line 100
x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))    # line 104
```

- `norm1`/`norm2` (line 87): `LayerNorm(dim, elementwise_affine=False)`.
- `self.attn` is `Attention`, which starts with its own `LayerNorm(dim)` with learned γ, β (lines 44, 57).
- `self.mlp` is `FeedForward`, which starts with `nn.LayerNorm(dim)` (line 22).

So each branch computes:

```
x ─ LN (no affine) ─ modulate: n·(1 + scale) + shift ─ LN (affine γ, β) ─ attention or MLP ─ × gate ─ + x
                     └──────── from the action ─────┘                                      └ action ┘
```

The same pattern appears in `AdaLNBlock` in [`stable_worldmodel/wm/prejepa/module.py`](stable_worldmodel/wm/prejepa/module.py#L364-L405) (lines 400–405), with the inner norms at lines 245 (`FeedForward`) and 272/288 (`Attention`).

## What the second LayerNorm removes

After `norm1`, a token `n` has zero mean and unit variance across its features. Modulation gives

```
y_i = n_i · (1 + s_i) + b_i
```

The inner LayerNorm subtracts `y`'s mean across features and divides by its standard deviation:

- **Uniform scale and shift are erased exactly.** If `s_i = s` and `b_i = b` for every feature, `y = (1 + s)·n + b` and `LN(y) = n`. The action has no effect.
- **Per-feature patterns survive, but rescaled.** `s` and `b` come from a linear layer, so they differ across features. The action can still change *which* features are emphasized, but not the token's overall offset or gain; the second LayerNorm re-standardizes both.
- **The gates are untouched.** `gate_msa` and `gate_mlp` multiply the sublayer *output*, after the inner LayerNorm. With the adaLN-zero init (`adaLN_modulation` zeroed, lines 93–94), every block starts as the identity and the gates carry the action signal early in training.

In the unconditioned `Block` (line 110), the double norm is harmless: `LN_affine(LN(x))` equals one affine LayerNorm up to eps.

## Origin

`Attention` and `FeedForward` follow the vit-pytorch style used by DINO-WM's ViT predictor, where each sublayer carries its own pre-norm. Wrapping them in DiT-style adaLN blocks without removing those inner norms produces the pattern. In DiT, the modulated input goes straight into the qkv projection and the MLP.

## Upstream status (checked 2026-10-21, `upstream/main` at 21446f1)

| File | Upstream `galilai-group/stable-worldmodel` | This fork |
| --- | --- | --- |
| `wm/lewm/module.py`, `ConditionalBlock` | Double norm (present since at least bcc8aa3, 2026-03-31) | Same |
| `wm/prejepa/module.py` | No AdaLN block; the plain `Transformer` uses only the inner pre-norms (single norm) | `AdaLNBlock` with the double norm (fork-only, copied from LeWM) |

The LeWM paper's own code release was not checked. If it matches upstream, the published LeWM results were trained with this double norm.

## Suggested fix and test

Make the inner norms optional in the conditional blocks only, defaulting to the current behavior so existing checkpoints still load:

```python
class Attention(nn.Module):
    def __init__(self, dim, ..., pre_norm=True):
        ...
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
```

`ConditionalBlock`/`AdaLNBlock` would build `Attention(..., pre_norm=False)` and a `FeedForward` without its leading LayerNorm when a new flag (e.g. `predictor.inner_norm=false`) is set.

Test: train ResLeWM with BN at λ 0.09 on noisy 100k with the flag off, and compare with the 26.6% mean-success reference on the standard `sched_n100` grid. Worth repeating for the DINO-WM CLS model once its baseline evals are in.

## Measured effect on trained LeWM checkpoints (2026-10-21)

Epoch-20 ResLeWM checkpoints, 1024 noisy-100k transitions. For every conditional block and sublayer, the action's change to the sublayer input before the inner LayerNorm, `‖modulate(n) − n‖`, is compared with what is left after it, `‖LN(modulate(n)) − LN(n)‖` (LN without its affine part, so both are in the same units).

| Checkpoint | Survives inner LN (attn / mlp, range over 6 blocks) | Uniform share of modulation energy | One-step error: true / shuffled / zero action |
| --- | --- | --- | --- |
| BN, λ 0.09 | 0.80–1.10 / 1.04–1.09 | 0.3–16% (median ~3%) | 0.050 / 0.983 / 0.839 |
| No norm, λ 0.009 | 0.98–1.02 / 1.03–1.05 | 0.3–7% | 0.014 / 0.985 / 0.797 |
| No norm, λ 0.09 | 0.84–1.00 / 0.99–1.02 | 0.5–17% | 0.035 / 0.871 / 0.840 |

- **The inner LayerNorm erases almost nothing.** The trained models put nearly all of their modulation into per-feature patterns, which survive (ratios near or slightly above 1, since re-standardizing can amplify). The token-uniform part that the inner LN removes is a few percent of the modulation energy. The only visible loss is block 0's attention (0.80–0.84).
- **The predictor relies heavily on the action.** Shuffling actions across the batch raises the one-step error from 0.01–0.05 to 0.87–0.99, about as bad as predicting no change at all (1.0).
- **Conclusion: low priority.** The double norm doesn't block action conditioning in practice, so a fix is unlikely to move planning success. Caveat: models trained with it may have learned to avoid uniform modulation; only a retrain would show whether a model without it would use that freedom.

Probe script (session scratchpad, not committed): it hooks each `ConditionalBlock`, recomputes the modulation from the block's input and condition, and evaluates the predictor with true, shuffled and zero actions.
