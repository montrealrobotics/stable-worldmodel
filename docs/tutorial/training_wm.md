title: Training A World Model
summary: Train or plug in a model that implements the planning cost interface, then evaluate it with model-predictive control.
sidebar_title: Train World-Model
---

This tutorial covers the second and third stages of a world-model experiment:
training a model from collected trajectories and using it for planning.

The important integration point is small: a model used by
`WorldModelPolicy` must provide `get_cost(info_dict, action_candidates)`.
Planning solvers such as CEM and MPPI search over action sequences and choose
the sequence with the lowest predicted cost.

## Prerequisites

Install the full package and collect the dataset from
[Collect Dataset](collect_data.md):

```bash
pip install 'stable-worldmodel[all]'
export STABLEWM_HOME=$PWD/.stablewm
```

On macOS arm64, `decord` may not provide a compatible wheel for the `all`
extra. For the CPU smoke path below, install the narrower runtime set:

```bash
pip install 'stable-worldmodel[train]' pygame pymunk shapely opencv-python-headless
```

This tutorial assumes the dataset exists at:

```text
$STABLEWM_HOME/datasets/tutorial_pusht.lance
```

## The cost-model contract

A cost model receives the current `World` info dict and a batch of candidate
action sequences. It returns one scalar cost per environment and candidate.
Lower is better.

```python
import torch


class MyWorldModel(torch.nn.Module):
    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        info_dict:
            Dict produced by World. Tensor values usually have shape
            (num_envs, history, ...).

        action_candidates:
            Tensor with shape
            (num_envs, num_samples, horizon, action_dim).

        returns:
            Cost tensor with shape (num_envs, num_samples).
        """
        costs = ...
        return costs
```

For goal-conditioned planning, `info_dict` usually contains `pixels`,
`goal`, `state`, and `goal_state` or related goal columns. Image models often
compare predicted future embeddings against a goal embedding; state models
often compare predicted state against `goal_state`.

## Load trajectory clips

Training starts from the same dataset API used everywhere else:

```python
import stable_worldmodel as swm

dataset = swm.data.load_dataset(
    'tutorial_pusht.lance',
    num_steps=8,
    frameskip=1,
    keys_to_load=['pixels', 'action', 'state'],
)
```

With PyTorch:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

batch = next(iter(loader))
print(batch['pixels'].shape)  # (B, T, C, H, W)
print(batch['action'].shape)  # (B, T, action_dim)
```

Lance datasets switch multiprocessing to the `spawn` start method for
fork-safety. If you use `num_workers > 0`, run the `DataLoader` code from a
normal Python file under an `if __name__ == '__main__':` guard. For notebooks,
REPLs, or heredoc snippets, set `num_workers=0`.

The first action in an episode may be `NaN` because there is no previous
environment action at reset. Training code should replace it before computing
losses:

```python
batch['action'] = torch.nan_to_num(batch['action'], 0.0)
```

## Use a reference training script

The repository ships reference training scripts under `scripts/train/`. For
example, `scripts/train/lewm.py` trains the LeWM baseline from image clips and
actions.

To predict ViT patch tokens instead of the CLS token, select `lewm_patch`:

```bash
python scripts/train/lewm.py --config-name lewm_patch \
  data.dataset.name=tutorial_pusht.lance \
  trainer.max_epochs=2 trainer.max_steps=-1
```

`LeWMPatch` keeps embeddings shaped `(B, T, P, D)` and predicts all `P`
patches of the next frame. It uses the same encoder, projection heads,
action conditioning, prediction MSE, and SIGReg weight as LEWM. The predictor
attends to all patches within the current and past frames, with learned
spatial and temporal positions; future frames are masked. The same projector
also processes the CLS token, retained as `cls_emb` with shape `(B, T, D)`.
SIGReg operates on these CLS embeddings across the batch at each timestep,
as in LEWM. The encoder and both sides of the patch prediction loss remain
trainable.

For the existing ManiSkill overhead vision settings, select
`--config-name vision_lewm_patch`. Both configs derive the patch count from
`img_size` and `patch_size` and reuse the existing dataset, training, and
checkpoint pipeline. Patch attention costs more memory than CLS attention;
adjust `loader.batch_size` for the image grid and available GPU memory.

The standard `scripts/plan/eval_wm.py` can load these checkpoints. Rollouts
retain the patch axis, and `GoalMSE` compares all patches of the final
prediction and goal frame. Use `GoalMSE(reduction='mean')` when comparing
cost scales across different patch counts.

For a short GPU smoke run:

```bash
python scripts/train/lewm.py \
  data.dataset.name=tutorial_pusht.lance \
  trainer.max_epochs=2 \
  loader.batch_size=32 \
  num_workers=2 \
  output_model_name=tutorial_lewm
```

For a CPU-only smoke run, override the trainer settings:

```bash
python scripts/train/lewm.py \
  data.dataset.name=tutorial_pusht.lance \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  loader.batch_size=8 \
  loader.prefetch_factor=null \
  loader.persistent_workers=false \
  num_workers=0 \
  output_model_name=tutorial_lewm_cpu
```

These commands are useful for checking that the dataset and training
dependencies are wired correctly. For real experiments, increase epochs,
batch size, image resolution, and model size according to your hardware.

## Train a LeWM behavior-cloning prior

Train a small Gaussian MLP on expert actions using a pretrained LeWM's frozen
encoder and projector. Inputs are the current latent and goal latent.
Only the policy head is optimized with Gaussian negative
log-likelihood; LeWM's prediction and SIGReg training remain unchanged.

From the repository root:

```bash
export STABLEWM_HOME=$PWD/.stablewm
python -m scripts.train.lewm_gcbc \
  world_model=lewm/weights_epoch_10.pt \
  data.dataset.name=pusht_expert_train.lance \
  subdir=lewm_prior wandb.enabled=false
```

Use the same dataset, observation columns, `img_size`, and action normalization
as the pretrained LeWM. Set `data.dataset.frameskip` to its action-block size:
five means five distinct environment actions, all supervised from the current
frame. Set `action_low` and `action_high` to the environment's physical bounds
(the defaults are PushT's `[-1, 1]`). The trainer normalizes these bounds using
the dataset's existing column scaler. It excludes nonfinite action blocks and
splits by episode. Future goals are sampled within the same episode, with an
equal mixture of uniform and geometric offsets; `goal_mean_steps=25` sets the
geometric mean before episode-end truncation, in environment steps.

For a CPU smoke run, add `trainer.accelerator=cpu trainer.precision=32-true
num_workers=0 loader.batch_size=8 trainer.max_epochs=1`.
Exports live at `checkpoints/lewm_prior/weights_epoch_N.pt` with `config.json`.
They contain the policy, frozen LeWM, and normalization statistics and load
through `load_pretrained` without the original LeWM checkpoint.

Evaluate all three modes with the existing dataset-start/goal pipeline:

```bash
# BC alone, executing each predicted block in order.
python -m scripts.plan.eval_wm \
  policy=lewm_prior/weights_epoch_10.pt +policy_mode=bc

# Original CEM; no prior is enabled by default.
python -m scripts.plan.eval_wm policy=lewm/weights_epoch_10.pt \
  +plan_config.history_len=3

# CEM initialized by the BC prior and its bundled LeWM.
python -m scripts.plan.eval_wm policy=lewm/weights_epoch_10.pt \
  +bc_prior=lewm_prior/weights_epoch_10.pt +plan_config.history_len=3
```

Use identical evaluation starts, goals, budgets, and action-block settings;
set `eval.dataset_name` to your evaluation dataset and match `eval.img_size`
to training. `plan_config.history_len` must match LeWM's context size for
planning. BC uses the current observation only. Evaluation uses Gaussian means;
`prior.get_action(info, horizon=5, sample=True)` samples a stochastic block.

CEM reuses its actor initialization/tail-fill interface rather than injecting
a 50/50 mixture, which would require a new solver sampling hook. The prior
rolls out LeWM with executed history and any previous plan prefix to initialize
the mean; CEM refines it using its existing distribution. Guided candidates
are clipped to normalized action bounds before scoring and elite selection.
Prefix rollouts currently replay history, so long planning horizons incur
quadratic predictor work. The prior supports flat LeWM embeddings, including
state LeWM; patch-token models are excluded.

## Save a custom model

If you train your own model, save it with the checkpoint format used by the
library:

```python
from stable_worldmodel.wm.utils import save_pretrained

model_config = {
    '_target_': 'my_project.models.MyWorldModel',
    'hidden_dim': 256,
    'action_dim': 2,
}

save_pretrained(
    model=model,
    run_name='tutorial_state_wm',
    config=model_config,
    filename='weights.pt',
)
```

`config` must be the Hydra instantiation config for the model itself. If your
training script has a larger config with keys such as `data`, `loader`, and
`trainer`, pass only the model sub-config or use `config_key='model'`.

## Evaluate with MPC

Load the model and wrap it with a solver-backed policy:

```python
import stable_worldmodel as swm
from stable_worldmodel.policy import PlanConfig, WorldModelPolicy
from stable_worldmodel.planning import CEMSolver
from stable_worldmodel.wm.utils import load_pretrained


device = 'cuda'
model = load_pretrained('tutorial_state_wm/weights.pt').to(device).eval()
model.requires_grad_(False)

world = swm.World(
    'swm/PushT-v1',
    num_envs=8,
    image_shape=(96, 96),
    max_episode_steps=100,
)

solver = CEMSolver(
    cost=model,
    num_samples=300,
    n_steps=5,
    device=device,
)

policy = WorldModelPolicy(
    solver=solver,
    config=PlanConfig(
        horizon=10,
        receding_horizon=5,
        action_block=1,
        warm_start=True,
    ),
)

world.set_policy(policy)
results = world.evaluate(episodes=32, seed=0, video='videos/tutorial_eval')
world.close()

print(results)
```

Use the same preprocessing at evaluation time that you used during training.
If you normalized actions or state columns, pass those reversible processors
to `WorldModelPolicy(process=...)`. If you normalized images, pass the image
transforms with `WorldModelPolicy(transform=...)`.

## Evaluate from dataset starts and goals

For goal-reaching experiments, it is common to initialize the simulator from
states stored in a dataset and ask the policy to reach a future dataset goal.
In this mode, `num_envs` must equal the number of requested episodes.

```python
dataset = swm.data.load_dataset(
    'tutorial_pusht.lance',
    num_steps=32,
    keys_to_load=['pixels', 'action', 'state'],
)

world = swm.World(
    'swm/PushT-v1',
    num_envs=4,
    image_shape=(96, 96),
    max_episode_steps=100,
)
world.set_policy(policy)

results = world.evaluate(
    dataset=dataset,
    episodes_idx=[0, 1, 2, 3],
    start_steps=[0, 5, 10, 15],
    goal_offset=25,
    eval_budget=50,
    callables=[
        {'method': '_set_state', 'args': {'state': {'value': 'state'}}},
        {
            'method': '_set_goal_state',
            'args': {'goal_state': {'value': 'goal_state'}},
        },
    ],
    video='videos/tutorial_dataset_eval',
)
```

Some environments need callables to restore internal simulator state from the
dataset. PushT supports `_set_state` and `_set_goal_state`; the planned
evaluation script in `scripts/plan/eval_wm.py` shows the full Hydra version
of that workflow.

## Debug checklist

- `get_cost()` must return finite costs with shape `(num_envs, num_samples)`.
- The solver's `device` must match the model device.
- `PlanConfig.horizon * action_block` should fit inside the evaluation
  budget.
- Use `warm_start=True` for faster receding-horizon planning once the model
  is stable.
- Start with small `num_samples` and short horizons while debugging, then
  increase them for final evaluation.
