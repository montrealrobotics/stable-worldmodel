import math
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch._functorch.config
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.data.filtering import filter_replay_episodes
from stable_worldmodel.planning.objective import GoalMSE
from stable_worldmodel.wm.diagnostics import encoder_gradient_diagnostics
from stable_worldmodel.wm.lewm.detached_rollout import detached_rollout
from stable_worldmodel.wm.loss import SIGReg, temporal_center
from stable_worldmodel.wm.utils import (
    _resolve,
    load_pretrained,
    save_pretrained,
)


def _vit_embed_dim(scale: str) -> int:
    dims = {
        'tiny': 192,
        'small': 384,
        'base': 768,
        'large': 1024,
        'huge': 1280,
    }
    if scale not in dims:
        raise ValueError(
            f"Unknown ViT encoder scale '{scale}'. "
            f'Expected one of {sorted(dims)}.'
        )
    return dims[scale]


if not OmegaConf.has_resolver('vit_embed_dim'):
    OmegaConf.register_new_resolver('vit_embed_dim', _vit_embed_dim)


def get_img_preprocessor(
    source: str, target: str, img_size: int = 224, uint8: bool = False
):
    """uint8=True leaves scaling and normalization to ``normalize_pixels``."""
    stats = {} if uint8 else dt.dataset_stats.ImageNet
    dtype = torch.uint8 if uint8 else torch.float32
    to_image = dt.transforms.ToImage(
        **stats, dtype=dtype, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class OfflineWandbManager(spt.Manager):
    """spt reloads an offline run's config from its previous offline run and
    crashes when there is none (a fresh run on a cluster without internet).
    The config is already logged by ``log_hyperparams``, so skip the reload."""

    def init_and_sync_wandb(self):
        import wandb

        if (
            wandb.run is not None
            and wandb.run.offline
            and self._wandb_previous_dir() is None
        ):
            return
        super().init_and_sync_wandb()


class SaveCkptCallback(Callback):
    """Save the model with save_pretrained every ``epoch_interval`` epochs.

    With ``step_interval``, also saves weights_step_N.pt every N optimizer
    steps (for step-based runs that last only a few epochs). Also saves the
    final weights when training ends off-interval, including a partial last
    epoch when ``max_steps`` stops training.
    """

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        step_interval=None,
        policy_cfg=None,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        # GFlowNet sampler config; the policy is saved under <run>/gfn_policy.
        self.policy_cfg = policy_cfg
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self._saved_epoch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, idx)
        # Count batches: with several optimizers global_step counts each one.
        fit_loop = getattr(trainer, 'fit_loop', None)
        step = (
            fit_loop.total_batch_idx + 1 if fit_loop else trainer.global_step
        )
        if (
            self.step_interval
            and trainer.is_global_zero
            and step > 0
            and step % self.step_interval == 0
        ):
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.cfg,
                filename=f'weights_step_{step}.pt',
            )
            self._save_policy(pl_module, f'weights_step_{step}.pt')

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        if trainer.is_global_zero and epoch % self.epoch_interval == 0:
            self._save(pl_module, epoch)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        # current_epoch counts finished epochs here, plus one if stopped
        # mid-epoch, matching the 1-based names used above.
        epoch = max(trainer.current_epoch, 1)
        if trainer.is_global_zero and self._saved_epoch != epoch:
            self._save(pl_module, epoch)

    def _save(self, pl_module, epoch):
        self._saved_epoch = epoch
        save_pretrained(
            pl_module.model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
        self._save_policy(pl_module, f'weights_epoch_{epoch}.pt')

    def _save_policy(self, pl_module, filename):
        policy = getattr(pl_module, 'gfn_policy', None)
        if policy is not None:
            save_pretrained(
                policy,
                run_name=f'{self.run_name}/gfn_policy',
                config=self.policy_cfg,
                filename=filename,
            )


def search_batch(batch, emb, cfg):
    """Check the clip length and keep the first ``search.batch_size`` clips."""
    H, search = cfg.wm.history_size, cfg.loss.search
    if emb.size(1) < H + search.horizon:
        raise ValueError(
            f'search loss needs num_steps >= {H + search.horizon}'
        )
    n = search.get('batch_size') or emb.size(0)
    batch = {
        k: v[:n] if torch.is_tensor(v) and v.ndim > 0 else v
        for k, v in batch.items()
    }
    return batch, emb[:n]


def plan_costs(model, batch, emb, cfg, plans, all_steps=False):
    """Goal MSE of each plan rolled out from the context: (B, S, T, A) -> (B, S).

    The goal is the clip frame ``horizon`` blocks after the last context
    frame, detached; gradients reach the model through the rollout.
    ``all_steps``: (B, S, T) costs, step k against the clip frame k blocks
    after the last context frame.
    """
    H, search = cfg.wm.history_size, cfg.loss.search
    T = search.horizon
    actions = batch['action']
    B = actions.size(0)
    goal = emb[:, H - 1 + T, None].detach()
    targets = emb[:, None, H : H + T].detach()  # (B, 1, T, ...)
    skip = ('action', 'emb', 'act_emb')
    heads = max(
        (m.heads for m in model.modules() if hasattr(m, 'heads')), default=1
    )

    def chunk_cost(plans):
        S = plans.size(1)

        def ctx(v, n=H):
            return v[:, None, :n].expand(B, S, *v[:, :n].shape[1:])

        info = {
            k: ctx(v)
            for k, v in batch.items()
            if torch.is_tensor(v) and v.ndim > 1 and k not in skip
        }
        info['emb'] = ctx(emb)
        info['action_history'] = ctx(actions, H - 1)
        if search.bptt or not torch.is_grad_enabled():
            info = model.rollout(info, plans, H)
        else:
            history = torch.cat([info['action_history'], plans], 2)
            preds = detached_rollout(
                model,
                info['emb'].flatten(0, 1),
                history.flatten(0, 1),
                T,
                H,
                all_grad=all_steps,
            )
            info['predicted_emb'] = preds.unflatten(0, (B, S))
        if all_steps:
            err = (info['predicted_emb'][:, :, -T:] - targets).pow(2)
            return err.flatten(3).sum(-1)  # (B, S, T), GoalMSE per step
        info['goal_emb'] = goal
        return GoalMSE()(info)  # (B, S)

    # Attention kernels fail past 65535 rollouts * heads (CUDA grid limit).
    # shortcut: needs B * heads <= 65535, chunk B too if not.
    step = max(1, 65535 // (heads * B))
    chunks = plans.split(step, dim=1)
    return torch.cat([chunk_cost(c) for c in chunks], dim=1)


def clip_plans(plans, cfg):
    """Clamp plans to the dataset action range when ``clip_actions`` is on."""
    search = cfg.loss.search
    if not search.get('clip_actions'):
        return plans
    low = plans.new_tensor(list(search.action_low))
    high = plans.new_tensor(list(search.action_high))
    return torch.maximum(torch.minimum(plans, high), low)


def search_loss(model, batch, emb, cfg):
    """InfoNCE over goal costs: the expert plan must beat a short CEM's elites.

    With ``all_steps`` the InfoNCE is averaged over every predicted step k,
    each against the clip frame k blocks ahead; CEM still ranks by the last.
    Without bptt each step gets only its one-step gradient from a detached
    prefix.
    """
    H, search = cfg.wm.history_size, cfg.loss.search
    T = search.horizon
    all_steps = search.get('all_steps', False)
    batch, emb = search_batch(batch, emb, cfg)
    actions = batch['action']
    B = actions.size(0)
    expert = actions[:, None, H - 1 : H - 1 + T]

    def cost(plans):
        return plan_costs(model, batch, emb, cfg, plans)

    # Mirror CEMSolver: start at N(0, var_scale), refit to the top-k each step.
    mean = torch.zeros_like(expert)
    std = torch.full_like(expert, search.var_scale)
    rows = torch.arange(B, device=actions.device)[:, None]
    elites = []
    with torch.no_grad():
        for _ in range(search.cem_steps):
            plans = mean + std * torch.randn(
                B,
                search.num_samples,
                *expert.shape[2:],
                device=actions.device,
                dtype=actions.dtype,
            )
            plans[:, 0] = mean[:, 0]
            plans = clip_plans(plans, cfg)
            idx = cost(plans).topk(search.topk, dim=1, largest=False).indices
            top = plans[rows, idx]
            elites.append(top)
            mean = top.mean(1, keepdim=True)
            std = top.std(1, correction=0, keepdim=True)
    # Expert-like negatives: action style alone can no longer identify the expert.
    perturbed = expert + search.get('perturb_std', 0.0) * torch.randn(
        B,
        search.get('num_perturbed', 0),
        *expert.shape[2:],
        device=actions.device,
        dtype=actions.dtype,
    )
    plans = torch.cat([expert, clip_plans(perturbed, cfg), *elites], 1)
    costs = plan_costs(model, batch, emb, cfg, plans, all_steps)  # expert 0
    if all_steps:  # one InfoNCE per step: (B, S, T) -> (B * T, S)
        costs = costs.transpose(1, 2).flatten(0, 1)
    target = costs.new_zeros(costs.size(0), dtype=torch.long)
    return F.cross_entropy(-costs.float() / search.temperature, target)


def gfn_temperature(gfn, step):
    """Reward temperature, annealed geometrically from tau_start to tau_end."""
    t = min(max(step, 0) / max(gfn.anneal_steps, 1), 1.0)
    return gfn.tau_start * (gfn.tau_end / gfn.tau_start) ** t


def gfn_log_reward(costs, plans, gfn, tau):
    """log R = -cost / tau, plus a N(0, prior_std^2 I) prior over the plan.

    Without the prior R / Z is not normalizable: far from the data the cost
    flattens out, so the target spreads over unbounded action space.
    With ``expert_threshold``, cost counts only above the expert's own cost
    (column 0) plus ``threshold_margin``: every plan the world model scores
    at least as well as the expert gets the same reward up to the prior, so
    the policy samples those violations evenly.
    With ``relative_cost``, costs are first divided by the expert's, so tau
    keeps its meaning while the world model rescales its costs (the cost
    ratio of samples to expert stays stable when both inflate).
    costs: (B, S), expert first; plans: (B, S, T, A), normalized actions.
    """
    cost = costs.float()
    if gfn.get('relative_cost'):
        cost = cost / cost[:, :1].clamp(min=1e-6)
    if gfn.get('expert_threshold'):
        margin = gfn.get('threshold_margin', 0.0)
        cost = (cost - cost[:, :1] - margin).clamp(min=0)
    log_r = -cost / tau
    if gfn.get('prior_std'):
        sq = plans.float().pow(2).sum((-1, -2))
        log_r = log_r - sq / (2 * gfn.prior_std**2)
    return log_r


def gfn_search_loss(model, policy, batch, emb, cfg, step, policy_only=False):
    """InfoNCE with GFlowNet negatives, plus the policy's trajectory balance.

    The policy samples plans in proportion to exp(-cost / tau) under the
    current world model, so the negatives cover every basin the model scores
    well. The world model gets only the InfoNCE gradient (samples and
    conditions are detached); the policy gets only trajectory balance on
    detached rewards. ``policy_only``: costs without grad, no InfoNCE.
    """
    H, search, gfn = cfg.wm.history_size, cfg.loss.search, cfg.loss.gfn
    T = search.horizon
    batch, emb = search_batch(batch, emb, cfg)
    expert = batch['action'][:, None, H - 1 : H - 1 + T]
    z0, z_goal = emb[:, H - 1].detach(), emb[:, H - 1 + T].detach()

    # The policy runs in fp32: TB sums log-probs over all T * A dimensions.
    fp32 = torch.autocast(z0.device.type, enabled=False)
    z0, z_goal = z0.float(), z_goal.float()
    with fp32:
        samples = policy.sample(
            z0, z_goal, gfn.num_samples, T, std_scale=gfn.explore_std
        ).to(expert.dtype)
    plans = torch.cat([expert, samples], 1)
    with torch.set_grad_enabled(torch.is_grad_enabled() and not policy_only):
        costs = plan_costs(model, batch, emb, cfg, plans)  # expert at index 0
    infonce = None
    if not policy_only:
        target = costs.new_zeros(costs.size(0), dtype=torch.long)
        infonce = F.cross_entropy(-costs.float() / search.temperature, target)

    tau = gfn_temperature(gfn, step)
    log_r = gfn_log_reward(costs.detach(), plans, gfn, tau)
    if not gfn.expert_tb:  # expert plans are valid off-policy TB samples
        plans, log_r = samples, log_r[:, 1:]
    with fp32:
        log_pi = policy.log_prob(z0, z_goal, plans.float())
        log_z = policy.log_z(z0, z_goal)[:, None]
    tb = (log_z + log_pi - log_r).pow(2).mean()
    return {
        'search_loss': infonce,
        'gfn_tb_loss': tb,
        'gfn_tau': tau,
        'gfn_log_z': log_z.detach().mean(),
        'gfn_cost_expert': costs[:, 0].detach().mean(),
        'gfn_cost_samples': costs[:, 1:].detach().mean(),
        'gfn_action_rms': samples.float().pow(2).mean().sqrt(),
        # Share of samples the world model scores better than the expert.
        'gfn_violation_frac': (costs[:, 1:] < costs[:, :1]).float().mean(),
        # > 0: the sampled plans beat the expert under the world model.
        'gfn_expert_margin': (costs[:, :1] - costs[:, 1:]).detach().mean(),
    }


def gfn_extra_policy_steps(self, batch, emb, cfg, step):
    """Fit the policy ``policy_steps - 1`` extra times on this batch.

    Runs before the joint step, which makes the last policy update together
    with the world-model update. The world model is frozen here, so ``emb``
    is reused and costs need no grad. Uses the raw optimizer so Lightning's
    step counters (and max_steps) still count one step per batch.
    """
    k = cfg.loss.gfn.get('policy_steps', 1)
    if k <= 1 or not torch.is_grad_enabled():
        return
    names = self._optimizer_index_to_name
    idx = next(i for i, n in names.items() if n == 'policy_opt')
    opt = self.optimizers()[idx].optimizer
    params = list(self.gfn_policy.parameters())
    clip = self.trainer.gradient_clip_val
    emb = emb.detach()
    for _ in range(k - 1):
        out = gfn_search_loss(
            self.model, self.gfn_policy, batch, emb, cfg, step, True
        )
        tb = out['gfn_tb_loss']
        opt.zero_grad(set_to_none=True)
        tb.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
    opt.zero_grad(set_to_none=True)


def normalize_pixels(batch):
    """Scale uint8 pixel columns on the GPU, as ToImage would on the CPU."""
    for key, value in batch.items():
        if key.startswith('pixels') and value.dtype == torch.uint8:
            stats = dt.dataset_stats.ImageNet
            shape = (3, 1, 1)
            mean = value.new_tensor(stats['mean'], dtype=torch.float32)
            std = value.new_tensor(stats['std'], dtype=torch.float32)
            batch[key] = (value / 255.0 - mean.view(shape)) / std.view(shape)
    return batch


def sigreg_weight(sigreg, progress):
    """SIGReg weight at a training progress in [0, 1].

    With a schedule, the weight holds until `schedule_start` and then decays
    to `final_weight` at the end of training. Keep `final_weight` above zero:
    SIGReg is the only term preventing representation collapse.
    """
    schedule = sigreg.get('schedule')
    if not schedule:
        return sigreg.weight
    start = sigreg.get('schedule_start', 0.0)
    t = min(max((progress - start) / max(1 - start, 1e-8), 0.0), 1.0)
    if schedule == 'cosine':
        t = (1 - math.cos(math.pi * t)) / 2
    elif schedule != 'linear':
        raise ValueError(f'unknown SIGReg schedule: {schedule}')
    return sigreg.weight + (sigreg.final_weight - sigreg.weight) * t


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    progress = 0.0
    if cfg.loss.sigreg.get('schedule'):
        total = max(1, self.trainer.estimated_stepping_batches)
        progress = self.global_step / total
    lambd = sigreg_weight(cfg.loss.sigreg, progress)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    normalize_pixels(batch)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, ..., D)
    act_emb = output['act_emb']

    ctx_info = {}
    ctx_info['emb'] = emb[:, :ctx_len]
    ctx_info['act_emb'] = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds : n_preds + ctx_len]  # label
    if getattr(self.model, 'one_shot', False):
        # cumulative act_emb; the last context frame uses one per target
        ctx_info['act_emb'] = act_emb[:, : ctx_len - 1 + n_preds]
        tgt_emb = emb[:, ctx_len : ctx_len + n_preds]

    pred_emb = self.model.predict(ctx_info)['preds']  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    # Patch models retain the CLS embedding for the standard LEWM regularizer.
    reg_emb = output.get('cls_emb', emb)
    window = cfg.loss.sigreg.get('temporal_window')
    if window:  # TC-LeWM: regularize residuals around the local temporal mean
        reg_emb = temporal_center(reg_emb, window)
    output['sigreg_loss'] = self.sigreg(reg_emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
    aux_yaw = cfg.loss.get('aux_yaw') or {}
    if aux_yaw.get('weight', 0) > 0:
        w, x, y, z = batch[aux_yaw.column][..., 3:7].float().unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        target = torch.stack([yaw.sin(), yaw.cos()], -1)  # (B, T, 2)
        output['aux_yaw_loss'] = F.mse_loss(self.aux_head(emb), target)
        output['loss'] = output['loss'] + aux_yaw.weight * output['aux_yaw_loss']
    search = getattr(cfg.loss, 'search', None)
    gfn_logs = {}
    if search and search.weight > 0 and search.get('sampler', 'cem') == 'gfn':
        step = self.trainer.fit_loop.total_batch_idx  # batches, not opt steps
        if stage == 'fit':
            gfn_extra_policy_steps(self, batch, emb, cfg, step)
        gfn = gfn_search_loss(
            self.model, self.gfn_policy, batch, emb, cfg, step
        )
        output['search_loss'] = gfn.pop('search_loss')
        output['gfn_tb_loss'] = gfn.pop('gfn_tb_loss')
        gfn_logs = gfn
        output['loss'] = (
            output['loss']
            + search.weight * output['search_loss']
            + output['gfn_tb_loss']
        )
    elif search and search.weight > 0:
        output['search_loss'] = search_loss(self.model, batch, emb, cfg)
        output['loss'] = output['loss'] + search.weight * output['search_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    losses_dict.update({f'{stage}/{k}': v for k, v in gfn_logs.items()})
    losses_dict[f'{stage}/sigreg_weight'] = float(lambd)
    if getattr(self.model, 'one_shot', False):
        # scale of the recurrent action state per horizon (drift check)
        future = ctx_info['act_emb'][:, ctx_len - 1 :].detach().float()
        norms = future.norm(dim=-1).mean(0)
        for k, v in enumerate(norms, 1):
            losses_dict[f'{stage}/act_emb_norm_h{k}'] = v
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    interval = cfg.get('gradient_diagnostics_interval', 100)
    if stage == 'fit' and torch.is_grad_enabled() and interval > 0:
        step = self.global_step
        if step % interval == 0 and step != getattr(
            self, '_last_gradient_diagnostics_step', None
        ):
            diagnostics = encoder_gradient_diagnostics(
                output['pred_loss'],
                output['sigreg_loss'],
                self.model.encoder.parameters(),
                lambd,
                search_loss=output.get('search_loss'),
                search_weight=search.weight if search else 0.0,
            )
            self.log_dict(
                {f'{stage}/{k}': v for k, v in diagnostics.items()},
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            # Log only once per optimizer step with gradient accumulation.
            self._last_gradient_diagnostics_step = step
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # TC-LeWM: load enough frames to fill one centering window.
    window = cfg.loss.sigreg.get('temporal_window')
    if window:
        with open_dict(cfg):
            cfg.data.dataset.num_steps = max(
                cfg.data.dataset.num_steps, window
            )

    # Proprio models: load their columns; they are normalized like actions.
    proprio_columns = list(cfg.model.get('proprio_columns') or [])
    with open_dict(cfg):
        for col in proprio_columns:
            if col not in cfg.data.dataset.keys_to_load:
                cfg.data.dataset.keys_to_load.append(col)

    # Yaw auxiliary: load the raw object pose; it must stay unnormalized.
    aux_yaw = cfg.loss.get('aux_yaw') or {}
    aux_col = aux_yaw.get('column') if aux_yaw.get('weight', 0) > 0 else None
    if aux_col and aux_col not in cfg.data.dataset.keys_to_load:
        with open_dict(cfg):
            cfg.data.dataset.keys_to_load.append(aux_col)

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    dataset_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    cache_dir = dataset_cfg.pop('cache_dir', None) or dataset_dir
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        # uint8 batches: 4x less host-to-GPU copy, normalized in the forward.
        get_img_preprocessor(
            source=col, target=col, img_size=cfg.img_size, uint8=True
        )
        for col in cfg.data.dataset.keys_to_load
        if col.startswith('pixels')
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels') or col == aux_col:
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            if col == 'action':
                action_scaler = normalizer.lambd  # the fitted scaler

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )
        if proprio_columns:
            cfg.model.proprio_encoder.input_dim = sum(
                dataset.get_dim(col) for col in proprio_columns
            )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    # Keep search-loss plans inside the dataset's action range (normalized,
    # tiled over the frameskip block): the world model never saw actions
    # outside it, and the env clips them anyway.
    if (cfg.loss.get('search') or {}).get('clip_actions'):
        raw = np.asarray(dataset.get_col_data('action'), dtype=np.float64)
        raw = raw[~np.isnan(raw).any(axis=1)]
        block = cfg.data.dataset.frameskip
        with open_dict(cfg):
            for name, bound in (('low', raw.min(0)), ('high', raw.max(0))):
                norm = action_scaler.transform(bound[None]).reshape(-1)
                cfg.loss.search[f'action_{name}'] = np.tile(
                    norm, block
                ).tolist()

    # Noisy-replay ratio sweeps: subset after fitting normalizers on all data.
    replay_filter = cfg.get('replay_filter')
    if replay_filter:
        dataset = filter_replay_episodes(dataset, **replay_filter)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val_cfg['num_workers'] = cfg.get('val_num_workers', val_cfg['num_workers'])
    if val_cfg['num_workers'] == 0:
        val_cfg.pop('persistent_workers', None)
        val_cfg.pop('prefetch_factor', None)
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    init_from = cfg.get('init_from')
    if init_from:
        # Post-training: keep the pretrained architecture and weights.
        world_model = load_pretrained(init_from)
        checkpoint_dir = swm.data.utils.get_cache_dir(sub_folder='checkpoints')
        with open_dict(cfg):
            cfg.model = _resolve(init_from, Path(checkpoint_dir))[1]
    else:
        world_model = hydra.utils.instantiate(cfg.model)
    if cfg.get('compile', False):
        # Gradient diagnostics backprop with retain_graph=True.
        torch._functorch.config.donated_buffer = False
        world_model.encode = torch.compile(world_model.encode)
        world_model.predict = torch.compile(world_model.predict)

    max_epochs = cfg.trainer.max_epochs
    total_steps = (
        max_epochs * len(train) if max_epochs > 0 else cfg.trainer.max_steps
    )
    extra_modules = {}
    if aux_col:
        extra_modules['aux_head'] = torch.nn.Linear(cfg.embed_dim, 2)
    search = cfg.loss.get('search') or {}
    use_gfn = search.get('weight', 0) > 0 and search.get('sampler') == 'gfn'
    optimizers = {
        'model_opt': {
            'modules': 'model|aux_head' if aux_col else 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(
                    1, int(cfg.get('warmup_ratio', 0.01) * total_steps)
                ),
                'max_steps': max(1, total_steps),
            },
            'interval': 'epoch',
        },
    }
    if use_gfn:
        extra_modules['gfn_policy'] = hydra.utils.instantiate(
            cfg.loss.gfn.policy
        )
        optimizers['policy_opt'] = {
            'modules': 'gfn_policy',
            'optimizer': dict(cfg.loss.gfn.optimizer),
            'scheduler': {'type': 'ConstantLR', 'factor': 1.0},
            'interval': 'epoch',
        }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        **extra_modules,
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=cfg.get('save_epoch_interval', 1),
        # Step-based runs may never reach an epoch interval.
        step_interval=cfg.get('save_step_interval')
        if cfg.trainer.max_steps > 0
        else None,
        policy_cfg=cfg.loss.gfn.policy if use_gfn else None,
    )

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    if use_gfn and trainer_cfg.get('max_steps', -1) > 0:
        # max_steps counts batches; Lightning's global_step advances once per
        # optimizer per batch, whether or not that optimizer stepped.
        trainer_cfg['max_steps'] = total_steps * len(optimizers)
    trainer = pl.Trainer(
        **trainer_cfg,
        callbacks=[save_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    configured_ckpt_path = cfg.get('ckpt_path', None)
    ckpt_path = (
        Path(configured_ckpt_path).expanduser().resolve()
        if configured_ckpt_path
        else run_dir / f'{cfg.output_model_name}_weights.ckpt'
    )
    if configured_ckpt_path and not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {ckpt_path}')
    manager = OfflineWandbManager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        weights_only=False,
    )

    manager()


if __name__ == '__main__':
    run()
