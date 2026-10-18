import math
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
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


class SaveCkptCallback(Callback):
    """Save the model with save_pretrained every ``epoch_interval`` epochs.

    Also saves the final weights when training ends off-interval, including
    a partial last epoch when ``max_steps`` stops training.
    """

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self._saved_epoch = None

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        if trainer.is_global_zero and epoch % self.epoch_interval == 0:
            self._save(pl_module.model, epoch)

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer, pl_module)
        # current_epoch counts finished epochs here, plus one if stopped
        # mid-epoch, matching the 1-based names used above.
        epoch = max(trainer.current_epoch, 1)
        if trainer.is_global_zero and self._saved_epoch != epoch:
            self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        self._saved_epoch = epoch
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def search_loss(model, batch, emb, cfg):
    """InfoNCE over goal costs: the expert plan must beat a short CEM's elites."""
    H, search = cfg.wm.history_size, cfg.loss.search
    T = search.horizon
    if emb.size(1) < H + T:
        raise ValueError(f'search loss needs num_steps >= {H + T}')
    actions = batch['action']
    B = actions.size(0)
    expert = actions[:, None, H - 1 : H - 1 + T]
    goal = emb[:, H - 1 + T, None].detach()
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
                model, info['emb'].flatten(0, 1), history.flatten(0, 1), T, H
            )
            info['predicted_emb'] = preds.unflatten(0, (B, S))
        info['goal_emb'] = goal
        return GoalMSE()(info)  # (B, S)

    def cost(plans):
        # Attention kernels fail past 65535 rollouts * heads (CUDA grid limit).
        # shortcut: needs B * heads <= 65535, chunk B too if not.
        step = max(1, 65535 // (heads * B))
        chunks = plans.split(step, dim=1)
        return torch.cat([chunk_cost(c) for c in chunks], dim=1)

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
    plans = torch.cat([expert, perturbed, *elites], 1)
    costs = cost(plans)  # expert at index 0
    target = costs.new_zeros(B, dtype=torch.long)
    return F.cross_entropy(-costs.float() / search.temperature, target)


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
    search = getattr(cfg.loss, 'search', None)
    if search and search.weight > 0:
        output['search_loss'] = search_loss(self.model, batch, emb, cfg)
        output['loss'] = output['loss'] + search.weight * output['search_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    losses_dict[f'{stage}/sigreg_weight'] = float(lambd)
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
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

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
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(
                    1, int(cfg.get('warmup_ratio', 0.01) * total_steps)
                ),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
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
    )

    trainer = pl.Trainer(
        **cfg.trainer,
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
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
        weights_only=False,
    )

    manager()


if __name__ == '__main__':
    run()
