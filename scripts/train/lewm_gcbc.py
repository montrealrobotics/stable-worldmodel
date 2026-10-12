"""Train a Gaussian BC prior independently from a pretrained LeWM."""

import os
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

import stable_worldmodel as swm
from scripts.train.lewm import SaveCkptCallback, get_img_preprocessor
from stable_worldmodel.data import column_normalizer
from stable_worldmodel.data.utils import _resolve_dataset
from stable_worldmodel.wm.utils import _resolve, load_pretrained


def bc_forward(self, batch, stage):
    distribution = self.model(batch)
    target = batch['action'][:, -1].to(distribution.mean)
    if (
        target.shape != distribution.mean.shape
        or not torch.isfinite(target).all()
    ):
        raise ValueError('Invalid expert action block')
    batch['loss'] = -distribution.log_prob(target).mean()
    if not torch.isfinite(batch['loss']):
        raise ValueError('Nonfinite BC loss')
    self.log_dict(
        {f'{stage}/bc_loss': batch['loss'].detach()},
        on_step=True,
        on_epoch=True,
        sync_dist=True,
        batch_size=target.size(0),
    )
    return batch


def seed_goals(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset.dataset
    dataset.rng = np.random.default_rng(torch.initial_seed())


def get_data(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop('name')
    cache_dir = dataset_cfg.pop('cache_dir', None) or os.getenv(
        'LOCAL_DATASET_DIR'
    )
    path = None
    if '://' not in name:
        path = _resolve_dataset(
            name, swm.data.get_cache_dir(cache_dir, sub_folder='datasets')
        )
    dataset_format = swm.data.detect_format(path) if path is not None else None
    if (
        dataset_format is not None
        and dataset_format.name == 'lance'
        and dataset_cfg.get('format') in (None, 'lance')
    ):
        from stable_worldmodel.data.goal_lance import BatchedGoalLanceDataset

        dataset_cfg.pop('format', None)
        dataset = BatchedGoalLanceDataset(path=path, **dataset_cfg)
    else:
        dataset = swm.data.load_dataset(
            name, cache_dir=cache_dir, **dataset_cfg
        )
    if not 0 < cfg.train_split < 1 or cfg.goal_mean_steps < dataset.frameskip:
        raise ValueError('Invalid train split or future-goal distance')
    transforms, normalization = [], {}
    for col in cfg.data.dataset.keys_to_load:
        if col.startswith('pixels'):
            transforms.append(get_img_preprocessor(col, col, cfg.img_size))
        else:
            transform = column_normalizer(dataset, col, col)
            scaler = transform.lambd
            normalization[col] = {
                'mean': scaler.mean.reshape(-1).tolist(),
                'std': scaler.std.reshape(-1).tolist(),
                'eps': scaler.eps,
            }
            transforms.append(transform)
    dataset.transform = spt.data.transforms.Compose(*transforms)
    # Native clip indexing keeps blocks in an episode; exclude terminal NaNs.
    raw_actions = np.asarray(dataset.get_col_data('action'))
    invalid = np.concatenate(
        ([0], np.cumsum(~np.isfinite(raw_actions).all(axis=1)))
    )
    dataset.clip_indices = [
        (ep, start)
        for ep, start in dataset.clip_indices
        if invalid[dataset.offsets[ep] + start + dataset.span]
        == invalid[dataset.offsets[ep] + start]
    ]
    goals = swm.data.GoalDataset(
        dataset,
        goal_probabilities=(0.0, 0.5, 0.5, 0.0),
        # Geometric distances average goal_mean_steps in environment steps.
        gamma=1 - dataset.frameskip / cfg.goal_mean_steps,
        current_goal_offset=dataset.num_steps,
        goal_keys={
            key: 'goal' if key == 'pixels' else f'goal_{key}'
            for key in cfg.data.dataset.keys_to_load
            if key != 'action'
        },
        seed=cfg.seed,
    )
    episodes = sorted({ep for ep, _ in goals.clip_indices})
    if len(episodes) < 2:
        raise ValueError('Need at least two episodes with valid future goals')
    generator = torch.Generator().manual_seed(cfg.seed)
    order = torch.randperm(len(episodes), generator=generator).tolist()
    count = min(
        len(episodes) - 1, max(1, int(cfg.train_split * len(episodes)))
    )
    train_episodes = {episodes[i] for i in order[:count]}
    train_indices, val_indices = [], []
    for idx, (ep, _) in enumerate(goals.clip_indices):
        (train_indices if ep in train_episodes else val_indices).append(idx)
    loader_cfg = dict(cfg.loader)
    if not loader_cfg['num_workers']:
        loader_cfg.update(persistent_workers=False, prefetch_factor=None)
    train = DataLoader(
        Subset(goals, train_indices),
        **loader_cfg,
        generator=generator,
        worker_init_fn=seed_goals,
    )
    val = DataLoader(
        Subset(goals, val_indices),
        **{**loader_cfg, 'shuffle': False, 'drop_last': False},
        worker_init_fn=seed_goals,
    )
    if not len(train):
        raise ValueError('No training batches; reduce loader.batch_size')
    return train, val, normalization, dataset.get_dim('action')


@hydra.main(version_base=None, config_path='./config', config_name='lewm_gcbc')
def run(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    world_model = load_pretrained(cfg.world_model)
    train, val, normalization, action_dim = get_data(cfg)
    checkpoint_dir = swm.data.get_cache_dir(sub_folder='checkpoints')
    _, world_model_cfg = _resolve(cfg.world_model, checkpoint_dir)
    action_scaler = swm.data.ZScoreScaler(**normalization['action'])
    with open_dict(cfg.model):
        cfg.model.world_model = world_model_cfg
        cfg.model.latent_dim = world_model.predictor.input_dim
        cfg.model.action_dim = action_dim
        cfg.model.action_block = cfg.data.dataset.frameskip
        cfg.model.action_low = (
            action_scaler.transform(np.asarray(cfg.action_low))
            .reshape(-1)
            .tolist()
        )
        cfg.model.action_high = (
            action_scaler.transform(np.asarray(cfg.action_high))
            .reshape(-1)
            .tolist()
        )
        cfg.model.normalization = normalization
    model = hydra.utils.instantiate(cfg.model, world_model=world_model)
    module = spt.Module(
        model=model,
        forward=bc_forward,
        optim={
            'policy_opt': {
                'modules': 'model.policy',
                'optimizer': dict(cfg.optimizer),
            }
        },
    )
    run_dir = Path(checkpoint_dir, cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml', resolve=True)
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(str(run_dir), cfg.model)],
        num_sanity_val_steps=1,
        logger=logger,
    )
    spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train, val=val),
    )()
    return run_dir


if __name__ == '__main__':
    run()
