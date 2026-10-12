"""LeWM GCBC dimensions, frozen training, native sampling and CEM seam."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from scripts.plan.eval_wm import make_policy
from scripts.train import lewm_gcbc
from stable_worldmodel.data import ZScoreScaler
from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.protocols import Actionable
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm_gcbc import LeWMGCBC
from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

ROOT = Path(__file__).resolve().parents[2]


def world_model_config():
    return {
        '_target_': 'stable_worldmodel.wm.lewm.LeWMState',
        'state_columns': ['state'],
        'encoder': {
            '_target_': 'torch.nn.Linear',
            'in_features': 2,
            'out_features': 4,
        },
        'projector': {'_target_': 'torch.nn.BatchNorm1d', 'num_features': 4},
        'predictor': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Predictor',
            'num_frames': 3,
            'input_dim': 4,
            'hidden_dim': 4,
            'depth': 1,
            'heads': 1,
            'dim_head': 4,
            'mlp_dim': 8,
            'dropout': 0.1,
        },
        'action_encoder': {
            '_target_': 'stable_worldmodel.wm.lewm.module.Embedder',
            'input_dim': 10,
            'emb_dim': 4,
        },
    }


@pytest.fixture
def prior():
    torch.manual_seed(1)
    return LeWMGCBC(
        hydra.utils.instantiate(world_model_config()),
        4,
        2,
        5,
        [-1, -1],
        [1, 1],
        hidden_dim=16,
    )


def info(history=1):
    return {
        'state': torch.randn(2, history, 2),
        'goal_state': torch.randn(2, 1, 2),
        'action_history': torch.randn(2, history - 1, 10),
    }


def test_dimensions_bounds_and_frozen_optimization(prior):
    prior.train()
    before = deepcopy(prior.world_model.state_dict())
    head_before = deepcopy(prior.policy.state_dict())
    observations = info()
    distribution = prior(observations)
    assert prior.policy.input_dim == 8
    assert distribution.mean.shape == (2, 10)
    assert distribution.log_prob(torch.zeros(2, 10)).shape == (2,)
    optimizer = torch.optim.Adam(prior.policy.parameters(), lr=0.01)
    loss = -distribution.log_prob(torch.zeros(2, 10)).mean()
    loss.backward()
    optimizer.step()
    assert not prior.world_model.training
    for name, value in prior.world_model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(
        not p.requires_grad and p.grad is None
        for p in prior.world_model.parameters()
    )
    assert any(
        not torch.equal(value, head_before[name])
        for name, value in prior.policy.state_dict().items()
    )
    actions = prior.get_action(observations, horizon=5, sample=True)
    assert actions.shape == (2, 5, 2)
    assert torch.isfinite(actions).all() and (actions.abs() <= 1).all()
    assert prior.get_action(observations).shape == (2, 2)
    with pytest.raises(ValueError, match='match LeWM'):
        LeWMGCBC(prior.world_model, 4, 2, 1, [-1, -1], [1, 1])


def test_autoregressive_history_and_prefix(prior, monkeypatch):
    observations = info(history=3)
    prefix = torch.full((2, 2, 10), 0.25)
    emb, _ = prior._encode(observations)
    expanded = {k: v.unsqueeze(1) for k, v in observations.items()}
    expanded['emb'] = emb.unsqueeze(1)
    expected_z = prior.world_model.rollout(expanded, prefix.unsqueeze(1))[
        'predicted_emb'
    ][:, 0, -1]
    actor_inputs = []
    distribution = prior.distribution

    def record(z, goal):
        actor_inputs.append(z.clone())
        return distribution(z, goal)

    monkeypatch.setattr(prior, 'distribution', record)
    histories = []
    hook = prior.world_model.predictor.register_forward_pre_hook(
        lambda module, args: histories.append(args[0].shape[1])
    )
    try:
        actions = prior.plan(observations, horizon=3, prefix_actions=prefix)
    finally:
        hook.remove()
    torch.testing.assert_close(actor_inputs[0], expected_z)
    assert histories and set(histories) == {3}
    assert actions.shape == (2, 3, 10) and (actions.abs() <= 1).all()


def test_image_encoder_and_bfloat16(prior):
    from transformers import ViTConfig, ViTModel

    encoder = ViTModel(
        ViTConfig(
            image_size=8,
            patch_size=4,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=8,
        )
    )
    model = LeWM(
        encoder,
        prior.world_model.predictor,
        prior.world_model.action_encoder,
        projector=prior.world_model.projector,
    )
    policy = LeWMGCBC(model, 4, 2, 5, [-1, -1], [1, 1], hidden_dim=16)
    observations = {
        'pixels': torch.randn(2, 1, 3, 8, 8),
        'goal': torch.randn(2, 1, 3, 8, 8),
    }
    assert policy(observations).mean.shape == (2, 10)
    policy.to(torch.bfloat16)
    actions = policy.plan(observations, horizon=2)
    assert actions.shape == (2, 2, 10) and torch.isfinite(actions).all()


@pytest.mark.parametrize('guided', [False, True])
def test_cem_with_and_without_prior(prior, guided):
    cfg = OmegaConf.create(
        {
            'plan_config': {
                'horizon': 3,
                'receding_horizon': 1,
                'action_block': 5,
            },
            'objective': {'_target_': 'stable_worldmodel.planning.GoalMSE'},
            'solver': {
                '_target_': 'stable_worldmodel.planning.solver.CEMSolver',
                'num_samples': 8,
                'n_steps': 2,
                'topk': 2,
                'batch_size': 2,
            },
        }
    )
    policy = make_policy(cfg, prior if guided else prior.world_model, {}, {})
    solver = policy.solver
    assert isinstance(solver.cost, Actionable) == guided
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, (2, 2)),
        n_envs=2,
        config=PlanConfig(horizon=3, receding_horizon=1, action_block=5),
    )
    observations = info(history=3)
    result = solver.solve(observations)
    actions = result['actions']
    assert actions.shape == (2, 3, 10) and torch.isfinite(actions).all()
    if guided:
        assert (actions.abs() <= 1).all()
        assert (
            solver.solve(observations, init_action=actions[:, 1:])[
                'actions'
            ].shape
            == actions.shape
        )


def test_bc_evaluation_executes_one_block(prior):
    cfg = OmegaConf.create(
        {'policy_mode': 'bc', 'plan_config': {'action_block': 5}}
    )
    policy = make_policy(cfg, prior, {}, {})
    env = SimpleNamespace(
        num_envs=2,
        single_action_space=gym.spaces.Box(-1, 1, (2,)),
        action_space=gym.spaces.Box(-1, 1, (2, 2)),
    )
    policy.set_env(env)
    observations = info()
    expected = prior.get_action(observations, horizon=5).numpy()
    raw = {k: v.numpy() for k, v in observations.items()}
    raw['goal'] = np.zeros((2, 1, 3, 8, 8), dtype=np.uint8)
    for step in range(5):
        np.testing.assert_allclose(
            policy.get_action(dict(raw)), expected[:, step], atol=1e-6
        )


def test_native_training_sampling_and_checkpoint(tmp_path, monkeypatch):
    lance = pytest.importorskip('lance')
    pa = pytest.importorskip('pyarrow')
    monkeypatch.setenv('STABLEWM_HOME', str(tmp_path))
    import stable_pretraining as spt

    monkeypatch.setattr(spt.get_config(), 'cache_dir', str(tmp_path / 'spt'))
    wm_cfg = world_model_config()
    wm = hydra.utils.instantiate(wm_cfg)
    frozen = deepcopy(wm.state_dict())
    save_pretrained(wm, 'wm', wm_cfg)
    steps = np.tile(np.arange(60, dtype=np.float32), 4)
    episodes = np.repeat(np.arange(4), 60)
    actions = np.stack([np.sin(steps), np.cos(steps)], axis=1).astype(
        np.float32
    )
    actions[steps == 59] = np.nan
    dataset_path = tmp_path / 'expert.lance'
    lance.write_dataset(
        pa.table(
            {
                'episode_idx': episodes,
                'step_idx': steps.astype(np.int64),
                'state': np.stack([episodes, steps], axis=1)
                .astype(np.float32)
                .tolist(),
                'action': actions.tolist(),
            }
        ),
        str(dataset_path),
    )
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / 'scripts/train/config')
    ):
        cfg = hydra.compose(
            config_name='lewm_gcbc',
            overrides=[
                'world_model=wm/weights.pt',
                f'data.dataset.name={dataset_path}',
                'data.dataset.keys_to_load=[state,action]',
                'data.dataset.keys_to_cache=[state,action]',
                'num_workers=0',
                'loader.batch_size=8',
                'loader.drop_last=false',
                'trainer.accelerator=cpu',
                'trainer.precision=32-true',
                'trainer.max_epochs=1',
                '+trainer.limit_train_batches=2',
                '+trainer.limit_val_batches=1',
                '+trainer.enable_progress_bar=false',
                '+trainer.enable_model_summary=false',
                'wandb.enabled=false',
                'subdir=prior',
                'model.hidden_dim=16',
            ],
        )
    train, val, stats, action_dim = lewm_gcbc.get_data(cfg)
    assert action_dim == 2
    goals = train.dataset.dataset
    from stable_worldmodel.data.goal_lance import BatchedGoalLanceDataset

    assert isinstance(goals.dataset, BatchedGoalLanceDataset)
    train_eps = {goals.clip_indices[i][0] for i in train.dataset.indices}
    val_eps = {goals.clip_indices[i][0] for i in val.dataset.indices}
    assert train_eps.isdisjoint(val_eps)
    state_scaler = ZScoreScaler(**stats['state'])
    action_scaler = ZScoreScaler(**stats['action'])
    distances = set()
    for idx in range(len(goals)):
        sample = goals[idx]
        state = state_scaler.inverse_transform(sample['state'])[0]
        goal = state_scaler.inverse_transform(sample['goal_state'])[0]
        assert round(state[0].item()) == round(goal[0].item())
        distance = round((goal[1] - state[1]).item())
        assert distance > 0 and distance % 5 == 0
        distances.add(distance)
        start = round(state[1].item())
        assert start + 5 <= 59
        target = action_scaler.inverse_transform(
            sample['action'].reshape(5, 2)
        )
        np.testing.assert_allclose(
            target, actions[start : start + 5], atol=1e-6
        )
    assert 25 in distances and len(distances) > 3
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = lewm_gcbc.run.__wrapped__(cfg)
    finally:
        torch.set_num_threads(old_threads)
    restored = load_pretrained(str(output / 'weights_epoch_1.pt'))
    for name, value in restored.world_model.state_dict().items():
        torch.testing.assert_close(value, frozen[name])
    assert restored.normalization == stats
    assert restored(next(iter(val))).mean.shape == (8, 10)
