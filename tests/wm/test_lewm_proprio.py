"""Image + proprio LeWM: config, fusion, residual prediction and planning."""

import importlib.util
import sys
from pathlib import Path

import hydra
import pytest
import torch

from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.wm.lewm import LeWMProprio, LeWMProprioRes

ROOT = Path(__file__).resolve().parents[2]


def load_train_script():
    """Import scripts/train/lewm.py, which registers the config resolvers."""
    spec = importlib.util.spec_from_file_location(
        'lewm_train_script', ROOT / 'scripts/train/lewm.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_train_script()


def tiny_model(target='LeWMProprioRes'):
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(ROOT / 'scripts/train/config')
    ):
        cfg = hydra.compose(
            config_name='lewm_proprio',
            overrides=[
                f'model._target_=stable_worldmodel.wm.lewm.{target}',
                'img_size=16',
                'patch_size=8',
                'wm.history_size=2',
                'proprio_embed_dim=8',
                'model.proprio_encoder.input_dim=10',
                'model.action_encoder.input_dim=2',
                'model.predictor.depth=1',
                'model.predictor.heads=2',
                'model.predictor.dim_head=8',
                'model.predictor.mlp_dim=32',
                'model.projector.hidden_dim=32',
                'model.pred_proj.hidden_dim=32',
            ],
        )
    return hydra.utils.instantiate(cfg.model).eval()


def batch(b=2, t=3):
    return {
        'pixels': torch.randn(b, t, 3, 16, 16),
        'obs_agent_qpos': torch.randn(b, t, 7),
        'obs_extra_tcp_pose': torch.randn(b, t, 3),
        'action': torch.randn(b, t, 2),
    }


@pytest.mark.parametrize('target', ['LeWMProprio', 'LeWMProprioRes'])
def test_proprio_changes_latent_and_trains(target):
    model = tiny_model(target)
    assert type(model).__name__ == target
    info = model.encode(batch())
    assert info['emb'].shape == (2, 3, 192)
    moved = batch()
    moved['pixels'] = info['pixels']
    moved['obs_agent_qpos'] = info['obs_agent_qpos'] + 1
    moved['obs_extra_tcp_pose'] = info['obs_extra_tcp_pose']
    assert not torch.allclose(model.encode(moved)['emb'], info['emb'])

    model.train()
    info = model.encode(batch())
    preds = model.predict(
        {'emb': info['emb'][:, :2], 'act_emb': info['act_emb'][:, :2]}
    )['preds']
    loss = (preds - info['emb'][:, 1:]).pow(2).mean()
    loss.backward()
    assert model.proprio_encoder.net[0].weight.grad.abs().sum() > 0


def test_residual_variant_adds_current_latent():
    model = tiny_model()
    assert isinstance(model, LeWMProprio)
    info = model.encode(batch())
    ctx = {'emb': info['emb'][:, :2], 'act_emb': info['act_emb'][:, :2]}
    delta = LeWMProprio.predict(model, dict(ctx))['preds']
    preds = LeWMProprioRes.predict(model, dict(ctx))['preds']
    torch.testing.assert_close(preds, ctx['emb'] + delta)


def test_planning_encodes_proprio_goal():
    model = tiny_model()
    info = {
        'pixels': torch.randn(1, 4, 2, 3, 16, 16),
        'obs_agent_qpos': torch.randn(1, 4, 2, 7),
        'obs_extra_tcp_pose': torch.randn(1, 4, 2, 3),
        'action_history': torch.randn(1, 4, 1, 2),
        'goal': torch.randn(1, 4, 1, 3, 16, 16),
        'goal_obs_agent_qpos': torch.randn(1, 4, 1, 7),
        'goal_obs_extra_tcp_pose': torch.randn(1, 4, 1, 3),
    }
    expected = model.encode(
        {
            'pixels': info['goal'][:, 0],
            'obs_agent_qpos': info['goal_obs_agent_qpos'][:, 0],
            'obs_extra_tcp_pose': info['goal_obs_extra_tcp_pose'][:, 0],
        }
    )['emb']
    cost = ShootingCostEvaluator(model, GoalMSE()).get_cost(
        info, torch.randn(1, 4, 3, 2)
    )
    assert cost.shape == (1, 4)
    torch.testing.assert_close(info['goal_emb'], expected)
    assert info['predicted_emb'].shape == (1, 4, 5, 192)

    missing = {k: v for k, v in info.items() if not k.startswith('goal_obs')}
    missing.pop('goal_emb')
    missing.pop('emb')
    with pytest.raises(KeyError):
        ShootingCostEvaluator(model, GoalMSE()).get_cost(
            missing, torch.randn(1, 4, 3, 2)
        )


def test_proprio_columns_required():
    with pytest.raises(ValueError):
        LeWMProprio(
            encoder=torch.nn.Linear(1, 1),
            predictor=torch.nn.Identity(),
            action_encoder=torch.nn.Identity(),
            proprio_encoder=torch.nn.Identity(),
            proprio_columns=[],
        )
