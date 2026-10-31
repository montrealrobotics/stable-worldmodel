import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stable_worldmodel.wm.lewm import LeWMState, PlanGFlowNet


def test_sample_and_log_prob_shapes():
    torch.manual_seed(0)
    policy = PlanGFlowNet(emb_dim=6, action_dim=3, horizon=4, hidden_dim=16)
    z0, z_goal = torch.randn(5, 6), torch.randn(5, 6)
    plans = policy.sample(z0, z_goal, n=7, std_scale=1.5)
    assert plans.shape == (5, 7, 4, 3)
    assert not plans.requires_grad
    log_pi = policy.log_prob(z0, z_goal, plans)
    assert log_pi.shape == (5, 7) and torch.isfinite(log_pi).all()
    assert policy.log_z(z0, z_goal).shape == (5,)
    log_pi.sum().backward()
    assert policy.head.weight.grad.abs().sum() > 0


def test_trajectory_balance_learns_reward():
    """TB on R = exp(-|a - c|^2 / tau) moves the samples onto c."""
    torch.manual_seed(0)
    policy = PlanGFlowNet(emb_dim=2, action_dim=2, horizon=2, hidden_dim=32)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    z = torch.zeros(16, 2)
    centre = torch.tensor([1.5, -1.0])

    def log_r(plans):
        return -(plans - centre).pow(2).sum((-1, -2)) / 0.1

    for _ in range(300):
        plans = policy.sample(z, z, n=8, std_scale=1.5)
        log_z = policy.log_z(z, z)[:, None]
        tb = (log_z + policy.log_prob(z, z, plans) - log_r(plans)).pow(2)
        opt.zero_grad()
        tb.mean().backward()
        opt.step()
    plans = policy.sample(z, z, n=64)
    assert (plans.mean((0, 1)) - centre).abs().max() < 0.2


class _Predictor(nn.Module):
    def forward(self, emb, act_emb):
        return emb + act_emb


def _identity(dim):
    layer = nn.Linear(dim, dim, bias=False)
    nn.init.eye_(layer.weight)
    return layer


@pytest.fixture
def lewm_script():
    pytest.importorskip('stable_pretraining')
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_gfn_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gfn_search_loss_splits_gradients(lewm_script):
    torch.manual_seed(0)
    H, T = 2, 3
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T,
                    'bptt': True,
                    'temperature': 1.0,
                    'batch_size': 3,
                },
                'gfn': {
                    'num_samples': 5,
                    'explore_std': 1.5,
                    'tau_start': 2.0,
                    'tau_end': 0.2,
                    'anneal_steps': 10,
                    'expert_tb': True,
                },
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    policy = PlanGFlowNet(emb_dim=2, action_dim=2, horizon=T, hidden_dim=8)
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], 1)
    batch = model.encode({'state': state, 'action': actions})

    out = lewm_script.gfn_search_loss(
        model, policy, batch, batch['emb'], cfg, step=5
    )
    assert out['gfn_tau'] == pytest.approx(2.0 * 0.1**0.5)
    # InfoNCE trains only the world model.
    out['search_loss'].backward(retain_graph=True)
    assert model.action_encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in policy.parameters())
    # Trajectory balance trains only the policy.
    model.zero_grad(set_to_none=True)
    out['gfn_tb_loss'].backward()
    assert policy.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
