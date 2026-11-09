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


def test_extra_policy_steps_update_only_the_policy(lewm_script):
    from types import SimpleNamespace

    torch.manual_seed(0)
    H, T = 2, 3
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {'horizon': T, 'bptt': True, 'temperature': 1.0},
                'gfn': {
                    'num_samples': 5,
                    'explore_std': 1.5,
                    'tau_start': 2.0,
                    'tau_end': 0.2,
                    'anneal_steps': 10,
                    'expert_tb': True,
                    'policy_steps': 4,
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
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    steps = []
    step = opt.step
    opt.step = lambda: (steps.append(1), step())[1]
    module = SimpleNamespace(
        model=model,
        gfn_policy=policy,
        _optimizer_index_to_name={0: 'model_opt', 1: 'policy_opt'},
        optimizers=lambda: [None, SimpleNamespace(optimizer=opt)],
        trainer=SimpleNamespace(gradient_clip_val=1.0),
    )
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], 1)
    batch = model.encode({'state': state, 'action': actions})
    before = [p.detach().clone() for p in policy.parameters()]

    lewm_script.gfn_extra_policy_steps(module, batch, batch['emb'], cfg, 5)
    assert len(steps) == 3  # policy_steps - 1; the joint step does the last
    assert any(
        not torch.equal(a, b) for a, b in zip(before, policy.parameters())
    )
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in policy.parameters())


def test_evaluator_clamps_candidates_to_action_bounds():
    from stable_worldmodel.planning.evaluator import ShootingCostEvaluator

    class _Model:
        def rollout(self, info, candidates):
            info['seen'] = candidates.clone()
            return info

    bounds = (torch.tensor([-1.0, -2.0]), torch.tensor([1.0, 2.0]))
    evaluator = ShootingCostEvaluator(
        _Model(), objective=None, encode_goal=None, action_bounds=bounds
    )
    candidates = torch.tensor([[[[5.0, -5.0], [0.5, 0.5]]]])
    info = evaluator._rollout({}, candidates)
    assert info['seen'].tolist() == [[[[1.0, -2.0], [0.5, 0.5]]]]


def test_gaussian_prior_keeps_flat_reward_normalizable(lewm_script):
    """Flat cost (as far from the data): with the prior TB fits N(0, 1);
    without it the widened exploration samples make the policy keep spreading.
    """
    log_reward = lewm_script.gfn_log_reward
    plans = torch.full((1, 1, 2, 3), 2.0)
    costs = torch.tensor([[10.0]])
    gfn = OmegaConf.create({'prior_std': 2.0})
    # -10 / 5 - (6 * 2^2) / (2 * 2^2)
    assert log_reward(costs, plans, gfn, tau=5.0).item() == pytest.approx(-5)
    no_prior = OmegaConf.create({'prior_std': None})
    assert log_reward(costs, plans, no_prior, 5.0).item() == pytest.approx(-2)

    def fitted_std(prior_std):
        torch.manual_seed(0)
        policy = PlanGFlowNet(
            emb_dim=2, action_dim=2, horizon=2, hidden_dim=32
        )
        opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
        z = torch.zeros(16, 2)
        gfn = OmegaConf.create({'prior_std': prior_std})
        for _ in range(300):
            plans = policy.sample(z, z, n=8, std_scale=1.5)
            log_r = log_reward(torch.zeros(16, 8), plans, gfn, tau=1.0)
            log_z = policy.log_z(z, z)[:, None]
            tb = (log_z + policy.log_prob(z, z, plans) - log_r).pow(2)
            opt.zero_grad()
            tb.mean().backward()
            opt.step()
        return policy.sample(z, z, n=64).std().item()

    assert fitted_std(1.0) == pytest.approx(1.0, abs=0.25)
    assert fitted_std(None) > 3.0


def test_expert_threshold_flattens_reward_below_expert(lewm_script):
    log_reward = lewm_script.gfn_log_reward
    costs = torch.tensor([[50.0, 40.0, 55.0, 80.0]])  # expert first
    plans = torch.zeros(1, 4, 2, 3)
    gfn = OmegaConf.create(
        {'prior_std': None, 'expert_threshold': True, 'threshold_margin': 10}
    )
    # Cost counts only above 50 + 10: the first three tie, 80 pays 20 / tau.
    assert log_reward(costs, plans, gfn, tau=5.0).tolist() == [
        [0.0, 0.0, 0.0, -4.0]
    ]
    gfn.expert_threshold = False
    assert log_reward(costs, plans, gfn, tau=5.0)[0, 1].item() == -8.0


def test_relative_cost_is_scale_free(lewm_script):
    log_reward = lewm_script.gfn_log_reward
    plans = torch.zeros(1, 3, 2, 3)
    gfn = OmegaConf.create(
        {'prior_std': None, 'relative_cost': True, 'expert_threshold': True}
    )
    costs = torch.tensor([[50.0, 40.0, 150.0]])  # expert first
    # Ratios 1, 0.8, 3: only the excess over 1 counts, divided by tau.
    expected = [[0.0, 0.0, -4.0]]
    assert log_reward(costs, plans, gfn, tau=0.5).tolist() == expected
    # Inflating every cost leaves the reward unchanged.
    assert log_reward(3 * costs, plans, gfn, tau=0.5).tolist() == expected
