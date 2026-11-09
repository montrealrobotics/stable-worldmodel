import importlib.util
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from stable_worldmodel.wm.lewm import LeWMState


class _Predictor(nn.Module):
    def forward(self, emb, act_emb):
        return emb + act_emb


def _identity(dim):
    layer = nn.Linear(dim, dim, bias=False)
    nn.init.eye_(layer.weight)
    return layer


@pytest.fixture
def search_loss():
    pytest.importorskip('stable_pretraining')
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_search_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.search_loss


@pytest.mark.parametrize('bptt', [True, False])
def test_expert_plan_ranks_first_and_trains(search_loss, bptt):
    torch.manual_seed(0)
    H, T, K = 2, 3, 15
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T,
                    'bptt': bptt,
                    'cem_steps': 5,
                    'num_samples': K,
                    'topk': 4,
                    'num_perturbed': 6,
                    'perturb_std': 0.5,
                    'var_scale': 1.0,
                    # Sharp: CEM elites approach the exact expert plan.
                    'temperature': 1e-3,
                }
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    # Exact dynamics: state_{t+1} = state_t + action_t, so the expert hits the goal.
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})

    loss = search_loss(model, batch, batch['emb'], cfg)
    assert loss < 0.5 * math.log(1 + 6 + 5 * 4)  # well below chance
    loss.backward()
    assert model.action_encoder.weight.grad.abs().sum() > 0
    # Without BPTT the last step's window holds only detached predictions.
    assert (model.encoder.weight.grad is not None) == bptt

    short = {k: v[:, : H + T - 1] for k, v in batch.items()}
    with pytest.raises(ValueError, match='num_steps'):
        search_loss(model, short, short['emb'], cfg)


@pytest.mark.parametrize('bptt', [True, False])
def test_all_steps_infonce(search_loss, bptt):
    torch.manual_seed(0)
    H, T = 2, 3
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T,
                    'bptt': bptt,
                    'cem_steps': 5,
                    'num_samples': 15,
                    'topk': 4,
                    'num_perturbed': 6,
                    'perturb_std': 0.5,
                    'var_scale': 1.0,
                    'temperature': 1e-3,
                    'all_steps': True,
                }
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})

    # Per-step costs: the last step is the usual goal cost, the expert is 0.
    plan_costs = search_loss.__globals__['plan_costs']
    plans = torch.randn(4, 5, T, 2)
    plans[:, 0] = actions[:, H - 1 : H - 1 + T]
    emb = batch['emb']
    per_step = plan_costs(model, batch, emb, cfg, plans, all_steps=True)
    assert per_step.shape == (4, 5, T)
    last = plan_costs(model, batch, emb, cfg, plans)
    torch.testing.assert_close(per_step[..., -1], last)
    assert per_step[:, 0].abs().max() < 1e-5

    loss = search_loss(model, batch, emb, cfg)
    assert loss < 0.5 * math.log(1 + 6 + 5 * 4)
    loss.backward()
    assert model.action_encoder.weight.grad.abs().sum() > 0


def test_detached_rollout_all_grad():
    from stable_worldmodel.wm.lewm.detached_rollout import detached_rollout

    torch.manual_seed(0)
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    initial = torch.randn(2, 2, 2)
    actions = torch.randn(2, 1 + 4, 2)
    preds = detached_rollout(model, initial, actions, 4, 2, all_grad=True)
    plain = detached_rollout(model, initial, actions, 4, 2)
    torch.testing.assert_close(preds, plain)
    # Default: an early step has no graph; all_grad: it does.
    plain[:, 1].sum().backward()
    assert model.action_encoder.weight.grad.abs().sum() == 0
    preds[:, 1].sum().backward()
    assert model.action_encoder.weight.grad.abs().sum() > 0


def test_uint8_pixels_match_float_preprocessing():
    pytest.importorskip('stable_pretraining')
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_uint8_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Dataset frames arrive as uint8 (T, C, H, W).
    frames = torch.randint(0, 256, (2, 3, 3, 16, 16), dtype=torch.uint8)

    def run(uint8):
        transform = module.get_img_preprocessor('pixels', 'pixels', 16, uint8)
        samples = [transform({'pixels': f.clone()})['pixels'] for f in frames]
        return module.normalize_pixels({'pixels': torch.stack(samples)})

    fast, slow = run(True)['pixels'], run(False)['pixels']
    assert fast.dtype == slow.dtype == torch.float32
    torch.testing.assert_close(fast, slow)
