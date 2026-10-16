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
