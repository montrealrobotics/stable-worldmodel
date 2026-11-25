import importlib.util
import math
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pose_probe', ROOT / 'scripts/plan/pose_probe.py')
pose_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose_probe)


def save_probe(tmp_path, D=8, hidden=16):
    torch.manual_seed(0)
    net = pose_probe.build_mlp(D, hidden, 6)
    path = tmp_path / 'probe.pt'
    torch.save({'state_dict': net.state_dict(), 'mu': torch.zeros(D), 'sd': torch.ones(D),
                'in_dim': D, 'hidden': hidden, 'out_dim': 6}, path)
    return path, net


def test_cost_shape_and_zero_at_goal(tmp_path):
    path, _ = save_probe(tmp_path)
    cost = pose_probe.PoseProbeCost(str(path))
    B, S, T, D = 2, 5, 3, 8
    goal = torch.randn(B, 1, D)
    pred = torch.randn(B, S, T, D)
    pred[:, 0, -1] = goal[:, 0]  # candidate 0 ends exactly at the goal
    out = cost({'predicted_emb': pred, 'goal_emb': goal})
    assert out.shape == (B, S)
    assert torch.allclose(out[:, 0], torch.zeros(B), atol=1e-5)
    assert (out[:, 1:] > 0).all()


def test_cost_uses_only_last_step(tmp_path):
    path, _ = save_probe(tmp_path)
    cost = pose_probe.PoseProbeCost(str(path))
    goal = torch.randn(1, 1, 8)
    pred = torch.randn(1, 4, 3, 8)
    a = cost({'predicted_emb': pred, 'goal_emb': goal})
    pred[:, :, :-1] = torch.randn_like(pred[:, :, :-1])
    assert torch.allclose(a, cost({'predicted_emb': pred, 'goal_emb': goal}))


def test_pose_cost_tolerance_units():
    z = torch.zeros(1)
    goal = {'txy': torch.zeros(1, 2), 'yaw': z, 'exy': torch.zeros(1, 2)}
    at_tol = {'txy': torch.tensor([[20.0, 0.0]]), 'yaw': torch.tensor([0.15]), 'exy': torch.tensor([[0.0, 20.0]])}
    assert torch.allclose(pose_probe.pose_cost(at_tol, goal), torch.tensor([3.0]), atol=1e-5)
    # yaw wraps: +pi and -pi are the same orientation
    a = {'txy': torch.zeros(1, 2), 'yaw': torch.tensor([math.pi - 0.01]), 'exy': torch.zeros(1, 2)}
    b = {'txy': torch.zeros(1, 2), 'yaw': torch.tensor([-math.pi + 0.01]), 'exy': torch.zeros(1, 2)}
    expected = (2 * math.sin(0.01) / (2 * math.sin(0.075))) ** 2  # 0.02 rad apart, not 2pi - 0.02
    assert abs(pose_probe.pose_cost(a, b).item() - expected) < 1e-4


def test_probe_decodes_scaled_pose(tmp_path):
    path, net = save_probe(tmp_path)
    probe = pose_probe.PoseProbe.load(path)
    z = torch.randn(3, 8)
    raw = net(z)
    out = probe(z)
    assert torch.allclose(out['txy'], 100 * raw[:, :2], atol=1e-5)
    assert torch.allclose(out['yaw'], torch.atan2(raw[:, 2], raw[:, 3]), atol=1e-6)


def test_hydra_config_instantiates(tmp_path, monkeypatch):
    path, _ = save_probe(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / 'scripts/plan'))
    cfg = OmegaConf.load(ROOT / 'scripts/plan/config/objective/pose_probe.yaml')
    cfg.probe_path = str(path)
    obj = hydra.utils.instantiate(cfg)
    assert type(obj).__name__ == 'PoseProbeCost'
