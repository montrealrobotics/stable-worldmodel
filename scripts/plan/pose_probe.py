"""Oracle-style planning cost: read the T and pusher pose from embeddings with a probe.

PoseProbeCost replaces the squared embedding distance (GoalMSE) in the planner. A small MLP
probe (fit by scripts/plan/fit_pose_probe.py on one checkpoint's embeddings) reads T x, y, yaw
and EEF x, y from the last predicted embedding of every candidate and from the goal embedding;
the cost is the pose error in units of the success tolerances:

    (|dT xy| / 20 mm)^2 + (chord(dyaw) / chord(0.15 rad))^2 + eef_weight * (|dEEF xy| / 20 mm)^2

so a plan that just meets every tolerance costs about 3. If planning succeeds much more often
with this cost on the same world model, the embedding distance is the bottleneck, not the
predictor. Used from scripts/plan/eval_wm.py:

    objective=pose_probe objective.probe_path=/path/to/<label>.pt

Stays importable on the eval venv (Python 3.10): eval_wm.py runs from scripts/plan, so Hydra
finds this module as `pose_probe`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

POSE_SCALE_MM = 100.0  # probe targets: T x, y and EEF x, y in units of 100 mm, yaw as (sin, cos)


def build_mlp(in_dim, hidden, out_dim):
    """The probe architecture (same as rotation_probe.MLP)."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class PoseProbe(nn.Module):
    """Embedding (..., D) -> T xy (mm), yaw (rad), EEF xy (mm)."""

    def __init__(self, in_dim, hidden, mu, sd, out_dim=6):
        super().__init__()
        self.net = build_mlp(in_dim, hidden, out_dim)
        self.register_buffer('mu', torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer('sd', torch.as_tensor(sd, dtype=torch.float32))

    @classmethod
    def load(cls, path):
        ck = torch.load(path, map_location='cpu')
        probe = cls(ck['in_dim'], ck['hidden'], ck['mu'], ck['sd'], ck.get('out_dim', 6))
        probe.net.load_state_dict(ck['state_dict'])
        return probe.eval()

    def forward(self, z):
        out = self.net((z.float() - self.mu) / self.sd)
        return {
            'txy': POSE_SCALE_MM * out[..., 0:2],
            'yaw': torch.atan2(out[..., 2], out[..., 3]),
            'exy': POSE_SCALE_MM * out[..., 4:6],
        }


def pose_cost(a, b, txy_tol_mm=20.0, yaw_tol_rad=0.15, eef_tol_mm=20.0, eef_weight=1.0):
    """Pose error between two decoded poses, in units of the success tolerances (squared)."""
    dyaw = a['yaw'] - b['yaw']
    chord = 2 * torch.sin(torch.remainder(dyaw + math.pi, 2 * math.pi).sub(math.pi).abs() / 2)
    return (
        ((a['txy'] - b['txy']).norm(dim=-1) / txy_tol_mm) ** 2
        + (chord / (2 * math.sin(yaw_tol_rad / 2))) ** 2
        + eef_weight * ((a['exy'] - b['exy']).norm(dim=-1) / eef_tol_mm) ** 2
    )


class PoseProbeCost(nn.Module):
    """Planning objective: probed pose error between the last predicted step and the goal.

    Reads ``predicted_emb`` (B, S, T, D) and ``goal_emb`` (B, T_goal, D); returns (B, S).
    """

    def __init__(
        self,
        probe_path: str,
        txy_tol_mm: float = 20.0,
        yaw_tol_rad: float = 0.15,
        eef_tol_mm: float = 20.0,
        eef_weight: float = 1.0,
        pred_key: str = 'predicted_emb',
        goal_key: str = 'goal_emb',
    ) -> None:
        super().__init__()
        self.probe = PoseProbe.load(probe_path)
        self.tols = dict(txy_tol_mm=txy_tol_mm, yaw_tol_rad=yaw_tol_rad, eef_tol_mm=eef_tol_mm, eef_weight=eef_weight)
        self.pred_key, self.goal_key = pred_key, goal_key

    @torch.no_grad()
    def forward(self, info_dict: dict) -> torch.Tensor:
        pred = info_dict[self.pred_key][:, :, -1]  # (B, S, D)
        goal = info_dict[self.goal_key][:, -1]  # (B, D)
        if self.probe.mu.device != pred.device:
            self.probe.to(pred.device)
        p, g = self.probe(pred), self.probe(goal[:, None])
        return pose_cost(p, g, **self.tols).to(pred.dtype)
