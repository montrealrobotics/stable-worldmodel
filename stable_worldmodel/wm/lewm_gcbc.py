"""Gaussian behavior-cloning prior in a frozen LeWM latent space."""

import torch
from torch import nn
from torch.distributions import Independent, Normal

from stable_worldmodel.planning.evaluator import flat_goal_encode

from .lewm.module import MLP


class LeWMGCBC(nn.Module):
    """Predict one block of normalized actions from [z, goal].

    ``action_low``/``action_high`` are normalized per-environment-action
    bounds. ``normalization`` stores the same column statistics used to train
    LeWM. Checkpoints include the frozen world model and these statistics.
    ``get_action`` uses environment steps; ``plan`` uses world-model blocks.
    """

    def __init__(
        self,
        world_model,
        latent_dim: int,
        action_dim: int,
        action_block: int,
        action_low,
        action_high,
        normalization=None,
        hidden_dim: int = 256,
    ):
        super().__init__()
        if min(latent_dim, action_dim, action_block, hidden_dim) < 1:
            raise ValueError('Policy dimensions must be positive')
        if world_model.action_encoder.input_dim != action_dim * action_block:
            raise ValueError('Action block does not match LeWM action encoder')
        self.world_model = world_model.eval().requires_grad_(False)
        self.action_dim = action_dim
        self.action_block = action_block
        self.normalization = normalization or {}
        self.policy = MLP(
            2 * latent_dim, hidden_dim, 2 * action_dim * action_block
        )
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if (
            low.shape != (action_dim,)
            or high.shape != low.shape
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not (low < high).all()
        ):
            raise ValueError(
                'Provide finite per-action bounds with low < high'
            )
        self.register_buffer('action_low', low.repeat(action_block))
        self.register_buffer('action_high', high.repeat(action_block))

    def train(self, mode=True):
        super().train(mode)
        self.world_model.eval()  # Keep dropout and batch statistics frozen.
        return self

    def distribution(self, z, goal):
        """Return a diagonal Gaussian over a flattened action block."""
        features = torch.cat([z, goal], dim=-1)
        features = features.to(next(self.policy.parameters()).dtype)
        mean, log_std = self.policy(features).chunk(2, dim=-1)
        return Independent(Normal(mean, log_std.clamp(-5, 2).exp()), 1)

    @torch.no_grad()
    def _encode(self, info):
        inputs = {
            k: v
            for k, v in info.items()
            if k not in ('action', 'action_history')
        }
        emb = self.world_model.encode(inputs)['emb']
        expanded = {
            k: v.unsqueeze(1) for k, v in info.items() if torch.is_tensor(v)
        }
        goal = flat_goal_encode(self.world_model, expanded)[:, -1]
        if emb.ndim != 3:
            raise ValueError('LeWMGCBC requires flat LeWM embeddings')
        return emb, goal

    def forward(self, info):
        """Encode the current observation and goal without LeWM gradients."""
        emb, goal = self._encode(info)
        return self.distribution(emb[:, -1], goal)

    def _action(self, distribution, sample=False):
        action = distribution.sample() if sample else distribution.mean
        return action.clamp(self.action_low, self.action_high)

    @torch.no_grad()
    def get_action(self, info, horizon=1, sample=False):
        """Return up to one action block for FeedForwardPolicy execution."""
        if not 1 <= horizon <= self.action_block:
            raise ValueError('Execution horizon must fit one action block')
        actions = self._action(self(info), sample).reshape(
            -1, self.action_block, self.action_dim
        )
        return actions[:, 0] if horizon == 1 else actions[:, :horizon]

    @torch.no_grad()
    def plan(self, info, horizon=1, prefix_actions=None, sample=False):
        """Autoregressive block proposals after an optional warm-start prefix.

        Reuse LeWM.rollout so prediction history, executed action history,
        and rollout normalization match the planner exactly.
        """
        if horizon < 1:
            raise ValueError('Planning horizon must be positive')
        device = next(self.policy.parameters()).device
        info = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in info.items()
        }
        emb, goal = self._encode(info)
        rollout_info = {
            k: v.unsqueeze(1) for k, v in info.items() if torch.is_tensor(v)
        }
        rollout_info['emb'] = emb.unsqueeze(1)
        actions = emb.new_empty(
            emb.size(0), 0, self.action_dim * self.action_block
        )
        if prefix_actions is not None:
            actions = prefix_actions.to(actions)
            if actions.ndim != 3 or actions.shape[::2] != (
                emb.size(0),
                self.action_dim * self.action_block,
            ):
                raise ValueError('Invalid warm-start action block dimensions')
        prefix_len = actions.size(1)
        for _ in range(horizon):
            z = emb[:, -1]
            if actions.size(1):
                # shortcut: replay the prefix, use incremental rollout if long
                # planning horizons make the quadratic predictor work costly.
                z = self.world_model.rollout(
                    dict(rollout_info), actions.unsqueeze(1)
                )['predicted_emb'][:, 0, -1]
            action = self._action(self.distribution(z, goal), sample)
            actions = torch.cat([actions, action.unsqueeze(1)], dim=1)
        return actions[:, prefix_len:]
