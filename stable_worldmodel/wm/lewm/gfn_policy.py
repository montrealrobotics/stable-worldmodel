import torch
import torch.nn.functional as F
from torch import distributions as D
from torch import nn


class PlanGFlowNet(nn.Module):
    """GFlowNet sampler over action plans, conditioned on (z_0, z_goal).

    Builds a plan block by block: a GRU carries the prefix and each block is
    drawn from a diagonal Gaussian mixture. Every prefix has a single parent,
    so log pi(plan) is the sum of the per-block log-probs and trajectory
    balance reads (log Z(z_0, z_goal) + log pi(plan) - log R(plan))^2.
    Trained to sample plans in proportion to R = exp(-cost / tau) under the
    world model, it explores that model's energy landscape.
    """

    def __init__(
        self,
        emb_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        num_components: int = 4,
        min_std: float = 0.05,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_components = num_components
        self.min_std = min_std
        self.cond = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(action_dim, hidden_dim)
        self.head = nn.Linear(
            hidden_dim, num_components * (1 + 2 * action_dim)
        )
        self.log_z_head = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _block_dist(self, h, std_scale=1.0):
        k, a = self.num_components, self.action_dim
        logits, mean, raw_std = self.head(h).split([k, k * a, k * a], dim=-1)
        std = (self.min_std + F.softplus(raw_std)) * std_scale
        comp = D.Independent(
            D.Normal(mean.unflatten(-1, (k, a)), std.unflatten(-1, (k, a))), 1
        )
        return D.MixtureSameFamily(D.Categorical(logits=logits), comp)

    def _init(self, z0, z_goal, n):
        """(B, D) conditions -> GRU state (B * n, hidden)."""
        h = self.cond(torch.cat([z0, z_goal], -1))
        return h.repeat_interleave(n, dim=0)

    def log_z(self, z0, z_goal):
        """(B, D) -> (B,) learned log partition function."""
        return self.log_z_head(torch.cat([z0, z_goal], -1)).squeeze(-1)

    def log_prob(self, z0, z_goal, plans):
        """plans: (B, S, T, A) -> (B, S) log pi(plan | z_0, z_goal)."""
        B, S, T, _ = plans.shape
        h = self._init(z0, z_goal, S)
        flat = plans.flatten(0, 1)
        prev = flat.new_zeros(B * S, self.action_dim)
        total = flat.new_zeros(B * S)
        for t in range(T):
            h = self.gru(prev, h)
            total = total + self._block_dist(h).log_prob(flat[:, t])
            prev = flat[:, t]
        return total.unflatten(0, (B, S))

    @torch.no_grad()
    def sample(self, z0, z_goal, n, horizon=None, std_scale=1.0):
        """(B, D) conditions -> (B, n, T, A) plans.

        ``std_scale`` > 1 widens every component for off-policy exploration;
        trajectory balance stays valid because log_prob scores the samples
        under the unscaled policy.
        """
        T = horizon or self.horizon
        h = self._init(z0, z_goal, n)
        prev = h.new_zeros(h.size(0), self.action_dim)
        blocks = []
        for _ in range(T):
            h = self.gru(prev, h)
            prev = self._block_dist(h, std_scale).sample()
            blocks.append(prev)
        return torch.stack(blocks, 1).unflatten(0, (z0.size(0), n))


__all__ = ['PlanGFlowNet']
