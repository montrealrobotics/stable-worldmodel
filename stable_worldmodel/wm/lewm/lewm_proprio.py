"""LeWM over images plus robot proprioception (joint angles, end-effector pose)."""

import torch
from einops import rearrange

from .lewm import LeWM
from .lewm_res import LeWMRes


class LeWMProprio(LeWM):
    """Fuse the image CLS embedding with an embedding of proprio columns.

    ``proprio_columns`` are concatenated in order along the feature axis,
    embedded by ``proprio_encoder``, concatenated with the CLS token and mapped
    by ``projector`` to the latent. Prediction, rollout and planning costs are
    shared with LeWM. Planning goals need the same columns as ``goal_<column>``.
    """

    def __init__(self, *args, proprio_encoder, proprio_columns, **kwargs):
        super().__init__(*args, **kwargs)
        if not proprio_columns:
            raise ValueError('LeWMProprio needs at least one proprio column')
        self.proprio_encoder = proprio_encoder
        self.proprio_columns = list(proprio_columns)

    def encode(self, info):
        """Encode pixels and proprio columns into one latent per frame."""
        dtype = next(self.encoder.parameters()).dtype
        pixels = self.observation(info).to(dtype)
        proprio = torch.cat(
            [info[key] for key in self.proprio_columns], dim=-1
        ).to(dtype)
        b = pixels.size(0)
        output = self.encoder(
            rearrange(pixels, 'b t ... -> (b t) ...'),
            interpolate_pos_encoding=True,
        )
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        proprio_emb = self.proprio_encoder(
            rearrange(proprio, 'b t d -> (b t) d')
        )
        emb = self.projector(torch.cat([pixels_emb, proprio_emb], dim=-1))
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info


class LeWMProprioRes(LeWMRes, LeWMProprio):
    """Proprio LeWM predicting residual changes in the embedding space."""


__all__ = ['LeWMProprio', 'LeWMProprioRes']
