import torch
from einops import rearrange

from .lewm import LeWM


class LeWMOneShotRes(LeWM):
    """LeWM predicting every future frame directly from the context frames.

    The action encoder must be cumulative over time (e.g.
    ``RecurrentActionEncoder``) and runs from the first context frame:
    act_emb[:, j] summarizes actions 0..j, where action j leaves frame j.
    For target k (1..K), context frame i < H-1 is conditioned on
    act_emb[:, i] as in LeWM, and the current frame z_t on act_emb[:, H-2+k],
    which covers every action up to t+k-1. Each target is predicted in one
    shot as z_{t+k} = z_t + f(context, actions); predictions are never fed
    back, so errors don't compound. With H = 1 it is z_t + f(z_t, h_k).
    """

    one_shot = True

    def predict(self, info):
        """
        emb: (B, H, D) context frames, the last is the current one
        act_emb: (B, H - 1 + K, A_emb) cumulative action embeddings
        preds: (B, K, D), one per future frame
        """
        emb, act_emb = info['emb'], info['act_emb']
        b, h = emb.shape[:2]
        k = act_emb.size(1) - (h - 1)
        if k < 1:
            raise ValueError(f'act_emb needs at least H={h} blocks')
        past = act_emb[:, None, : h - 1].expand(-1, k, -1, -1)
        cond = torch.cat([past, act_emb[:, h - 1 :, None]], dim=2)
        x = emb[:, None].expand(-1, k, -1, -1).reshape(b * k, h, -1)
        preds = self.predictor(x, cond.reshape(b * k, h, -1))[:, -1]
        preds = self.pred_proj(preds).reshape(b, k, -1)
        info['preds'] = emb[:, -1:] + preds
        return info

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int | None = None):
        """Predict all T future frames at once.
        pixels: (B, S, H, C, h, w) — H context frames
        action_sequence: (B, S, T, action_dim) — strictly-future candidates
        info['action_history']: (B, S, H - 1, action_dim) — executed action
            blocks between the context frames (required when H > 1)
        Returns ``info`` with ``predicted_emb`` of shape (B, S, H + T, D);
        the first H entries are the encoded context frames.
        """
        H = self.observation(info).size(2)
        B, S = action_sequence.shape[:2]
        act_past = info.get('action_history')
        if act_past is None:
            act_past = action_sequence.new_zeros(
                B, S, 0, action_sequence.size(-1)
            )
        assert act_past.size(2) == H - 1, (
            f'action_history must hold H-1={H - 1} executed blocks, '
            f'got {act_past.size(2)}'
        )
        info['action'] = torch.cat(
            [act_past, action_sequence[:, :, :1]], dim=2
        )

        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            initial_emb = _init['emb'].detach()
            info['emb'] = initial_emb.unsqueeze(1).expand(
                B, S, *initial_emb.shape[1:]
            )

        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        actions = torch.cat([act_past, action_sequence], dim=2)
        actions = rearrange(actions, 'b s ... -> (b s) ...')
        preds = self.predict(
            {'emb': emb_init, 'act_emb': self.action_encoder(actions)}
        )['preds']
        emb = torch.cat([emb_init, preds], dim=1)  # (BS, H + T, D)
        info['predicted_emb'] = rearrange(
            emb, '(b s) ... -> b s ...', b=B, s=S
        )
        return info


__all__ = ['LeWMOneShotRes']
