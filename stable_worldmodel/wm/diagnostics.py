from collections.abc import Iterable

import torch


def encoder_gradient_diagnostics(
    pred_loss: torch.Tensor,
    sigreg_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    sigreg_weight: float,
    eps: float = 1e-8,
    search_loss: torch.Tensor | None = None,
    search_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compare unscaled loss gradients without changing parameter ``.grad``.

    Retain the forward graph for the usual backward pass. Unused parameters
    contribute zeros at their original positions in all gradient vectors.
    Reduce in FP32 outside autocast, including when training uses FP16/BF16.
    A zero norm gives a cosine similarity of zero. When ``search_loss`` is
    given, also compare its gradient against the prediction and SIGReg ones.
    """
    parameters = tuple(p for p in parameters if p.requires_grad)

    def gradients(loss):
        if not parameters or not loss.requires_grad:
            return (None,) * len(parameters)
        return torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

    # Use the original losses, before Lightning's backward loss scaling.
    losses = {'pred': pred_loss, 'sigreg': sigreg_loss}
    if search_loss is not None:
        losses['search'] = search_loss
    grads = {k: gradients(v) for k, v in losses.items()}
    names = list(grads)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    with (
        torch.no_grad(),
        torch.autocast(device_type=pred_loss.device.type, enabled=False),
    ):
        zero = pred_loss.new_zeros((), dtype=torch.float32)
        sq = {k: zero.clone() for k in names}
        dot = {pair: zero.clone() for pair in pairs}
        for per_param in zip(*grads.values()):
            g = {
                k: v.float() for k, v in zip(names, per_param) if v is not None
            }
            for k, v in g.items():
                sq[k] += v.square().sum()
            for a, b in pairs:
                if a in g and b in g:
                    dot[a, b] += (g[a] * g[b]).sum()

        norm = {k: v.sqrt() for k, v in sq.items()}

        def cosine(a, b):
            product = norm[a] * norm[b]
            denominator = torch.where(
                product > 0, product, torch.ones_like(product)
            )
            return (dot[a, b] / denominator).clamp(-1, 1)

        metrics = {
            'pred_grad_norm': norm['pred'],
            'sigreg_grad_norm': norm['sigreg'],
            'relative_grad_magnitude': (
                sigreg_weight * norm['sigreg'] / (norm['pred'] + eps)
            ),
            'grad_cosine_similarity': cosine('pred', 'sigreg'),
        }
        if search_loss is not None:
            metrics |= {
                'search_grad_norm': norm['search'],
                'search_relative_grad_magnitude': (
                    search_weight * norm['search'] / (norm['pred'] + eps)
                ),
                'search_pred_grad_cosine_similarity': cosine('pred', 'search'),
                'search_sigreg_grad_cosine_similarity': cosine(
                    'sigreg', 'search'
                ),
            }
        return metrics
