import pytest
import torch

from stable_worldmodel.wm.loss import SIGReg, temporal_center


def test_temporal_center_zero_mean_per_window():
    z = torch.randn(3, 5, 7)
    r = temporal_center(z, window=2)
    for sl in (slice(0, 2), slice(2, 4), slice(4, 5)):
        torch.testing.assert_close(
            r[:, sl], z[:, sl] - z[:, sl].mean(1, keepdim=True)
        )


def test_temporal_center_removes_persistent_component():
    z = torch.randn(4, 4, 8)
    shift = torch.randn(4, 1, 8) * 10
    torch.testing.assert_close(
        temporal_center(z + shift, window=4), temporal_center(z, window=4)
    )


def test_temporal_center_rejects_trivial_window():
    with pytest.raises(ValueError):
        temporal_center(torch.randn(2, 4, 3), window=1)


def test_sigreg_on_residuals_backpropagates():
    z = torch.randn(16, 4, 8, requires_grad=True)
    loss = SIGReg(knots=5, num_proj=8)(temporal_center(z, 4).transpose(0, 1))
    loss.backward()
    assert torch.isfinite(loss) and z.grad.abs().sum() > 0
