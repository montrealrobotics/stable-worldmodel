"""SIGReg weight schedules decay from weight to final_weight over training."""

import pytest
from omegaconf import OmegaConf

pytest.importorskip('stable_pretraining')
from scripts.train.lewm import sigreg_weight


def config(**kwargs):
    return OmegaConf.create({'weight': 0.1, 'final_weight': 0.01, **kwargs})


def test_no_schedule_keeps_weight_constant():
    sigreg = OmegaConf.create({'weight': 0.09})
    assert sigreg_weight(sigreg, 0.0) == sigreg_weight(sigreg, 1.0) == 0.09


@pytest.mark.parametrize('schedule', ['linear', 'cosine'])
def test_schedule_moves_from_weight_to_final_weight(schedule):
    sigreg = config(schedule=schedule)
    assert sigreg_weight(sigreg, 0.0) == pytest.approx(0.1)
    assert sigreg_weight(sigreg, 0.5) == pytest.approx(0.055)
    assert sigreg_weight(sigreg, 1.0) == pytest.approx(0.01)
    assert sigreg_weight(sigreg, 1.5) == pytest.approx(0.01)
    weights = [sigreg_weight(sigreg, t / 10) for t in range(11)]
    assert weights == sorted(weights, reverse=True)


def test_cosine_decays_slowly_at_the_ends():
    sigreg = config(schedule='cosine')
    assert sigreg_weight(sigreg, 0.1) > sigreg_weight(
        config(schedule='linear'), 0.1
    )


def test_schedule_holds_until_start():
    sigreg = config(schedule='linear', schedule_start=0.5)
    assert sigreg_weight(sigreg, 0.25) == pytest.approx(0.1)
    assert sigreg_weight(sigreg, 0.75) == pytest.approx(0.055)
    assert sigreg_weight(sigreg, 1.0) == pytest.approx(0.01)


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match='unknown SIGReg schedule'):
        sigreg_weight(config(schedule='step'), 0.5)
