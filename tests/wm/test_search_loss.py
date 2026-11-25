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


@pytest.mark.parametrize('bptt', [True, False])
def test_all_steps_infonce(search_loss, bptt):
    torch.manual_seed(0)
    H, T = 2, 3
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T,
                    'bptt': bptt,
                    'cem_steps': 5,
                    'num_samples': 15,
                    'topk': 4,
                    'num_perturbed': 6,
                    'perturb_std': 0.5,
                    'var_scale': 1.0,
                    'temperature': 1e-3,
                    'all_steps': True,
                }
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})

    # Per-step costs: the last step is the usual goal cost, the expert is 0.
    plan_costs = search_loss.__globals__['plan_costs']
    plans = torch.randn(4, 5, T, 2)
    plans[:, 0] = actions[:, H - 1 : H - 1 + T]
    emb = batch['emb']
    per_step = plan_costs(model, batch, emb, cfg, plans, all_steps=True)
    assert per_step.shape == (4, 5, T)
    last = plan_costs(model, batch, emb, cfg, plans)
    torch.testing.assert_close(per_step[..., -1], last)
    assert per_step[:, 0].abs().max() < 1e-5

    loss = search_loss(model, batch, emb, cfg)
    assert loss < 0.5 * math.log(1 + 6 + 5 * 4)
    loss.backward()
    assert model.action_encoder.weight.grad.abs().sum() > 0


def test_detached_rollout_all_grad():
    from stable_worldmodel.wm.lewm.detached_rollout import detached_rollout

    torch.manual_seed(0)
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    initial = torch.randn(2, 2, 2)
    actions = torch.randn(2, 1 + 4, 2)
    preds = detached_rollout(model, initial, actions, 4, 2, all_grad=True)
    plain = detached_rollout(model, initial, actions, 4, 2)
    torch.testing.assert_close(preds, plain)
    # Default: an early step has no graph; all_grad: it does.
    plain[:, 1].sum().backward()
    assert model.action_encoder.weight.grad.abs().sum() == 0
    preds[:, 1].sum().backward()
    assert model.action_encoder.weight.grad.abs().sum() > 0


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


def test_search_on_main_clips(search_loss):
    """main_clips: MSE + SIGReg on the trimmed clips, InfoNCE on the first
    search.batch_size clips of the same batch with all their frames."""
    from types import SimpleNamespace

    g = search_loss.__globals__
    H, T, B, N = 1, 3, 5, 3
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H, 'num_preds': 1},
            'gradient_diagnostics_interval': 0,
            'loss': {
                'sigreg': {'weight': 0.5},
                'search': {
                    'weight': 0.1,
                    'horizon': T,
                    'bptt': True,
                    'cem_steps': 2,
                    'num_samples': 8,
                    'topk': 3,
                    'num_perturbed': 4,
                    'perturb_std': 0.5,
                    'var_scale': 1.0,
                    'temperature': 1.0,
                    'batch_size': N,
                    'main_clips': True,
                },
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    sigreg = lambda x: x.pow(2).mean()  # noqa: E731
    module = SimpleNamespace(
        model=model, sigreg=sigreg, log_dict=lambda *a, **k: None, global_step=1
    )
    torch.manual_seed(0)
    state = torch.randn(B, H + T, 2)
    action = torch.randn(B, H + T, 2)

    torch.manual_seed(1)
    out = g['lejepa_forward'](
        module, {'state': state.clone(), 'action': action.clone()}, 'fit', cfg
    )

    # MSE + SIGReg: base objective on the first H + 1 frames of every clip.
    short = model.encode({'state': state[:, : H + 1], 'action': action[:, : H + 1]})
    emb = short['emb']
    pred = model.predict({'emb': emb[:, :H], 'act_emb': short['act_emb'][:, :H]})
    torch.testing.assert_close(
        out['pred_loss'], (pred['preds'] - emb[:, 1 : 1 + H]).pow(2).mean()
    )
    torch.testing.assert_close(out['sigreg_loss'], sigreg(emb.transpose(0, 1)))
    assert out['emb'].shape[:2] == (B, H + 1)

    # InfoNCE: first N clips of the same batch, all frames.
    torch.manual_seed(1)
    sub = model.encode({'state': state[:N], 'action': action[:N]})
    expected = search_loss(model, sub, sub['emb'], cfg)
    torch.testing.assert_close(out['search_loss'], expected)
    torch.testing.assert_close(
        out['loss'],
        out['pred_loss'] + 0.5 * out['sigreg_loss'] + 0.1 * expected,
    )

    # Validation: no search loss.
    val = g['lejepa_forward'](
        module, {'state': state.clone(), 'action': action.clone()}, 'validate', cfg
    )
    assert 'search_loss' not in val


@pytest.mark.parametrize('all_frames', [None, False, True])
def test_sigreg_on_prediction_frames(search_loss, all_frames):
    """Post-training clips hold history + horizon frames; SIGReg sees only the
    history + num_preds frames the prediction loss uses, or every loaded frame
    with sigreg.all_frames."""
    from types import SimpleNamespace

    g = search_loss.__globals__
    H, T, B = 1, 3, 4
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H, 'num_preds': 1},
            'gradient_diagnostics_interval': 0,
            'loss': {'sigreg': {'weight': 0.5}},
        }
    )
    if all_frames is not None:  # None: key absent (older configs)
        cfg.loss.sigreg.all_frames = all_frames
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    sigreg = lambda x: x.pow(2).mean()  # noqa: E731
    module = SimpleNamespace(
        model=model, sigreg=sigreg, log_dict=lambda *a, **k: None, global_step=1
    )
    state = torch.randn(B, H + T, 2)
    action = torch.randn(B, H + T, 2)
    out = g['lejepa_forward'](module, {'state': state, 'action': action}, 'fit', cfg)
    emb = out['emb']
    assert emb.shape[1] == H + T
    frames = H + T if all_frames else H + 1
    torch.testing.assert_close(
        out['sigreg_loss'], sigreg(emb[:, :frames].transpose(0, 1))
    )


@pytest.mark.parametrize('P', [2, 0])
def test_in_batch_negatives_without_cem(search_loss, P):
    """in_batch adds other clips' expert plans (never the clip's own, no
    repeats); cem_steps 0 skips CEM, so the plans are expert + P + I.
    P = 0: the in-batch plans are the only negatives."""
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_inbatch_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    H, T, B, I = 2, 3, 6, 4
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T,
                    'bptt': True,
                    'cem_steps': 0,
                    'num_samples': 8,
                    'topk': 2,
                    'num_perturbed': P,
                    'perturb_std': 0.5,
                    'var_scale': 1.0,
                    'temperature': 1.0,
                    'in_batch': I,
                }
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
    )
    actions = torch.randn(B, H + T, 2)
    state = torch.cat([torch.zeros(B, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})

    plans = module.search_negatives(model, batch, batch['emb'], cfg)
    assert plans.shape == (B, 1 + P + I, T, 2)
    experts = actions[:, H - 1 : H - 1 + T]
    for i in range(B):
        # Each in-batch plan is exactly one other clip's expert plan.
        match = (plans[i, 1 + P :, None] == experts[None]).flatten(2).all(-1)
        owners = match.float().argmax(1)
        assert match.any(1).all()
        assert i not in owners.tolist()
        assert len(set(owners.tolist())) == I

    loss = search_loss(model, batch, batch['emb'], cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.action_encoder.weight.grad.abs().sum() > 0

    cfg.loss.search.in_batch = B  # only B - 1 other clips exist
    with pytest.raises(ValueError, match='in_batch'):
        module.search_negatives(model, batch, batch['emb'], cfg)


def test_search_anchor_loss():
    """The expert rollout is anchored to the clip's real frames: zero for exact
    dynamics, positive and trainable otherwise; targets are detached."""
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_anchor_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    H, T, B = 2, 3, 4
    cfg = OmegaConf.create(
        {'wm': {'history_size': H}, 'loss': {'search': {'horizon': T, 'anchor_weight': 0.5}}}
    )
    model = LeWMState(encoder=_identity(2), predictor=_Predictor(), action_encoder=_identity(2))
    actions = torch.randn(B, H + T, 2)
    state = torch.cat([torch.zeros(B, 1, 2), actions[:, :-1].cumsum(1)], dim=1)

    batch = model.encode({'state': state, 'action': actions})
    assert module.search_anchor_loss(model, batch, batch['emb'], cfg) < 1e-10

    with torch.no_grad():
        model.action_encoder.weight.mul_(1.5)  # wrong dynamics: rollouts overshoot
    batch = model.encode({'state': state, 'action': actions})
    loss = module.search_anchor_loss(model, batch, batch['emb'], cfg)
    assert loss > 0
    loss.backward()
    assert model.action_encoder.weight.grad.abs().sum() > 0

    # Detached targets: with H context frames out of the graph, the encoder gets no gradient.
    model.zero_grad()
    batch = model.encode({'state': state, 'action': actions})
    emb = torch.cat([batch['emb'][:, :H].detach(), batch['emb'][:, H:]], 1)  # targets keep their graph
    module.search_anchor_loss(model, batch, emb, cfg).backward()
    assert model.encoder.weight.grad is None or model.encoder.weight.grad.abs().sum() == 0

    out = {'loss': torch.zeros(())}
    module.add_search_anchor(out, model, batch, batch['emb'], cfg)
    torch.testing.assert_close(out['loss'], 0.5 * out['search_anchor_loss'])
    cfg.loss.search.anchor_weight = 0
    out = {'loss': torch.zeros(())}
    module.add_search_anchor(out, model, batch, batch['emb'], cfg)
    assert 'search_anchor_loss' not in out


@pytest.mark.parametrize('detach_goal', [True, False])
def test_detach_goal_flag(search_loss, detach_goal):
    """detach_goal false lets the InfoNCE gradient reach the encoder through the goal frame."""
    torch.manual_seed(0)
    H, T = 2, 3
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T, 'bptt': True, 'cem_steps': 0, 'num_samples': 8, 'topk': 2,
                    'num_perturbed': 4, 'perturb_std': 0.5, 'var_scale': 1.0, 'temperature': 1.0,
                    'detach_goal': detach_goal,
                }
            },
        }
    )
    model = LeWMState(encoder=_identity(2), predictor=_Predictor(), action_encoder=_identity(2))
    with torch.no_grad():
        model.action_encoder.weight.mul_(1.5)  # imperfect dynamics: the goal term has a gradient
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})
    # Context frames out of the graph: any encoder gradient comes through the goal.
    emb = torch.cat([batch['emb'][:, :H].detach(), batch['emb'][:, H:]], 1)
    search_loss(model, batch, emb, cfg).backward()
    grad = model.encoder.weight.grad
    reached = grad is not None and grad.abs().sum() > 0
    assert reached == (not detach_goal)


@pytest.mark.parametrize('anchor_detach', [True, False])
def test_anchor_detach_flag(anchor_detach):
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_anchor_flag_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    H, T = 2, 3
    cfg = OmegaConf.create(
        {'wm': {'history_size': H}, 'loss': {'search': {'horizon': T, 'anchor_detach': anchor_detach}}}
    )
    model = LeWMState(encoder=_identity(2), predictor=_Predictor(), action_encoder=_identity(2))
    with torch.no_grad():
        model.action_encoder.weight.mul_(1.5)
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})
    emb = torch.cat([batch['emb'][:, :H].detach(), batch['emb'][:, H:]], 1)
    module.search_anchor_loss(model, batch, emb, cfg).backward()
    grad = model.encoder.weight.grad
    assert (grad is not None and grad.abs().sum() > 0) == (not anchor_detach)


def test_detach_goal_same_costs(search_loss):
    """The undetached cost path gives exactly GoalMSE's values; only gradients differ."""
    torch.manual_seed(0)
    H, T = 2, 3
    plan_costs = search_loss.__globals__['plan_costs']
    cfg = OmegaConf.create({'wm': {'history_size': H}, 'loss': {'search': {'horizon': T, 'bptt': True}}})
    model = LeWMState(encoder=_identity(2), predictor=_Predictor(), action_encoder=_identity(2))
    actions = torch.randn(4, H + T, 2)
    state = torch.cat([torch.zeros(4, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})
    plans = torch.randn(4, 6, T, 2)
    fixed = plan_costs(model, batch, batch['emb'], cfg, plans)
    cfg.loss.search.detach_goal = False
    torch.testing.assert_close(plan_costs(model, batch, batch['emb'], cfg, plans), fixed)


@pytest.mark.parametrize('bn_eval', [False, True])
def test_bn_eval_flag(bn_eval):
    """bn_eval: BatchNorm in the rollout uses running statistics inside the search
    loss (costs equal an eval-mode model's), leaves them unchanged, still trains
    the affine weights, and returns to train mode. Off: the negatives' rollouts
    update the running statistics."""
    pytest.importorskip('stable_pretraining')
    script = Path(__file__).resolve().parents[2] / 'scripts/train/lewm.py'
    spec = importlib.util.spec_from_file_location('lewm_bn_eval_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    H, T, B = 2, 3, 6
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': H},
            'loss': {
                'search': {
                    'horizon': T,
                    'bptt': True,
                    'cem_steps': 2,
                    'num_samples': 8,
                    'topk': 2,
                    'num_perturbed': 3,
                    'perturb_std': 0.5,
                    'var_scale': 1.0,
                    'temperature': 1.0,
                    'bn_eval': bn_eval,
                    'anchor_weight': 1.0,
                }
            },
        }
    )
    model = LeWMState(
        encoder=_identity(2),
        predictor=_Predictor(),
        action_encoder=_identity(2),
        pred_proj=nn.BatchNorm1d(2),
    )
    bn = model.pred_proj
    bn.running_mean.fill_(0.3)
    bn.running_var.fill_(2.0)
    model.train()
    actions = torch.randn(B, H + T, 2)
    state = torch.cat([torch.zeros(B, 1, 2), actions[:, :-1].cumsum(1)], dim=1)
    batch = model.encode({'state': state, 'action': actions})
    stats = (bn.running_mean.clone(), bn.running_var.clone())

    torch.manual_seed(1)
    loss = module.search_loss(model, batch, batch['emb'], cfg)
    anchor = module.search_anchor_loss(model, batch, batch['emb'], cfg)
    (loss + anchor).backward()
    assert bn.training and model.training
    unchanged = torch.equal(bn.running_mean, stats[0]) and torch.equal(
        bn.running_var, stats[1]
    )
    assert unchanged == bn_eval
    assert bn.weight.grad.abs().sum() > 0

    if bn_eval:  # same costs as a fully eval-mode model, i.e. as in planning
        plans = torch.randn(B, 5, T, 2)
        with module.batchnorm_eval(model):
            costs = module.plan_costs(model, batch, batch['emb'], cfg, plans)
        model.eval()
        ref = module.plan_costs(model, batch, batch['emb'], cfg, plans)
        model.train()
        torch.testing.assert_close(costs, ref)
