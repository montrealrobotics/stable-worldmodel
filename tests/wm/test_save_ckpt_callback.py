from types import SimpleNamespace

from torch import nn


def test_save_ckpt_callback_step_interval(monkeypatch):
    import scripts.train.lewm as lewm

    saved = []
    monkeypatch.setattr(
        lewm,
        'save_pretrained',
        lambda *a, filename, **k: saved.append(filename),
    )
    callback = lewm.SaveCkptCallback(
        'run', {}, epoch_interval=5, step_interval=2
    )
    module = SimpleNamespace(model=nn.Linear(1, 1))
    for step in range(1, 6):
        trainer = SimpleNamespace(global_step=step, is_global_zero=True)
        callback.on_train_batch_end(trainer, module, None, None, step)
    callback.on_train_end(
        SimpleNamespace(current_epoch=1, is_global_zero=True), module
    )
    assert saved == [
        'weights_step_2.pt',
        'weights_step_4.pt',
        'weights_epoch_1.pt',
    ]
