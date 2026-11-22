import torch

from stable_worldmodel.wm.utils import match_hf_vit_keys

LEGACY = [
    'encoder.embeddings.cls_token',
    'encoder.encoder.layer.0.attention.attention.query.weight',
    'encoder.encoder.layer.0.attention.attention.key.bias',
    'encoder.encoder.layer.0.attention.attention.value.weight',
    'encoder.encoder.layer.0.attention.output.dense.weight',
    'encoder.encoder.layer.0.intermediate.dense.weight',
    'encoder.encoder.layer.11.output.dense.bias',
    'encoder.encoder.layer.11.layernorm_before.weight',
    'encoder.layernorm.weight',
    'predictor.layers.0.attn.weight',  # not a ViT key: never renamed
]
CURRENT = [
    'encoder.embeddings.cls_token',
    'encoder.layers.0.attention.q_proj.weight',
    'encoder.layers.0.attention.k_proj.bias',
    'encoder.layers.0.attention.v_proj.weight',
    'encoder.layers.0.attention.o_proj.weight',
    'encoder.layers.0.mlp.fc1.weight',
    'encoder.layers.11.mlp.fc2.bias',
    'encoder.layers.11.layernorm_before.weight',
    'encoder.layernorm.weight',
    'predictor.layers.0.attn.weight',
]


def _sd(keys):
    return {k: torch.tensor(float(i)) for i, k in enumerate(keys)}


def test_legacy_checkpoint_into_current_model():
    out = match_hf_vit_keys(_sd(LEGACY), CURRENT)
    assert list(out) == CURRENT
    assert [v.item() for v in out.values()] == list(range(len(CURRENT)))


def test_current_checkpoint_into_legacy_model():
    out = match_hf_vit_keys(_sd(CURRENT), LEGACY)
    assert list(out) == LEGACY


def test_matching_or_unfixable_keys_unchanged():
    sd = _sd(CURRENT)
    assert match_hf_vit_keys(sd, CURRENT) is sd
    assert match_hf_vit_keys(sd, CURRENT + ['extra']) is sd
