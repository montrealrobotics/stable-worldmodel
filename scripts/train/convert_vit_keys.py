"""Rewrite checkpoints to the transformers >= 5.12 ViT weight names.

transformers 5.12 renamed Hugging Face ViT weights from
``encoder.encoder.layer.N.attention.attention.query`` (and
``intermediate`` / ``output.dense``) to ``encoder.layers.N.attention.q_proj``
(``mlp.fc1`` / ``mlp.fc2``). Runs trained with an older transformers (Mila's
cluv venv had 5.6.2) save the legacy names, which ``load_pretrained`` cannot
load under a newer transformers (strict ``load_state_dict``). This script
rewrites the files.

Each file is checked before it is written: the renamed keys must equal those of
the model built from the run's ``config.json`` in the new naming. Under an
older transformers the model's own keys are renamed first, so the script runs
in either venv (e.g. Mila's cluv .venv). Files already in the new naming
are skipped. Files are replaced atomically; ``--backup`` keeps
``<file>.legacy``.

    python scripts/train/convert_vit_keys.py RUN_DIR [RUN_DIR ...] [--dry-run]
    python scripts/train/convert_vit_keys.py last.ckpt --config RUN_DIR/config.json

A directory converts every ``*.pt`` in it. A Lightning ``.ckpt`` (resume
state) needs ``--config``: the ``config.json`` of the weights folder of the
same run; only its ``model.`` weights are renamed.
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
import transformers
from hydra.utils import instantiate
from packaging.version import Version

# Hugging Face ViT renamed its weights between transformers 5.6 and 5.12:
# (legacy, current) pieces of a key, matched inside '<prefix>.encoder.layer.N.'
# vs '<prefix>.layers.N.'. 'attention.output.dense' must come before
# 'output.dense'.
_HF_VIT_RENAMES = (
    ('attention.attention.query.', 'attention.q_proj.'),
    ('attention.attention.key.', 'attention.k_proj.'),
    ('attention.attention.value.', 'attention.v_proj.'),
    ('attention.output.dense.', 'attention.o_proj.'),
    ('intermediate.dense.', 'mlp.fc1.'),
    ('output.dense.', 'mlp.fc2.'),
)


def _hf_vit_key(key, to_current):
    src, dst = (
        (r'(.*\.)?encoder\.layer\.(\d+)\.(.*)', r'\1layers.\2.')
        if to_current
        else (r'(.*\.)?layers\.(\d+)\.(.*)', r'\1encoder.layer.\2.')
    )
    m = re.fullmatch(src, key)
    if m is None:
        return key
    rest = m.group(3)
    for legacy, current in _HF_VIT_RENAMES:
        old, new = (legacy, current) if to_current else (current, legacy)
        if rest.startswith(old):
            rest = new + rest[len(old) :]
            break
    return m.expand(dst) + rest


def match_hf_vit_keys(state_dict, expected_keys):
    """Rename HF ViT encoder weights to the naming the model expects.

    Checkpoints trained with transformers < 5.12 store
    ``encoder.encoder.layer.N.attention.attention.query`` where newer versions
    build ``encoder.layers.N.attention.q_proj`` (and the reverse). Returns the
    state dict unchanged when its keys already match or no rename makes them.
    """
    expected = set(expected_keys)
    if set(state_dict) == expected:
        return state_dict
    renamed = {}
    for k, v in state_dict.items():
        if k not in expected:  # only touch keys the model lacks
            k = next(
                (
                    n
                    for n in (_hf_vit_key(k, d) for d in (True, False))
                    if n in expected
                ),
                k,
            )
        renamed[k] = v
    if set(renamed) == expected:
        return renamed
    return state_dict


def expected_keys(config_path, cache):
    """The model's state-dict keys in the transformers >= 5.12 naming."""
    if config_path not in cache:
        with open(config_path) as f:
            keys = set(instantiate(json.load(f)).state_dict())
        if Version(transformers.__version__) < Version('5.12'):
            keys = {_hf_vit_key(k, to_current=True) for k in keys}
        cache[config_path] = keys
    return cache[config_path]


def convert(path, config_path, cache, dry_run, backup):
    lightning = path.suffix == '.ckpt'
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    full = ckpt['state_dict'] if lightning else ckpt
    expected = expected_keys(config_path, cache)
    if lightning:  # the world model sits under 'model.' in the LightningModule
        sd = {k[6:]: v for k, v in full.items() if k.startswith('model.')}
    else:
        sd = full
    if set(sd) == expected:
        return 'already current'
    renamed = match_hf_vit_keys(sd, expected)
    if set(renamed) != expected:
        missing = sorted(expected - set(renamed))[:3]
        extra = sorted(set(renamed) - expected)[:3]
        return f'SKIPPED, keys do not match the model: missing {missing}, unexpected {extra}'
    if lightning:
        new = {k: v for k, v in full.items() if not k.startswith('model.')}
        new.update({f'model.{k}': v for k, v in renamed.items()})
        ckpt['state_dict'] = new
    else:
        ckpt = renamed
    if dry_run:
        return 'would convert'
    if backup:
        shutil.copy2(path, path.with_name(path.name + '.legacy'))
    tmp = path.with_name(path.name + '.tmp')
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    return 'converted'


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument(
        'paths', nargs='+', type=Path, help='.pt/.ckpt files or run folders'
    )
    p.add_argument(
        '--config', type=Path, help='config.json (default: next to each .pt)'
    )
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--backup', action='store_true', help='keep <file>.legacy')
    args = p.parse_args()

    files = []
    for path in args.paths:
        files += sorted(path.glob('*.pt')) if path.is_dir() else [path]
    cache, failed = {}, False
    for f in files:
        config = args.config or f.parent / 'config.json'
        if f.suffix == '.ckpt' and args.config is None:
            status = 'SKIPPED, a .ckpt needs --config'
        else:
            status = convert(f, config, cache, args.dry_run, args.backup)
        failed |= status.startswith('SKIPPED')
        print(f'{f}: {status}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
