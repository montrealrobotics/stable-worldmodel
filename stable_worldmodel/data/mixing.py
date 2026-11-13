"""Mix extra episodes (e.g. on-policy simulator rollouts) into a training set."""

from functools import partial

import torch


def _block_scale(scaler, dim, x):
    shape = x.shape
    return scaler(x.reshape(-1, dim)).reshape(shape)


def block_normalizer(scaler, dim, source='action', target='action'):
    """Apply a per-action scaler to rows that hold several actions.

    On-policy rollouts store one row per action block, i.e. ``frameskip``
    raw actions concatenated (width ``frameskip * dim``). ``scaler`` is the
    one fitted on the main dataset's ``dim``-wide raw actions, so both
    sources are normalized identically.
    """
    from stable_pretraining.data.transforms import WrapTorchTransform

    return WrapTorchTransform(
        partial(_block_scale, scaler, dim), source=source, target=target
    )


class WithSearchClips(torch.utils.data.Dataset):
    """Each batch of ``main`` clips also carries ``n`` random longer clips.

    The main loss keeps its own clips (e.g. two frames, every episode);
    the search loss gets ``n`` clips of its own length per batch under
    ``batch['search']``. Use with :func:`collate_with_search`.
    """

    def __init__(self, main, clips, n: int):
        if len(clips) == 0:
            raise ValueError('no clips long enough for the search loss')
        self.main = main
        self.clips = clips
        self.n = n

    def __len__(self):
        return len(self.main)

    def __getitem__(self, idx):
        return self.main[idx]

    def __getitems__(self, indices):
        if callable(getattr(self.main, '__getitems__', None)):
            items = self.main.__getitems__(list(indices))
        else:
            items = [self.main[i] for i in indices]
        # torch RNG: seeded per DataLoader worker, so runs are reproducible.
        pick = torch.randint(len(self.clips), (self.n,)).tolist()
        if callable(getattr(self.clips, '__getitems__', None)):
            search = self.clips.__getitems__(pick)
        else:
            search = [self.clips[i] for i in pick]
        return {'main': items, 'search': search}


def collate_with_search(data):
    """Collate :class:`WithSearchClips` output: main batch + ``search`` dict."""
    from torch.utils.data import default_collate

    if not isinstance(data, dict):
        return default_collate(data)
    batch = default_collate(data['main'])
    batch['search'] = default_collate(data['search'])
    return batch


class MixDataset(torch.utils.data.Dataset):
    """``main`` plus a virtual tail of ``extra`` samples, so that a shuffled
    sampler draws about ``extra_frac`` of each batch from ``extra``.

    The tail cycles through ``extra``; its length is set by ``extra_frac``,
    not by the size of ``extra``.
    """

    def __init__(self, main, extra, extra_frac: float):
        if not 0.0 < extra_frac < 1.0:
            raise ValueError('extra_frac must be in (0, 1)')
        if len(extra) == 0:
            raise ValueError('extra dataset has no clips')
        self.main = main
        self.extra = extra
        self.n_main = len(main)
        self.n_extra = max(
            1, round(self.n_main * extra_frac / (1.0 - extra_frac))
        )

    def __len__(self):
        return self.n_main + self.n_extra

    def _route(self, idx):
        if idx < self.n_main:
            return 0, idx
        return 1, (idx - self.n_main) % len(self.extra)

    def __getitem__(self, idx):
        src, local = self._route(idx)
        return (self.main, self.extra)[src][local]

    def __getitems__(self, indices):
        routed = [self._route(int(i)) for i in indices]
        out = [None] * len(routed)
        for src, ds in enumerate((self.main, self.extra)):
            pos = [k for k, (s, _) in enumerate(routed) if s == src]
            if not pos:
                continue
            local = [routed[k][1] for k in pos]
            if callable(getattr(ds, '__getitems__', None)):
                items = ds.__getitems__(local)
            else:
                items = [ds[i] for i in local]
            for k, item in zip(pos, items):
                out[k] = item
        return out
