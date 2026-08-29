"""Compatibility shims that let the vendored trainer run on modern torch.

The trainer in ``training/`` is vendored **unmodified** (see
``training/VENDORED.md``), which means API drift between its 2023-era
torch and the installed one is repaired here, before the vendored modules
import — never by editing them. One function, called by any orchestration
that is about to ``import train``.

Current shims:

- ``torch._utils._accumulate`` — removed in torch 2.x; ``dataset.py``
  imports it for its ``random_split``-style subset arithmetic. It was a
  running-sum generator, which :func:`itertools.accumulate` reproduces.
- ``DataLoader(prefetch_factor=…)`` with ``num_workers=0`` — the trainer
  passes ``prefetch_factor`` unconditionally; modern torch raises unless
  workers are enabled. Single-process loading is exactly what a macOS
  spike wants (spawned workers would re-import ``dataset.py`` without
  the shim above), so the wrapper drops the argument when it would
  raise.
- ``data_loader_iter.next()`` — a Python-2-ism ``dataset.py`` calls on
  torch's loader iterators, which lost their ``next`` method in torch
  1.13. Restored as an alias for ``__next__`` on the base iterator
  class, covering both the single- and multi-process variants.
"""

from __future__ import annotations

import itertools


def install() -> None:
    """Install every shim the vendored trainer needs. Idempotent."""
    import torch._utils
    import torch.utils.data as torch_data

    if not hasattr(torch._utils, "_accumulate"):

        def _accumulate(iterable, fn=None):
            return itertools.accumulate(iterable, fn) if fn else itertools.accumulate(iterable)

        torch._utils._accumulate = _accumulate

    from torch.utils.data import dataloader as _dataloader_module

    base_iter = getattr(_dataloader_module, "_BaseDataLoaderIter", None)
    if base_iter is not None and not hasattr(base_iter, "next"):
        base_iter.next = base_iter.__next__

    if not getattr(torch_data.DataLoader, "_tetrak_hy_compat", False):
        _RealDataLoader = torch_data.DataLoader

        class DataLoader(_RealDataLoader):  # noqa: N801 - torch's own name
            _tetrak_hy_compat = True

            def __init__(self, *args, **kwargs):
                if not kwargs.get("num_workers"):
                    kwargs.pop("prefetch_factor", None)
                super().__init__(*args, **kwargs)

        torch_data.DataLoader = DataLoader
