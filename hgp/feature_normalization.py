"""Per-feature z-score normalization for HGBO-DSE PyG samples.

Stats are computed from the training split only and applied in-place to every
split (train, val, test) before DataLoader collation. Only ``data.x`` (per-node
features) and ``data.hls_attr`` (per-graph HLS metadata) are normalized;
``arch_attr`` and ``arch_aware_arch_metadata`` are skipped because their builders
in ``hgp/board_utils.py`` and ``hgp/arch_aware_arch.py`` already hand-scale the
values to O(1).

The output of :func:`compute_feature_stats` is suitable for storing inside a
training checkpoint or run-summary JSON; :func:`stats_to_jsonable` and
:func:`stats_from_jsonable` round-trip the dict through plain Python lists.

Note on inference compatibility: a checkpoint trained with normalization
enabled cannot be loaded by ``bome/pred/*.py`` without applying these stats
first — those modules read ``data.x`` and ``data['hls_attr']`` directly. Use
:func:`apply_feature_stats` on the inference batch with the stats stored in
the checkpoint payload (key ``feature_stats``).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import torch


_EPS = 1e-6


def compute_feature_stats(samples: Iterable) -> dict[str, torch.Tensor]:
    """Compute mean/std for ``x`` and ``hls_attr`` across ``samples``.

    Stats are returned as float32 CPU tensors. The standard deviation is
    clamped from below at ``1e-6`` so columns with zero variance (e.g. an
    always-zero HLS estimate) survive division.
    """
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("compute_feature_stats: empty sample sequence")

    x_all = torch.cat([s.x.to(torch.float32) for s in sample_list], dim=0)
    hls_all = torch.cat([s["hls_attr"].to(torch.float32) for s in sample_list], dim=0)
    return {
        "x_mean": x_all.mean(dim=0).contiguous(),
        "x_std": x_all.std(dim=0).clamp_min(_EPS).contiguous(),
        "hls_attr_mean": hls_all.mean(dim=0).contiguous(),
        "hls_attr_std": hls_all.std(dim=0).clamp_min(_EPS).contiguous(),
    }


def apply_feature_stats(samples: Iterable, stats: Mapping[str, torch.Tensor]) -> None:
    """Standardize ``x`` and ``hls_attr`` on each sample in place."""
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    hls_mean = stats["hls_attr_mean"]
    hls_std = stats["hls_attr_std"]
    for sample in samples:
        sample.x = (sample.x.to(torch.float32) - x_mean) / x_std
        sample["hls_attr"] = (sample["hls_attr"].to(torch.float32) - hls_mean) / hls_std


def stats_to_jsonable(stats: Mapping[str, torch.Tensor] | None) -> dict[str, list] | None:
    if stats is None:
        return None
    return {key: value.detach().cpu().tolist() for key, value in stats.items()}


def stats_from_jsonable(payload: Mapping[str, list] | None) -> dict[str, torch.Tensor] | None:
    if payload is None:
        return None
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in payload.items()}
