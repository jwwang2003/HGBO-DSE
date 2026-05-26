"""Multi-board training helpers for the architecture-aware HGBO-DSE model.

This module is the bridge between the per-board RapidWright caches produced
by :mod:`hgp.arch_aware_arch` / :mod:`hgp.board_fabric` and the
:class:`hgp.hier_arch_model.ArchAwareHierNet` forward pass when the dataset
contains samples from more than one FPGA part.

The single-board legacy path is unchanged: callers that supply one device get
back exactly one cache and the training loop behaves as before.

Public API:
    - :func:`prepare_arch_caches` — load per-device arch payloads to a torch device
    - :func:`prepare_board_fabric_caches` — same, but for the board-fabric graphs
    - :func:`board_devices_for_batch` — extract the per-sample board_device list
    - :func:`dispatch_arch_aware_payload` — stack per-sample layout+metadata
    - :func:`dispatch_fabric_payload` — choose the right fabric graph for a batch

The dispatch helpers raise ``KeyError`` if a batch contains a board that has
not been cached, which is the desired loud failure mode when training data
and architecture caches drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from hgp.arch_aware_arch import (
    ARCH_AWARE_LAYOUT_COLS,
    ARCH_AWARE_METADATA_DIM,
    ARCH_AWARE_TILE_SLOTS,
    arch_aware_arch_to_device,
    ensure_arch_aware_arch_cache,
    load_arch_aware_arch_cache,
)
from hgp.board_fabric import (
    board_fabric_to_device,
    ensure_board_fabric_cache,
    load_board_fabric_cache,
)
from hgp.board_utils import normalize_device_name


def prepare_arch_caches(
    devices: Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    torch_device: torch.device | None = None,
    force: bool = False,
) -> dict[str, dict[str, object]]:
    """Ensure & load the arch-aware payload for each device, moved to ``torch_device``.

    Returns a dict keyed by canonical (normalized) device name. Each value is
    the dict produced by :func:`hgp.arch_aware_arch.load_arch_aware_arch_cache`.
    """
    payloads: dict[str, dict[str, object]] = {}
    for device in devices:
        canonical = normalize_device_name(device)
        cache_path = ensure_arch_aware_arch_cache(canonical, cache_dir=cache_dir, force=force)
        payload = load_arch_aware_arch_cache(cache_path)
        if torch_device is not None:
            payload = arch_aware_arch_to_device(payload, torch_device)
        payloads[canonical] = payload
    return payloads


def prepare_board_fabric_caches(
    devices: Iterable[str],
    *,
    cache_dir: str | Path | None = None,
    torch_device: torch.device | None = None,
    force: bool = False,
) -> dict[str, dict[str, object]]:
    """Ensure & load the board-fabric graph for each device, moved to ``torch_device``."""
    payloads: dict[str, dict[str, object]] = {}
    for device in devices:
        canonical = normalize_device_name(device)
        cache_path = ensure_board_fabric_cache(canonical, cache_dir=cache_dir, force=force)
        payload = load_board_fabric_cache(cache_path)
        if torch_device is not None:
            payload = board_fabric_to_device(payload, torch_device)
        payloads[canonical] = payload
    return payloads


def board_devices_for_batch(batch) -> list[str] | None:
    """Return the per-sample ``board_device`` list for a PyG ``Batch`` (or ``None``).

    PyG's collation turns each sample's ``board_device`` string into a list at
    the batch level. When the input dataset has been augmented through
    :func:`hgp.board_utils.attach_board_profile`, every sample carries this
    field. We accept both list-of-strings (multi-sample batch) and a bare
    string (single-sample batch / pre-batched data).
    """
    value = getattr(batch, "board_device", None)
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    # Some PyG collators may produce torch tensors of bytes. Best-effort fallback.
    try:
        return [str(item) for item in value]
    except TypeError:
        return None


def dispatch_arch_aware_payload(
    arch_caches: dict[str, dict[str, object]],
    devices: list[str],
) -> dict[str, torch.Tensor]:
    """Stack per-device layout/metadata tensors in the requested order.

    For a batch whose samples come from devices ``[d0, d1, d0, d2]``, produce
    tensors of shape ``[4, ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS]``
    and ``[4, ARCH_AWARE_METADATA_DIM]`` respectively, where row ``i`` matches
    ``devices[i]``. The :class:`ArchAwareArchitectureEncoder` consumes both
    tensors batched along dim 0.
    """
    if not devices:
        raise ValueError("dispatch_arch_aware_payload requires at least one device")

    layouts: list[torch.Tensor] = []
    metadatas: list[torch.Tensor] = []
    for device in devices:
        canonical = normalize_device_name(device)
        try:
            payload = arch_caches[canonical]
        except KeyError as err:
            raise KeyError(
                "No arch-aware cache loaded for device {!r} (have: {})".format(
                    canonical, sorted(arch_caches)
                )
            ) from err
        layout = payload["arch_aware_arch_layout"]
        metadata = payload["arch_aware_arch_metadata"]

        if not torch.is_tensor(layout):
            layout = torch.as_tensor(layout)
        if not torch.is_tensor(metadata):
            metadata = torch.as_tensor(metadata)

        # Layout cache shape is [cols, 1, slots]. Add the leading batch dim.
        if layout.dim() == 3:
            layouts.append(layout.unsqueeze(0))
        elif layout.dim() == 4 and layout.size(0) == 1:
            layouts.append(layout)
        else:
            raise ValueError(
                "Unexpected arch-aware layout shape {} for device {!r}".format(
                    tuple(layout.shape), canonical
                )
            )

        if metadata.dim() == 1:
            metadata = metadata.unsqueeze(0)
        if metadata.dim() != 2 or metadata.size(0) != 1:
            raise ValueError(
                "Unexpected arch-aware metadata shape {} for device {!r}".format(
                    tuple(metadata.shape), canonical
                )
            )
        metadatas.append(metadata)

    return {
        "arch_aware_arch_layout": torch.cat(layouts, dim=0),
        "arch_aware_arch_metadata": torch.cat(metadatas, dim=0),
    }


def dispatch_fabric_payload(
    fabric_caches: dict[str, dict[str, object]],
    devices: list[str],
) -> dict[str, object]:
    """Pick the fabric payload for the (homogeneous) batch.

    The fabric encoder is GNN-based and expects a single device's graph at a
    time; mixing fabrics inside one batch is not supported. This helper enforces
    that constraint by raising ``ValueError`` if ``devices`` mixes boards.
    """
    if not devices:
        raise ValueError("dispatch_fabric_payload requires at least one device")
    canonicals = {normalize_device_name(device) for device in devices}
    if len(canonicals) > 1:
        raise ValueError(
            "Fabric mode requires homogeneous batches (one board per batch); "
            "got {}. Use --arch-mode arch-aware for heterogeneous batches.".format(sorted(canonicals))
        )
    canonical = next(iter(canonicals))
    try:
        return fabric_caches[canonical]
    except KeyError as err:
        raise KeyError(
            "No fabric cache loaded for device {!r} (have: {})".format(
                canonical, sorted(fabric_caches)
            )
        ) from err


# Re-exported for documentation / type hint purposes.
__all__ = [
    "ARCH_AWARE_LAYOUT_COLS",
    "ARCH_AWARE_METADATA_DIM",
    "ARCH_AWARE_TILE_SLOTS",
    "prepare_arch_caches",
    "prepare_board_fabric_caches",
    "board_devices_for_batch",
    "dispatch_arch_aware_payload",
    "dispatch_fabric_payload",
]
