from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch_geometric.data import Data


DEFAULT_BOARD_DEVICE = "xc7vx485tffg1761-2"

DEVICE_ALIASES = {
    "xc7vx485t-ffg1761-2": DEFAULT_BOARD_DEVICE,
    "xc7vx485tffg1761-2": DEFAULT_BOARD_DEVICE,
}

ARCH_ATTR_FIELDS = (
    "lut_m",
    "ff_m",
    "dsp_k",
    "bram_k",
    "tech_node_100nm",
    "vccint",
    "resource_balance_lut",
    "resource_balance_ff",
    "resource_balance_dsp",
    "resource_balance_bram",
    "is_series7",
    "is_ultrascale",
    "is_ultrascale_plus",
)


@dataclass(frozen=True)
class BoardProfile:
    device: str
    family: str
    lut_count: int
    ff_count: int
    dsp_count: int
    bram_count: int
    tech_node_nm: int
    vccint: float


_BOARD_PROFILES = {
    DEFAULT_BOARD_DEVICE: BoardProfile(
        device=DEFAULT_BOARD_DEVICE,
        family="virtex7",
        lut_count=303600,
        ff_count=607200,
        dsp_count=2800,
        bram_count=1030,
        tech_node_nm=28,
        vccint=1.0,
    )
}


def normalize_device_name(device: str | None) -> str:
    if not device:
        return DEFAULT_BOARD_DEVICE
    normalized = str(device).strip()
    return DEVICE_ALIASES.get(normalized, normalized)


def resolve_board_profile(device: str | None = None) -> BoardProfile:
    normalized = normalize_device_name(device)
    profile = _BOARD_PROFILES.get(normalized)
    if profile is None:
        raise ValueError(
            "Unsupported board device {!r}. Available devices: {}".format(
                normalized,
                sorted(_BOARD_PROFILES),
            )
        )
    return replace(profile)


def _safe_ratio(value: int, total: int) -> float:
    return float(value) / float(total) if total else 0.0


def board_feature_values(profile: BoardProfile) -> list[float]:
    total_resources = (
        profile.lut_count
        + profile.ff_count
        + profile.dsp_count
        + profile.bram_count
    )
    family = profile.family.lower()
    return [
        profile.lut_count / 1_000_000.0,
        profile.ff_count / 1_000_000.0,
        profile.dsp_count / 1_000.0,
        profile.bram_count / 1_000.0,
        profile.tech_node_nm / 100.0,
        profile.vccint,
        _safe_ratio(profile.lut_count, total_resources),
        _safe_ratio(profile.ff_count, total_resources),
        _safe_ratio(profile.dsp_count, total_resources),
        _safe_ratio(profile.bram_count, total_resources),
        1.0 if "7" in family else 0.0,
        1.0 if family == "ultrascale" else 0.0,
        1.0 if family == "ultrascaleplus" else 0.0,
    ]


def board_feature_tensor(profile: BoardProfile, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([board_feature_values(profile)], dtype=torch.float32, device=device)


def _payload_tensor(payload: dict[str, object], key: str, dtype: torch.dtype) -> torch.Tensor:
    value = payload[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().clone().to(dtype=dtype)


def attach_board_profile(
    sample: Data,
    profile: BoardProfile,
    *,
    arch_aware_arch: dict[str, object] | None = None,
) -> Data:
    augmented = sample.clone()
    augmented.arch_attr = board_feature_tensor(profile)
    augmented.board_device = profile.device
    augmented.board_family = profile.family
    augmented.board_arch_device = profile.device
    if arch_aware_arch is not None:
        augmented.arch_aware_arch_layout = _payload_tensor(arch_aware_arch, "arch_aware_arch_layout", torch.float32)
        augmented.arch_aware_arch_metadata = _payload_tensor(arch_aware_arch, "arch_aware_arch_metadata", torch.float32)
        augmented.arch_aware_arch_device = str(arch_aware_arch.get("device", profile.device))
        augmented.arch_aware_clock_region = arch_aware_arch.get("arch_aware_clock_region")
    return augmented


def augment_dataset(
    samples: Iterable[Data],
    device: str | None = None,
    *,
    arch_aware_arch: dict[str, object] | None = None,
) -> list[Data]:
    profile = resolve_board_profile(device)
    return [attach_board_profile(sample, profile, arch_aware_arch=arch_aware_arch) for sample in samples]


def save_augmented_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str | None = None,
    arch_aware_arch: dict[str, object] | None = None,
) -> list[Data]:
    samples = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(samples, Sequence):
        raise TypeError("Expected a sequence of PyG Data samples in {}".format(input_path))
    augmented = augment_dataset(samples, device=device, arch_aware_arch=arch_aware_arch)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, output)
    return augmented
