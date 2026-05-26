"""Board profile registry and feature encoding for HGBO-DSE.

Each supported FPGA part is registered as a :class:`BoardProfile`. Profile values
are looked up via :func:`resolve_board_profile`, and the per-board feature
vector (LUT/FF/DSP/BRAM counts, tech node, voltage, family one-hot) is built by
:func:`board_feature_values` / :func:`board_feature_tensor`. The same vector is
also embedded in the arch-aware metadata produced by
:mod:`hgp.arch_aware_arch`, which uses :func:`board_family_one_hot` to keep the
family classification logic in one place.

To add a new board: register it in :data:`_BOARD_PROFILES` with the canonical
Xilinx part string as both the dict key and ``BoardProfile.device``. Counts come
from the relevant Xilinx datasheet (CLB LUTs, CLB FFs, DSP slices, 36Kb BRAM
blocks). The ``family`` string must contain a token recognized by
:func:`board_family_one_hot` so that the family one-hot is non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch_geometric.data import Data


DEFAULT_BOARD_DEVICE = "xc7vx485tffg1761-2"


# Aliases mapping user-visible spellings (with/without dashes) to the canonical
# device key in :data:`_BOARD_PROFILES`. Keep the canonical form (the dict key)
# matching the device string Vivado/RapidWright uses for that part.
DEVICE_ALIASES = {
    # VC707 / Virtex-7
    "xc7vx485t-ffg1761-2": DEFAULT_BOARD_DEVICE,
    # KCU105 / Kintex UltraScale
    "xcku040_ffva1156_2_e": "xcku040-ffva1156-2-e",
    # VCU118 / Virtex UltraScale+
    "xcvu9p_flga2104_2_i": "xcvu9p-flga2104-2-i",
    # ZCU102 / Zynq UltraScale+
    "xczu9eg_ffvb1156_2_e": "xczu9eg-ffvb1156-2-e",
}


# Field labels for :func:`board_feature_values`. The order matches the float
# tensor produced by :func:`board_feature_tensor`; consumers index by name via
# ``ARCH_ATTR_FIELDS.index(...)``.
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
    """Static specification of an FPGA part.

    All counts are taken from the canonical Xilinx datasheet:
    CLB LUTs (not "system logic cells"), CLB flip-flops, DSP slices, and 36Kb
    BRAM blocks.
    """

    device: str
    family: str
    lut_count: int
    ff_count: int
    dsp_count: int
    bram_count: int
    tech_node_nm: int
    vccint: float


_BOARD_PROFILES = {
    # VC707 — Virtex-7, 28nm. 1.0V Vccint.
    DEFAULT_BOARD_DEVICE: BoardProfile(
        device=DEFAULT_BOARD_DEVICE,
        family="virtex7",
        lut_count=303600,
        ff_count=607200,
        dsp_count=2800,
        bram_count=1030,
        tech_node_nm=28,
        vccint=1.0,
    ),
    # KCU105 — Kintex UltraScale (xcku040), 20nm. 0.95V Vccint.
    "xcku040-ffva1156-2-e": BoardProfile(
        device="xcku040-ffva1156-2-e",
        family="kintex_ultrascale",
        lut_count=242400,
        ff_count=484800,
        dsp_count=1920,
        bram_count=600,
        tech_node_nm=20,
        vccint=0.95,
    ),
    # VCU118 — Virtex UltraScale+ (xcvu9p), 16nm. 0.85V Vccint.
    "xcvu9p-flga2104-2-i": BoardProfile(
        device="xcvu9p-flga2104-2-i",
        family="virtex_ultrascale_plus",
        lut_count=1182240,
        ff_count=2364480,
        dsp_count=6840,
        bram_count=2160,
        tech_node_nm=16,
        vccint=0.85,
    ),
    # ZCU102 — Zynq UltraScale+ (xczu9eg), 16nm. 0.85V Vccint.
    "xczu9eg-ffvb1156-2-e": BoardProfile(
        device="xczu9eg-ffvb1156-2-e",
        family="zynq_ultrascale_plus",
        lut_count=274080,
        ff_count=548160,
        dsp_count=2520,
        bram_count=912,
        tech_node_nm=16,
        vccint=0.85,
    ),
}


def normalize_device_name(device: str | None) -> str:
    """Return the canonical key in :data:`_BOARD_PROFILES` for ``device``.

    Falls back to :data:`DEFAULT_BOARD_DEVICE` when ``device`` is empty.
    """
    if not device:
        return DEFAULT_BOARD_DEVICE
    normalized = str(device).strip()
    return DEVICE_ALIASES.get(normalized, normalized)


def equivalent_device_names(device: str | None) -> tuple[str, ...]:
    """Return canonical and alias spellings that may appear on disk."""
    normalized = normalize_device_name(device)
    names = [normalized]
    if device:
        requested = str(device).strip()
        if requested and requested not in names:
            names.append(requested)
    for alias, canonical in DEVICE_ALIASES.items():
        if canonical == normalized and alias not in names:
            names.append(alias)
    return tuple(names)


def resolve_board_profile(device: str | None = None) -> BoardProfile:
    """Return a copy of the :class:`BoardProfile` registered for ``device``."""
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


def registered_board_devices() -> tuple[str, ...]:
    """Return canonical device keys for every registered board."""
    return tuple(_BOARD_PROFILES)


def _safe_ratio(value: int, total: int) -> float:
    return float(value) / float(total) if total else 0.0


def board_family_one_hot(family: str) -> tuple[float, float, float]:
    """Map a profile family string to ``(is_series7, is_ultrascale, is_ultrascale_plus)``.

    The classification is substring-based on a normalized family name (lowercase,
    underscores collapsed) so that family strings such as ``virtex7``,
    ``kintex_ultrascale``, ``virtex_ultrascale_plus``, and
    ``zynq_ultrascale_plus`` all classify correctly. ``ultrascale_plus`` is
    considered a refinement of ``ultrascale`` — it sets only ``is_ultrascale_plus``.

    Returns a 3-tuple of floats so callers can splice it directly into a feature
    vector.
    """
    normalized = family.lower().replace("-", "").replace(" ", "")
    is_us_plus = "ultrascaleplus" in normalized or "ultrascale_plus" in normalized
    is_us = "ultrascale" in normalized and not is_us_plus
    is_s7 = (not is_us) and (not is_us_plus) and ("7" in normalized)
    return (1.0 if is_s7 else 0.0, 1.0 if is_us else 0.0, 1.0 if is_us_plus else 0.0)


def board_feature_values(profile: BoardProfile) -> list[float]:
    """Build the 13-dim ``arch_attr`` feature vector for ``profile``.

    Layout matches :data:`ARCH_ATTR_FIELDS`. The first six entries are absolute
    sizes (scaled to convenient units), the next four are intra-board resource
    balances, and the final three are the family one-hot from
    :func:`board_family_one_hot`.
    """
    total_resources = (
        profile.lut_count
        + profile.ff_count
        + profile.dsp_count
        + profile.bram_count
    )
    is_s7, is_us, is_us_plus = board_family_one_hot(profile.family)
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
        is_s7,
        is_us,
        is_us_plus,
    ]


def board_feature_tensor(profile: BoardProfile, *, device: torch.device | None = None) -> torch.Tensor:
    """Return :func:`board_feature_values` as a ``[1, 13]`` float32 tensor."""
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
    """Return a clone of ``sample`` augmented with board metadata.

    Adds three string fields (``board_device``, ``board_family``,
    ``board_arch_device``) and one float tensor (``arch_attr``). When
    ``arch_aware_arch`` is supplied, also attaches the layout/metadata tensors
    produced by :mod:`hgp.arch_aware_arch`.
    """
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
    """Apply :func:`attach_board_profile` to every sample in ``samples``."""
    profile = resolve_board_profile(device)
    return [attach_board_profile(sample, profile, arch_aware_arch=arch_aware_arch) for sample in samples]


def save_augmented_dataset(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: str | None = None,
    arch_aware_arch: dict[str, object] | None = None,
) -> list[Data]:
    """Load a list of PyG samples, augment them with board metadata, save."""
    samples = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(samples, Sequence):
        raise TypeError("Expected a sequence of PyG Data samples in {}".format(input_path))
    augmented = augment_dataset(samples, device=device, arch_aware_arch=arch_aware_arch)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, output)
    return augmented
