"""Augment HGBO-DSE PyG datasets with one or more FPGA architecture profiles.

Inputs are the raw HLS-derived PyG sample shards under ``<input_root>/std/``
and ``<input_root>/rdc/``. For each ``--device`` requested, the script:

  1. Ensures the RapidWright-derived arch-aware cache (and, optionally, the
     fabric-graph cache) for that device exists under ``<cache-dir>``. The
     arch-aware extraction also exports SVG views of the full device fabric,
     the selected clock-region fabric, and the compressed fabric tensor.
  2. Loads each sample shard, attaches the board profile + arch tensors via
     :func:`hgp.board_utils.augment_dataset`, and writes the augmented shards.

By default the augmented shards are written under
``<output_root>/<device>/std_arch/`` and ``<output_root>/<device>/rdc_arch/``.
This per-device layout is required when training across multiple boards: the
legacy single-device layout (``<output_root>/std_arch/``) overwrites between
runs and is no longer the default. Pass ``--legacy-layout`` to opt back into
the flat layout for backwards-compatible single-device experiments.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from hgp.arch_aware_arch import arch_aware_svg_paths, ensure_arch_aware_arch_cache, load_arch_aware_arch_cache
    from hgp.board_fabric import ensure_board_fabric_cache
    from hgp.board_utils import (
        DEFAULT_BOARD_DEVICE,
        equivalent_device_names,
        normalize_device_name,
        save_augmented_dataset,
    )
except ImportError:  # pragma: no cover - supports running from hgp/data_process
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from hgp.arch_aware_arch import arch_aware_svg_paths, ensure_arch_aware_arch_cache, load_arch_aware_arch_cache
    from hgp.board_fabric import ensure_board_fabric_cache
    from hgp.board_utils import (
        DEFAULT_BOARD_DEVICE,
        equivalent_device_names,
        normalize_device_name,
        save_augmented_dataset,
    )


HGBO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = HGBO_ROOT / "dataset"
ARCH_AWARE_MODE = "arch-aware"


def _augment_split(
    input_root: Path,
    output_root: Path,
    split: str,
    device: str,
    arch_aware_payload: dict[str, object] | None,
    *,
    layout: str = "per_device",
) -> list[Path]:
    """Augment one ``std`` or ``rdc`` split for a single ``device``.

    With ``layout='per_device'`` (the default), augmented shards are written
    under ``<output_root>/<device>/<split>_arch/<bench>.pt`` so that multiple
    devices can coexist in one tree. With ``layout='legacy'`` the shards are
    written flat under ``<output_root>/<split>_arch/<bench>.pt`` and re-runs
    for a different device overwrite the same files.
    """
    input_dir = _resolve_split_input_dir(input_root, split, device)
    if layout == "per_device":
        output_dir = output_root / normalize_device_name(device) / f"{split}_arch"
    elif layout == "legacy":
        output_dir = output_root / f"{split}_arch"
    else:
        raise ValueError("layout must be 'per_device' or 'legacy'; got {!r}".format(layout))

    written: list[Path] = []
    if input_dir is None:
        return written

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.glob("*.pt")):
        output_path = output_dir / input_path.name
        save_augmented_dataset(input_path, output_path, device=device, arch_aware_arch=arch_aware_payload)
        written.append(output_path)
    _copy_arch_aware_svgs(arch_aware_payload, output_dir)
    return written


def _copy_arch_aware_svgs(arch_aware_payload: dict[str, object] | None, output_dir: Path) -> list[Path]:
    """Copy device-level arch-aware SVGs next to generated dataset shards."""
    if not arch_aware_payload:
        return []

    raw_paths = arch_aware_payload.get("arch_aware_svg_paths")
    if not isinstance(raw_paths, dict):
        return []

    copied: list[Path] = []
    for source in raw_paths.values():
        source_path = Path(str(source))
        if not source_path.exists():
            continue
        destination = output_dir / source_path.name
        if source_path.resolve() == destination.resolve():
            copied.append(destination)
            continue
        shutil.copy2(source_path, destination)
        copied.append(destination)
    return copied


def _resolve_split_input_dir(input_root: Path, split: str, device: str) -> Path | None:
    """Find flat or per-device input shards for ``split`` and ``device``."""
    candidates = [input_root / split]
    candidates.extend(input_root / name / split for name in equivalent_device_names(device))
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.pt")):
            return candidate
    return None


def augment_dataset_dirs(
    input_root: str | Path,
    output_root: str | Path,
    *,
    device: str = DEFAULT_BOARD_DEVICE,
    cache_dir: str | Path | None = None,
    force_cache: bool = False,
    arch_mode: str = ARCH_AWARE_MODE,
    clock_region: str | None = None,
    layout: str = "per_device",
) -> dict[str, list[Path]]:
    """Augment ``std/`` and ``rdc/`` directories for a single device.

    See module docstring for output layout. Returns a dict ``{"std": [...], "rdc": [...]}``
    listing every shard written.
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    cache_root = Path(cache_dir) if cache_dir is not None else output_path / "board_arch"
    arch_aware_payload = None
    if arch_mode in (ARCH_AWARE_MODE, "both"):
        arch_aware_cache = ensure_arch_aware_arch_cache(
            device,
            cache_dir=cache_root,
            force=force_cache,
            clock_region=clock_region,
        )
        arch_aware_payload = load_arch_aware_arch_cache(arch_aware_cache)
        if "arch_aware_svg_paths" not in arch_aware_payload:
            arch_aware_payload["arch_aware_svg_paths"] = {
                key: str(path)
                for key, path in arch_aware_svg_paths(device, cache_dir=cache_root).items()
            }
    if arch_mode in ("fabric", "both"):
        ensure_board_fabric_cache(device, cache_dir=cache_root, force=force_cache)
    return {
        "std": _augment_split(input_path, output_path, "std", device, arch_aware_payload, layout=layout),
        "rdc": _augment_split(input_path, output_path, "rdc", device, arch_aware_payload, layout=layout),
    }


def augment_dataset_dirs_multi(
    input_root: str | Path,
    output_root: str | Path,
    *,
    devices: list[str],
    cache_dir: str | Path | None = None,
    force_cache: bool = False,
    arch_mode: str = ARCH_AWARE_MODE,
    clock_region: str | None = None,
    layout: str = "per_device",
) -> dict[str, dict[str, list[Path]]]:
    """Run :func:`augment_dataset_dirs` for each device, returning per-device results."""
    normalized_devices = [normalize_device_name(device) for device in devices]
    if layout == "legacy" and len(set(normalized_devices)) > 1:
        raise ValueError(
            "layout='legacy' clobbers shared output paths and cannot be used "
            "with more than one device; use layout='per_device' for multi-board runs."
        )
    results: dict[str, dict[str, list[Path]]] = {}
    for device in normalized_devices:
        if device in results:
            continue
        results[device] = augment_dataset_dirs(
            input_root,
            output_root,
            device=device,
            cache_dir=cache_dir,
            force_cache=force_cache,
            arch_mode=arch_mode,
            clock_region=clock_region,
            layout=layout,
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``gen_dataset_board.py``."""
    parser = argparse.ArgumentParser(description="Attach FPGA architecture profiles to HGBO-DSE datasets.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="dataset root containing std/ and rdc/ shards (default: <repo>/dataset)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="dataset root where augmented shards are written (default: <repo>/dataset)",
    )
    parser.add_argument(
        "--device",
        action="append",
        default=None,
        help=(
            "FPGA part used to derive the architecture profile. Repeat to "
            "augment for multiple boards in one invocation. Defaults to "
            "{!r} when omitted.".format(DEFAULT_BOARD_DEVICE)
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="directory for the extracted RapidWright fabric/arch cache "
        "and generated SVG fabric views (default: <output_root>/board_arch)",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="re-extract the RapidWright fabric/arch cache even if present",
    )
    parser.add_argument(
        "--arch-mode",
        choices=[ARCH_AWARE_MODE, "fabric", "both"],
        default=ARCH_AWARE_MODE,
        help="which architecture cache to ensure: arch-aware tile layout, "
        "fabric graph, or both",
    )
    parser.add_argument(
        "--clock-region",
        default=None,
        help="optional RapidWright clock region for arch-aware extraction",
    )
    parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="write augmented shards under <output>/std_arch (single device only); "
        "default is per-device subdirs that support multi-board training.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    devices = args.device or [DEFAULT_BOARD_DEVICE]
    layout = "legacy" if args.legacy_layout else "per_device"

    results = augment_dataset_dirs_multi(
        args.input_root,
        args.output_root,
        devices=devices,
        cache_dir=args.cache_dir,
        force_cache=args.force_cache,
        arch_mode=args.arch_mode,
        clock_region=args.clock_region,
        layout=layout,
    )
    for device, written in results.items():
        total = sum(len(paths) for paths in written.values())
        print("Wrote {} architecture-aware dataset files for {}".format(total, device))
        for split, paths in written.items():
            for path in paths:
                print("{} ({}): {}".format(split, device, path))


if __name__ == "__main__":
    main()
