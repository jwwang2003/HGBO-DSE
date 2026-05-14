from __future__ import annotations

import argparse
from pathlib import Path

try:
    from hgp.arch_aware_arch import ensure_arch_aware_arch_cache, load_arch_aware_arch_cache
    from hgp.board_fabric import ensure_board_fabric_cache
    from hgp.board_utils import DEFAULT_BOARD_DEVICE, save_augmented_dataset
except ImportError:  # pragma: no cover - supports running from hgp/data_process
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from hgp.arch_aware_arch import ensure_arch_aware_arch_cache, load_arch_aware_arch_cache
    from hgp.board_fabric import ensure_board_fabric_cache
    from hgp.board_utils import DEFAULT_BOARD_DEVICE, save_augmented_dataset


HGBO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = HGBO_ROOT / "dataset"
ARCH_AWARE_MODE = "arch-aware"


def _augment_split(
    input_root: Path,
    output_root: Path,
    split: str,
    device: str,
    arch_aware_payload: dict[str, object] | None,
) -> list[Path]:
    input_dir = input_root / split
    output_dir = output_root / f"{split}_arch"
    written: list[Path] = []
    if not input_dir.is_dir():
        return written

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in sorted(input_dir.glob("*.pt")):
        output_path = output_dir / input_path.name
        save_augmented_dataset(input_path, output_path, device=device, arch_aware_arch=arch_aware_payload)
        written.append(output_path)
    return written


def augment_dataset_dirs(
    input_root: str | Path,
    output_root: str | Path,
    *,
    device: str = DEFAULT_BOARD_DEVICE,
    cache_dir: str | Path | None = None,
    force_cache: bool = False,
    arch_mode: str = ARCH_AWARE_MODE,
    clock_region: str | None = None,
) -> dict[str, list[Path]]:
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
    if arch_mode in ("fabric", "both"):
        ensure_board_fabric_cache(device, cache_dir=cache_root, force=force_cache)
    return {
        "std": _augment_split(input_path, output_path, "std", device, arch_aware_payload),
        "rdc": _augment_split(input_path, output_path, "rdc", device, arch_aware_payload),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach one FPGA architecture profile to HGBO-DSE datasets.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_DATASET_ROOT, help="dataset root containing std/ and rdc/")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DATASET_ROOT, help="dataset root where std_arch/ and rdc_arch/ are written")
    parser.add_argument("--device", default=DEFAULT_BOARD_DEVICE, help="FPGA part used by the raw HGBO-DSE labels")
    parser.add_argument("--cache-dir", default=None, help="directory for the extracted RapidWright fabric cache")
    parser.add_argument("--force-cache", action="store_true", help="re-extract the RapidWright fabric cache")
    parser.add_argument("--arch-mode", choices=[ARCH_AWARE_MODE, "fabric", "both"], default=ARCH_AWARE_MODE)
    parser.add_argument("--clock-region", default=None, help="optional RapidWright clock region for arch-aware extraction")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    written = augment_dataset_dirs(
        args.input_root,
        args.output_root,
        device=args.device,
        cache_dir=args.cache_dir,
        force_cache=args.force_cache,
        arch_mode=args.arch_mode,
        clock_region=args.clock_region,
    )
    total = sum(len(paths) for paths in written.values())
    print("Wrote {} architecture-aware dataset files for {}".format(total, args.device))
    for split, paths in written.items():
        for path in paths:
            print("{}: {}".format(split, path))


if __name__ == "__main__":
    main()
