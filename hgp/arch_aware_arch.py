from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import torch

from hgp.board_utils import BoardProfile, DEFAULT_BOARD_DEVICE, normalize_device_name, resolve_board_profile
from hgp.rapidwright_env import require_rapidwright_device


ARCH_AWARE_LAYOUT_COLS = 360
ARCH_AWARE_LAYOUT_ROWS = 80
ARCH_AWARE_TILE_SLOTS = 4
ARCH_AWARE_METADATA_DIM = 21
DEFAULT_ARCH_AWARE_CACHE_DIR = Path(__file__).resolve().parents[1] / "dataset" / "board_arch"

ARCH_AWARE_TILE_TYPE_TO_ID = {
    "PAD": 0,
    "CLEL": 1,
    "CLEM": 2,
    "INT": 3,
    "INT_INTERFACE": 4,
    "DSP": 5,
    "BRAM": 6,
    "BRK": 7,
}
ARCH_AWARE_TILE_ID_TO_TYPE = {value: key for key, value in ARCH_AWARE_TILE_TYPE_TO_ID.items()}

_XY_RE = re.compile(r"(?:^|_)X(?P<x>-?\d+)Y(?P<y>-?\d+)(?:_|$)")
_CLOCK_REGION_RE = re.compile(r"X(?P<x>-?\d+)Y(?P<y>-?\d+)")


def _tokens(value: object) -> set[str]:
    normalized = str(value).upper().replace("-", "_").replace(".", "_")
    return {token for token in normalized.split("_") if token}


def _name(value: object) -> str:
    if hasattr(value, "getName"):
        return str(value.getName())
    return str(value)


def classify_arch_aware_tile(tile_or_name: object) -> str | None:
    tile_name = _name(tile_or_name)
    tile_type = tile_or_name.getTileTypeEnum() if hasattr(tile_or_name, "getTileTypeEnum") else ""
    tokens = _tokens(tile_name) | _tokens(tile_type)

    if "BRAM" in tokens or "RAMB" in tokens:
        return "BRAM"
    if "DSP" in tokens:
        return "DSP"
    if "CLBLL" in tokens or "CLEL" in tokens:
        return "CLEL"
    if "CLBLM" in tokens or "CLEM" in tokens or ("CLE" in tokens and "M" in tokens):
        return "CLEM"
    if "INT" in tokens and "INTERFACE" in tokens:
        return "INT_INTERFACE"
    if "INT" in tokens:
        return "INT"
    if str(tile_type).upper() == "NULL":
        return "BRK"
    return None


def sinusoidal_positional_encoding(
    length: int = ARCH_AWARE_LAYOUT_COLS,
    dim: int = ARCH_AWARE_TILE_SLOTS,
) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    indexes = torch.arange(0, dim, 2, dtype=torch.float32)
    div_term = torch.exp(indexes * (-math.log(10000.0) / dim))
    encoding = torch.zeros((length, dim), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.view(length, 1, dim)


def _safe_ratio(value: float, denominator: float) -> float:
    return float(value) / float(denominator) if denominator else 0.0


def build_arch_aware_metadata(
    profile: BoardProfile,
    *,
    raw_rows: int,
    raw_cols: int,
    valid_rows: int,
    valid_cols: int,
    fsr_row: int = 0,
    fsr_col: int = 0,
    fsr_count: int = 1,
) -> torch.Tensor:
    total_resources = profile.lut_count + profile.ff_count + profile.dsp_count + profile.bram_count
    family = profile.family.lower()
    values = [
        profile.lut_count / 1_000_000.0,
        profile.ff_count / 1_000_000.0,
        profile.dsp_count / 1_000.0,
        profile.bram_count / 1_000.0,
        profile.tech_node_nm / 100.0,
        profile.vccint,
        _safe_ratio(raw_rows, ARCH_AWARE_LAYOUT_ROWS),
        _safe_ratio(raw_cols, ARCH_AWARE_LAYOUT_COLS),
        _safe_ratio(valid_rows, ARCH_AWARE_LAYOUT_ROWS),
        _safe_ratio(valid_cols, ARCH_AWARE_LAYOUT_COLS),
        _safe_ratio(fsr_row, max(fsr_count - 1, 1)),
        _safe_ratio(fsr_col, max(fsr_count - 1, 1)),
        float(fsr_count),
        _safe_ratio(profile.lut_count, total_resources),
        _safe_ratio(profile.ff_count, total_resources),
        _safe_ratio(profile.dsp_count, total_resources),
        _safe_ratio(profile.bram_count, total_resources),
        1.0 if "7" in family else 0.0,
        1.0 if family == "ultrascale" else 0.0,
        1.0 if family == "ultrascaleplus" else 0.0,
        1.0,
    ]
    if len(values) != ARCH_AWARE_METADATA_DIM:
        raise RuntimeError("arch-aware metadata dimension mismatch")
    return torch.tensor([values], dtype=torch.float32)


def _iter_tiles(device: object) -> Iterable[object]:
    if hasattr(device, "getTiles"):
        try:
            for row in device.getTiles():
                for tile in row:
                    if tile is not None:
                        yield tile
            return
        except AttributeError:
            pass
    for tile in device.getAllTiles():
        if tile is not None:
            yield tile


def _clock_regions(device: object) -> list[object]:
    if not hasattr(device, "getClockRegions"):
        return []
    regions: list[object] = []
    pending = list(device.getClockRegions())
    while pending:
        region = pending.pop(0)
        if region is None:
            continue
        if hasattr(region, "getName") or hasattr(region, "containsTile"):
            regions.append(region)
            continue
        if isinstance(region, Iterable) and not isinstance(region, (str, bytes)):
            pending = list(region) + pending
            continue
        regions.append(region)
    return regions


def _select_clock_region(device: object, clock_region: str | None) -> object | None:
    if clock_region is not None:
        if hasattr(device, "getClockRegion"):
            return device.getClockRegion(clock_region)
        for region in _clock_regions(device):
            if _clock_region_name(region) == clock_region:
                return region
        raise ValueError("Clock region {!r} is not available".format(clock_region))

    regions = _clock_regions(device)
    if not regions:
        return None
    positions = [_clock_region_position(region) for region in regions]
    center_row = sum(row for row, _ in positions) / len(positions)
    center_col = sum(col for _, col in positions) / len(positions)
    return min(
        regions,
        key=lambda region: (
            abs(_clock_region_position(region)[0] - center_row)
            + abs(_clock_region_position(region)[1] - center_col),
            _clock_region_name(region) or "",
        ),
    )


def _clock_region_name(clock_region: object | None) -> str | None:
    if clock_region is None:
        return None
    if hasattr(clock_region, "getName"):
        return str(clock_region.getName())
    return str(clock_region)


def _clock_region_position(clock_region: object | None) -> tuple[int, int]:
    name = _clock_region_name(clock_region) or ""
    match = _CLOCK_REGION_RE.search(name)
    if not match:
        return 0, 0
    return int(match.group("y")), int(match.group("x"))


def _tile_x_y(tile: object) -> tuple[int, int]:
    if hasattr(tile, "getColumn") and hasattr(tile, "getRow"):
        try:
            return int(tile.getColumn()), int(tile.getRow())
        except AttributeError:
            pass
    if hasattr(tile, "getTileXCoordinate") and hasattr(tile, "getTileYCoordinate"):
        try:
            return int(tile.getTileXCoordinate()), int(tile.getTileYCoordinate())
        except AttributeError:
            pass

    match = _XY_RE.search(_name(tile))
    if not match:
        return 0, 0
    return int(match.group("x")), int(match.group("y"))


def _compress_layout(raw_layout: torch.Tensor) -> torch.Tensor:
    compressed = torch.zeros((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32)
    for col in range(raw_layout.shape[0]):
        column = raw_layout[col]
        non_empty_rows = torch.nonzero(column.any(dim=1), as_tuple=False).view(-1)
        if non_empty_rows.numel() == 0:
            continue
        first_row = int(non_empty_rows[0].item())
        compressed[col, 0] = column[first_row].to(torch.float32)
    return compressed


def extract_arch_aware_architecture(device: str | None = None, *, clock_region: str | None = None) -> dict[str, object]:
    device_name = normalize_device_name(device)
    profile = resolve_board_profile(device_name)
    device_class = require_rapidwright_device()
    raw_device = device_class.getDevice(device_name)
    if raw_device is None:
        raise RuntimeError("Could not load RapidWright device {!r}".format(device_name))

    selected_clock_region = _select_clock_region(raw_device, clock_region)
    cells: dict[tuple[int, int], list[int]] = {}
    for tile in _iter_tiles(raw_device):
        if selected_clock_region is not None and hasattr(selected_clock_region, "containsTile"):
            if not bool(selected_clock_region.containsTile(tile)):
                continue
        tile_class = classify_arch_aware_tile(tile)
        if tile_class is None:
            continue
        x, y = _tile_x_y(tile)
        cells.setdefault((x, y), []).append(ARCH_AWARE_TILE_TYPE_TO_ID[tile_class])

    if not cells:
        region_name = _clock_region_name(selected_clock_region)
        raise RuntimeError("RapidWright device {!r} produced no arch-aware tiles for {}".format(device_name, region_name))

    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_x = max(x for x, _ in cells)
    max_y = max(y for _, y in cells)
    raw_cols = max_x - min_x + 1
    raw_rows = max_y - min_y + 1

    raw_layout = torch.zeros(
        (ARCH_AWARE_LAYOUT_COLS, ARCH_AWARE_LAYOUT_ROWS, ARCH_AWARE_TILE_SLOTS),
        dtype=torch.long,
    )
    valid_cols = min(raw_cols, ARCH_AWARE_LAYOUT_COLS)
    valid_rows = min(raw_rows, ARCH_AWARE_LAYOUT_ROWS)
    for (x, y), tile_ids in cells.items():
        col = x - min_x
        row = y - min_y
        if col >= ARCH_AWARE_LAYOUT_COLS or row >= ARCH_AWARE_LAYOUT_ROWS:
            continue
        for slot, tile_id in enumerate(tile_ids[:ARCH_AWARE_TILE_SLOTS]):
            raw_layout[col, row, slot] = int(tile_id)

    positional_encoding = sinusoidal_positional_encoding()
    compressed = _compress_layout(raw_layout) + positional_encoding
    fsr_row, fsr_col = _clock_region_position(selected_clock_region)
    fsr_count = max(len(_clock_regions(raw_device)), 1)
    metadata = build_arch_aware_metadata(
        profile,
        raw_rows=raw_rows,
        raw_cols=raw_cols,
        valid_rows=valid_rows,
        valid_cols=valid_cols,
        fsr_row=fsr_row,
        fsr_col=fsr_col,
        fsr_count=fsr_count,
    )

    return {
        "device": device_name,
        "arch_aware_arch_layout_raw": raw_layout,
        "arch_aware_arch_layout": compressed.to(torch.float32),
        "arch_aware_arch_metadata": metadata,
        "arch_aware_arch_positional_encoding": positional_encoding,
        "arch_aware_valid_rows": valid_rows,
        "arch_aware_valid_cols": valid_cols,
        "arch_aware_clock_region": _clock_region_name(selected_clock_region),
        "arch_aware_tile_type_to_id": dict(ARCH_AWARE_TILE_TYPE_TO_ID),
    }


def arch_aware_arch_cache_path(device: str | None = None, *, cache_dir: str | Path | None = None) -> Path:
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_ARCH_AWARE_CACHE_DIR
    return cache_root / "{}_arch_aware_arch.pt".format(normalize_device_name(device))


def ensure_arch_aware_arch_cache(
    device: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    force: bool = False,
    clock_region: str | None = None,
) -> Path:
    output_path = arch_aware_arch_cache_path(device, cache_dir=cache_dir)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = extract_arch_aware_architecture(device, clock_region=clock_region)
    torch.save(payload, output_path)
    return output_path


def load_arch_aware_arch_cache(path_or_device: str | Path | None = None, *, cache_dir: str | Path | None = None) -> dict[str, object]:
    candidate = Path(path_or_device) if path_or_device is not None else arch_aware_arch_cache_path(DEFAULT_BOARD_DEVICE, cache_dir=cache_dir)
    if candidate.suffix != ".pt":
        candidate = arch_aware_arch_cache_path(str(path_or_device), cache_dir=cache_dir)
    if not candidate.exists():
        candidate = ensure_arch_aware_arch_cache(
            str(path_or_device) if path_or_device is not None else DEFAULT_BOARD_DEVICE,
            cache_dir=cache_dir,
        )
    return torch.load(candidate, map_location="cpu", weights_only=False)


def arch_aware_arch_to_device(payload: dict[str, object], device: torch.device) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            converted[key] = value.to(device)
        else:
            converted[key] = value
    return converted
