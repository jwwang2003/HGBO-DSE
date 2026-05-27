from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import torch

from hgp.board_utils import (
    BoardProfile,
    DEFAULT_BOARD_DEVICE,
    board_family_one_hot,
    normalize_device_name,
    resolve_board_profile,
)


ARCH_AWARE_LAYOUT_COLS = 360
ARCH_AWARE_LAYOUT_ROWS = 80
ARCH_AWARE_TILE_SLOTS = 4
ARCH_AWARE_METADATA_DIM = 21
DEFAULT_ARCH_AWARE_CACHE_DIR = Path(__file__).resolve().parents[1] / "dataset" / "board_arch"
ARCH_AWARE_SVG_FILENAMES = {
    "full_device_fabric": "{device}_arch_aware_full_device_fabric.svg",
    "clock_region_fabric": "{device}_arch_aware_clock_region_fabric.svg",
    "compressed_fabric": "{device}_arch_aware_compressed_fabric.svg",
}

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
ARCH_AWARE_TILE_COLORS = {
    "PAD": "#f8fafc",
    "CLEL": "#3b82f6",
    "CLEM": "#22c55e",
    "INT": "#94a3b8",
    "INT_INTERFACE": "#f59e0b",
    "DSP": "#ef4444",
    "BRAM": "#8b5cf6",
    "BRK": "#111827",
}

_XY_RE = re.compile(r"(?:^|_)X(?P<x>-?\d+)Y(?P<y>-?\d+)(?:_|$)")
_CLOCK_REGION_RE = re.compile(r"X(?P<x>-?\d+)Y(?P<y>-?\d+)")


def require_rapidwright_device() -> object:
    from hgp.rapidwright_env import require_rapidwright_device as _require_rapidwright_device

    return _require_rapidwright_device()


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
    is_series7, is_ultrascale, is_ultrascale_plus = board_family_one_hot(profile.family)
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
        is_series7,
        is_ultrascale,
        is_ultrascale_plus,
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


def _extract_arch_aware_cells(
    raw_device: object,
    *,
    clock_region: object | None = None,
) -> dict[tuple[int, int], list[int]]:
    cells: dict[tuple[int, int], list[int]] = {}
    for tile in _iter_tiles(raw_device):
        if clock_region is not None and hasattr(clock_region, "containsTile"):
            if not bool(clock_region.containsTile(tile)):
                continue
        tile_class = classify_arch_aware_tile(tile)
        if tile_class is None:
            continue
        x, y = _tile_x_y(tile)
        cells.setdefault((x, y), []).append(ARCH_AWARE_TILE_TYPE_TO_ID[tile_class])
    return cells


def _cells_bounds(cells: dict[tuple[int, int], list[int]]) -> tuple[int, int, int, int]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    max_x = max(x for x, _ in cells)
    max_y = max(y for _, y in cells)
    return min_x, min_y, max_x, max_y


def _slot_tile_ids(tile_ids: Iterable[int]) -> list[int]:
    values = [int(tile_id) for tile_id in tile_ids if int(tile_id) != ARCH_AWARE_TILE_TYPE_TO_ID["PAD"]]
    if not values:
        return [ARCH_AWARE_TILE_TYPE_TO_ID["PAD"]]
    return values[:ARCH_AWARE_TILE_SLOTS]


def _svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_svg_header(
    *,
    title: str,
    subtitle: str,
    width: int,
    height: int,
) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(
            width, height
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="18" y="26" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#111827">{}</text>'.format(
            _svg_escape(title)
        ),
        '<text x="18" y="46" font-family="Arial, sans-serif" font-size="12" fill="#475569">{}</text>'.format(
            _svg_escape(subtitle)
        ),
    ]


def _append_svg_legend(lines: list[str], *, x: int, y: int) -> None:
    cursor_x = x
    for tile_type in ARCH_AWARE_TILE_TYPE_TO_ID:
        color = ARCH_AWARE_TILE_COLORS[tile_type]
        lines.append(
            '<rect x="{0}" y="{1}" width="12" height="12" fill="{2}" stroke="#ffffff" stroke-width="1"/>'.format(
                cursor_x, y, color
            )
        )
        lines.append(
            '<text x="{0}" y="{1}" font-family="Arial, sans-serif" font-size="11" fill="#334155">{2}</text>'.format(
                cursor_x + 16, y + 10, _svg_escape(tile_type)
            )
        )
        cursor_x += 16 + len(tile_type) * 7 + 14


def _write_cells_svg(
    cells: dict[tuple[int, int], list[int]],
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    cell_size: int = 4,
) -> Path:
    if not cells:
        raise ValueError("cannot render empty fabric cells")
    min_x, min_y, max_x, max_y = _cells_bounds(cells)
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1
    margin_left = 18
    margin_top = 70
    width = max(760, margin_left * 2 + cols * cell_size)
    height = margin_top + rows * cell_size + 52
    lines = _render_svg_header(title=title, subtitle=subtitle, width=width, height=height)
    _append_svg_legend(lines, x=18, y=height - 30)
    lines.append(
        '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>'.format(
            margin_left, margin_top, cols * cell_size, rows * cell_size
        )
    )
    for (x, y), tile_ids in sorted(cells.items()):
        col = x - min_x
        row = max_y - y
        slots = _slot_tile_ids(tile_ids)
        slot_width = cell_size / len(slots)
        for slot_index, tile_id in enumerate(slots):
            tile_type = ARCH_AWARE_TILE_ID_TO_TYPE.get(tile_id, "PAD")
            color = ARCH_AWARE_TILE_COLORS[tile_type]
            lines.append(
                '<rect x="{0:.2f}" y="{1}" width="{2:.2f}" height="{3}" fill="{4}"/>'.format(
                    margin_left + col * cell_size + slot_index * slot_width,
                    margin_top + row * cell_size,
                    slot_width,
                    cell_size,
                    color,
                )
            )
    lines.append("</svg>")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_layout_svg(
    raw_layout: torch.Tensor,
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    valid_cols: int,
    valid_rows: int,
    cell_size: int = 6,
) -> Path:
    valid_cols = min(int(valid_cols), raw_layout.shape[0])
    valid_rows = min(int(valid_rows), raw_layout.shape[1])
    margin_left = 18
    margin_top = 70
    width = max(760, margin_left * 2 + valid_cols * cell_size)
    height = margin_top + valid_rows * cell_size + 52
    lines = _render_svg_header(title=title, subtitle=subtitle, width=width, height=height)
    _append_svg_legend(lines, x=18, y=height - 30)
    lines.append(
        '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>'.format(
            margin_left, margin_top, valid_cols * cell_size, valid_rows * cell_size
        )
    )
    for col in range(valid_cols):
        for row in range(valid_rows):
            slots = _slot_tile_ids(int(tile_id) for tile_id in raw_layout[col, row].view(-1))
            if slots != [ARCH_AWARE_TILE_TYPE_TO_ID["PAD"]]:
                slot_width = cell_size / len(slots)
                svg_row = valid_rows - row - 1
                for slot_index, tile_id in enumerate(slots):
                    tile_type = ARCH_AWARE_TILE_ID_TO_TYPE.get(tile_id, "PAD")
                    color = ARCH_AWARE_TILE_COLORS[tile_type]
                    lines.append(
                        '<rect x="{0:.2f}" y="{1}" width="{2:.2f}" height="{3}" fill="{4}"/>'.format(
                            margin_left + col * cell_size + slot_index * slot_width,
                            margin_top + svg_row * cell_size,
                            slot_width,
                            cell_size,
                            color,
                        )
                    )
    lines.append("</svg>")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _write_compressed_layout_svg(
    compressed_layout: torch.Tensor,
    output_path: str | Path,
    *,
    title: str,
    subtitle: str,
    valid_cols: int,
    cell_size: int = 5,
) -> Path:
    layout = compressed_layout.detach().cpu()
    positional_encoding = sinusoidal_positional_encoding().to(layout.dtype)
    if layout.shape == positional_encoding.shape:
        layout = layout - positional_encoding
    valid_cols = min(int(valid_cols), layout.shape[0])
    rows = ARCH_AWARE_TILE_SLOTS
    margin_left = 18
    margin_top = 70
    width = max(760, margin_left * 2 + valid_cols * cell_size)
    height = margin_top + rows * cell_size + 52
    lines = _render_svg_header(title=title, subtitle=subtitle, width=width, height=height)
    _append_svg_legend(lines, x=18, y=height - 30)
    lines.append(
        '<rect x="{0}" y="{1}" width="{2}" height="{3}" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>'.format(
            margin_left, margin_top, valid_cols * cell_size, rows * cell_size
        )
    )
    for col in range(valid_cols):
        slot_values = layout[col, 0].round().to(torch.long).view(-1)
        for slot_index in range(rows):
            tile_id = int(slot_values[slot_index].item())
            tile_type = ARCH_AWARE_TILE_ID_TO_TYPE.get(tile_id, "PAD")
            color = ARCH_AWARE_TILE_COLORS[tile_type]
            lines.append(
                '<rect x="{0}" y="{1}" width="{2}" height="{2}" fill="{3}"/>'.format(
                    margin_left + col * cell_size,
                    margin_top + slot_index * cell_size,
                    cell_size,
                    color,
                )
            )
    lines.append("</svg>")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def arch_aware_svg_paths(device: str | None = None, *, cache_dir: str | Path | None = None) -> dict[str, Path]:
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_ARCH_AWARE_CACHE_DIR
    device_name = normalize_device_name(device)
    return {
        key: cache_root / template.format(device=device_name)
        for key, template in ARCH_AWARE_SVG_FILENAMES.items()
    }


def arch_aware_svgs_exist(device: str | None = None, *, cache_dir: str | Path | None = None) -> bool:
    return all(path.exists() for path in arch_aware_svg_paths(device, cache_dir=cache_dir).values())


def _export_arch_aware_svgs(
    *,
    device_name: str,
    full_cells: dict[tuple[int, int], list[int]],
    payload: dict[str, object],
    cache_dir: str | Path | None = None,
) -> dict[str, Path]:
    svg_paths = arch_aware_svg_paths(device_name, cache_dir=cache_dir)
    clock_region = payload.get("arch_aware_clock_region") or "full device"
    valid_cols = int(payload["arch_aware_valid_cols"])
    valid_rows = int(payload["arch_aware_valid_rows"])
    _write_cells_svg(
        full_cells,
        svg_paths["full_device_fabric"],
        title="{} full device fabric".format(device_name),
        subtitle="RapidWright tile classes before clock-region cropping",
        cell_size=2,
    )
    _write_layout_svg(
        payload["arch_aware_arch_layout_raw"],
        svg_paths["clock_region_fabric"],
        title="{} clock-region fabric".format(device_name),
        subtitle="{} rebased into {}x{} raw tensor".format(clock_region, valid_cols, valid_rows),
        valid_cols=valid_cols,
        valid_rows=valid_rows,
        cell_size=6,
    )
    _write_compressed_layout_svg(
        payload["arch_aware_arch_layout"],
        svg_paths["compressed_fabric"],
        title="{} compressed fabric".format(device_name),
        subtitle="First non-empty tile row per column plus sinusoidal positional encoding",
        valid_cols=valid_cols,
        cell_size=5,
    )
    return svg_paths


def _compress_layout(raw_layout: torch.Tensor) -> torch.Tensor:
    """Collapse the ``[cols, rows, slots]`` raw layout into ``[cols, 1, slots]``.

    For each column we keep only the first non-empty row's tile-id slots. This
    discards Y information on purpose — the downstream encoder treats the
    fabric as a 1D column sequence with positional encoding. Callers that need
    the full 2D layout should read ``arch_aware_arch_layout_raw`` instead.
    """
    compressed = torch.zeros((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32)
    for col in range(raw_layout.shape[0]):
        column = raw_layout[col]
        non_empty_rows = torch.nonzero(column.any(dim=1), as_tuple=False).view(-1)
        if non_empty_rows.numel() == 0:
            continue
        first_row = int(non_empty_rows[0].item())
        compressed[col, 0] = column[first_row].to(torch.float32)
    return compressed


def extract_arch_aware_architecture(
    device: str | None = None,
    *,
    clock_region: str | None = None,
    svg_cache_dir: str | Path | None = None,
) -> dict[str, object]:
    """Extract the arch-aware architecture payload for ``device``.

    Walks RapidWright's tile grid (optionally limited to a single clock region),
    classifies each tile via :func:`classify_arch_aware_tile`, and packs the
    resulting tile-id grid into a fixed ``[ARCH_AWARE_LAYOUT_COLS,
    ARCH_AWARE_LAYOUT_ROWS, ARCH_AWARE_TILE_SLOTS]`` raw tensor. The compressed
    1D-per-column layout (with positional encoding) and a 21-dim metadata vector
    derived via :func:`build_arch_aware_metadata` are returned alongside.

    A warning is printed if the device's tile grid exceeds the fixed layout
    dimensions — large parts (e.g. xcvu9p) are truncated silently otherwise.
    """
    device_name = normalize_device_name(device)
    profile = resolve_board_profile(device_name)
    device_class = require_rapidwright_device()
    raw_device = device_class.getDevice(device_name)
    if raw_device is None:
        raise RuntimeError("Could not load RapidWright device {!r}".format(device_name))

    selected_clock_region = _select_clock_region(raw_device, clock_region)
    full_cells = _extract_arch_aware_cells(raw_device)
    cells = _extract_arch_aware_cells(raw_device, clock_region=selected_clock_region)

    if not cells:
        region_name = _clock_region_name(selected_clock_region)
        raise RuntimeError("RapidWright device {!r} produced no arch-aware tiles for {}".format(device_name, region_name))

    min_x, min_y, max_x, max_y = _cells_bounds(cells)
    raw_cols = max_x - min_x + 1
    raw_rows = max_y - min_y + 1

    if raw_cols > ARCH_AWARE_LAYOUT_COLS or raw_rows > ARCH_AWARE_LAYOUT_ROWS:
        import warnings

        warnings.warn(
            "Device {!r} fabric ({}x{}) exceeds fixed arch-aware layout "
            "({}x{}); tiles outside the bounds are silently dropped.".format(
                device_name, raw_cols, raw_rows, ARCH_AWARE_LAYOUT_COLS, ARCH_AWARE_LAYOUT_ROWS,
            ),
            stacklevel=2,
        )

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

    payload: dict[str, object] = {
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
    if svg_cache_dir is not None:
        svg_paths = _export_arch_aware_svgs(
            device_name=device_name,
            full_cells=full_cells,
            payload=payload,
            cache_dir=svg_cache_dir,
        )
        payload["arch_aware_svg_paths"] = {key: str(path) for key, path in svg_paths.items()}
    return payload


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
    if output_path.exists() and not force and arch_aware_svgs_exist(device, cache_dir=cache_dir):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = extract_arch_aware_architecture(device, clock_region=clock_region, svg_cache_dir=output_path.parent)
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
