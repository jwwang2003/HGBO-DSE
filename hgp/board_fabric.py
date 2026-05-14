from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from hgp.board_utils import DEFAULT_BOARD_DEVICE, BoardProfile, normalize_device_name, resolve_board_profile


TILE_CLASS_ORDER = ("CLEL", "CLEM", "INT", "INT_INTERFACE", "DSP", "BRAM", "BRK")
SITE_CLASS_ORDER = (
    "SLICE",
    "DSP",
    "BRAM",
    "IO",
    "FIFO",
    "BUF",
    "PLL",
    "MMCM",
    "GT",
    "PCIE",
    "XADC",
    "SYSMON",
    "OTHER",
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "dataset" / "board_arch"


def require_rapidwright_device() -> object:
    from hgp.rapidwright_env import require_rapidwright_device as _require_rapidwright_device

    return _require_rapidwright_device()


@dataclass(frozen=True)
class BoardFabricGraph:
    device: str
    node_features: list[list[float]]
    edge_index: list[tuple[int, int]]
    edge_features: list[list[float]]
    graph_features: list[float]
    node_coords: list[tuple[int, int]]
    layout_rows: int
    layout_cols: int


def _tokens(value: object) -> set[str]:
    normalized = str(value).upper().replace("-", "_").replace(".", "_")
    return {token for token in normalized.split("_") if token}


def classify_tile(tile: object) -> str | None:
    tile_type = tile.getTileTypeEnum() if hasattr(tile, "getTileTypeEnum") else ""
    tokens = _tokens(tile.getName()) | _tokens(tile_type)

    if str(tile_type).upper() == "NULL":
        return "BRK"
    if "BRAM" in tokens:
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
    return None


def classify_site(site: object) -> str:
    site_type = str(site.getSiteTypeEnum()).upper() if hasattr(site, "getSiteTypeEnum") else ""
    site_name = str(site.getName()).upper() if hasattr(site, "getName") else ""

    if "SLICEL" in site_type or "SLICEM" in site_type or site_type.startswith("SLICE"):
        return "SLICE"
    if "DSP" in site_type:
        return "DSP"
    if "RAMB" in site_type or "BRAM" in site_type:
        return "BRAM"
    if "IOB" in site_type or site_type.startswith("IO") or "IOB" in site_name:
        return "IO"
    if "FIFO" in site_type:
        return "FIFO"
    if "BUFG" in site_type or "BUFH" in site_type or site_type.startswith("BUF"):
        return "BUF"
    if "PLL" in site_type:
        return "PLL"
    if "MMCM" in site_type:
        return "MMCM"
    if site_type.startswith("GT"):
        return "GT"
    if "PCIE" in site_type:
        return "PCIE"
    if "XADC" in site_type:
        return "XADC"
    if "SYSMON" in site_type:
        return "SYSMON"
    return "OTHER"


def _tile_grid_x(tile: object) -> int:
    if hasattr(tile, "getColumn"):
        return int(tile.getColumn())
    if hasattr(tile, "getTileXCoordinate"):
        return int(tile.getTileXCoordinate())
    return 0


def _tile_grid_y(tile: object) -> int:
    if hasattr(tile, "getRow"):
        return int(tile.getRow())
    if hasattr(tile, "getTileYCoordinate"):
        return int(tile.getTileYCoordinate())
    return 0


def _iter_tiles(device: object) -> Iterable[object]:
    if hasattr(device, "getTiles"):
        for row in device.getTiles():
            for tile in row:
                if tile is not None:
                    yield tile
        return
    for tile in device.getAllTiles():
        if tile is not None:
            yield tile


def _profile_graph_features(profile: BoardProfile, layout_rows: int, layout_cols: int) -> list[float]:
    return [
        profile.lut_count / 1_000_000.0,
        profile.ff_count / 1_000_000.0,
        profile.dsp_count / 1_000.0,
        profile.bram_count / 1_000.0,
        profile.tech_node_nm / 100.0,
        profile.vccint,
        float(layout_rows),
        float(layout_cols),
    ]


def _node_features(cell: dict[str, object], max_x: int, max_y: int) -> list[float]:
    tile_counts = {name: 0.0 for name in TILE_CLASS_ORDER}
    for tile_class in cell["tile_classes"]:
        tile_counts[str(tile_class)] += 1.0

    site_counts = {name: 0.0 for name in SITE_CLASS_ORDER}
    for site_class in cell["site_classes"]:
        site_counts[str(site_class)] += 1.0

    x = int(cell["x"])
    y = int(cell["y"])
    return [
        *(tile_counts[name] for name in TILE_CLASS_ORDER),
        *(site_counts[name] for name in SITE_CLASS_ORDER),
        float(len(cell["tile_classes"])),
        float(len(cell["site_classes"])),
        float(x) / max(max_x, 1),
        float(y) / max(max_y, 1),
    ]


def extract_board_fabric(device: str | None = None) -> BoardFabricGraph:
    device_name = normalize_device_name(device)
    profile = resolve_board_profile(device_name)
    device_class = require_rapidwright_device()
    raw_device = device_class.getDevice(device_name)
    if raw_device is None:
        raise RuntimeError(f"Could not load RapidWright device {device_name!r}")

    cells: dict[tuple[int, int], dict[str, object]] = {}
    for tile in _iter_tiles(raw_device):
        tile_class = classify_tile(tile)
        if tile_class is None:
            continue
        x = _tile_grid_x(tile)
        y = _tile_grid_y(tile)
        cell = cells.setdefault((x, y), {"x": x, "y": y, "tile_classes": [], "site_classes": []})
        cell["tile_classes"].append(tile_class)
        raw_sites = tile.getSites() if hasattr(tile, "getSites") else ()
        for site in raw_sites or ():
            if site is not None:
                cell["site_classes"].append(classify_site(site))

    if not cells:
        raise RuntimeError(f"RapidWright device {device_name!r} produced no fabric cells")

    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    rebased: dict[tuple[int, int], dict[str, object]] = {}
    for (x, y), cell in cells.items():
        rebased_cell = dict(cell)
        rebased_cell["x"] = x - min_x
        rebased_cell["y"] = y - min_y
        rebased[(x - min_x, y - min_y)] = rebased_cell

    coords = sorted(rebased)
    coord_to_id = {coord: index for index, coord in enumerate(coords)}
    max_x = max(x for x, _ in coords)
    max_y = max(y for _, y in coords)
    node_features = [_node_features(rebased[coord], max_x, max_y) for coord in coords]
    tile_hist = {name: 0 for name in TILE_CLASS_ORDER}
    site_hist = {name: 0 for name in SITE_CLASS_ORDER}
    for cell in rebased.values():
        for tile_class in cell["tile_classes"]:
            tile_hist[str(tile_class)] += 1
        for site_class in cell["site_classes"]:
            site_hist[str(site_class)] += 1

    edge_index: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for x, y in coords:
        src = coord_to_id[(x, y)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            dst = coord_to_id.get((x + dx, y + dy))
            if dst is None:
                continue
            edge_index.append((src, dst))
            edge_features.append([float(dx), float(dy), float(abs(dx) + abs(dy)), 1.0])

    layout_rows = max_y + 1
    layout_cols = max_x + 1
    node_count = len(node_features)
    edge_count = len(edge_index)
    possible_edges = max(node_count * max(node_count - 1, 1), 1)
    return BoardFabricGraph(
        device=device_name,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        graph_features=[
            *(_profile_graph_features(profile, layout_rows, layout_cols)),
            float(node_count),
            float(edge_count),
            float(edge_count) / max(node_count, 1),
            float(edge_count) / float(possible_edges),
            *(float(tile_hist[name]) / max(node_count, 1) for name in TILE_CLASS_ORDER),
            *(float(site_hist[name]) / max(node_count, 1) for name in SITE_CLASS_ORDER),
        ],
        node_coords=coords,
        layout_rows=layout_rows,
        layout_cols=layout_cols,
    )


def board_fabric_cache_path(device: str | None = None, *, cache_dir: str | Path | None = None) -> Path:
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return cache_root / f"{normalize_device_name(device)}_fabric_graph.pt"


def _graph_to_payload(graph: BoardFabricGraph) -> dict[str, object]:
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32)
    if edge_attr.numel() == 0:
        edge_attr = torch.empty((0, 4), dtype=torch.float32)

    return {
        "device": graph.device,
        "arch_x": torch.tensor(graph.node_features, dtype=torch.float32),
        "arch_edge_index": edge_index,
        "arch_edge_attr": edge_attr,
        "arch_graph_attr": torch.tensor([graph.graph_features], dtype=torch.float32),
        "arch_node_coords": torch.tensor(graph.node_coords, dtype=torch.long),
        "layout_rows": graph.layout_rows,
        "layout_cols": graph.layout_cols,
    }


def ensure_board_fabric_cache(device: str | None = None, *, cache_dir: str | Path | None = None, force: bool = False) -> Path:
    output_path = board_fabric_cache_path(device, cache_dir=cache_dir)
    if output_path.exists() and not force:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = extract_board_fabric(device)
    torch.save(_graph_to_payload(graph), output_path)
    return output_path


def load_board_fabric_cache(path_or_device: str | Path | None = None, *, cache_dir: str | Path | None = None) -> dict[str, object]:
    candidate = Path(path_or_device) if path_or_device is not None else board_fabric_cache_path(DEFAULT_BOARD_DEVICE, cache_dir=cache_dir)
    if candidate.suffix != ".pt":
        candidate = board_fabric_cache_path(str(path_or_device), cache_dir=cache_dir)
    if not candidate.exists():
        candidate = ensure_board_fabric_cache(str(path_or_device) if path_or_device is not None else DEFAULT_BOARD_DEVICE, cache_dir=cache_dir)
    return torch.load(candidate, map_location="cpu", weights_only=False)


def board_fabric_to_device(payload: dict[str, object], device: torch.device) -> dict[str, object]:
    converted: dict[str, object] = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            converted[key] = value.to(device)
        else:
            converted[key] = value
    return converted
