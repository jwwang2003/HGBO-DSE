from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from hgp.board_utils import DEFAULT_BOARD_DEVICE, resolve_board_profile  # noqa: E402
from hgp import arch_aware_arch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class FakeTile:
    def __init__(self, name, tile_type=None, column=None, row=None):
        self._name = name
        self._tile_type = tile_type if tile_type is not None else name.split("_X", 1)[0]
        self._column = column
        self._row = row

    def getName(self):
        return self._name

    def getTileTypeEnum(self):
        return self._tile_type

    def getColumn(self):
        if self._column is None:
            raise AttributeError("column not set")
        return self._column

    def getRow(self):
        if self._row is None:
            raise AttributeError("row not set")
        return self._row


class FakeClockRegion:
    def __init__(self, name, allowed_names):
        self._name = name
        self._allowed_names = set(allowed_names)

    def getName(self):
        return self._name

    def containsTile(self, tile):
        return tile.getName() in self._allowed_names


class FakeDevice:
    def __init__(self, tiles, clock_regions, *, expose_tiles=True):
        self._tiles = tiles
        self._clock_regions = clock_regions
        self._expose_tiles = expose_tiles

    def getClockRegions(self):
        return self._clock_regions

    def getClockRegion(self, name):
        return next(region for region in self._clock_regions if region.getName() == name)

    def getTiles(self):
        if not self._expose_tiles:
            raise AttributeError("getTiles unavailable")
        return self._tiles

    def getAllTiles(self):
        return [tile for row in self._tiles for tile in row if tile is not None]


class FakeNestedClockRegionDevice(FakeDevice):
    def getClockRegions(self):
        return [self._clock_regions]


def test_classify_arch_aware_tile_reduces_rapidwright_names_to_fixed_categories():
    assert arch_aware_arch.classify_arch_aware_tile("CLEL_R_X0Y0") == "CLEL"
    assert arch_aware_arch.classify_arch_aware_tile("CLBLM_L_X0Y0") == "CLEM"
    assert arch_aware_arch.classify_arch_aware_tile("CLEM_X0Y0") == "CLEM"
    assert arch_aware_arch.classify_arch_aware_tile("INT_INTERFACE_L_X0Y0") == "INT_INTERFACE"
    assert arch_aware_arch.classify_arch_aware_tile("INT_X0Y0") == "INT"
    assert arch_aware_arch.classify_arch_aware_tile("DSP_R_X0Y0") == "DSP"
    assert arch_aware_arch.classify_arch_aware_tile("BRAM_X0Y0") == "BRAM"
    assert arch_aware_arch.classify_arch_aware_tile(FakeTile("DSP_X0Y0", tile_type="NULL")) == "DSP"
    assert arch_aware_arch.classify_arch_aware_tile(FakeTile("BRAM_X0Y0", tile_type="NULL")) == "BRAM"
    assert arch_aware_arch.classify_arch_aware_tile(FakeTile("UNUSED_X0Y0", tile_type="NULL")) == "BRK"
    assert arch_aware_arch.classify_arch_aware_tile("PAD_X0Y0") is None


def test_sinusoidal_positional_encoding_shape_and_initial_values():
    encoding = arch_aware_arch.sinusoidal_positional_encoding()

    assert encoding.shape == (
        arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
        1,
        arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
    )
    assert encoding.dtype == torch.float32
    assert encoding[0, 0].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_build_arch_aware_metadata_uses_normalized_board_profile_fields():
    profile = resolve_board_profile(DEFAULT_BOARD_DEVICE)

    metadata = arch_aware_arch.build_arch_aware_metadata(
        profile,
        raw_rows=40,
        raw_cols=180,
        valid_rows=20,
        valid_cols=90,
        fsr_row=1,
        fsr_col=2,
        fsr_count=4,
    )

    assert metadata.shape == (1, arch_aware_arch.ARCH_AWARE_METADATA_DIM)
    assert metadata.dtype == torch.float32
    assert metadata[0, 0].item() == pytest.approx(0.3036)
    assert metadata[0, 1].item() == pytest.approx(0.6072)
    assert metadata[0, 2].item() == pytest.approx(2.8)
    assert metadata[0, 3].item() == pytest.approx(1.03)
    assert metadata[0, 4].item() == pytest.approx(0.28)
    assert metadata[0, 5].item() == pytest.approx(1.0)
    assert metadata[0, 6].item() == pytest.approx(40 / arch_aware_arch.ARCH_AWARE_LAYOUT_ROWS)
    assert metadata[0, 7].item() == pytest.approx(180 / arch_aware_arch.ARCH_AWARE_LAYOUT_COLS)
    assert metadata[0, 17].item() == pytest.approx(1.0)
    assert metadata[0, 18].item() == pytest.approx(0.0)
    assert metadata[0, 19].item() == pytest.approx(0.0)


def test_extract_arch_aware_architecture_with_fake_device(monkeypatch):
    selected_tiles = [
        FakeTile("CLEM_X10Y20"),
        FakeTile("INT_X10Y20"),
        FakeTile("CLEL_X11Y20"),
        FakeTile("BRAM_X12Y21"),
    ]
    ignored_tile = FakeTile("DSP_X99Y99")
    clock_region = FakeClockRegion(
        "CR_X0Y0",
        [tile.getName() for tile in selected_tiles],
    )
    fake_device = FakeDevice(
        [
            [selected_tiles[0], selected_tiles[1]],
            [selected_tiles[2], None],
            [selected_tiles[3], ignored_tile],
        ],
        [clock_region],
    )
    fake_device_class = SimpleNamespace(getDevice=lambda device_name: fake_device)
    monkeypatch.setattr(arch_aware_arch, "require_rapidwright_device", lambda: fake_device_class)

    payload = arch_aware_arch.extract_arch_aware_architecture(DEFAULT_BOARD_DEVICE)

    assert payload["device"] == DEFAULT_BOARD_DEVICE
    assert payload["arch_aware_clock_region"] == "CR_X0Y0"
    assert payload["arch_aware_arch_layout_raw"].shape == (
        arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
        arch_aware_arch.ARCH_AWARE_LAYOUT_ROWS,
        arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
    )
    assert payload["arch_aware_arch_layout_raw"].dtype == torch.long
    assert payload["arch_aware_arch_layout"].shape == (
        arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
        1,
        arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
    )
    assert payload["arch_aware_arch_layout"].dtype == torch.float32
    assert payload["arch_aware_arch_metadata"].shape == (1, arch_aware_arch.ARCH_AWARE_METADATA_DIM)
    assert payload["arch_aware_arch_positional_encoding"].shape == (
        arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
        1,
        arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
    )
    assert payload["arch_aware_valid_rows"] == 2
    assert payload["arch_aware_valid_cols"] == 3
    assert payload["arch_aware_arch_layout_raw"][0, 0].tolist() == [
        arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["CLEM"],
        arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["INT"],
        0,
        0,
    ]
    assert payload["arch_aware_arch_layout_raw"][1, 0, 0].item() == arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["CLEL"]
    assert payload["arch_aware_arch_layout_raw"][2, 1, 0].item() == arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["BRAM"]

    compressed_ids = payload["arch_aware_arch_layout"] - payload["arch_aware_arch_positional_encoding"]
    assert compressed_ids[0, 0].tolist() == pytest.approx([
        arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["CLEM"],
        arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["INT"],
        0.0,
        0.0,
    ])
    assert compressed_ids[2, 0, 0].item() == pytest.approx(arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["BRAM"])


def test_extract_arch_aware_architecture_selects_central_clock_region_by_default(monkeypatch):
    edge_tile = FakeTile("DSP_X0Y0")
    central_tile = FakeTile("INT_X5Y5")
    fake_device = FakeDevice(
        [[edge_tile, central_tile]],
        [
            FakeClockRegion("CR_X0Y0", [edge_tile.getName()]),
            FakeClockRegion("CR_X1Y1", [central_tile.getName()]),
            FakeClockRegion("CR_X2Y2", []),
        ],
    )
    fake_device_class = SimpleNamespace(getDevice=lambda device_name: fake_device)
    monkeypatch.setattr(arch_aware_arch, "require_rapidwright_device", lambda: fake_device_class)

    payload = arch_aware_arch.extract_arch_aware_architecture(DEFAULT_BOARD_DEVICE)

    assert payload["arch_aware_clock_region"] == "CR_X1Y1"
    assert payload["arch_aware_arch_layout_raw"][0, 0, 0].item() == arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["INT"]


def test_extract_arch_aware_architecture_flattens_nested_clock_region_arrays(monkeypatch):
    edge_tile = FakeTile("DSP_X0Y0")
    central_tile = FakeTile("INT_X5Y5")
    fake_device = FakeNestedClockRegionDevice(
        [[edge_tile, central_tile]],
        [
            FakeClockRegion("CR_X0Y0", [edge_tile.getName()]),
            FakeClockRegion("CR_X1Y1", [central_tile.getName()]),
            FakeClockRegion("CR_X2Y2", []),
        ],
    )
    fake_device_class = SimpleNamespace(getDevice=lambda device_name: fake_device)
    monkeypatch.setattr(arch_aware_arch, "require_rapidwright_device", lambda: fake_device_class)

    payload = arch_aware_arch.extract_arch_aware_architecture(DEFAULT_BOARD_DEVICE)

    assert payload["arch_aware_clock_region"] == "CR_X1Y1"
    assert payload["arch_aware_valid_rows"] == 1
    assert payload["arch_aware_valid_cols"] == 1


def test_extract_arch_aware_architecture_supports_get_all_tiles(monkeypatch):
    tiles = [[FakeTile("INT_X3Y7")]]
    clock_region = FakeClockRegion("CR_X0Y0", ["INT_X3Y7"])
    fake_device = FakeDevice(tiles, [clock_region], expose_tiles=False)
    fake_device_class = SimpleNamespace(getDevice=lambda device_name: fake_device)
    monkeypatch.setattr(arch_aware_arch, "require_rapidwright_device", lambda: fake_device_class)

    payload = arch_aware_arch.extract_arch_aware_architecture(DEFAULT_BOARD_DEVICE, clock_region="CR_X0Y0")

    assert payload["arch_aware_arch_layout_raw"][0, 0, 0].item() == arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID["INT"]
    assert payload["arch_aware_clock_region"] == "CR_X0Y0"


def test_arch_aware_arch_cache_round_trips_payload_with_tensors(tmp_path, monkeypatch):
    payload = {
        "device": DEFAULT_BOARD_DEVICE,
        "arch_aware_arch_layout_raw": torch.zeros(
            (
                arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
                arch_aware_arch.ARCH_AWARE_LAYOUT_ROWS,
                arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
            ),
            dtype=torch.long,
        ),
        "arch_aware_arch_layout": torch.ones(
            (
                arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
                1,
                arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
            ),
            dtype=torch.float32,
        ),
        "arch_aware_arch_metadata": torch.arange(arch_aware_arch.ARCH_AWARE_METADATA_DIM, dtype=torch.float32).view(1, -1),
        "arch_aware_arch_positional_encoding": torch.zeros(
            (
                arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
                1,
                arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
            ),
            dtype=torch.float32,
        ),
        "arch_aware_valid_rows": 1,
        "arch_aware_valid_cols": 1,
        "arch_aware_clock_region": None,
        "arch_aware_tile_type_to_id": dict(arch_aware_arch.ARCH_AWARE_TILE_TYPE_TO_ID),
    }
    monkeypatch.setattr(arch_aware_arch, "extract_arch_aware_architecture", lambda device=None, **kwargs: payload)

    cache_path = arch_aware_arch.ensure_arch_aware_arch_cache(DEFAULT_BOARD_DEVICE, cache_dir=tmp_path)
    loaded = arch_aware_arch.load_arch_aware_arch_cache(cache_path)
    moved = arch_aware_arch.arch_aware_arch_to_device(loaded, torch.device("cpu"))

    assert cache_path == tmp_path / f"{DEFAULT_BOARD_DEVICE}_arch_aware_arch.pt"
    assert loaded["device"] == DEFAULT_BOARD_DEVICE
    assert torch.equal(loaded["arch_aware_arch_layout_raw"], payload["arch_aware_arch_layout_raw"])
    assert torch.equal(loaded["arch_aware_arch_layout"], payload["arch_aware_arch_layout"])
    assert torch.equal(loaded["arch_aware_arch_metadata"], payload["arch_aware_arch_metadata"])
    assert all(value.device.type == "cpu" for value in moved.values() if torch.is_tensor(value))


def test_cached_architecture_loads_do_not_initialize_rapidwright(tmp_path):
    arch_cache = tmp_path / f"{DEFAULT_BOARD_DEVICE}_arch_aware_arch.pt"
    fabric_cache = tmp_path / f"{DEFAULT_BOARD_DEVICE}_fabric_graph.pt"
    torch.save(
        {
            "device": DEFAULT_BOARD_DEVICE,
            "arch_aware_arch_layout": torch.ones(
                (
                    arch_aware_arch.ARCH_AWARE_LAYOUT_COLS,
                    1,
                    arch_aware_arch.ARCH_AWARE_TILE_SLOTS,
                ),
                dtype=torch.float32,
            ),
            "arch_aware_arch_metadata": torch.ones((1, arch_aware_arch.ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
        },
        arch_cache,
    )
    torch.save(
        {
            "device": DEFAULT_BOARD_DEVICE,
            "arch_x": torch.ones((1, 24), dtype=torch.float32),
            "arch_edge_index": torch.empty((2, 0), dtype=torch.long),
            "arch_edge_attr": torch.empty((0, 4), dtype=torch.float32),
            "arch_graph_attr": torch.ones((1, 32), dtype=torch.float32),
        },
        fabric_cache,
    )

    script = f"""
import sys
from hgp.arch_aware_arch import load_arch_aware_arch_cache
from hgp.board_fabric import load_board_fabric_cache

load_arch_aware_arch_cache({str(arch_cache)!r})
load_board_fabric_cache({str(fabric_cache)!r})
if "hgp.rapidwright_env" in sys.modules:
    raise SystemExit("RapidWright environment initialized while loading existing caches")
"""

    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)
