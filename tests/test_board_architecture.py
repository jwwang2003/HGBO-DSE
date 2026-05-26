import pytest
from pathlib import Path
from types import SimpleNamespace


torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")
Data = pyg_data.Data

from hgp.board_utils import (  # noqa: E402
    ARCH_ATTR_FIELDS,
    DEFAULT_BOARD_DEVICE,
    attach_board_profile,
    augment_dataset,
    board_feature_tensor,
    equivalent_device_names,
    resolve_board_profile,
)
from hgp.data_process.gen_dataset_board import augment_dataset_dirs  # noqa: E402
from hgp.data_process import gen_dataset_board  # noqa: E402
from hgp.data_process import gen_dataset_std  # noqa: E402
from hgp.data_process.gen_dataframe import generate_dataframe  # noqa: E402
from hgp.dataset_utils import generate_dataset  # noqa: E402
from hgp.arch_aware_arch import ARCH_AWARE_LAYOUT_COLS, ARCH_AWARE_METADATA_DIM, ARCH_AWARE_TILE_SLOTS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_default_board_profile_matches_raw_dataset_target():
    profile = resolve_board_profile("xc7vx485t-ffg1761-2")

    assert profile.device == DEFAULT_BOARD_DEVICE
    assert profile.family == "virtex7"
    assert profile.tech_node_nm == 28
    assert profile.vccint == pytest.approx(1.0)
    assert profile.lut_count == 303600
    assert profile.ff_count == 607200
    assert profile.dsp_count == 2800
    assert profile.bram_count == 1030

    arch_attr = board_feature_tensor(profile)

    assert arch_attr.shape == (1, len(ARCH_ATTR_FIELDS))
    assert arch_attr.dtype == torch.float32
    assert arch_attr[0, ARCH_ATTR_FIELDS.index("lut_m")].item() == pytest.approx(0.3036)
    assert arch_attr[0, ARCH_ATTR_FIELDS.index("ff_m")].item() == pytest.approx(0.6072)
    assert arch_attr[0, ARCH_ATTR_FIELDS.index("is_series7")].item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "device,expected_family,expected_one_hot_field",
    [
        ("xcku040_ffva1156_2_e", "kintex_ultrascale", "is_ultrascale"),
        ("xcku040-ffva1156-2-e", "kintex_ultrascale", "is_ultrascale"),
        ("xcvu9p-flga2104-2-i", "virtex_ultrascale_plus", "is_ultrascale_plus"),
        ("xczu9eg-ffvb1156-2-e", "zynq_ultrascale_plus", "is_ultrascale_plus"),
    ],
)
def test_thesis_board_profiles_classify_family_correctly(
    device, expected_family, expected_one_hot_field
):
    profile = resolve_board_profile(device)
    assert profile.family == expected_family

    arch_attr = board_feature_tensor(profile)
    one_hot_indices = {
        "is_series7": ARCH_ATTR_FIELDS.index("is_series7"),
        "is_ultrascale": ARCH_ATTR_FIELDS.index("is_ultrascale"),
        "is_ultrascale_plus": ARCH_ATTR_FIELDS.index("is_ultrascale_plus"),
    }
    set_value = arch_attr[0, one_hot_indices[expected_one_hot_field]].item()
    assert set_value == pytest.approx(1.0)
    for field, index in one_hot_indices.items():
        if field == expected_one_hot_field:
            continue
        assert arch_attr[0, index].item() == pytest.approx(0.0), (
            "Board {} should not flag {}".format(device, field)
        )


def test_equivalent_device_names_include_canonical_and_alias():
    assert equivalent_device_names("xcku040_ffva1156_2_e") == (
        "xcku040-ffva1156-2-e",
        "xcku040_ffva1156_2_e",
    )


def test_attach_board_profile_preserves_sample_fields():
    sample = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        hls_attr=torch.tensor([[989.0, 1039.0, 0.0, 0.0, 0.0, 5.393]]),
        y=torch.tensor([[478.0, 1033.0, 0.0, 0.0, 0.0, 0.0, 3.985, 0.248]]),
        bench_name="bfs",
        prj_name="prj_0",
    )

    augmented = attach_board_profile(sample, resolve_board_profile(DEFAULT_BOARD_DEVICE))

    assert augmented is not sample
    assert "arch_attr" not in sample
    assert torch.equal(augmented.x, sample.x)
    assert torch.equal(augmented.edge_index, sample.edge_index)
    assert torch.equal(augmented.edge_attr, sample.edge_attr)
    assert torch.equal(augmented.hls_attr, sample.hls_attr)
    assert torch.equal(augmented.y, sample.y)
    assert augmented.bench_name == "bfs"
    assert augmented.prj_name == "prj_0"
    assert augmented.board_device == DEFAULT_BOARD_DEVICE
    assert augmented.arch_attr.shape == (1, len(ARCH_ATTR_FIELDS))


def test_attach_board_profile_adds_architecture_cache_reference():
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
    )

    augmented = attach_board_profile(sample, resolve_board_profile(DEFAULT_BOARD_DEVICE))

    assert augmented.board_device == DEFAULT_BOARD_DEVICE
    assert augmented.arch_attr.shape == (1, len(ARCH_ATTR_FIELDS))
    assert augmented.board_arch_device == DEFAULT_BOARD_DEVICE
    assert "board_arch_device" not in sample


def test_attach_board_profile_adds_arch_aware_architecture_tensors():
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
    )
    arch_aware_payload = {
        "device": DEFAULT_BOARD_DEVICE,
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32) * 2.0,
        "arch_aware_clock_region": "CR_X0Y0",
    }

    augmented = attach_board_profile(
        sample,
        resolve_board_profile(DEFAULT_BOARD_DEVICE),
        arch_aware_arch=arch_aware_payload,
    )

    assert augmented.arch_aware_arch_layout.shape == (ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS)
    assert augmented.arch_aware_arch_metadata.shape == (1, ARCH_AWARE_METADATA_DIM)
    assert torch.equal(augmented.arch_aware_arch_layout, arch_aware_payload["arch_aware_arch_layout"])
    assert torch.equal(augmented.arch_aware_arch_metadata, arch_aware_payload["arch_aware_arch_metadata"])
    assert augmented.arch_aware_arch_device == DEFAULT_BOARD_DEVICE
    assert augmented.arch_aware_clock_region == "CR_X0Y0"
    assert "arch_aware_arch_layout" not in sample


def test_augment_dataset_attaches_same_default_board_to_all_samples():
    samples = [
        Data(
            x=torch.tensor([[float(index)]]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 2), dtype=torch.float32),
            hls_attr=torch.tensor([[float(index), 10.0]]),
            y=torch.tensor([[float(index + 1)]]),
        )
        for index in range(2)
    ]

    augmented = augment_dataset(samples, device="xc7vx485t-ffg1761-2")

    assert len(augmented) == 2
    assert all(sample.board_device == DEFAULT_BOARD_DEVICE for sample in augmented)
    assert all(sample.arch_attr.shape == (1, len(ARCH_ATTR_FIELDS)) for sample in augmented)
    assert all("arch_attr" not in sample for sample in samples)


def test_ensure_board_fabric_cache_writes_one_file_for_default_board(tmp_path, monkeypatch):
    import hgp.board_fabric as board_fabric

    fake_graph = SimpleNamespace(
        device=DEFAULT_BOARD_DEVICE,
        node_features=[[1.0, 0.0]],
        edge_index=[(0, 0)],
        edge_features=[[0.0, 0.0, 0.0, 1.0]],
        graph_features=[1.0, 2.0, 3.0],
        node_coords=[(0, 0)],
        layout_rows=1,
        layout_cols=1,
    )
    monkeypatch.setattr(board_fabric, "extract_board_fabric", lambda *args, **kwargs: fake_graph)

    cache_path = board_fabric.ensure_board_fabric_cache(DEFAULT_BOARD_DEVICE, cache_dir=tmp_path)
    loaded = board_fabric.load_board_fabric_cache(cache_path)

    assert cache_path == tmp_path / f"{DEFAULT_BOARD_DEVICE}_fabric_graph.pt"
    assert loaded["device"] == DEFAULT_BOARD_DEVICE
    assert loaded["arch_x"].shape == (1, 2)
    assert loaded["arch_edge_index"].shape == (2, 1)
    assert loaded["arch_edge_attr"].shape == (1, 4)
    assert loaded["arch_graph_attr"].shape == (1, 3)


def test_generate_dataframe_accepts_board_device(tmp_path):
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(0, x=[1.0, 2.0])
    graph.add_node(1, x=[3.0, 4.0])
    graph.add_edge(0, 1, edge_attr=[1.0, 0.0])

    dataframe = generate_dataframe(
        graph,
        metric_list=[478.0, 1033.0],
        hls_attr=[989.0, 5.393],
        bench_name="bfs",
        prj_name="prj_0",
        df_store_path=tmp_path / "sample.pt",
        board_device="xc7vx485t-ffg1761-2",
    )

    loaded = torch.load(tmp_path / "sample.pt", map_location="cpu", weights_only=False)

    assert dataframe.board_device == DEFAULT_BOARD_DEVICE
    assert dataframe.arch_attr.shape == (1, len(ARCH_ATTR_FIELDS))
    assert loaded.board_device == DEFAULT_BOARD_DEVICE
    assert torch.equal(loaded.arch_attr, dataframe.arch_attr)
    assert loaded.board_arch_device == DEFAULT_BOARD_DEVICE
    assert torch.equal(loaded.hls_attr, torch.tensor([[989.0, 5.393]]))


def test_gen_dataset_board_defaults_resolve_under_project_root():
    args = gen_dataset_board.build_parser().parse_args([])

    assert Path(args.input_root) == ROOT / "dataset"
    assert Path(args.output_root) == ROOT / "dataset"


def test_augment_dataset_dirs_writes_std_and_rdc_arch_outputs(tmp_path):
    input_root = tmp_path / "dataset"
    std_dir = input_root / "std"
    rdc_dir = input_root / "rdc"
    std_dir.mkdir(parents=True)
    rdc_dir.mkdir(parents=True)
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    torch.save([sample], std_dir / "bfs.pt")
    torch.save([sample], rdc_dir / "bfs.pt")
    arch_aware_payload = {
        "device": DEFAULT_BOARD_DEVICE,
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
        "arch_aware_clock_region": "CR_X0Y0",
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gen_dataset_board, "ensure_board_fabric_cache", lambda *args, **kwargs: tmp_path / "fabric.pt")
    monkeypatch.setattr(gen_dataset_board, "ensure_arch_aware_arch_cache", lambda *args, **kwargs: tmp_path / "arch_aware.pt")
    monkeypatch.setattr(gen_dataset_board, "load_arch_aware_arch_cache", lambda *args, **kwargs: arch_aware_payload)

    written = augment_dataset_dirs(input_root, input_root, device="xc7vx485t-ffg1761-2")
    monkeypatch.undo()

    expected_root = input_root / DEFAULT_BOARD_DEVICE
    assert written == {
        "std": [expected_root / "std_arch" / "bfs.pt"],
        "rdc": [expected_root / "rdc_arch" / "bfs.pt"],
    }
    std_arch = torch.load(expected_root / "std_arch" / "bfs.pt", map_location="cpu", weights_only=False)
    rdc_arch = torch.load(expected_root / "rdc_arch" / "bfs.pt", map_location="cpu", weights_only=False)
    assert std_arch[0].board_device == DEFAULT_BOARD_DEVICE
    assert rdc_arch[0].board_device == DEFAULT_BOARD_DEVICE
    assert std_arch[0].arch_attr.shape == (1, len(ARCH_ATTR_FIELDS))
    assert rdc_arch[0].arch_attr.shape == (1, len(ARCH_ATTR_FIELDS))
    assert std_arch[0].board_arch_device == DEFAULT_BOARD_DEVICE
    assert rdc_arch[0].board_arch_device == DEFAULT_BOARD_DEVICE
    assert std_arch[0].arch_aware_arch_layout.shape == (ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS)
    assert rdc_arch[0].arch_aware_arch_metadata.shape == (1, ARCH_AWARE_METADATA_DIM)


def test_augment_dataset_dirs_legacy_layout_writes_flat_paths(tmp_path):
    input_root = tmp_path / "dataset"
    std_dir = input_root / "std"
    rdc_dir = input_root / "rdc"
    std_dir.mkdir(parents=True)
    rdc_dir.mkdir(parents=True)
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    torch.save([sample], std_dir / "bfs.pt")
    torch.save([sample], rdc_dir / "bfs.pt")
    arch_aware_payload = {
        "device": DEFAULT_BOARD_DEVICE,
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
        "arch_aware_clock_region": "CR_X0Y0",
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gen_dataset_board, "ensure_board_fabric_cache", lambda *args, **kwargs: tmp_path / "fabric.pt")
    monkeypatch.setattr(gen_dataset_board, "ensure_arch_aware_arch_cache", lambda *args, **kwargs: tmp_path / "arch_aware.pt")
    monkeypatch.setattr(gen_dataset_board, "load_arch_aware_arch_cache", lambda *args, **kwargs: arch_aware_payload)

    written = augment_dataset_dirs(
        input_root,
        input_root,
        device="xc7vx485t-ffg1761-2",
        layout="legacy",
    )
    monkeypatch.undo()

    assert written == {
        "std": [input_root / "std_arch" / "bfs.pt"],
        "rdc": [input_root / "rdc_arch" / "bfs.pt"],
    }


def test_augment_dataset_dirs_multi_writes_per_device_subdirs(tmp_path):
    from hgp.data_process.gen_dataset_board import augment_dataset_dirs_multi

    input_root = tmp_path / "dataset"
    (input_root / "std").mkdir(parents=True)
    (input_root / "rdc").mkdir(parents=True)
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    torch.save([sample], input_root / "std" / "bfs.pt")
    torch.save([sample], input_root / "rdc" / "bfs.pt")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gen_dataset_board, "ensure_board_fabric_cache", lambda *args, **kwargs: tmp_path / "fabric.pt")
    monkeypatch.setattr(
        gen_dataset_board,
        "ensure_arch_aware_arch_cache",
        lambda device, **kwargs: tmp_path / f"{device}_arch_aware.pt",
    )
    monkeypatch.setattr(
        gen_dataset_board,
        "load_arch_aware_arch_cache",
        lambda *args, **kwargs: {
            "device": DEFAULT_BOARD_DEVICE,
            "arch_aware_arch_layout": torch.ones(
                (ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS),
                dtype=torch.float32,
            ),
            "arch_aware_arch_metadata": torch.ones(
                (1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32,
            ),
        },
    )

    devices = [DEFAULT_BOARD_DEVICE, "xcku040_ffva1156_2_e"]
    results = augment_dataset_dirs_multi(input_root, input_root, devices=devices)
    monkeypatch.undo()

    expected_devices = [DEFAULT_BOARD_DEVICE, "xcku040-ffva1156-2-e"]
    assert sorted(results) == sorted(expected_devices)
    for device in expected_devices:
        assert (input_root / device / "std_arch" / "bfs.pt").is_file()
        assert (input_root / device / "rdc_arch" / "bfs.pt").is_file()


def test_augment_dataset_dirs_multi_deduplicates_device_aliases(tmp_path):
    from hgp.data_process.gen_dataset_board import augment_dataset_dirs_multi

    calls = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        gen_dataset_board,
        "augment_dataset_dirs",
        lambda *args, device, **kwargs: calls.append(device) or {"std": [], "rdc": []},
    )

    results = augment_dataset_dirs_multi(
        tmp_path,
        tmp_path,
        devices=["xcku040_ffva1156_2_e", "xcku040-ffva1156-2-e"],
    )
    monkeypatch.undo()

    assert calls == ["xcku040-ffva1156-2-e"]
    assert sorted(results) == ["xcku040-ffva1156-2-e"]


def test_gen_dataset_board_reads_per_device_input_shards(tmp_path):
    input_root = tmp_path / "dataset"
    std_dir = input_root / "xcku040_ffva1156_2_e" / "std"
    rdc_dir = input_root / "xcku040_ffva1156_2_e" / "rdc"
    std_dir.mkdir(parents=True)
    rdc_dir.mkdir(parents=True)
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    torch.save([sample], std_dir / "bfs.pt")
    torch.save([sample], rdc_dir / "bfs.pt")
    arch_aware_payload = {
        "device": "xcku040-ffva1156-2-e",
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(gen_dataset_board, "ensure_arch_aware_arch_cache", lambda *args, **kwargs: tmp_path / "arch.pt")
    monkeypatch.setattr(gen_dataset_board, "load_arch_aware_arch_cache", lambda *args, **kwargs: arch_aware_payload)

    written = gen_dataset_board.augment_dataset_dirs(
        input_root,
        input_root,
        device="xcku040_ffva1156_2_e",
    )
    monkeypatch.undo()

    expected_root = input_root / "xcku040-ffva1156-2-e"
    assert written == {
        "std": [expected_root / "std_arch" / "bfs.pt"],
        "rdc": [expected_root / "rdc_arch" / "bfs.pt"],
    }
    std_arch = torch.load(expected_root / "std_arch" / "bfs.pt", map_location="cpu", weights_only=False)
    assert std_arch[0].board_device == "xcku040-ffva1156-2-e"


def test_gen_dataset_std_writes_canonical_per_device_dirs(tmp_path):
    std_by_key = {("bfs", "xcku040_ffva1156_2_e"): {0: str(tmp_path / "std.pt")}}
    rdc_by_key = {("bfs", "xcku040_ffva1156_2_e"): {0: str(tmp_path / "rdc.pt")}}
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    torch.save(sample, tmp_path / "std.pt")
    torch.save(sample, tmp_path / "rdc.pt")

    gen_dataset_std._write_shards(
        std_by_key,
        rdc_by_key,
        output_root=tmp_path / "dataset",
        device_override=None,
        layout="per_device",
    )

    assert (tmp_path / "dataset" / "xcku040-ffva1156-2-e" / "std" / "bfs.pt").is_file()
    assert (tmp_path / "dataset" / "xcku040-ffva1156-2-e" / "rdc" / "bfs.pt").is_file()


def test_generate_dataset_loads_pyg_data_with_current_torch_defaults(tmp_path):
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 2), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0, 2.0]]),
        y=torch.tensor([[3.0]]),
    )
    torch.save([sample], tmp_path / "bfs.pt")

    loaded = generate_dataset(str(tmp_path), ["bfs.pt"])

    assert len(loaded) == 1
    assert torch.equal(loaded[0].x, sample.x)
    assert torch.equal(loaded[0].hls_attr, sample.hls_attr)
