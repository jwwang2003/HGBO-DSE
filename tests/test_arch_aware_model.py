import pytest
from pathlib import Path


torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")
pyg_loader = pytest.importorskip("torch_geometric.loader")
Data = pyg_data.Data
DataLoader = pyg_loader.DataLoader

from hgp.hier_arch_model import (  # noqa: E402
    ArchAwareHierNet,
    ArchAwareArchitectureEncoder,
    TARGET_SPECS,
    _default_dataset_dir,
    _resolve_device,
    _set_cpu_threads,
    _prepare_board_training_input,
    build_parser,
    evaluate,
    train_epoch,
)
from hgp.arch_aware_arch import ARCH_AWARE_LAYOUT_COLS, ARCH_AWARE_METADATA_DIM, ARCH_AWARE_TILE_SLOTS  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def make_fake_board_graph(node_shift=0.0):
    return {
        "arch_x": torch.tensor(
            [
                [1.0 + node_shift, 0.0, 0.5] + [0.0] * 21,
                [0.0, 1.0 + node_shift, 0.25] + [0.0] * 21,
            ],
            dtype=torch.float32,
        ),
        "arch_edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "arch_edge_attr": torch.tensor(
            [
                [1.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        "arch_graph_attr": torch.tensor([[0.3036, 0.6072, 2.8, 1.03] + [0.0] * 28], dtype=torch.float32),
    }


class FakeBatch:
    def __init__(self):
        self.x = torch.ones((1, 2), dtype=torch.float32)
        self.edge_index = torch.empty((2, 0), dtype=torch.long)
        self.batch = torch.zeros(1, dtype=torch.long)
        self.num_graphs = 1
        self.values = {
            "hls_attr": torch.ones((1, 6), dtype=torch.float32),
            "arch_attr": torch.ones((1, 13), dtype=torch.float32),
            "arch_aware_arch_layout": torch.ones((1, ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
            "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
            "y": torch.ones((1, 8), dtype=torch.float32),
        }

    def __getitem__(self, key):
        return self.values[key]

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.batch = self.batch.to(device)
        self.values = {key: value.to(device) for key, value in self.values.items()}
        return self


class FakeLoader:
    def __init__(self):
        self.dataset = [FakeBatch()]

    def __iter__(self):
        return iter(self.dataset)


class NonFiniteModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.value = value

    def forward(self, *args):
        return self.weight * torch.tensor([self.value], dtype=torch.float32, device=self.weight.device)


def test_default_dataset_dir_resolves_under_hgbo_project_root():
    assert _default_dataset_dir("std_arch") == ROOT / "dataset" / "std_arch"


def test_parser_exposes_training_stability_flags():
    args = build_parser().parse_args([])

    assert args.lr == pytest.approx(0.001)
    assert args.lr_decay_factor == pytest.approx(0.9)
    assert args.lr_decay_interval == 10
    assert args.grad_clip == pytest.approx(1.0)
    assert args.print_predictions is False
    assert args.deterministic_eval is False
    assert args.init_checkpoint is None
    assert args.summary_path is None
    assert args.arch_mode == "arch-aware"
    assert args.fabric_mode == "cached"
    assert args.conv_type == "gine"
    assert args.device == "cpu"
    assert args.num_workers == 0
    assert args.cpu_threads is None


def test_resolve_device_can_force_cpu_even_when_cuda_is_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert _resolve_device("cpu") == torch.device("cpu")
    assert _resolve_device("auto") == torch.device("cuda")
    assert _resolve_device("cuda") == torch.device("cuda")


def test_set_cpu_threads_configures_torch_thread_count():
    previous = torch.get_num_threads()
    try:
        _set_cpu_threads(1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)


def test_arch_aware_architecture_encoder_uses_layout_and_metadata():
    torch.manual_seed(0)
    encoder = ArchAwareArchitectureEncoder(hidden_dim=8, output_dim=6)
    layout = torch.zeros((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32)
    metadata = torch.zeros((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32)

    out_1 = encoder({"arch_aware_arch_layout": layout, "arch_aware_arch_metadata": metadata})
    out_2 = encoder({"arch_aware_arch_layout": layout + 1.0, "arch_aware_arch_metadata": metadata})
    out_3 = encoder({"arch_aware_arch_layout": layout, "arch_aware_arch_metadata": metadata + 1.0})

    assert out_1.shape == (1, 6)
    assert not torch.allclose(out_1, out_2)
    assert not torch.allclose(out_1, out_3)


def test_prepare_board_training_input_can_keep_fabric_encoder_trainable():
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        conv_type="sage",
        hls_dim=6,
        arch_dim=13,
        arch_node_dim=24,
        arch_edge_dim=4,
        arch_graph_dim=32,
        fabric_hidden_dim=4,
        drop_out=0.0,
    )
    board_graph = make_fake_board_graph()

    cached_input = _prepare_board_training_input(model, board_graph, "cached")
    trainable_input = _prepare_board_training_input(model, board_graph, "trainable")

    assert torch.is_tensor(cached_input)
    assert cached_input.requires_grad is False
    assert trainable_input is board_graph


def test_train_epoch_backprops_through_trainable_board_fabric_graph():
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        conv_type="sage",
        hls_dim=6,
        arch_dim=13,
        arch_node_dim=24,
        arch_edge_dim=4,
        arch_graph_dim=32,
        fabric_hidden_dim=4,
        drop_out=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_epoch(
        model,
        FakeLoader(),
        optimizer,
        torch.device("cpu"),
        TARGET_SPECS["lut"],
        make_fake_board_graph(),
        epoch=0,
        grad_clip=1.0,
    )

    assert any(param.grad is not None for param in model.board_encoder.parameters())


def test_fabric_mode_ignores_arch_aware_tensors_attached_to_dataset_samples():
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        conv_type="sage",
        hls_dim=6,
        arch_dim=13,
        arch_node_dim=24,
        arch_edge_dim=4,
        arch_graph_dim=32,
        fabric_hidden_dim=4,
        arch_mode="fabric",
        drop_out=0.0,
    )
    sample = Data(
        x=torch.ones((1, 2), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        hls_attr=torch.ones((1, 6), dtype=torch.float32),
        arch_attr=torch.ones((1, 13), dtype=torch.float32),
        arch_aware_arch_layout=torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        arch_aware_arch_metadata=torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
        y=torch.ones((1, 8), dtype=torch.float32),
    )
    loader = DataLoader([sample], batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        TARGET_SPECS["lut"],
        make_fake_board_graph(),
        epoch=0,
        grad_clip=1.0,
    )

    assert any(param.grad is not None for param in model.board_encoder.parameters())


def test_train_epoch_backprops_through_arch_aware_architecture_encoder():
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        conv_type="sage",
        hls_dim=6,
        arch_dim=13,
        arch_mode="arch-aware",
        arch_aware_hidden_dim=8,
        arch_aware_output_dim=4,
        drop_out=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_epoch(
        model,
        FakeLoader(),
        optimizer,
        torch.device("cpu"),
        TARGET_SPECS["lut"],
        {"arch_aware_arch_layout": torch.zeros((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS)), "arch_aware_arch_metadata": torch.zeros((1, ARCH_AWARE_METADATA_DIM))},
        epoch=0,
        grad_clip=1.0,
    )

    assert any(param.grad is not None for param in model.arch_aware_encoder.parameters())


def test_train_epoch_rejects_non_finite_model_output():
    model = NonFiniteModel(float("nan"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with pytest.raises(RuntimeError, match="non-finite model output.*epoch 3.*batch 0"):
        train_epoch(
            model,
            FakeLoader(),
            optimizer,
            torch.device("cpu"),
            TARGET_SPECS["lut"],
            torch.zeros((1, 32)),
            epoch=3,
            grad_clip=1.0,
        )


def test_evaluate_rejects_non_finite_metric():
    model = NonFiniteModel(float("inf"))

    with pytest.raises(RuntimeError, match="non-finite model output.*evaluation.*batch 0"):
        evaluate(
            model,
            FakeLoader(),
            torch.device("cpu"),
            TARGET_SPECS["lut"],
            torch.zeros((1, 32)),
            epoch=4,
        )


def test_arch_aware_model_uses_board_features_in_forward():
    torch.manual_seed(0)
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=2,
        conv_type="sage",
        hls_dim=2,
        arch_dim=13,
        arch_node_dim=24,
        arch_graph_dim=32,
        drop_out=0.0,
    )
    x = torch.tensor([[1.0, 0.5], [0.3, 0.7], [0.9, 0.1]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    hls_attr = torch.tensor([[989.0, 5.393]], dtype=torch.float32)
    arch_attr = torch.tensor([[0.3036, 0.6072, 2.8, 1.03, 0.28, 1.0, 0.4, 0.5, 0.1, 0.2, 1.0, 0.0, 0.0]], dtype=torch.float32)
    shifted_arch_attr = arch_attr + 0.25

    board_graph = make_fake_board_graph()
    out_1 = model(x, edge_index, batch, hls_attr, board_graph, arch_attr)
    out_2 = model(x, edge_index, batch, hls_attr, board_graph, shifted_arch_attr)

    assert out_1.shape == (1, 1)
    assert out_2.shape == (1, 1)
    assert not torch.allclose(out_1, out_2)


def test_arch_aware_model_uses_board_fabric_graph_in_forward():
    torch.manual_seed(0)
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=2,
        conv_type="sage",
        hls_dim=2,
        arch_dim=13,
        arch_node_dim=24,
        arch_edge_dim=4,
        arch_graph_dim=32,
        drop_out=0.0,
    )
    x = torch.tensor([[1.0, 0.5], [0.3, 0.7], [0.9, 0.1]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    hls_attr = torch.tensor([[989.0, 5.393]], dtype=torch.float32)
    arch_attr = torch.tensor(
        [[0.3036, 0.6072, 2.8, 1.03, 0.28, 1.0, 0.4, 0.5, 0.1, 0.2, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    arch_graph_a = make_fake_board_graph(node_shift=0.0)
    arch_graph_b = make_fake_board_graph(node_shift=0.25)

    out_1 = model(x, edge_index, batch, hls_attr, arch_graph_a, arch_attr)
    out_2 = model(x, edge_index, batch, hls_attr, arch_graph_b, arch_attr)

    assert out_1.shape == (1, 1)
    assert out_2.shape == (1, 1)
    assert not torch.allclose(out_1, out_2)


def test_arch_aware_model_uses_arch_aware_layout_in_forward():
    torch.manual_seed(0)
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=2,
        conv_type="sage",
        hls_dim=2,
        arch_dim=13,
        arch_mode="arch-aware",
        arch_aware_hidden_dim=8,
        arch_aware_output_dim=4,
        drop_out=0.0,
    )
    x = torch.tensor([[1.0, 0.5], [0.3, 0.7], [0.9, 0.1]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    hls_attr = torch.tensor([[989.0, 5.393]], dtype=torch.float32)
    arch_attr = torch.tensor(
        [[0.3036, 0.6072, 2.8, 1.03, 0.28, 1.0, 0.4, 0.5, 0.1, 0.2, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    arch_aware_a = {
        "arch_aware_arch_layout": torch.zeros((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.zeros((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
    }
    arch_aware_b = {
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.zeros((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
    }

    out_1 = model(x, edge_index, batch, hls_attr, arch_aware_a, arch_attr)
    out_2 = model(x, edge_index, batch, hls_attr, arch_aware_b, arch_attr)

    assert out_1.shape == (1, 1)
    assert out_2.shape == (1, 1)
    assert not torch.allclose(out_1, out_2)


def test_arch_aware_gine_design_encoder_uses_edge_attributes_in_forward():
    torch.manual_seed(0)
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=2,
        conv_type="gine",
        design_edge_dim=2,
        hls_dim=2,
        arch_dim=13,
        arch_mode="arch-aware",
        arch_aware_hidden_dim=8,
        arch_aware_output_dim=4,
        drop_out=0.0,
    )
    x = torch.tensor([[1.0, 0.5], [0.3, 0.7], [0.9, 0.1]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.tensor([[1.0, 0.0], [1.0, 0.0], [4.0, 1.0], [4.0, 1.0]], dtype=torch.float32)
    shifted_edge_attr = edge_attr + torch.tensor([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    batch = torch.zeros(3, dtype=torch.long)
    hls_attr = torch.tensor([[989.0, 5.393]], dtype=torch.float32)
    arch_attr = torch.zeros((1, 13), dtype=torch.float32)
    arch_aware_input = {
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
    }

    out_1 = model(x, edge_index, batch, hls_attr, arch_aware_input, arch_attr, edge_attr=edge_attr)
    out_2 = model(x, edge_index, batch, hls_attr, arch_aware_input, arch_attr, edge_attr=shifted_edge_attr)

    assert out_1.shape == (1, 1)
    assert out_2.shape == (1, 1)
    assert not torch.allclose(out_1, out_2)


def test_arch_aware_mode_ignores_legacy_scalar_arch_attr_branch():
    torch.manual_seed(0)
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=2,
        conv_type="sage",
        hls_dim=2,
        arch_dim=13,
        arch_mode="arch-aware",
        arch_aware_hidden_dim=8,
        arch_aware_output_dim=4,
        drop_out=0.0,
    )
    x = torch.tensor([[1.0, 0.5], [0.3, 0.7], [0.9, 0.1]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.zeros(3, dtype=torch.long)
    hls_attr = torch.tensor([[989.0, 5.393]], dtype=torch.float32)
    arch_aware_input = {
        "arch_aware_arch_layout": torch.ones((ARCH_AWARE_LAYOUT_COLS, 1, ARCH_AWARE_TILE_SLOTS), dtype=torch.float32),
        "arch_aware_arch_metadata": torch.ones((1, ARCH_AWARE_METADATA_DIM), dtype=torch.float32),
    }
    arch_attr = torch.zeros((1, 13), dtype=torch.float32)
    shifted_arch_attr = torch.ones((1, 13), dtype=torch.float32)

    out_1 = model(x, edge_index, batch, hls_attr, arch_aware_input, arch_attr)
    out_2 = model(x, edge_index, batch, hls_attr, arch_aware_input, shifted_arch_attr)

    assert torch.allclose(out_1, out_2)


def test_arch_aware_mode_does_not_register_legacy_scalar_arch_mlp_parameters():
    model = ArchAwareHierNet(
        in_channels=2,
        hidden_channels=4,
        num_layers=1,
        conv_type="sage",
        hls_dim=2,
        arch_dim=13,
        arch_mode="arch-aware",
        arch_aware_hidden_dim=8,
        arch_aware_output_dim=4,
        drop_out=0.0,
    )

    assert model.arch_mlp is None
    assert all(not name.startswith("arch_mlp") for name, _ in model.named_parameters())
