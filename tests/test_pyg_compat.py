import os

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_legacy_sag_pooling_has_original_state_dict_shape():
    from hgp.pyg_compat import LegacySAGPooling

    pool = LegacySAGPooling(64, ratio=0.5)

    assert list(pool.state_dict()) == [
        "gnn.lin_rel.weight",
        "gnn.lin_rel.bias",
        "gnn.lin_root.weight",
    ]


def test_legacy_sag_pooling_matches_new_sag_pooling_with_positive_selector():
    from hgp.pyg_compat import LegacySAGPooling
    from torch_geometric.nn.pool import SAGPooling as PyGSAGPooling

    torch.manual_seed(7)
    legacy_pool = LegacySAGPooling(3, ratio=0.5)
    pyg_pool = PyGSAGPooling(3, ratio=0.5)
    pyg_pool.gnn.load_state_dict(legacy_pool.gnn.state_dict())
    pyg_pool.select.weight.data.fill_(1.0)

    x = torch.tensor(
        [
            [0.2, 1.0, -0.4],
            [1.5, -0.3, 0.7],
            [-0.8, 0.4, 1.1],
            [0.9, 0.2, -1.3],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 2, 3, 0, 2, 0],
        ],
        dtype=torch.long,
    )
    batch = torch.zeros(x.size(0), dtype=torch.long)

    legacy_out = legacy_pool(x, edge_index, batch=batch)
    pyg_out = pyg_pool(x, edge_index, batch=batch)

    for legacy_value, pyg_value in zip(legacy_out, pyg_out):
        if legacy_value is None:
            assert pyg_value is None
        else:
            assert torch.equal(legacy_value, pyg_value)


def test_sag_pooling_alias_uses_legacy_mode_when_enabled(monkeypatch):
    monkeypatch.setenv("HGBO_LEGACY_SAGPOOL", "1")

    from hgp.pyg_compat import get_sag_pooling_class

    pool_cls = get_sag_pooling_class()
    assert pool_cls.__name__ == "LegacySAGPooling"
