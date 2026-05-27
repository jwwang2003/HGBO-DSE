import json

import pytest

from scripts import audit_power_labels


def _write_ppa(path, impl):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"IMPL": impl}), encoding="utf-8")


def test_audit_raw_power_labels_accepts_consistent_power_triplet(tmp_path):
    _write_ppa(
        tmp_path / "raw" / "bfs" / "xc7" / "script" / "ppa_0.json",
        {"PWR": 1.5, "PWR_DYNAMIC": 0.4, "PWR_STATIC": 1.1},
    )

    stats = audit_power_labels.audit_raw_power_labels(tmp_path / "raw")

    assert stats["seen"] == 1
    assert stats["ok"] == 1
    assert stats["failed"] == 0


def test_audit_raw_power_labels_flags_missing_and_inconsistent_power(tmp_path):
    ppa_path = tmp_path / "raw" / "bfs" / "xc7" / "script" / "ppa_0.json"
    _write_ppa(ppa_path, {"PWR": 1.5, "PWR_DYNAMIC": 0.7})

    stats = audit_power_labels.audit_raw_power_labels(tmp_path / "raw")

    assert stats["failed"] == 1
    assert any("missing_or_nonfinite:PWR_STATIC" in issue for issue in stats["issues"][str(ppa_path)])


def test_audit_sample_y_uses_total_and_dynamic_power_indices():
    torch = pytest.importorskip("torch")
    data = pytest.importorskip("torch_geometric.data")
    sample = data.Data(y=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.4]]))

    issues = audit_power_labels._audit_sample_y(sample, path=__file__, sample_index=0)

    assert issues == []
    assert audit_power_labels.TOTAL_POWER_INDEX == 7
    assert audit_power_labels.DYNAMIC_POWER_INDEX == 8


def test_std_shard_paths_find_legacy_and_arch_datasets(tmp_path):
    paths = [
        tmp_path / "dataset" / "std" / "bfs.pt",
        tmp_path / "dataset" / "std_arch" / "bfs.pt",
        tmp_path / "dataset" / "xc7" / "std" / "atax.pt",
        tmp_path / "dataset" / "xc7" / "std_arch" / "atax.pt",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    assert audit_power_labels._std_shard_paths(tmp_path / "dataset") == sorted(paths)
