import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from hgp.reporting.compare_training_graphs import (  # noqa: E402
    _get_split,
    _load_split,
    build_parser,
    build_consistency_summary,
    write_consistency_summary,
    write_results,
)


def _history(train_metric, test_metric):
    return [
        {
            "epoch": 0,
            "train_loss": float(train_metric),
            "test_loss": float(test_metric),
            "train_metric": float(train_metric),
            "test_metric": float(test_metric),
            "best_test_metric": float(test_metric),
        },
        {
            "epoch": 1,
            "train_loss": float(train_metric) * 0.9,
            "test_loss": float(test_metric) * 0.9,
            "train_metric": float(train_metric) * 0.9,
            "test_metric": float(test_metric) * 0.9,
            "best_test_metric": float(test_metric) * 0.9,
        },
    ]


def _minimal_results():
    targets = {}
    for index, target in enumerate(["lut", "ff", "dsp", "bram", "cp", "power"]):
        metric = "mae" if target in {"dsp", "bram"} else "mape"
        original_best = 1.0 + index
        arch_best = original_best * (0.8 if target != "cp" else 1.2)
        targets[target] = {
            "metric_name": metric,
            "original": {
                "history": _history(original_best * 1.2, original_best),
                "best_test_metric": original_best,
            },
            "architecture_aware": {
                "history": _history(arch_best * 1.2, arch_best),
                "best_test_metric": arch_best,
            },
        }
    return {
        "settings": {
            "epochs": 2,
            "seed": 128,
            "lr": 0.001,
            "grad_clip": 1.0,
            "arch_mode": "arch-aware",
            "fabric_mode": "cached",
            "board_device": "xc7vx485tffg1761-2",
        },
        "targets": targets,
    }


def test_training_graph_loader_ignores_non_pt_dataset_files(tmp_path, monkeypatch):
    import hgp.reporting.compare_training_graphs as report
    from torch_geometric.data import Data

    dataset_dir = tmp_path / "dataset" / "std"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "README.md").write_text("not a torch file\n", encoding="utf-8")
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0]]),
        y=torch.tensor([[1.0]]),
    )
    torch.save([sample] * 10, dataset_dir / "bfs.pt")
    monkeypatch.setattr(report, "HGBO_ROOT", tmp_path)

    train_ds, test_ds = _load_split("std", seed=128)

    assert len(train_ds) == 8
    assert len(test_ds) == 2


def test_training_graph_loader_does_not_print_dataset_preview(tmp_path, monkeypatch, capsys):
    import hgp.reporting.compare_training_graphs as report
    from torch_geometric.data import Data

    dataset_dir = tmp_path / "dataset" / "std"
    dataset_dir.mkdir(parents=True)
    sample = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        hls_attr=torch.tensor([[1.0]]),
        y=torch.tensor([[1.0]]),
    )
    torch.save([sample] * 10, dataset_dir / "bfs.pt")
    monkeypatch.setattr(report, "HGBO_ROOT", tmp_path)

    _load_split("std", seed=128)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_get_split_reuses_cached_dataset_split(monkeypatch):
    import hgp.reporting.compare_training_graphs as report

    calls = []

    def fake_load_split(dataset_subdir, seed):
        calls.append((dataset_subdir, seed))
        return ["train"], ["test"]

    monkeypatch.setattr(report, "_load_split", fake_load_split)
    cache = {}

    first = _get_split(cache, "std", 128)
    second = _get_split(cache, "std", 128)

    assert first == (["train"], ["test"])
    assert second is first
    assert calls == [("std", 128)]


def test_write_results_generates_report_graph_artifacts(tmp_path):
    paths = write_results(_minimal_results(), tmp_path)

    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    assert Path(paths["training_curves"]).is_file()
    assert Path(paths["best_test_delta"]).is_file()

    loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert loaded["settings"]["epochs"] == 2
    assert "<svg" in Path(paths["training_curves"]).read_text(encoding="utf-8")
    assert "Architecture-Aware Best Test Change" in Path(paths["best_test_delta"]).read_text(encoding="utf-8")


def test_write_results_honors_target_subset(tmp_path):
    results = _minimal_results()
    results["targets"] = {"lut": results["targets"]["lut"]}

    paths = write_results(results, tmp_path)

    csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
    curves_text = Path(paths["training_curves"]).read_text(encoding="utf-8")
    assert "lut,Original" in csv_text
    assert "ff,Original" not in csv_text
    assert ">LUT</text>" in curves_text
    assert ">FF</text>" not in curves_text
    assert "MAPE is used for LUT, FF, CP, and Power" not in curves_text


def test_write_results_rejects_unknown_target_keys(tmp_path):
    results = _minimal_results()
    results["targets"] = {"lut": results["targets"]["lut"], "unknown": results["targets"]["lut"]}

    with pytest.raises(ValueError, match="Unknown target"):
        write_results(results, tmp_path)


def test_build_consistency_summary_reports_per_seed_winners():
    first = _minimal_results()
    second = _minimal_results()
    first["settings"]["seed"] = 128
    second["settings"]["seed"] = 256
    second["targets"]["lut"]["architecture_aware"]["best_test_metric"] = 1.2

    summary = build_consistency_summary([first, second])

    assert summary["settings"]["seeds"] == [128, 256]
    assert summary["settings"]["arch_mode"] == "arch-aware"
    assert summary["targets"]["lut"]["architecture_aware_win_count"] == 1
    assert summary["targets"]["lut"]["original_win_count"] == 1
    assert summary["targets"]["lut"]["consistent_winner"] == "mixed"
    assert summary["targets"]["ff"]["architecture_aware_win_count"] == 2
    assert summary["targets"]["ff"]["consistent_winner"] == "architecture-aware"
    assert summary["targets"]["cp"]["original_win_count"] == 2
    assert summary["overall"]["target_run_count"] == 12


def test_build_consistency_summary_honors_target_subset(tmp_path):
    first = _minimal_results()
    second = _minimal_results()
    first["targets"] = {"lut": first["targets"]["lut"]}
    second["targets"] = {"lut": second["targets"]["lut"]}

    summary = build_consistency_summary([first, second])
    paths = write_consistency_summary(summary, tmp_path)

    assert list(summary["targets"]) == ["lut"]
    assert summary["overall"]["target_count"] == 1
    assert summary["overall"]["target_run_count"] == 2
    assert "lut,mape" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_build_consistency_summary_reports_execution_settings():
    first = _minimal_results()
    second = _minimal_results()
    first["settings"].update(
        {
            "seed": 128,
            "batch_size": 40,
            "device_request": "cpu",
            "device": "cpu",
            "cpu_threads": 16,
            "num_workers": 0,
            "weight_decay": 0.001,
        }
    )
    second["settings"].update(
        {
            "seed": 256,
            "batch_size": 40,
            "device_request": "cpu",
            "device": "cpu",
            "cpu_threads": 16,
            "num_workers": 0,
            "weight_decay": 0.001,
        }
    )

    summary = build_consistency_summary([first, second])

    assert summary["settings"]["device_request"] == "cpu"
    assert summary["settings"]["device"] == "cpu"
    assert summary["settings"]["cpu_threads"] == 16
    assert summary["settings"]["num_workers"] == 0
    assert summary["settings"]["batch_size"] == 40
    assert summary["settings"]["weight_decay"] == 0.001


def test_write_consistency_summary_generates_json_and_csv(tmp_path):
    first = _minimal_results()
    second = _minimal_results()
    first["settings"]["seed"] = 128
    second["settings"]["seed"] = 256
    summary = build_consistency_summary([first, second])

    paths = write_consistency_summary(summary, tmp_path)

    assert Path(paths["json"]).is_file()
    assert Path(paths["csv"]).is_file()
    loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert loaded["settings"]["seeds"] == [128, 256]
    assert "architecture_aware_win_count" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_compare_parser_defaults_to_cpu_training():
    args = build_parser().parse_args([])

    assert args.device == "cpu"
