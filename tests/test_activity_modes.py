import json
from pathlib import Path

from hgp.data_process import gen_dataset_std


def test_activity_mode_none_skips_switching_cache(tmp_path):
    cache_dir = tmp_path / "switching_activity"
    cache_dir.mkdir()
    (cache_dir / "atax_switching_activity.json").write_text(
        '{"by_opcode": {"add": {"sa": 1.0, "ar": 0.5}}}',
        encoding="utf-8",
    )

    loaded = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="none",
        cache_dir=cache_dir,
    )

    assert loaded is None


def test_activity_mode_shuffled_preserves_metrics_but_breaks_mapping(tmp_path):
    cache_dir = tmp_path / "switching_activity"
    cache_dir.mkdir()
    payload = {
        "by_opcode": {
            "add": {"sa": 1.0, "ar": 0.1},
            "mul": {"sa": 2.0, "ar": 0.2},
            "sub": {"sa": 3.0, "ar": 0.3},
        }
    }
    (cache_dir / "atax_switching_activity.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    opcode = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="opcode",
        cache_dir=cache_dir,
    )
    shuffled_a = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="shuffled",
        seed_parts=("bench", "dev", 0),
        cache_dir=cache_dir,
    )
    shuffled_b = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="shuffled",
        seed_parts=("bench", "dev", 0),
        cache_dir=cache_dir,
    )

    assert opcode == payload["by_opcode"]
    assert shuffled_a == shuffled_b
    assert sorted(shuffled_a) == sorted(payload["by_opcode"])
    assert sorted(shuffled_a.values(), key=lambda item: item["sa"]) == sorted(
        payload["by_opcode"].values(),
        key=lambda item: item["sa"],
    )


def test_build_tasks_threads_activity_mode(tmp_path):
    bench_path = tmp_path / "bench" / "device"
    script_dir = bench_path / "script"
    script_dir.mkdir(parents=True)
    (script_dir / "ppa_7.json").write_text("{}", encoding="utf-8")

    tasks = gen_dataset_std._build_tasks(
        [("bench", "device", bench_path)],
        device_override=None,
        emit_dot=False,
        timeout_s=5,
        activity_mode="shuffled",
    )

    assert tasks == [("bench", "device", 7, str(bench_path), "device", False, 5, "shuffled")]


def test_infer_top_name_ignores_pipeline_bind_and_sched_files(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    for name in ["atax_Pipeline_lp1.adb", "atax.bind.adb", "atax.sched.adb", "atax.adb"]:
        Path(graph_dir / name).write_text("", encoding="utf-8")

    assert gen_dataset_std._infer_top_name(graph_dir) == "atax"
