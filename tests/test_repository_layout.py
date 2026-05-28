from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_docs_are_grouped_by_purpose():
    assert (ROOT / "docs" / "setup" / "development.md").is_file()
    assert (ROOT / "docs" / "setup" / "dockerfile.md").is_file()
    assert (ROOT / "docs" / "reports" / "v2-production-reproduction.md").is_file()
    assert (ROOT / "docs" / "reports" / "arch-aware-hgbo-dse-report.md").is_file()
    assert (ROOT / "docs" / "experiments" / "dynamic-power-vc707-holdout-v2.json").is_file()


def test_repository_root_excludes_obsolete_experiment_clutter():
    forbidden_root_entries = [
        ".idea",
        "_old",
        "context0",
        "experiments",
        "model",
        "model_smoke",
        "README.dev.md",
        "README.dockerfile.md",
        "ARCH_AWARE_HGBO_DSE_REPORT.md",
        "V2_PRODUCTION_REPRODUCTION.md",
    ]

    assert [name for name in forbidden_root_entries if (ROOT / name).exists()] == []

