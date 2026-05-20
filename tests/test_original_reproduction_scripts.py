from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_original_baseline_wrapper_seeds_legacy_training_scripts():
    script = (ROOT / "scripts" / "train_original_paper_baseline.sh").read_text(encoding="utf-8")

    assert "HGP_TORCH_SEED" in script
    assert "torch.manual_seed(seed)" in script
    assert "torch.cuda.manual_seed_all(seed)" in script
    assert "runpy.run_path(script, run_name=\"__main__\")" in script


def test_new_stack_reference_tuning_script_pins_legacy_sagpool_and_isolates_outputs():
    script = (ROOT / "scripts" / "run_new_stack_reference_tuning.sh").read_text(encoding="utf-8")

    assert "HGBO_LEGACY_SAGPOOL=1" in script
    assert "--init-from-builtins" in script
    assert "--weight-decay \"$weight_decay\"" in script
    assert "dsp_lr002_wd0_decay095_clip1_${DSP_FULL_EPOCHS}ep" in script
    assert "RUN_MAPE_TARGETS" in script
    assert "RUN_BRAM_DEFAULT" in script
    assert "tuning_summary.csv" in script
    assert "Skipping non-empty output" in script


def test_new_stack_hgbo_dse_flow_script_has_original_and_arch_aware_stages():
    script = (ROOT / "scripts" / "run_new_stack_hgbo_dse_flow.sh").read_text(encoding="utf-8")

    assert "HGBO_LEGACY_SAGPOOL=1" in script
    assert "DRY_RUN" in script
    assert "RUN_REFERENCE_EVAL" in script
    assert "RUN_ORIGINAL_TARGETS" in script
    assert "RUN_ARCH_VERIFY" in script
    assert "hgp.reporting.original_stable_training" in script
    assert "hgp.reporting.compare_training_graphs" in script
    assert "--deterministic-eval" in script
    assert "--init-from-builtins" in script
    assert "--arch-mode \"$ARCH_MODE\"" in script
    assert "flow_summary.csv" in script
    assert "Skipping non-empty output" in script
    assert 'stats["settings"]["target"]' in script
    assert 'stats["settings"]["metric"]' in script
    assert 'stats["mae"]' in script
    assert 'stats["abs_error"]["median"]' in script
