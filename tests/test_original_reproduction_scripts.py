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
    assert "BRAM_HLS_RESIDUAL_INDEX" in script
    assert "--hls-residual-index" in script
    assert "LOADER_RNG_MODE:=isolated" in script
    assert "--loader-rng-mode \"$LOADER_RNG_MODE\"" in script
    assert "tuning_summary.csv" in script
    assert "Skipping non-empty output" in script


def test_new_stack_hgbo_dse_flow_script_has_original_and_arch_aware_stages():
    script = (ROOT / "scripts" / "run_new_stack_hgbo_dse_flow.sh").read_text(encoding="utf-8")

    assert "HGBO_LEGACY_SAGPOOL=1" in script
    assert "DRY_RUN" in script
    assert "RUN_REFERENCE_EVAL" in script
    assert "RUN_ORIGINAL_TARGETS" in script
    assert "RUN_ARCH_VERIFY" in script
    assert "RUN_BRAM_RESIDUAL_CALIBRATOR" in script
    assert "REFERENCE_DETERMINISTIC_EVAL" in script
    assert "ORIGINAL_TRAIN_DETERMINISTIC_EVAL:=0" in script
    assert "LOADER_RNG_MODE:=isolated" in script
    assert "--loader-rng-mode \"$loader_rng_mode\"" in script
    assert "BRAM_TARGET_TRANSFORM" in script
    assert "BRAM_TRAIN_SAMPLER" in script
    assert "BRAM_HLS_RESIDUAL_INDEX" in script
    assert "hgp.reporting.original_stable_training" in script
    assert "hgp.reporting.bram_residual_calibrator" in script
    assert "hgp.reporting.compare_training_graphs" in script
    assert "--deterministic-eval" in script
    assert "--init-from-builtins" in script
    assert "--target-transform \"$BRAM_TARGET_TRANSFORM\"" in script
    assert "--train-sampler \"$BRAM_TRAIN_SAMPLER\"" in script
    assert "--hls-residual-index \"$BRAM_HLS_RESIDUAL_INDEX\"" in script
    assert "--arch-mode \"$ARCH_MODE\"" in script
    assert "flow_summary.csv" in script
    assert "Skipping non-empty output" in script
    assert 'stats["settings"]["target"]' in script
    assert 'stats["settings"]["metric"]' in script
    assert 'stats["mae"]' in script
    assert 'stats["abs_error"]["median"]' in script


def test_v2_production_reproduction_wrapper_runs_new_stack_flow_and_checkpoint_eval():
    script = (ROOT / "scripts" / "run_v2_production_reproduction.sh").read_text(encoding="utf-8")

    assert "scripts/run_new_stack_hgbo_dse_flow.sh" in script
    assert "RUN_ORIGINAL_MAPE_TARGETS:=1" in script
    assert "RUN_ORIGINAL_DSP:=1" in script
    assert "RUN_BRAM_RESIDUAL_CALIBRATOR:=1" in script
    assert "LOADER_RNG_MODE:=isolated" in script
    assert "ORIGINAL_TRAIN_DETERMINISTIC_EVAL:=0" in script
    assert "run_checkpoint_eval" in script
    assert "--deterministic-eval" in script
    assert "--init-checkpoint-dir" in script


def test_v2_arch_embedding_experiment_wrapper_runs_arch_training_and_eval():
    script = (ROOT / "scripts" / "run_v2_arch_embedding_experiment.sh").read_text(encoding="utf-8")

    assert "hgp.hier_arch_model" in script
    assert "ARCH_MODE:=arch-aware" in script
    assert "ARCH_CONV_TYPE:=sage" in script
    assert "ARCH_LR_DECAY_FACTOR:=0.95" in script
    assert "ARCH_WEIGHT_DECAY:=0.0" in script
    assert "RUN_DETERMINISTIC_ARCH_EVAL:=1" in script
    assert "--summary-path" in script
    assert "--init-checkpoint" in script
    assert "--deterministic-eval" in script
    assert "hgp.reporting.bram_residual_calibrator" in script
    assert "flow_summary.csv" in script
