import pytest


torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data

from hgp.reporting import original_stable_training as stable  # noqa: E402
from hgp.reporting.checkpoint_prediction_stats import summarize_error_buckets, summarize_values  # noqa: E402


def test_mae_targets_use_conservative_default_learning_rate():
    args = stable.build_parser().parse_args(["--targets", "dsp", "bram", "--epochs", "1"])

    assert stable.target_learning_rate("dsp", args) == pytest.approx(0.001)
    assert stable.target_learning_rate("bram", args) == pytest.approx(0.001)


def test_mape_targets_keep_default_learning_rate():
    args = stable.build_parser().parse_args(["--targets", "lut", "--epochs", "1", "--lr", "0.005"])

    assert stable.target_learning_rate("lut", args) == pytest.approx(0.005)


def test_learning_rate_schedule_can_disable_decay():
    args = stable.build_parser().parse_args(["--lr-decay-factor", "1.0", "--lr-decay-interval", "10"])

    assert not stable.should_decay_learning_rate(epoch=10, args=args)


def test_learning_rate_schedule_uses_interval():
    args = stable.build_parser().parse_args(["--lr-decay-factor", "0.9", "--lr-decay-interval", "5"])

    assert stable.should_decay_learning_rate(epoch=5, args=args)
    assert not stable.should_decay_learning_rate(epoch=6, args=args)


def test_test_loader_options_preserve_paper_style_by_default():
    args = stable.build_parser().parse_args([])

    assert stable.test_loader_options(args) == {"shuffle": True, "drop_last": True}


def test_deterministic_eval_uses_full_unshuffled_test_loader():
    args = stable.build_parser().parse_args(["--deterministic-eval"])

    assert stable.test_loader_options(args) == {"shuffle": False, "drop_last": False}


def test_loss_weights_default_to_ones():
    args = stable.build_parser().parse_args([])
    true_y = torch.tensor([0.0, 0.5, 5.0, 50.0, 200.0])

    weights = stable.loss_weights(true_y, args)

    assert torch.equal(weights, torch.ones_like(true_y))


def test_bram_tail_loss_weights_prioritize_high_bram_samples():
    args = stable.build_parser().parse_args(
        [
            "--loss-weighting",
            "bram-tail",
            "--bram-positive-weight",
            "2",
            "--bram-mid-weight",
            "4",
            "--bram-high-weight",
            "16",
            "--bram-extreme-weight",
            "32",
        ]
    )
    true_y = torch.tensor([0.0, 0.5, 5.0, 50.0, 200.0])

    weights = stable.loss_weights(true_y, args)

    assert torch.equal(weights, torch.tensor([1.0, 2.0, 4.0, 16.0, 32.0]))


def test_training_loss_uses_normalized_weighted_huber_loss():
    args = stable.build_parser().parse_args(["--loss-weighting", "bram-tail", "--bram-high-weight", "10"])
    pred = torch.tensor([0.0, 0.0])
    true_y = torch.tensor([0.0, 50.0])

    loss = stable.training_loss(pred, true_y, args)

    expected_per_sample = torch.nn.functional.huber_loss(pred, true_y, reduction="none")
    expected = (expected_per_sample * torch.tensor([1.0, 10.0])).sum() / 11.0
    assert loss == pytest.approx(expected)


def test_target_transform_defaults_to_identity():
    args = stable.build_parser().parse_args([])
    values = torch.tensor([0.0, 1.0, 10.0])

    assert torch.equal(stable.transform_targets(values, args), values)
    assert torch.equal(stable.metric_predictions(values, args), values)


def test_log1p_target_transform_round_trips_predictions_and_clamps_negative_values():
    args = stable.build_parser().parse_args(["--target-transform", "log1p"])
    true_y = torch.tensor([0.0, 1.0, 10.0])
    transformed = stable.transform_targets(true_y, args)

    assert torch.allclose(transformed, torch.log1p(true_y))
    assert torch.allclose(stable.metric_predictions(transformed, args), true_y)
    assert stable.metric_predictions(torch.tensor([-2.0]), args).item() == pytest.approx(0.0)


def test_training_loss_uses_target_transform_before_huber_loss():
    args = stable.build_parser().parse_args(["--target-transform", "log1p"])
    pred = torch.log1p(torch.tensor([0.0, 10.0]))
    true_y = torch.tensor([0.0, 20.0])

    loss = stable.training_loss(pred, true_y, args)

    expected = torch.nn.functional.huber_loss(pred, torch.log1p(true_y), reduction="none").mean()
    assert loss == pytest.approx(expected)


def test_metric_values_use_inverse_transformed_predictions():
    args = stable.build_parser().parse_args(["--target-transform", "log1p"])
    pred = torch.log1p(torch.tensor([0.0, 10.0]))
    true_y = torch.tensor([0.0, 20.0])

    metric = stable.metric_value(pred, true_y, "mae", args)

    assert metric == pytest.approx(torch.tensor(5.0))


def _bram_sample(value):
    y = torch.zeros((1, 8), dtype=torch.float32)
    y[0, stable.ORIGINAL_TARGET_SPECS["bram"].target_index] = value
    return Data(y=y)


def test_train_sample_weights_default_to_none():
    args = stable.build_parser().parse_args([])
    dataset = [_bram_sample(0.0), _bram_sample(64.0)]

    assert stable.train_sample_weights(dataset, stable.ORIGINAL_TARGET_SPECS["bram"], args) is None


def test_bram_bucket_balanced_sampler_gives_each_present_bucket_equal_total_weight():
    args = stable.build_parser().parse_args(["--train-sampler", "bram-bucket-balanced"])
    dataset = [
        _bram_sample(0.0),
        _bram_sample(0.0),
        _bram_sample(0.5),
        _bram_sample(5.0),
        _bram_sample(50.0),
        _bram_sample(200.0),
        _bram_sample(200.0),
    ]

    weights = stable.train_sample_weights(dataset, stable.ORIGINAL_TARGET_SPECS["bram"], args)

    assert weights is not None
    true_y = torch.tensor([0.0, 0.0, 0.5, 5.0, 50.0, 200.0, 200.0])
    bucket_masks = [
        true_y == 0,
        (true_y > 0) & (true_y <= 1),
        (true_y > 1) & (true_y <= 10),
        (true_y > 10) & (true_y <= 100),
        true_y > 100,
    ]
    bucket_totals = [weights[mask].sum().item() for mask in bucket_masks]
    assert bucket_totals == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])


def test_bram_nonzero_balanced_sampler_gives_zero_and_nonzero_equal_total_weight():
    args = stable.build_parser().parse_args(["--train-sampler", "bram-nonzero-balanced"])
    dataset = [
        _bram_sample(0.0),
        _bram_sample(0.0),
        _bram_sample(5.0),
        _bram_sample(50.0),
        _bram_sample(200.0),
    ]

    weights = stable.train_sample_weights(dataset, stable.ORIGINAL_TARGET_SPECS["bram"], args)

    true_y = torch.tensor([0.0, 0.0, 5.0, 50.0, 200.0])
    assert weights[true_y == 0].sum().item() == pytest.approx(1.0)
    assert weights[true_y > 0].sum().item() == pytest.approx(1.0)


def test_prediction_stats_summary_uses_population_std_and_quantiles():
    summary = summarize_values(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(1.11803398875)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["max"] == pytest.approx(4.0)


def test_prediction_stats_bucketizes_bram_tail_errors():
    true = torch.tensor([0.0, 0.5, 5.0, 50.0, 200.0])
    abs_error = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    buckets = summarize_error_buckets(true, abs_error)

    assert [bucket["count"] for bucket in buckets] == [1, 1, 1, 1, 1]
    assert buckets[0]["bucket"] == "true == 0"
    assert buckets[-1]["mae"] == pytest.approx(0.5)


def test_safe_optimizer_step_rejects_nonfinite_loss_and_restores_weights():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    before = stable.snapshot_model_state(model)

    result = stable.safe_optimizer_step(
        loss=torch.tensor(float("nan"), requires_grad=True),
        model=model,
        optimizer=optimizer,
        grad_clip=1.0,
        fallback_state=before,
    )

    assert not result.applied
    assert result.reason == "non-finite loss"
    assert stable.model_state_all_finite(model)
    assert torch.equal(model.weight.detach(), before["weight"])


def test_builtin_initial_checkpoint_path_uses_target_checkpoint_name():
    args = stable.build_parser().parse_args(["--targets", "dsp", "--init-from-builtins"])

    assert stable.initial_checkpoint_path("dsp", args) == stable.HGBO_ROOT / "hgp" / "model" / "dsp_mae_h64_d0_checkpoint_test.pt"


def test_initial_checkpoint_dir_overrides_builtin_checkpoint_path(tmp_path):
    args = stable.build_parser().parse_args(
        [
            "--targets",
            "bram",
            "--init-from-builtins",
            "--init-checkpoint-dir",
            str(tmp_path),
        ]
    )

    assert stable.initial_checkpoint_path("bram", args) == tmp_path / "bram_mae_h64_d0_checkpoint_test.pt"


def test_load_initial_checkpoint_loads_model_state(tmp_path):
    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    for parameter in source.parameters():
        parameter.data.fill_(0.25)
    for parameter in target.parameters():
        parameter.data.zero_()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": source.state_dict(), "epoch": 7}, checkpoint_path)

    metadata = stable.load_initial_checkpoint(target, checkpoint_path, torch.device("cpu"))

    assert metadata["path"] == str(checkpoint_path)
    assert metadata["epoch"] == 7
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)
