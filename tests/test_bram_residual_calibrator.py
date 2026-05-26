import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data

from hgp.reporting import bram_residual_calibrator as calibrator  # noqa: E402


def _sample(*, true_bram, hls_bram):
    y = torch.zeros((1, 8), dtype=torch.float32)
    y[0, 3] = true_bram
    return Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.long),
        hls_attr=torch.tensor([[10.0, 20.0, 30.0, hls_bram, 0.0, 5.0]], dtype=torch.float32),
        y=y,
    )


def test_build_feature_matrix_uses_hls_bram_residual_labels():
    dataset = [_sample(true_bram=64.0, hls_bram=60.0), _sample(true_bram=5.0, hls_bram=9.0)]

    features, true, hls, residual = calibrator.build_feature_matrix(dataset, hls_index=3)

    assert features.shape == (2, 15)
    assert true.tolist() == pytest.approx([64.0, 5.0])
    assert hls.tolist() == pytest.approx([60.0, 9.0])
    assert residual.tolist() == [4, -4]
    assert features[0, :6].tolist() == pytest.approx([10.0, 20.0, 30.0, 60.0, 0.0, 5.0])
    assert features[0, -3:].tolist() == pytest.approx([2.0, 3.0, 60.0])


def test_evaluate_predictions_reports_bram_mae_and_residual_accuracy():
    true = np.array([64.0, 5.0, 0.0])
    hls = np.array([60.0, 9.0, 0.0])
    residual = np.array([4, -4, 0])
    predicted_residual = np.array([4, -3, 0])

    result = calibrator.evaluate_predictions(true=true, hls=hls, residual=residual, predicted_residual=predicted_residual)

    assert result["mae"] == pytest.approx(1.0 / 3.0)
    assert result["residual_accuracy"] == pytest.approx(2.0 / 3.0)
    assert result["pred"]["max"] == pytest.approx(64.0)
    assert result["true_buckets"][0]["bucket"] == "true == 0"
