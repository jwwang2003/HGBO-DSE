import pytest


torch = pytest.importorskip("torch")
Data = pytest.importorskip("torch_geometric.data").Data

from hgp.dataset_utils import generate_dataset  # noqa: E402


def test_generate_dataset_ignores_non_pt_files_and_sorts_inputs(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    sample_a = Data(x=torch.tensor([[1.0]]), y=torch.tensor([[1.0]]))
    sample_b = Data(x=torch.tensor([[2.0]]), y=torch.tensor([[2.0]]))
    torch.save([sample_b], dataset_dir / "b.pt")
    torch.save([sample_a], dataset_dir / "a.pt")
    (dataset_dir / "README.md").write_text("not a torch dataset\n", encoding="utf-8")

    dataset = generate_dataset(str(dataset_dir), ["README.md", "b.pt", "a.pt"])

    assert [float(data.y.item()) for data in dataset] == [1.0, 2.0]
