import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_pyproject_declares_hgbo_runtime_dependencies():
    pyproject = read_pyproject()
    dependencies = set(pyproject["project"]["dependencies"])

    assert "optuna==3.2.0" in dependencies
    assert "torch==2.6.0" in dependencies
    assert "torchvision==0.21.0" in dependencies
    assert "torchaudio==2.6.0" in dependencies
    assert "torch-geometric==2.6.0" in dependencies
    assert "torch-scatter==2.1.2" in dependencies
    assert "torch-sparse==0.6.18" in dependencies
    assert "torch==2.6.0+cpu" not in dependencies


def test_pyproject_routes_pytorch_and_pyg_wheels_to_uv_indexes():
    pyproject = read_pyproject()
    sources = pyproject["tool"]["uv"]["sources"]
    indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}

    assert sources["torch"]["index"] == "pytorch-cpu"
    assert sources["torchvision"]["index"] == "pytorch-cpu"
    assert sources["torchaudio"]["index"] == "pytorch-cpu"
    assert sources["torch-scatter"]["index"] == "pyg-torch-260-cpu"
    assert sources["torch-sparse"]["index"] == "pyg-torch-260-cpu"
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True
    assert indexes["pyg-torch-260-cpu"]["url"] == "https://data.pyg.org/whl/torch-2.6.0+cpu.html"
    assert indexes["pyg-torch-260-cpu"]["format"] == "flat"
    assert indexes["pyg-torch-260-cpu"]["explicit"] is True


def test_dockerfile_and_docs_use_uv_sync():
    dockerfile = (ROOT / "Dockerfile").read_text()
    readme = (ROOT / "README.dev.md").read_text()
    docker_readme = (ROOT / "README.dockerfile.md").read_text()

    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --no-install-project" in dockerfile
    assert "uv pip install -r requirements.txt" not in dockerfile
    assert "uv sync" in readme
    assert "uv pip install -r requirements.txt" not in readme
    assert "uv sync --no-install-project" in docker_readme
    assert "COPY requirements.txt" not in docker_readme
