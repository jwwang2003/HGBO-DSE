try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9
    import tomli as tomllib
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
    assert "rapidwright" in dependencies
    assert "jpype1==1.6.0" in dependencies
    assert "torch==2.6.0+cpu" not in dependencies


def test_pyproject_routes_pytorch_and_pyg_wheels_to_cuda_uv_indexes():
    pyproject = read_pyproject()
    sources = pyproject["tool"]["uv"]["sources"]
    indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}

    assert sources["torch"]["index"] == "pytorch-cu124"
    assert sources["torchvision"]["index"] == "pytorch-cu124"
    assert sources["torchaudio"]["index"] == "pytorch-cu124"
    assert sources["torch-scatter"]["index"] == "pyg-torch-260-cu124"
    assert sources["torch-sparse"]["index"] == "pyg-torch-260-cu124"
    assert indexes["pytorch-cu124"]["url"] == "https://download.pytorch.org/whl/cu124"
    assert indexes["pytorch-cu124"]["explicit"] is True
    assert indexes["pyg-torch-260-cu124"]["url"] == "https://data.pyg.org/whl/torch-2.6.0+cu124.html"
    assert indexes["pyg-torch-260-cu124"]["format"] == "flat"
    assert indexes["pyg-torch-260-cu124"]["explicit"] is True


def test_pyproject_routes_rapidwright_to_local_source():
    pyproject = read_pyproject()
    sources = pyproject["tool"]["uv"]["sources"]

    assert sources["rapidwright"]["path"] == "../RapidWright/python"
    assert sources["rapidwright"]["editable"] is True


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


def test_readme_mentions_rapidwright_cache_workflow():
    readme = (ROOT / "README.md").read_text()
    dev_readme = (ROOT / "README.dev.md").read_text()

    assert "RapidWright" in readme
    assert "std_arch" in readme
    assert "rdc_arch" in readme
    assert "dataset/board_arch" in dev_readme
