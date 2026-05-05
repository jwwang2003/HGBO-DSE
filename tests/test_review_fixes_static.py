import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_temp_folder_manager_import_path_is_live():
    from backend.tempFolderManager import TempFolderManager

    assert hasattr(TempFolderManager, "create_temp_folder")
    assert hasattr(TempFolderManager, "clean_old_folders")


def test_entrypoint_skips_xilinx_setup_by_default(tmp_path):
    result = subprocess.run(
        [str(REPO / "entrypoint.sh"), "bash", "-lc", "echo started"],
        env={**os.environ, "HOME": str(tmp_path), "XILINX_INSTALL": ""},
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert "started" in result.stdout
    bashrc = tmp_path / ".bashrc"
    assert not bashrc.exists() or "settings64.sh" not in bashrc.read_text()


def test_entrypoint_requires_xilinx_only_when_requested(tmp_path):
    result = subprocess.run(
        [str(REPO / "entrypoint.sh"), "bash", "-lc", "echo unreachable"],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "REQUIRE_XILINX": "1",
            "XILINX_INSTALL": str(tmp_path / "missing-xilinx"),
        },
        text=True,
        capture_output=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Xilinx settings file not found" in result.stderr


def test_backend_admin_credentials_are_not_hard_coded():
    source = (REPO / "backend" / "entry.py").read_text()

    assert 'ADMIN_USER = "admin"' not in source
    assert 'ADMIN_PASS = "s3cr3t"' not in source
    assert "HGBO_ADMIN_USER" in source
    assert "HGBO_ADMIN_PASS" in source


def test_websocket_background_task_is_async():
    source = (REPO / "backend" / "entry.py").read_text()

    assert "async def long_task" in source
    assert "asyncio.get_event_loop()" not in source


def test_celery_app_uses_real_env_driven_app_and_compose_uses_mcp_server():
    celery_source = (REPO / "helpers" / "celery_app.py").read_text()
    compose_source = (REPO / "docker-compose.yaml").read_text()
    tasks_source = (REPO / "backend" / "tasks.py").read_text()

    assert "CELERY_BROKER_URL" in celery_source
    assert "CELERY_RESULT_BACKEND" in celery_source
    assert "redis://localhost:6379/0" in celery_source
    assert 'include=[\'backend.tasks\']' in celery_source
    assert '@app.task(name="backend.tasks.predict_impl_ppa")' in tasks_source
    assert "getGNNPred" in tasks_source
    assert "uv run python -m backend.mcp_server" in compose_source
    assert "HGBO_MCP_SECRET_DIR=/var/lib/hgbo-mcp" in compose_source
    assert "celery -A" not in compose_source
    assert "redis://redis:6379/0" not in compose_source


def test_dockerfile_uses_uv_for_python_dependencies():
    dockerfile = (REPO / "Dockerfile").read_text()

    assert "uv pip install" in dockerfile or "uv sync" in dockerfile
    assert "pip install --upgrade pip" not in dockerfile


def test_pyproject_declares_uv_project_metadata():
    pyproject = (REPO / "pyproject.toml").read_text()

    assert 'name = "hgbo-dse"' in pyproject
    assert "[tool.uv]" in pyproject


def test_impl_verification_entrypoint_replays_selected_trials():
    source = (REPO / "bome" / "impl_verify.py").read_text()

    assert "selected_trials.json" in source
    assert '"impl"' in source
    assert "class FixedTrial" in source
    assert "config_tree_space" in source
    assert "genDirConfig" in source
    assert "VitisHLSRunner" in source
    assert "getPPA" in source
    assert "normalizePCA" in source
    assert "normalizePLCA" in source
    assert "impl_verification.json" in source
    assert '"results"' in source
    assert '"study"' in source
    assert '"implementation"' in source
    assert "Completed implementation for trial" in source
