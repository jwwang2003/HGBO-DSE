from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]


def load_compose():
    return yaml.safe_load((REPO / "docker-compose.yaml").read_text())


def normalize_environment(service):
    environment = service.get("environment") or {}
    if isinstance(environment, list):
        return dict(item.split("=", 1) for item in environment)
    return environment


def test_compose_defines_mcp_inference_service():
    compose = load_compose()
    services = compose["services"]

    assert set(services) == {"mcp-inference"}
    mcp_service = services["mcp-inference"]
    assert mcp_service["build"] == "."
    assert mcp_service["working_dir"] == "/app"
    assert mcp_service["command"] == "uv run python -m backend.mcp_server"
    assert normalize_environment(mcp_service) == {
        "HGBO_MCP_SECRET_DIR": "/var/lib/hgbo-mcp",
        "HGBO_MCP_HOST": "0.0.0.0",
        "HGBO_MCP_PORT": "8000",
    }
    assert mcp_service["ports"] == ["8000:8000"]
    assert mcp_service["volumes"] == ["hgbo_mcp_data:/var/lib/hgbo-mcp"]


def test_compose_uses_named_mcp_data_volume_without_workspace_mounts():
    compose_source = (REPO / "docker-compose.yaml").read_text()
    compose = load_compose()

    assert "hgbo_mcp_data" in compose["volumes"]
    assert "${HGBO_SHARED_HOST_ROOT" not in compose_source
    assert "HGBO_SHARED_CONTAINER_ROOT" not in compose_source
    assert ".compass" not in compose_source
    assert "redis" not in compose["services"]
    assert "celery" not in compose["services"]
