from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import backend.mcp_client as mcp_client
import bome.inference as inference


def _set_remote_env(monkeypatch, timeout=None):
    monkeypatch.setenv("HGBO_MCP_URL", "https://mcp.example.test/mcp")
    monkeypatch.setenv("HGBO_MCP_API_KEY", "secret-key")
    if timeout is None:
        monkeypatch.delenv("HGBO_REMOTE_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("HGBO_REMOTE_TIMEOUT_SEC", str(timeout))


def _ppa(**overrides):
    result = {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4, "CP": 5, "PWR": 6}
    result.update(overrides)
    return result


def test_read_remote_config_requires_url(monkeypatch):
    monkeypatch.delenv("HGBO_MCP_URL", raising=False)
    monkeypatch.setenv("HGBO_MCP_API_KEY", "secret-key")

    with pytest.raises(RuntimeError, match="HGBO_MCP_URL"):
        mcp_client.read_remote_config()


def test_read_remote_config_requires_api_key(monkeypatch):
    monkeypatch.setenv("HGBO_MCP_URL", "https://mcp.example.test/mcp")
    monkeypatch.delenv("HGBO_MCP_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="HGBO_MCP_API_KEY"):
        mcp_client.read_remote_config()


def test_read_remote_config_uses_default_timeout(monkeypatch):
    _set_remote_env(monkeypatch)

    config = mcp_client.read_remote_config()

    assert config.url == "https://mcp.example.test/mcp"
    assert config.api_key == "secret-key"
    assert config.timeout == 600


def test_read_remote_config_parses_timeout(monkeypatch):
    _set_remote_env(monkeypatch, timeout=17.5)

    config = mcp_client.read_remote_config()

    assert config.timeout == 17.5


def test_read_remote_config_rejects_invalid_timeout(monkeypatch):
    _set_remote_env(monkeypatch, timeout="slow")

    with pytest.raises(RuntimeError, match="HGBO_REMOTE_TIMEOUT_SEC"):
        mcp_client.read_remote_config()


def test_jsonrpc_result_to_dict_prefers_structured_content_dict():
    result = mcp_client.jsonrpc_result_to_dict(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "result": {"structuredContent": {"status": "ok"}},
        }
    )

    assert result == {"status": "ok"}


def test_jsonrpc_result_to_dict_parses_first_text_json_object():
    result = mcp_client.jsonrpc_result_to_dict(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps({"status": "ok"})},
                ],
            },
        }
    )

    assert result == {"status": "ok"}


def test_jsonrpc_result_to_dict_raises_jsonrpc_error_message():
    with pytest.raises(RuntimeError, match="Tool failed"):
        mcp_client.jsonrpc_result_to_dict(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "error": {"code": -32603, "message": "Tool failed"},
            }
        )


def test_jsonrpc_result_to_dict_requires_json_object():
    with pytest.raises(RuntimeError, match="JSON object"):
        mcp_client.jsonrpc_result_to_dict(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps(["not", "an", "object"])},
                    ],
                },
            }
        )


def test_call_tool_posts_json_rpc_and_returns_dict(monkeypatch):
    monkeypatch.setenv("HGBO_CONTEXT_ID", "ctx-123")
    requests = []

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "id": "req-1",
                "result": {"structuredContent": {"status": "ok"}},
            }

        async def text(self):
            return ""

    class FakeSession:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, headers, json):
            requests.append((url, headers, json, self.timeout.total))
            return FakeResponse()

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeSession)

    result = asyncio.run(
        mcp_client.call_tool(
            "https://mcp.example.test/mcp",
            "secret-key",
            "ping",
            {},
            42,
        )
    )

    assert result == {"status": "ok"}
    assert len(requests) == 1
    url, headers, payload, timeout = requests[0]
    assert url == "https://mcp.example.test/mcp"
    assert headers["Authorization"] == "Bearer secret-key"
    assert payload["jsonrpc"] == "2.0"
    assert payload["id"] == "ctx-123"
    assert payload["method"] == "tools/call"
    assert payload["params"] == {"name": "ping", "arguments": {}}
    assert timeout == 42


def test_call_tool_maps_client_error_to_unavailable_runtime_error(monkeypatch):
    class FakeSession:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, headers, json):
            raise mcp_client.aiohttp.ClientError("connection refused")

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(RuntimeError, match="MCP server unavailable"):
        asyncio.run(
            mcp_client.call_tool(
                "https://mcp.example.test/mcp",
                "secret-key",
                "ping",
                {},
                42,
            )
        )


def test_call_tool_maps_timeout_to_runtime_error(monkeypatch):
    class FakeSession:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, headers, json):
            raise asyncio.TimeoutError()

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(RuntimeError, match="MCP request timed out"):
        asyncio.run(
            mcp_client.call_tool(
                "https://mcp.example.test/mcp",
                "secret-key",
                "ping",
                {},
                42,
            )
        )


def test_call_tool_maps_unauthorized_response(monkeypatch):
    class FakeResponse:
        status = 401

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def text(self):
            return "unauthorized"

    class FakeSession:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, headers, json):
            return FakeResponse()

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(RuntimeError, match="MCP authentication failed"):
        asyncio.run(
            mcp_client.call_tool(
                "https://mcp.example.test/mcp",
                "bad-key",
                "ping",
                {},
                42,
            )
        )


def test_call_tool_non_auth_http_error_does_not_include_response_body(monkeypatch):
    class FakeResponse:
        status = 500

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def text(self):
            return "debug echo Authorization: Bearer leaked-secret-token"

    class FakeSession:
        def __init__(self, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, headers, json):
            return FakeResponse()

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeSession)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            mcp_client.call_tool(
                "https://mcp.example.test/mcp",
                "secret-key",
                "ping",
                {},
                42,
            )
        )

    message = str(exc_info.value)
    assert message == "MCP request failed with HTTP 500"
    assert "Authorization" not in message
    assert "Bearer" not in message
    assert "leaked-secret-token" not in message


def test_list_tools_async_posts_tools_list_json_rpc_method(monkeypatch):
    _set_remote_env(monkeypatch, timeout=19)
    monkeypatch.setenv("HGBO_CONTEXT_ID", "ctx-list")
    requests = []

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def json(self):
            return {
                "jsonrpc": "2.0",
                "id": "ctx-list",
                "result": {"tools": [{"name": "ping"}]},
            }

        async def text(self):
            return ""

    class FakeSession:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, headers, json):
            requests.append((url, headers, json, self.timeout.total))
            return FakeResponse()

    monkeypatch.setattr(mcp_client.aiohttp, "ClientSession", FakeSession)

    result = asyncio.run(mcp_client.list_tools_async())

    assert result == {"tools": [{"name": "ping"}]}
    assert len(requests) == 1
    url, headers, payload, timeout = requests[0]
    assert url == "https://mcp.example.test/mcp"
    assert headers["Authorization"] == "Bearer secret-key"
    assert payload == {"jsonrpc": "2.0", "id": "ctx-list", "method": "tools/list"}
    assert timeout == 19.0


def test_ping_async_calls_ping_tool(monkeypatch):
    _set_remote_env(monkeypatch, timeout=11)
    calls = []

    async def fake_call_tool(url, api_key, tool_name, arguments, timeout):
        calls.append((url, api_key, tool_name, arguments, timeout))
        return {"status": "ok"}

    monkeypatch.setattr(mcp_client, "call_tool", fake_call_tool)

    result = asyncio.run(mcp_client.ping_async())

    assert result == {"status": "ok"}
    assert calls == [
        ("https://mcp.example.test/mcp", "secret-key", "ping", {}, 11.0),
    ]


def test_main_list_tools_prints_sorted_json(monkeypatch, capsys):
    async def fake_list_tools_async():
        return {"z": 2, "a": 1}

    monkeypatch.setattr(
        mcp_client,
        "list_tools_async",
        fake_list_tools_async,
        raising=False,
    )

    assert mcp_client.main(["--list-tools"]) == 0
    assert capsys.readouterr().out == '{"a": 1, "z": 2}\n'


def test_main_ping_prints_sorted_json(monkeypatch, capsys):
    async def fake_ping_async():
        return {"z": 2, "a": 1}

    monkeypatch.setattr(mcp_client, "ping_async", fake_ping_async)

    assert mcp_client.main(["--ping"]) == 0
    assert capsys.readouterr().out == '{"a": 1, "z": 2}\n'


def test_predict_impl_ppa_async_packages_graph_and_validates_result(monkeypatch, tmp_path):
    _set_remote_env(monkeypatch, timeout=23)
    monkeypatch.setenv("HGBO_CONTEXT_ID", "ctx-123")
    project = tmp_path / "project"
    project.mkdir()
    calls = []

    def fake_create_graph_archive_base64(prj_path):
        calls.append(("archive", prj_path))
        return "encoded-graph"

    async def fake_call_tool(url, api_key, tool_name, arguments, timeout):
        calls.append((url, api_key, tool_name, arguments, timeout))
        return _ppa(ignored=99)

    monkeypatch.setattr(
        mcp_client,
        "create_graph_archive_base64",
        fake_create_graph_archive_base64,
    )
    monkeypatch.setattr(mcp_client, "call_tool", fake_call_tool)

    result = asyncio.run(
        mcp_client.predict_impl_ppa_async(project, (7, 8, 9), "bfs")
    )

    assert result == _ppa()
    assert calls == [
        ("archive", project),
        (
            "https://mcp.example.test/mcp",
            "secret-key",
            "predict_impl_ppa",
            {
                "case": "bfs",
                "hls_attr": [7, 8, 9],
                "graph_archive_base64": "encoded-graph",
                "archive_format": "zip",
            },
            23.0,
        ),
    ]


def test_predict_impl_ppa_async_maps_missing_ppa_keys_to_runtime_error(
    monkeypatch,
    tmp_path,
):
    _set_remote_env(monkeypatch)
    project = tmp_path / "project"
    project.mkdir()

    monkeypatch.setattr(
        mcp_client,
        "create_graph_archive_base64",
        lambda prj_path: "encoded-graph",
    )

    async def fake_call_tool(url, api_key, tool_name, arguments, timeout):
        return {"LUT": 1}

    monkeypatch.setattr(mcp_client, "call_tool", fake_call_tool)

    with pytest.raises(RuntimeError, match="invalid PPA response") as exc_info:
        asyncio.run(mcp_client.predict_impl_ppa_async(project, [1, 2, 3], "bfs"))

    assert "missing" in str(exc_info.value)


def test_dispatch_remote_inference_uses_mcp_client(monkeypatch):
    calls = []

    def fake_predict_impl_ppa(prj_path, hls_attr, case, timeout=None):
        calls.append((prj_path, hls_attr, case, timeout))
        return _ppa()

    monkeypatch.setattr(mcp_client, "predict_impl_ppa", fake_predict_impl_ppa)

    result = inference.dispatch_remote_inference(
        "/tmp/project",
        (1, 2, 3),
        "bfs",
        timeout=13,
    )

    assert result == _ppa()
    assert calls == [("/tmp/project", [1, 2, 3], "bfs", 13)]


def test_hls_dse_inference_mode_help_mentions_mcp_not_celery():
    source = (
        Path(__file__).resolve().parents[1] / "bome" / "hls_dse.py"
    ).read_text(encoding="utf-8")

    assert "dispatch it to a Celery worker" not in source
    assert "MCP service" in source
