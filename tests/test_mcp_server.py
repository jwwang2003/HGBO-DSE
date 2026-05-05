from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import types
import zipfile

import pytest

import backend.mcp_server as mcp_server
from backend.mcp_auth import ApiKeyStore


def _graph_archive_base64() -> str:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("graph/kernel.adb", "graph data")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_predict_impl_ppa_impl_extracts_payload_and_cleans_up(tmp_path, monkeypatch):
    temp_root = tmp_path / "request-work"
    calls = []
    fake_module = types.ModuleType("bome.hgp_pred")

    def fake_get_gnn_pred(prj_path, hls_attr, case):
        calls.append((prj_path, hls_attr, case))
        assert (temp_root / "prj" / "graph" / "kernel.adb").read_text(
            encoding="utf-8"
        ) == "graph data"
        return {
            "LUT": 1,
            "FF": 2,
            "DSP": 3,
            "BRAM": 4,
            "CP": 5,
            "PWR": 6,
            "ignored": 7,
        }

    fake_module.getGNNPred = fake_get_gnn_pred
    monkeypatch.setitem(sys.modules, "bome.hgp_pred", fake_module)

    result = mcp_server.predict_impl_ppa_impl(
        case="bfs",
        hls_attr=(10, 20, 30),
        graph_archive_base64=_graph_archive_base64(),
        request_id="req-1",
        temp_root=temp_root,
    )

    assert result == {"LUT": 1, "FF": 2, "DSP": 3, "BRAM": 4, "CP": 5, "PWR": 6}
    assert calls == [(str(temp_root / "prj"), [10, 20, 30], "bfs")]
    assert not temp_root.exists()


def test_predict_impl_ppa_impl_rejects_unsupported_archive_format(tmp_path):
    with pytest.raises(ValueError, match="archive_format"):
        mcp_server.predict_impl_ppa_impl(
            case="bfs",
            hls_attr=[1, 2, 3],
            graph_archive_base64=_graph_archive_base64(),
            archive_format="tar",
            temp_root=tmp_path / "request-work",
        )


def test_json_rpc_tools_list_includes_ping_and_predict_impl_ppa():
    response = mcp_server.handle_json_rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )

    names = {tool["name"] for tool in response["result"]["tools"]}
    assert {"ping", "predict_impl_ppa"} <= names


def test_json_rpc_initialize_returns_mcp_metadata():
    response = mcp_server.handle_json_rpc(
        {"jsonrpc": "2.0", "id": "init-1", "method": "initialize"}
    )

    result = response["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"]


def test_json_rpc_initialized_notification_returns_no_response():
    assert (
        mcp_server.handle_json_rpc(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )
        is None
    )


def test_json_rpc_initialized_with_id_returns_success_response():
    response = mcp_server.handle_json_rpc(
        {"jsonrpc": "2.0", "id": "initialized-1", "method": "notifications/initialized"}
    )

    assert response == {"jsonrpc": "2.0", "id": "initialized-1", "result": {}}


def test_json_rpc_no_id_tools_list_notification_returns_no_response():
    assert (
        mcp_server.handle_json_rpc({"jsonrpc": "2.0", "method": "tools/list"})
        is None
    )


def test_json_rpc_batch_notifications_return_no_response():
    assert (
        mcp_server.handle_json_rpc(
            [
                {"jsonrpc": "2.0", "method": "tools/list"},
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            ]
        )
        is None
    )


@pytest.mark.parametrize("bad_id", [{"nested": "id"}, ["bad-id"]])
def test_json_rpc_rejects_invalid_id_type(bad_id):
    response = mcp_server.handle_json_rpc(
        {"jsonrpc": "2.0", "id": bad_id, "method": "tools/list"}
    )

    assert response["id"] is None
    assert response["error"]["code"] == -32600


@pytest.mark.parametrize("bad_id", [float("nan"), float("inf"), float("-inf")])
def test_json_rpc_rejects_non_finite_float_id_without_serialization_failure(bad_id):
    response = mcp_server.handle_json_rpc(
        {"jsonrpc": "2.0", "id": bad_id, "method": "tools/list"}
    )

    assert response["id"] is None
    assert response["error"]["code"] == -32600
    json.dumps(response, allow_nan=False)


def test_json_rpc_tools_call_ping_returns_structured_content():
    response = mcp_server.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "ping-1",
            "method": "tools/call",
            "params": {"name": "ping", "arguments": {}},
        }
    )

    result = response["result"]
    assert result["structuredContent"] == {"status": "ok"}
    assert json.loads(result["content"][0]["text"]) == {"status": "ok"}


def test_json_rpc_tools_call_ping_rejects_non_dict_arguments():
    response = mcp_server.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "ping-bad-args",
            "method": "tools/call",
            "params": {"name": "ping", "arguments": []},
        }
    )

    assert response["error"]["code"] == -32602


def test_json_rpc_unknown_tool_returns_error():
    response = mcp_server.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "bad-tool",
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert response["error"]["code"] == -32601
    assert "result" not in response


@pytest.mark.parametrize(
    "arguments",
    [
        {"case": "bfs", "hls_attr": [1, 2, 3]},
        {"case": 1, "hls_attr": [1, 2, 3], "graph_archive_base64": "payload"},
        {"case": "bfs", "hls_attr": "bad", "graph_archive_base64": "payload"},
        {"case": "bfs", "hls_attr": [1, 2, 3], "graph_archive_base64": 42},
        {
            "case": "bfs",
            "hls_attr": [1, 2, 3],
            "graph_archive_base64": "payload",
            "archive_format": [],
        },
    ],
)
def test_json_rpc_predict_rejects_missing_or_wrong_argument_types(
    arguments, monkeypatch
):
    calls = []

    def fail_if_called(**kwargs):
        calls.append(kwargs)
        raise AssertionError("predict_impl_ppa_impl should not be called")

    monkeypatch.setattr(mcp_server, "predict_impl_ppa_impl", fail_if_called)

    response = mcp_server.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "predict-bad-args",
            "method": "tools/call",
            "params": {
                "name": "predict_impl_ppa",
                "arguments": arguments,
            },
        }
    )

    assert response["error"]["code"] == -32602
    assert calls == []


def test_json_rpc_tool_exception_returns_sanitized_error(monkeypatch):
    def fail_predict(**kwargs):
        raise RuntimeError("api key secret traceback")

    monkeypatch.setattr(mcp_server, "predict_impl_ppa_impl", fail_predict)

    response = mcp_server.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "predict-1",
            "method": "tools/call",
            "params": {
                "name": "predict_impl_ppa",
                "arguments": {
                    "case": "bfs",
                    "hls_attr": [1, 2, 3],
                    "graph_archive_base64": "payload",
                },
            },
        }
    )

    encoded = json.dumps(response).lower()
    assert response["error"]["code"] == -32603
    assert "api key" not in encoded
    assert "secret" not in encoded
    assert "traceback" not in encoded


async def _call_asgi(app, method, path, headers=None, body=b""):
    sent = []
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "headers": headers or [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    status = sent[0]["status"]
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, response_body


def test_create_app_starlette_path_returns_parse_error_for_invalid_utf8_body(
    tmp_path, monkeypatch
):
    class FakeRoute:
        def __init__(self, path, endpoint, methods):
            self.path = path
            self.endpoint = endpoint
            self.methods = methods

    class FakeRequest:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return json.loads(self._body.decode("utf-8"))

        async def body(self):
            return self._body

    class FakeJSONResponse:
        def __init__(self, payload, status_code=200):
            self.status_code = status_code
            self.body = json.dumps(payload).encode("utf-8")

    class FakeResponse:
        def __init__(self, status_code=200):
            self.status_code = status_code
            self.body = b""

    class FakeStarlette:
        def __init__(self, routes):
            self.routes = routes

        async def __call__(self, scope, receive, send):
            body = await mcp_server._read_body(receive)
            for route in self.routes:
                if route.path == scope["path"] and scope["method"] in route.methods:
                    response = await route.endpoint(FakeRequest(body))
                    await send(
                        {
                            "type": "http.response.start",
                            "status": response.status_code,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (
                                    b"content-length",
                                    str(len(response.body)).encode("ascii"),
                                ),
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": response.body})
                    return
            await mcp_server._send_empty(send, 404)

    fake_root = types.ModuleType("starlette")
    fake_root.__path__ = []
    fake_applications = types.ModuleType("starlette.applications")
    fake_applications.Starlette = FakeStarlette
    fake_requests = types.ModuleType("starlette.requests")
    fake_requests.Request = FakeRequest
    fake_responses = types.ModuleType("starlette.responses")
    fake_responses.JSONResponse = FakeJSONResponse
    fake_responses.Response = FakeResponse
    fake_routing = types.ModuleType("starlette.routing")
    fake_routing.Route = FakeRoute

    monkeypatch.setitem(sys.modules, "starlette", fake_root)
    monkeypatch.setitem(sys.modules, "starlette.applications", fake_applications)
    monkeypatch.setitem(sys.modules, "starlette.requests", fake_requests)
    monkeypatch.setitem(sys.modules, "starlette.responses", fake_responses)
    monkeypatch.setitem(sys.modules, "starlette.routing", fake_routing)

    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    app = mcp_server.create_app(store=store)

    status, body = asyncio.run(
        _call_asgi(
            app,
            "POST",
            "/mcp",
            headers=[(b"x-api-key", b"secret"), (b"content-type", b"application/json")],
            body=b"\xff",
        )
    )

    assert status == 400
    response = json.loads(body)
    assert response["error"]["code"] == -32700


def test_create_app_rejects_raw_json_nan_id_without_500(tmp_path):
    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    app = mcp_server.create_app(store=store)

    status, body = asyncio.run(
        _call_asgi(
            app,
            "POST",
            "/mcp",
            headers=[(b"x-api-key", b"secret"), (b"content-type", b"application/json")],
            body=b'{"jsonrpc":"2.0","id":NaN,"method":"tools/list"}',
        )
    )

    assert status in (200, 400)
    response = json.loads(body)
    assert response["error"]["code"] in (-32700, -32600)


def test_create_app_protects_mcp_and_leaves_healthz_unprotected(tmp_path):
    store = ApiKeyStore(tmp_path / "api_key")
    store.write_api_key_for_tests("secret")
    app = mcp_server.create_app(store=store)

    health_status, health_body = asyncio.run(_call_asgi(app, "GET", "/healthz"))
    assert health_status == 200
    assert json.loads(health_body) == {"status": "ok"}

    missing_status, _ = asyncio.run(
        _call_asgi(
            app,
            "POST",
            "/mcp",
            body=json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            ).encode("utf-8"),
        )
    )
    assert missing_status == 401

    ok_status, ok_body = asyncio.run(
        _call_asgi(
            app,
            "POST",
            "/mcp",
            headers=[(b"x-api-key", b"secret"), (b"content-type", b"application/json")],
            body=json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
            ).encode("utf-8"),
        )
    )
    assert ok_status == 200
    assert json.loads(ok_body)["id"] == 2


def test_main_show_api_key_prints_persistent_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HGBO_MCP_SECRET_DIR", str(tmp_path))

    assert mcp_server.main(["--show-api-key"]) == 0
    first = capsys.readouterr().out.strip()
    assert first
    assert (tmp_path / "api_key").read_text(encoding="utf-8").strip() == first

    assert mcp_server.main(["--show-api-key"]) == 0
    second = capsys.readouterr().out.strip()
    assert second == first
