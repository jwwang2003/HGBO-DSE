from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

from backend.mcp_auth import ApiKeyAuthMiddleware, ApiKeyStore
from backend.mcp_payload import extract_graph_archive_base64, validate_ppa_result


PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "hgbo-dse-mcp"
SERVER_VERSION = "0.1.0"
DEFAULT_SECRET_DIR = "/var/lib/hgbo-mcp"
SECRET_DIR_ENV = "HGBO_MCP_SECRET_DIR"
API_KEY_FILENAME = "api_key"

JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603

PathLike = Union[str, os.PathLike]
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def ping() -> Dict[str, str]:
    return {"status": "ok"}


def predict_impl_ppa_impl(
    case: str,
    hls_attr: Sequence[Any],
    graph_archive_base64: str,
    archive_format: str = "zip",
    request_id: str = "",
    temp_root: Optional[PathLike] = None,
) -> Dict[str, Any]:
    if archive_format != "zip":
        raise ValueError("archive_format must be 'zip'")

    root_path = _make_temp_root(request_id, temp_root)
    try:
        prj_path = root_path / "prj"
        prj_path.mkdir(parents=True, exist_ok=True)
        extract_graph_archive_base64(graph_archive_base64, prj_path)

        from bome.hgp_pred import getGNNPred

        result = getGNNPred(str(prj_path), list(hls_attr), case)
        return validate_ppa_result(result)
    finally:
        shutil.rmtree(str(root_path), ignore_errors=True)


def handle_json_rpc(message: Any) -> Any:
    if isinstance(message, list):
        if not message:
            return _json_rpc_error(None, JSONRPC_INVALID_REQUEST, "Invalid Request")
        responses = []
        for item in message:
            response = _handle_single_json_rpc(item)
            if response is not None:
                responses.append(response)
        return responses or None

    return _handle_single_json_rpc(message)


def create_app(store: Optional[ApiKeyStore] = None) -> ApiKeyAuthMiddleware:
    api_key_store = store or _default_api_key_store()
    app = _build_mcp_app()
    return ApiKeyAuthMiddleware(app, store=api_key_store, protected_prefix="/mcp")


def _build_mcp_app() -> ASGIApp:
    try:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Route
    except ModuleNotFoundError:
        return _fallback_asgi_app

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse(ping())

    async def mcp_endpoint(request: Request) -> Response:
        try:
            body = await request.body()
            payload = _loads_strict_json(body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return JSONResponse(
                _json_rpc_error(None, JSONRPC_PARSE_ERROR, "Parse error"),
                status_code=400,
            )

        response = handle_json_rpc(payload)
        if response is None:
            return Response(status_code=204)
        return JSONResponse(response)

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/mcp", mcp_endpoint, methods=["POST"]),
        ]
    )


async def _fallback_asgi_app(scope: Scope, receive: Receive, send: Send) -> None:
    if scope.get("type") != "http":
        await _send_empty(send, 404)
        return

    method = scope.get("method", "")
    path = scope.get("path", "")
    if method == "GET" and path == "/healthz":
        await _send_json(send, ping())
        return

    if method == "POST" and path == "/mcp":
        body = await _read_body(receive)
        try:
            payload = _loads_strict_json(body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            await _send_json(
                send,
                _json_rpc_error(None, JSONRPC_PARSE_ERROR, "Parse error"),
                status=400,
            )
            return

        response = handle_json_rpc(payload)
        if response is None:
            await _send_empty(send, 204)
            return
        await _send_json(send, response)
        return

    await _send_empty(send, 404)


async def _read_body(receive: Receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        if message.get("type") != "http.request":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_json(send: Send, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _send_empty(send: Send, status: int) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-length", b"0")],
        }
    )
    await send({"type": "http.response.body", "body": b""})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the HGBO-DSE MCP JSON-RPC server")
    parser.add_argument("--host", default=os.environ.get("HGBO_MCP_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("HGBO_MCP_PORT", "8000")),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("HGBO_MCP_LOG_LEVEL", "info"),
    )
    parser.add_argument(
        "--show-api-key",
        action="store_true",
        help="Print the persistent MCP API key and exit",
    )
    args = parser.parse_args(argv)

    if args.show_api_key:
        print(_default_api_key_store().ensure_api_key())
        return 0

    import uvicorn

    uvicorn.run(
        "backend.mcp_server:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


def _handle_single_json_rpc(message: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(message, dict):
        return _json_rpc_error(None, JSONRPC_INVALID_REQUEST, "Invalid Request")

    has_id = "id" in message
    request_id = message.get("id") if has_id else None
    if has_id and not _is_valid_json_rpc_id(request_id):
        return _json_rpc_error(None, JSONRPC_INVALID_REQUEST, "Invalid Request")

    method = message.get("method")
    if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
        return _json_rpc_error(request_id, JSONRPC_INVALID_REQUEST, "Invalid Request")

    if not has_id:
        return None

    if method == "notifications/initialized":
        return _json_rpc_success(request_id, {})
    if method == "initialize":
        return _json_rpc_success(request_id, _initialize_result())
    if method == "tools/list":
        return _json_rpc_success(request_id, {"tools": _tool_definitions()})
    if method == "tools/call":
        return _handle_tools_call(request_id, message.get("params"))

    return _json_rpc_error(
        request_id,
        JSONRPC_METHOD_NOT_FOUND,
        "Method not found",
    )


def _handle_tools_call(request_id: Any, params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return _json_rpc_error(request_id, JSONRPC_INVALID_PARAMS, "Invalid params")

    tool_name = params.get("name")
    arguments = params["arguments"] if "arguments" in params else {}
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return _json_rpc_error(request_id, JSONRPC_INVALID_PARAMS, "Invalid params")

    if tool_name == "ping":
        structured = ping()
        return _json_rpc_success(request_id, _tool_call_result(structured))

    if tool_name == "predict_impl_ppa":
        predict_params = _validate_predict_impl_ppa_arguments(arguments)
        if predict_params is None:
            return _json_rpc_error(request_id, JSONRPC_INVALID_PARAMS, "Invalid params")

        try:
            structured = predict_impl_ppa_impl(
                case=predict_params["case"],
                hls_attr=predict_params["hls_attr"],
                graph_archive_base64=predict_params["graph_archive_base64"],
                archive_format=predict_params["archive_format"],
                request_id=str(request_id or ""),
            )
        except Exception:
            return _json_rpc_error(
                request_id,
                JSONRPC_INTERNAL_ERROR,
                "Tool execution failed",
            )
        return _json_rpc_success(request_id, _tool_call_result(structured))

    return _json_rpc_error(request_id, JSONRPC_METHOD_NOT_FOUND, "Tool not found")


def _validate_predict_impl_ppa_arguments(
    arguments: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    required = ("case", "hls_attr", "graph_archive_base64")
    if any(name not in arguments for name in required):
        return None

    case = arguments["case"]
    hls_attr = arguments["hls_attr"]
    graph_archive_base64 = arguments["graph_archive_base64"]
    archive_format = arguments.get("archive_format", "zip")

    if not isinstance(case, str):
        return None
    if not isinstance(hls_attr, list):
        return None
    if not isinstance(graph_archive_base64, str):
        return None
    if not isinstance(archive_format, str):
        return None

    return {
        "case": case,
        "hls_attr": hls_attr,
        "graph_archive_base64": graph_archive_base64,
        "archive_format": archive_format,
    }


def _is_valid_json_rpc_id(request_id: Any) -> bool:
    if request_id is None:
        return True
    if isinstance(request_id, bool):
        return False
    if isinstance(request_id, float):
        return math.isfinite(request_id)
    return isinstance(request_id, (str, int, float))


def _loads_strict_json(body: bytes) -> Any:
    return json.loads(
        body.decode("utf-8"),
        parse_constant=_reject_json_constant,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant: {0}".format(value))


def _tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "name": "ping",
            "description": "Return server liveness status.",
            "inputSchema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "name": "predict_impl_ppa",
            "description": "Predict implementation PPA from an HGP graph archive.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "case": {"type": "string"},
                    "hls_attr": {"type": "array"},
                    "graph_archive_base64": {"type": "string"},
                    "archive_format": {"type": "string", "default": "zip"},
                },
                "required": ["case", "hls_attr", "graph_archive_base64"],
                "additionalProperties": False,
            },
        },
    ]


def _tool_call_result(structured: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "structuredContent": structured,
        "content": [
            {
                "type": "text",
                "text": json.dumps(structured, sort_keys=True),
            }
        ],
    }


def _initialize_result() -> Dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
    }


def _json_rpc_success(request_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _json_rpc_error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _make_temp_root(request_id: str, temp_root: Optional[PathLike]) -> Path:
    if temp_root is not None:
        root_path = Path(temp_root)
        root_path.mkdir(parents=True, exist_ok=True)
        return root_path

    prefix = "hgbo-mcp-"
    safe_request_id = _safe_request_id(request_id)
    if safe_request_id:
        prefix = "hgbo-mcp-{0}-".format(safe_request_id)
    return Path(tempfile.mkdtemp(prefix=prefix))


def _safe_request_id(request_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in ("-", "_") else "-"
        for character in str(request_id)
    )[:32]


def _default_api_key_store() -> ApiKeyStore:
    secret_dir = Path(os.environ.get(SECRET_DIR_ENV, DEFAULT_SECRET_DIR))
    return ApiKeyStore(secret_dir / API_KEY_FILENAME)


if __name__ == "__main__":
    raise SystemExit(main())
