from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import aiohttp

from backend.mcp_payload import create_graph_archive_base64, validate_ppa_result


MCP_URL_ENV = "HGBO_MCP_URL"
MCP_API_KEY_ENV = "HGBO_MCP_API_KEY"
REMOTE_TIMEOUT_ENV = "HGBO_REMOTE_TIMEOUT_SEC"
CONTEXT_ID_ENV = "HGBO_CONTEXT_ID"
DEFAULT_REMOTE_TIMEOUT_SEC = 600.0

PathLike = Union[str, os.PathLike]


@dataclass(frozen=True)
class RemoteConfig:
    url: str
    api_key: str
    timeout: float


def read_remote_config() -> RemoteConfig:
    url = _required_env(MCP_URL_ENV)
    api_key = _required_env(MCP_API_KEY_ENV)
    return RemoteConfig(
        url=url,
        api_key=api_key,
        timeout=_read_timeout(),
    )


def jsonrpc_result_to_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError("MCP response was not a JSON object")

    error = response.get("error")
    if error is not None:
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        else:
            message = str(error)
        raise RuntimeError(message)

    result = response.get("result")
    if isinstance(result, dict):
        structured_content = result.get("structuredContent")
        if isinstance(structured_content, dict):
            return structured_content

        parsed_text = _parse_first_text_content(result.get("content"))
        if parsed_text is not None:
            return parsed_text

    raise RuntimeError("MCP response did not contain a JSON object result")


async def call_tool(
    url: str,
    api_key: str,
    tool_name: str,
    arguments: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    response = await _post_json_rpc_method(
        url,
        api_key,
        "tools/call",
        {
            "name": tool_name,
            "arguments": arguments,
        },
        timeout,
    )
    return jsonrpc_result_to_dict(response)


async def call_method(
    url: str,
    api_key: str,
    method: str,
    params: Optional[Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    response = await _post_json_rpc_method(url, api_key, method, params, timeout)
    return _jsonrpc_response_result_to_dict(response)


async def _post_json_rpc_method(
    url: str,
    api_key: str,
    method: str,
    params: Optional[Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    request = {
        "jsonrpc": "2.0",
        "id": _json_rpc_request_id(),
        "method": method,
    }
    if params is not None:
        request["params"] = params

    headers = {
        "Authorization": "Bearer {0}".format(api_key),
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    try:
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, headers=headers, json=request) as response:
                if response.status == 401:
                    raise RuntimeError("MCP authentication failed")
                if response.status >= 400:
                    raise RuntimeError(
                        "MCP request failed with HTTP {0}".format(response.status)
                    )

                try:
                    payload = await response.json()
                except asyncio.TimeoutError:
                    raise
                except Exception as exc:
                    raise RuntimeError("MCP response was not valid JSON") from exc
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            "MCP request timed out after {0} seconds".format(timeout)
        ) from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError("MCP server unavailable: {0}".format(exc)) from exc

    return payload


async def predict_impl_ppa_async(
    prj_path: PathLike,
    hls_attr: Sequence[Any],
    case: str,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    config = read_remote_config()
    graph_archive_base64 = create_graph_archive_base64(prj_path)
    arguments = {
        "case": case,
        "hls_attr": list(hls_attr),
        "graph_archive_base64": graph_archive_base64,
        "archive_format": "zip",
    }
    result = await call_tool(
        config.url,
        config.api_key,
        "predict_impl_ppa",
        arguments,
        config.timeout if timeout is None else timeout,
    )
    try:
        return validate_ppa_result(result)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "invalid PPA response from MCP server: {0}".format(exc)
        ) from exc


def predict_impl_ppa(
    prj_path: PathLike,
    hls_attr: Sequence[Any],
    case: str,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    return asyncio.run(predict_impl_ppa_async(prj_path, hls_attr, case, timeout=timeout))


async def ping_async(timeout: Optional[float] = None) -> Dict[str, Any]:
    config = read_remote_config()
    return await call_tool(
        config.url,
        config.api_key,
        "ping",
        {},
        config.timeout if timeout is None else timeout,
    )


async def list_tools_async(timeout: Optional[float] = None) -> Dict[str, Any]:
    config = read_remote_config()
    return await call_method(
        config.url,
        config.api_key,
        "tools/list",
        None,
        config.timeout if timeout is None else timeout,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HGBO-DSE MCP JSON-RPC client")
    parser.add_argument("--ping", action="store_true", help="Call the remote ping tool")
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List remote MCP tools",
    )
    args = parser.parse_args(argv)

    if args.ping:
        result = asyncio.run(ping_async())
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.list_tools:
        result = asyncio.run(list_tools_async())
        print(json.dumps(result, sort_keys=True))
        return 0

    parser.print_help()
    return 0


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError("{0} is required for remote MCP inference".format(name))
    return value.strip()


def _read_timeout() -> float:
    raw_timeout = os.getenv(REMOTE_TIMEOUT_ENV, str(int(DEFAULT_REMOTE_TIMEOUT_SEC)))
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "{0} must be a positive number of seconds".format(REMOTE_TIMEOUT_ENV)
        ) from exc

    if timeout <= 0:
        raise RuntimeError(
            "{0} must be a positive number of seconds".format(REMOTE_TIMEOUT_ENV)
        )
    return timeout


def _json_rpc_request_id() -> str:
    context_id = os.getenv(CONTEXT_ID_ENV, "").strip()
    if context_id:
        return context_id
    return str(uuid.uuid4())


def _jsonrpc_response_result_to_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError("MCP response was not a JSON object")

    error = response.get("error")
    if error is not None:
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        else:
            message = str(error)
        raise RuntimeError(message)

    result = response.get("result")
    if isinstance(result, dict):
        return result
    raise RuntimeError("MCP response did not contain a JSON object result")


def _parse_first_text_content(content: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(content, list):
        return None

    for item in content:
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            try:
                parsed = json.loads(item["text"])
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "MCP response text content did not contain a JSON object"
                ) from exc
            if not isinstance(parsed, dict):
                raise RuntimeError(
                    "MCP response text content did not contain a JSON object"
                )
            return parsed
    return None


async def _response_text(response: Any) -> str:
    try:
        return await response.text()
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
