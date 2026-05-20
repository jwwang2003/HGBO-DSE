from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


VIVADO_EXECUTION_MODE_ENV = "HGBO_VIVADO_EXECUTION_MODE"
VIVADO_MCP_URL_ENV = "HGBO_VIVADO_MCP_URL"
VIVADO_MCP_HOST_ENV = "HGBO_VIVADO_MCP_HOST"
VIVADO_MCP_PORT_ENV = "HGBO_VIVADO_MCP_PORT"
VIVADO_MCP_API_KEY_ENV = "HGBO_VIVADO_MCP_API_KEY"
VIVADO_MCP_WORKSPACE_ENV = "HGBO_VIVADO_MCP_WORKSPACE"
VIVADO_MCP_TOOL_PROFILE_ENV = "HGBO_VIVADO_MCP_TOOL_PROFILE"
VIVADO_MCP_RESOURCE_SLOTS_ENV = "HGBO_VIVADO_MCP_RESOURCE_SLOTS"
VIVADO_MCP_POLL_INTERVAL_ENV = "HGBO_VIVADO_MCP_POLL_INTERVAL_SEC"
VIVADO_MCP_DEFAULT_PORT = "8080"
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "timed_out"}


@dataclass
class VivadoMcpRunResult:
    returncode: int
    stdout: str
    stderr: str


def is_vivado_mcp_enabled() -> bool:
    return os.getenv(VIVADO_EXECUTION_MODE_ENV, "").strip().lower() == "mcp"


def check_vivado_mcp_connection(timeout: float = 60.0) -> VivadoMcpRunResult:
    try:
        result = call_tool("vivado_versions", {}, timeout=timeout)
        return VivadoMcpRunResult(0, json.dumps(result, indent=2), "")
    except Exception as exc:
        return VivadoMcpRunResult(1, "", str(exc))


def run_vitis_hls_tcl(
    tcl_script: str,
    context: Optional[str],
    timeout: float = 3600.0,
) -> VivadoMcpRunResult:
    workspace_root = os.path.abspath(os.getenv(VIVADO_MCP_WORKSPACE_ENV, "."))
    script_path = _workspace_relative_path(workspace_root, tcl_script)
    arguments = {
        "workspace": ".",
        "flow": {
            "type": "tcl_script",
            "tool_profile": os.getenv(VIVADO_MCP_TOOL_PROFILE_ENV, "vitis_hls.legacy"),
            "script_path": script_path,
        },
        "resource_slots": _positive_int_env(VIVADO_MCP_RESOURCE_SLOTS_ENV, 1),
        "timeout_seconds": int(timeout),
        "artifacts": [
            "**/*.rpt",
            "**/*.log",
            "**/*.xml",
            "**/*.json",
        ],
    }
    submitted = call_tool("vivado_submit_job", arguments, timeout=timeout)
    job_id = submitted.get("job_id") or submitted.get("jobId")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError("Vivado MCP submit response did not include job_id")

    status = _wait_for_job(job_id, timeout=timeout)
    logs = _read_job_logs(job_id, timeout=timeout)
    state = str(status.get("state", "")).lower()
    return VivadoMcpRunResult(0 if state == "succeeded" else 1, logs, "" if state == "succeeded" else logs)


def call_tool(tool_name: str, arguments: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    response = _post_json_rpc_method(
        "tools/call",
        {
            "name": tool_name,
            "arguments": arguments,
        },
        timeout=timeout,
    )
    return _jsonrpc_result_to_dict(response)


def _wait_for_job(job_id: str, timeout: float) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    interval = _positive_float_env(VIVADO_MCP_POLL_INTERVAL_ENV, 2.0)

    while True:
        status = call_tool("vivado_job_status", {"job_id": job_id}, timeout=timeout)
        state = str(status.get("state", "")).lower()
        if state in TERMINAL_STATES:
            return status
        if time.monotonic() >= deadline:
            raise RuntimeError("Vivado MCP job {0} timed out".format(job_id))
        time.sleep(interval)


def _read_job_logs(job_id: str, timeout: float) -> str:
    try:
        result = call_tool("vivado_job_logs", {"job_id": job_id, "tail_lines": 2000}, timeout=timeout)
        if isinstance(result, dict):
            text = result.get("text") or result.get("logs") or result.get("stdout")
            if isinstance(text, str):
                return text
            return json.dumps(result, indent=2)
        return str(result)
    except Exception as exc:
        return "Failed to read Vivado MCP logs for job {0}: {1}".format(job_id, exc)


def _post_json_rpc_method(method: str, params: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request_body = json.dumps({
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": method,
        "params": params,
    }).encode("utf-8")
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    api_key = os.getenv(VIVADO_MCP_API_KEY_ENV, "").strip()
    if api_key:
        headers["Authorization"] = "Bearer {0}".format(api_key)

    request = urllib.request.Request(_read_vivado_mcp_url(), data=request_body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError("Vivado MCP request failed with HTTP {0}".format(exc.code)) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError("Vivado MCP server unavailable: {0}".format(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("Vivado MCP request timed out after {0} seconds".format(timeout)) from exc


def _jsonrpc_result_to_dict(response: Dict[str, Any]) -> Dict[str, Any]:
    error = response.get("error")
    if error is not None:
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            raise RuntimeError(error["message"])
        raise RuntimeError(str(error))

    result = response.get("result")
    if isinstance(result, dict):
        structured_content = result.get("structuredContent")
        if isinstance(structured_content, dict):
            return structured_content
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                try:
                    parsed = json.loads(first["text"])
                except json.JSONDecodeError:
                    return {"text": first["text"]}
                if isinstance(parsed, dict):
                    return parsed

    raise RuntimeError("Vivado MCP response did not contain a JSON object result")


def _read_vivado_mcp_url() -> str:
    explicit_url = os.getenv(VIVADO_MCP_URL_ENV, "").strip()
    if explicit_url:
        return explicit_url

    host = os.getenv(VIVADO_MCP_HOST_ENV, "localhost").strip() or "localhost"
    port = os.getenv(VIVADO_MCP_PORT_ENV, VIVADO_MCP_DEFAULT_PORT).strip() or VIVADO_MCP_DEFAULT_PORT
    return "http://{0}:{1}/mcp".format(host, port)


def _workspace_relative_path(workspace_root: str, candidate: str) -> str:
    absolute_candidate = os.path.abspath(candidate)
    return os.path.relpath(absolute_candidate, workspace_root)


def _positive_int_env(name: str, fallback: int) -> int:
    try:
        value = int(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _positive_float_env(name: str, fallback: float) -> float:
    try:
        value = float(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback
