from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

MISSING_OPERATOR_VALUES = {"", "none", "not_exist", "null", "nan"}


def clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def is_valid_operator_name(value) -> bool:
    return clean_cell(value).lower() not in MISSING_OPERATOR_VALUES


def hardware_operator_key(row: dict, *, node_id: str | None = None) -> str:
    """Return the FSMD/RTL hardware-operator identity for a CDFG row/node."""
    rtl_name = clean_cell(row.get("rtl_name"))
    if is_valid_operator_name(rtl_name):
        return "rtl:{}".format(rtl_name)
    if node_id is None:
        node_id = clean_cell(row.get("node_id"))
    return "node:{}".format(node_id)


def read_cdfg_rows(cdfg_node_csv: Path | None) -> list[dict]:
    if cdfg_node_csv is None or not cdfg_node_csv.exists():
        return []
    try:
        with cdfg_node_csv.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def metric_pair(metrics: dict | None) -> dict[str, float] | None:
    if not metrics:
        return None
    return {
        "sa": float(metrics.get("sa", 0.0)),
        "ar": float(metrics.get("ar", 0.0)),
    }


def mean_metric(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"sa": 0.0, "ar": 0.0}
    return {
        "sa": sum(item["sa"] for item in metrics) / len(metrics),
        "ar": sum(item["ar"] for item in metrics) / len(metrics),
    }


def row_trace_id(row: dict) -> str:
    for key in ("trace_id", "op_id", "trace_op_id"):
        value = clean_cell(row.get(key))
        if value:
            return value
    return ""


def activity_for_cdfg_row(
    row: dict,
    by_op_id: dict,
    by_opcode: dict,
    by_node_id: dict | None = None,
) -> dict[str, float] | None:
    node_id = clean_cell(row.get("node_id"))
    if node_id:
        metrics = metric_pair((by_node_id or {}).get(node_id))
        if metrics is not None:
            return metrics
        metrics = metric_pair(by_op_id.get(node_id))
        if metrics is not None:
            return metrics
    trace_id = row_trace_id(row)
    if trace_id:
        metrics = metric_pair(by_op_id.get(trace_id))
        if metrics is not None:
            return metrics
    return metric_pair(by_opcode.get(clean_cell(row.get("opcode"))))


def node_activity_from_cdfg_csv(by_op_id: dict, cdfg_node_csv: Path | None) -> dict:
    """Compatibility mapper for old caches that relied on CDFG row order."""
    if not by_op_id:
        return {}
    rows = read_cdfg_rows(cdfg_node_csv)
    if not rows:
        return {}
    result: dict[str, dict[str, float]] = {}
    for op_id, row in enumerate(rows):
        metrics = metric_pair(by_op_id.get(str(op_id)))
        node_id = clean_cell(row.get("node_id"))
        if not metrics or not node_id:
            continue
        result[node_id] = metrics
    return result


def node_activity_from_operator_merge(
    *,
    by_op_id: dict,
    by_opcode: dict,
    cdfg_node_csv: Path | None,
    by_node_id: dict | None = None,
) -> dict:
    """Project trace/opcode activity through FSMD RTL operator sharing."""
    rows = read_cdfg_rows(cdfg_node_csv)
    if not rows:
        return {}

    by_operator: dict[str, list[dict[str, float]]] = defaultdict(list)
    node_to_operator: dict[str, str] = {}
    for row in rows:
        node_id = clean_cell(row.get("node_id"))
        if not node_id:
            continue
        metrics = activity_for_cdfg_row(row, by_op_id, by_opcode, by_node_id=by_node_id)
        if metrics is None:
            continue
        operator_key = hardware_operator_key(row, node_id=node_id)
        node_to_operator[node_id] = operator_key
        by_operator[operator_key].append(metrics)

    merged_by_operator = {
        operator_key: mean_metric(metrics)
        for operator_key, metrics in by_operator.items()
    }
    return {
        node_id: dict(merged_by_operator[operator_key])
        for node_id, operator_key in node_to_operator.items()
        if operator_key in merged_by_operator
    }
