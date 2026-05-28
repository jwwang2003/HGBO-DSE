from __future__ import annotations

import glob
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from hgp.data_process.operator_activity import clean_cell, hardware_operator_key


RESOURCE_TYPE_IDS = {
    "none": 0,
    "logic": 1,
    "memory": 2,
    "dsp": 3,
    "call": 4,
    "control": 5,
}


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _resource_type(row_or_attrs: dict) -> str:
    opcode = clean_cell(row_or_attrs.get("opcode")).lower()
    core = clean_cell(row_or_attrs.get("core_name") or row_or_attrs.get("core")).lower()
    op_type = clean_cell(row_or_attrs.get("op_type")).lower()
    rtl_name = clean_cell(row_or_attrs.get("rtl_name")).lower()

    if opcode == "call" or rtl_name.startswith("grp_"):
        return "call"
    if opcode in {"load", "store", "alloca"} or core == "ram" or "ram" in op_type or "memory" in core:
        return "memory"
    if "dsp" in core or _to_float(row_or_attrs.get("dsp"), 0.0) > 0:
        return "dsp"
    if opcode in {"br", "switch", "phi"}:
        return "control"
    if opcode or core or op_type:
        return "logic"
    return "none"


def _adb_kind(path: Path) -> str:
    name = path.name[:-4] if path.name.endswith(".xml") else path.name
    if ".bind." in name:
        return "bind"
    if ".sched." in name:
        return "sched"
    return "adb"


def _canonical_adb_stem(path: Path) -> str:
    if path.name.endswith(".xml"):
        path = Path(path.name[:-4])
    stem = path.stem
    if stem.endswith(".bind"):
        return stem[: -len(".bind")]
    if stem.endswith(".sched"):
        return stem[: -len(".sched")]
    return stem


def _iter_adb_roots(graph_dir: Path):
    """Yield (prefix, stem, kind, XML root) matching CDFG prefix order."""
    graph_dir = Path(graph_dir)
    prefix = 0
    for adb_path in sorted(glob.glob(str(graph_dir / "*.adb"))):
        adb_path = Path(adb_path)
        name = Path(adb_path).name
        if ".bind." in name or ".sched." in name:
            continue
        stem = _canonical_adb_stem(adb_path)
        bases = [
            graph_dir / "{}.adb".format(stem),
            graph_dir / "{}.bind.adb".format(stem),
            graph_dir / "{}.sched.adb".format(stem),
        ]
        candidates = []
        for base in bases:
            candidates.append(Path(str(base) + ".xml"))
            candidates.append(base)
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                root = ET.parse(candidate).getroot()
            except Exception:
                continue
            yield prefix, stem, _adb_kind(candidate), root
        prefix += 1


def _operation_name(operation) -> str:
    stg = operation.find("StgValue/ssdm")
    if stg is not None:
        name = clean_cell(stg.get("name"))
        if name:
            return name
    node = operation.find("Node")
    if node is not None and node.text:
        text = node.text.strip()
        if "%" in text:
            return text.split("%", 1)[1].split("=", 1)[0].strip()
    return ""


def parse_fsmd_bindings(graph_dir: Path | str) -> dict[str, dict[str, object]]:
    """Parse Vitis ADB schedule/binding metadata keyed by CDFG node id.

    The existing CDFG builder prefixes node ids by sorted top/submodule ADB
    order. ADB operation entries refer to local node ids, so the same prefixing
    gives stable ids such as ``0_28``.
    """
    graph_dir = Path(graph_dir)
    bindings: dict[str, dict[str, object]] = {}
    for prefix, module_name, artifact_kind, root in _iter_adb_roots(graph_dir):
        for operation in root.findall(".//state_list/state/operation"):
            node = operation.find("Node")
            if node is None:
                continue
            local_id = clean_cell(node.get("id"))
            if not local_id:
                continue
            node_id = "{}_{}".format(prefix, local_id)
            core = clean_cell(operation.findtext("core"))
            bindings[node_id] = {
                "fsmd_module": module_name,
                "fsmd_artifact": artifact_kind,
                "fsmd_state": _safe_int(operation.get("st_id") or operation.get("state")),
                "fsmd_stage": _safe_int(operation.get("stage")),
                "fsmd_latency": _safe_int(operation.get("lat")),
                "fsmd_operation_id": _safe_int(operation.get("id")),
                "fsmd_node_local_id": _safe_int(local_id),
                "fsmd_core": core,
                "fsmd_mem_port_count": len(clean_cell(operation.findtext("MemPortIdVec")).split()),
                "fsmd_op_name": _operation_name(operation),
            }
    return bindings


def annotate_graph_with_fsmd(DG, graph_dir: Path | str) -> dict[str, int]:
    """Attach FSMD/binding numeric metadata to graph nodes in-place."""
    bindings = parse_fsmd_bindings(graph_dir)
    operator_members: dict[str, list[str]] = defaultdict(list)
    for node_id, node in DG.nodes(data=True):
        node.update(
            {
                "fsmd_state": 0,
                "fsmd_stage": 0,
                "fsmd_latency": 0,
                "fsmd_mem_port_count": 0,
                "operator_shared_count": 1,
                "operator_resource_type": RESOURCE_TYPE_IDS["none"],
            }
        )
        binding = bindings.get(str(node_id))
        if binding:
            node.update(binding)
            if binding.get("fsmd_core") and not clean_cell(node.get("core_name")):
                node["core_name"] = binding["fsmd_core"]
        resource_type = _resource_type(node)
        node["operator_resource_type"] = RESOURCE_TYPE_IDS.get(resource_type, 0)
        operator_members[hardware_operator_key(node, node_id=str(node_id))].append(str(node_id))

    for members in operator_members.values():
        count = len(members)
        for node_id in members:
            if node_id in DG.nodes:
                DG.nodes[node_id]["operator_shared_count"] = count

    return {
        "bindings": len(bindings),
        "annotated_nodes": sum(1 for node_id in DG.nodes if str(node_id) in bindings),
        "operator_groups": len(operator_members),
        "shared_operator_groups": sum(1 for members in operator_members.values() if len(members) > 1),
    }


def fsmd_edge_attrs(src_node: dict, dst_node: dict) -> list[float]:
    src_state = _to_float(src_node.get("fsmd_state"))
    dst_state = _to_float(dst_node.get("fsmd_state"))
    src_stage = _to_float(src_node.get("fsmd_stage"))
    dst_stage = _to_float(dst_node.get("fsmd_stage"))
    src_operator = clean_cell(src_node.get("operator_key")) or hardware_operator_key(src_node)
    dst_operator = clean_cell(dst_node.get("operator_key")) or hardware_operator_key(dst_node)
    same_operator = 1.0 if src_operator and src_operator == dst_operator and src_operator != "node:" else 0.0
    return [
        max(dst_state - src_state, 0.0),
        max(dst_stage - src_stage, 0.0),
        same_operator,
    ]


def summarize_fsmd_nodes(DG) -> dict[str, int]:
    resource_types = Counter()
    for _, node in DG.nodes(data=True):
        rid = int(_to_float(node.get("operator_resource_type"), 0.0))
        name = next((key for key, value in RESOURCE_TYPE_IDS.items() if value == rid), "none")
        resource_types[name] += 1
    return dict(resource_types)
