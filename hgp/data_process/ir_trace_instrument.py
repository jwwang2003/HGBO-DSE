"""Automatic LLVM-IR trace instrumentation for CDFG switching activity.

The hand-maintained ``*_instr.c`` files log source-level operations by a small
integer id.  ATAPP's tracing path is closer to the HLS IR/FSMD boundary: trace
IR operators, map them to CDFG/RTL operators, and then compute SA/AR from those
operator values.  This module provides the mechanical part of that flow:

* map CDFG node names such as ``mul57_1`` to LLVM SSA names like ``%mul57``;
* inject ``printf("SA <node_id> <bits> <value>")`` calls after matched IR ops;
* optionally build executable host IR from the raw kernel source so the trace
  can run without hand-maintained instrumented C kernels.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from hgp.data_process.operator_activity import clean_cell, read_cdfg_rows

TRACEABLE_OPCODES = {
    "add",
    "and",
    "ashr",
    "fcmp",
    "fadd",
    "fdiv",
    "fmul",
    "fsub",
    "icmp",
    "load",
    "lshr",
    "mul",
    "mux",
    "or",
    "select",
    "shl",
    "sdiv",
    "srem",
    "sub",
    "udiv",
    "urem",
    "xor",
    "zext",
    "sext",
    "trunc",
}

IR_SOURCE_PRIORITY = (
    "{top}.bc",
    "{top}.g.bc",
    "a.g.ld.5.gdce.bc",
    "a.g.ld.4.m2.bc",
    "a.g.ld.3.fpc.bc",
    "a.g.ld.2.m1.bc",
    "a.g.ld.1.lower.bc",
    "a.g.ld.0.bc",
    "a.g.bc",
    "a.pp.bc",
    "{top}.ll",
    "apatb_{top}_ir.bc",
    "apatb_{top}_ir.ll",
)


@dataclass(frozen=True)
class IRInstruction:
    line_no: int
    result_name: str
    opcode: str
    value_type: str
    value_ref: str
    text: str


@dataclass(frozen=True)
class TracePoint:
    trace_id: str
    node_id: str
    node_name: str
    cdfg_opcode: str
    cdfg_bitwidth: int
    rtl_name: str
    ir_name: str
    ir_opcode: str
    ir_type: str
    line_no: int


def normalize_ir_name(name: str) -> str:
    """Normalize SSA/CDFG operator names for cross-tool matching."""
    value = clean_cell(name)
    if value.startswith("%"):
        value = value[1:]
    value = value.replace(".", "_").lower()
    # HLS CDFGs often clone source operators as mul_1/add58_1 after unroll or
    # scheduling.  Strip only trailing numeric clone suffixes.
    while re.search(r"_\d+$", value):
        value = re.sub(r"_\d+$", "", value)
    return value


def opcode_compatible(cdfg_opcode: str, ir_opcode: str) -> bool:
    cdfg = clean_cell(cdfg_opcode).lower()
    ir = clean_cell(ir_opcode).lower()
    if cdfg == ir:
        return True
    if cdfg == "mux" and ir == "select":
        return True
    if cdfg == "select" and ir == "mux":
        return True
    if cdfg.startswith("f") and cdfg[1:] == ir:
        return True
    if ir.startswith("f") and ir[1:] == cdfg:
        return True
    return False


def _strip_instruction_suffix(text: str) -> str:
    return text.split(";", 1)[0].strip()


def _first_type_token(text: str) -> str:
    parts = text.strip().split(None, 1)
    return parts[0] if parts else ""


def _result_type(opcode: str, rest: str) -> str:
    rest = rest.strip()
    if opcode == "load":
        return rest.split(",", 1)[0].strip()
    if opcode in {"icmp", "fcmp"}:
        return "i1"
    if opcode == "select":
        match = re.match(r"\S+\s+\S+\s*,\s*([^,\s]+)\s+", rest)
        return match.group(1) if match else ""
    if opcode in {"zext", "sext", "trunc"}:
        match = re.search(r"\bto\s+([^,\s]+)", rest)
        return match.group(1) if match else ""
    if opcode == "bitcast":
        match = re.search(r"\bto\s+([^,\s]+)", rest)
        return match.group(1) if match else ""

    # Binary operations may include flags before the type: add nsw i32 ...
    tokens = rest.split()
    skip_flags = {
        "exact",
        "fast",
        "nnan",
        "ninf",
        "nsw",
        "nuw",
        "reassoc",
        "contract",
        "afn",
        "arcp",
    }
    for token in tokens:
        if token in skip_flags:
            continue
        return token
    return ""


def parse_ir_instructions(ir_text: str) -> list[IRInstruction]:
    instructions: list[IRInstruction] = []
    result_re = re.compile(
        r"^\s*(%[-A-Za-z0-9_.$]+)\s*=\s+"
        r"(fadd|fsub|fmul|fdiv|add|sub|mul|udiv|sdiv|urem|srem|"
        r"icmp|fcmp|select|load|shl|lshr|ashr|and|or|xor|"
        r"zext|sext|trunc|bitcast)\b\s*(.*)$"
    )
    for line_no, line in enumerate(ir_text.splitlines(), start=1):
        instruction = _strip_instruction_suffix(line)
        match = result_re.match(instruction)
        if not match:
            continue
        result_name, opcode, rest = match.groups()
        opcode = opcode.lower()
        value_type = _result_type(opcode, rest)
        if not value_type:
            continue
        instructions.append(
            IRInstruction(
                line_no=line_no,
                result_name=result_name[1:],
                opcode=opcode,
                value_type=value_type,
                value_ref=result_name,
                text=line.rstrip(),
            )
        )
    return instructions


def _row_bitwidth(row: dict, instruction: IRInstruction) -> int:
    try:
        bitwidth = int(float(clean_cell(row.get("bitwidth"))))
    except (TypeError, ValueError):
        bitwidth = 0
    return bitwidth or bitwidth_for_type(instruction.value_type)


def build_trace_points(cdfg_node_csv: Path | None, ir_text: str) -> list[TracePoint]:
    """Build CDFG-node trace points by matching node names to LLVM SSA names."""
    rows = read_cdfg_rows(cdfg_node_csv)
    if not rows:
        return []

    by_name: dict[str, list[IRInstruction]] = {}
    for instruction in parse_ir_instructions(ir_text):
        by_name.setdefault(normalize_ir_name(instruction.result_name), []).append(instruction)

    trace_points: list[TracePoint] = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        node_id = clean_cell(row.get("node_id"))
        node_name = clean_cell(row.get("node_name"))
        cdfg_opcode = clean_cell(row.get("opcode")).lower()
        if not node_id or not node_name or cdfg_opcode not in TRACEABLE_OPCODES:
            continue
        candidates = by_name.get(normalize_ir_name(node_name), [])
        candidates = [
            instruction
            for instruction in candidates
            if opcode_compatible(cdfg_opcode, instruction.opcode)
        ]
        if not candidates:
            continue
        instruction = candidates[0]
        key = (node_id, instruction.result_name, instruction.line_no)
        if key in seen:
            continue
        seen.add(key)
        trace_points.append(
            TracePoint(
                trace_id=node_id,
                node_id=node_id,
                node_name=node_name,
                cdfg_opcode=cdfg_opcode,
                cdfg_bitwidth=_row_bitwidth(row, instruction),
                rtl_name=clean_cell(row.get("rtl_name")),
                ir_name=instruction.result_name,
                ir_opcode=instruction.opcode,
                ir_type=instruction.value_type,
                line_no=instruction.line_no,
            )
        )
    return trace_points


def bitwidth_for_type(value_type: str) -> int:
    value_type = value_type.strip()
    if value_type == "half":
        return 16
    if value_type in {"float", "bfloat"}:
        return 32
    if value_type == "double":
        return 64
    match = re.fullmatch(r"i(\d+)", value_type)
    if match:
        return int(match.group(1))
    return 64


def _llvm_c_string(value: str) -> tuple[str, int]:
    raw = value.encode("utf-8") + b"\x00"
    chunks: list[str] = []
    for byte in raw:
        if 32 <= byte <= 126 and byte not in {34, 92}:
            chunks.append(chr(byte))
        else:
            chunks.append("\\{:02X}".format(byte))
    return "".join(chunks), len(raw)


def _uses_opaque_pointers(ir_text: str) -> bool:
    return bool(re.search(r"\bptr\b", ir_text))


def _global_header(trace_points: list[TracePoint], *, opaque_ptrs: bool) -> tuple[list[str], dict[str, tuple[str, int]]]:
    fmt_text, fmt_len = _llvm_c_string("SA %s %d %llu\n")
    lines = [
        '@__sa_trace_fmt = private unnamed_addr constant [{} x i8] c"{}"'.format(fmt_len, fmt_text),
    ]
    id_globals: dict[str, tuple[str, int]] = {}
    for idx, trace_point in enumerate(trace_points):
        escaped, length = _llvm_c_string(trace_point.trace_id)
        global_name = "__sa_trace_id_{}".format(idx)
        lines.append(
            '@{} = private unnamed_addr constant [{} x i8] c"{}"'.format(
                global_name,
                length,
                escaped,
            )
        )
        id_globals[trace_point.trace_id] = (global_name, length)
    if opaque_ptrs:
        lines.append("declare i32 @printf(ptr, ...)")
    else:
        lines.append("declare i32 @printf(i8*, ...)")
    return lines, id_globals


def _fmt_gep(fmt_len: int, *, opaque_ptrs: bool) -> str:
    if opaque_ptrs:
        return "ptr getelementptr inbounds ([{} x i8], ptr @__sa_trace_fmt, i64 0, i64 0)".format(fmt_len)
    return (
        "i8* getelementptr inbounds ([{0} x i8], [{0} x i8]* @__sa_trace_fmt, "
        "i64 0, i64 0)"
    ).format(fmt_len)


def _id_gep(global_name: str, length: int, *, opaque_ptrs: bool) -> str:
    if opaque_ptrs:
        return "ptr getelementptr inbounds ([{} x i8], ptr @{}, i64 0, i64 0)".format(
            length,
            global_name,
        )
    return "i8* getelementptr inbounds ([{0} x i8], [{0} x i8]* @{1}, i64 0, i64 0)".format(
        length,
        global_name,
    )


def _int_value_lines(
    *,
    value_type: str,
    value_ref: str,
    prefix: str,
    indent: str,
) -> tuple[list[str], str]:
    value_type = value_type.strip()
    bitwidth = bitwidth_for_type(value_type)
    if value_type in {"half", "bfloat"}:
        bits_name = "%{}_half_bits".format(prefix)
        wide_name = "%{}_u64".format(prefix)
        return [
            "{}{} = bitcast {} {} to i16".format(indent, bits_name, value_type, value_ref),
            "{}{} = zext i16 {} to i64".format(indent, wide_name, bits_name),
        ], wide_name
    if value_type in {"float", "bfloat"}:
        bits_name = "%{}_f32_bits".format(prefix)
        wide_name = "%{}_u64".format(prefix)
        return [
            "{}{} = bitcast {} {} to i32".format(indent, bits_name, value_type, value_ref),
            "{}{} = zext i32 {} to i64".format(indent, wide_name, bits_name),
        ], wide_name
    if value_type == "double":
        bits_name = "%{}_f64_bits".format(prefix)
        return [
            "{}{} = bitcast double {} to i64".format(indent, bits_name, value_ref),
        ], bits_name
    if re.fullmatch(r"i\d+", value_type):
        if bitwidth < 64:
            wide_name = "%{}_u64".format(prefix)
            return [
                "{}{} = zext {} {} to i64".format(indent, wide_name, value_type, value_ref),
            ], wide_name
        if bitwidth > 64:
            trunc_name = "%{}_u64".format(prefix)
            return [
                "{}{} = trunc {} {} to i64".format(indent, trunc_name, value_type, value_ref),
            ], trunc_name
        return [], value_ref
    if value_type == "ptr":
        wide_name = "%{}_ptr_u64".format(prefix)
        return [
            "{}{} = ptrtoint ptr {} to i64".format(indent, wide_name, value_ref),
        ], wide_name
    return [], value_ref


def _trace_lines(
    trace_point: TracePoint,
    *,
    idx: int,
    id_global: tuple[str, int],
    opaque_ptrs: bool,
    indent: str,
) -> list[str]:
    prefix = "__sa_trace_{}".format(idx)
    value_lines, value_ref = _int_value_lines(
        value_type=trace_point.ir_type,
        value_ref="%{}".format(trace_point.ir_name),
        prefix=prefix,
        indent=indent,
    )
    if value_ref.startswith("%") and not value_lines and trace_point.ir_type not in {"i64", "double"}:
        return []
    fmt_text, fmt_len = _llvm_c_string("SA %s %d %llu\n")
    del fmt_text
    bits = bitwidth_for_type(trace_point.ir_type)
    global_name, id_len = id_global
    printf_ptr = "ptr" if opaque_ptrs else "i8*"
    call_result = "%{}_printf".format(prefix)
    call = "{}{} = call i32 ({}, ...) @printf({}, {}, i32 {}, i64 {})".format(
        indent,
        call_result,
        printf_ptr,
        _fmt_gep(fmt_len, opaque_ptrs=opaque_ptrs),
        _id_gep(global_name, id_len, opaque_ptrs=opaque_ptrs),
        bits,
        value_ref,
    )
    return value_lines + [call]


def instrument_ir_text(ir_text: str, trace_points: list[TracePoint]) -> str:
    """Return LLVM IR text with printf trace calls inserted after trace points."""
    if not trace_points:
        return ir_text

    opaque_ptrs = _uses_opaque_pointers(ir_text)
    header_lines, id_globals = _global_header(trace_points, opaque_ptrs=opaque_ptrs)
    by_line: dict[int, list[TracePoint]] = {}
    for trace_point in trace_points:
        by_line.setdefault(trace_point.line_no, []).append(trace_point)

    original_lines = ir_text.splitlines()
    insert_at = 0
    for idx, line in enumerate(original_lines):
        stripped = line.strip()
        if stripped.startswith("define ") or stripped.startswith("declare "):
            insert_at = idx
            break
    else:
        insert_at = 0

    output: list[str] = []
    output.extend(original_lines[:insert_at])
    if output and output[-1].strip():
        output.append("")
    output.extend(header_lines)
    output.append("")

    trace_idx = 0
    for line_no, line in enumerate(original_lines[insert_at:], start=insert_at + 1):
        output.append(line)
        for trace_point in by_line.get(line_no, []):
            indent = re.match(r"^\s*", line).group(0)
            output.extend(
                _trace_lines(
                    trace_point,
                    idx=trace_idx,
                    id_global=id_globals[trace_point.trace_id],
                    opaque_ptrs=opaque_ptrs,
                    indent=indent,
                )
            )
            trace_idx += 1
    return "\n".join(output) + "\n"


def write_manifest(path: Path, trace_points: list[TracePoint], *, ir_source: str) -> None:
    payload = {
        "ir_source": ir_source,
        "trace_count": len(trace_points),
        "trace_points": [asdict(trace_point) for trace_point in trace_points],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_hls_ir_source(graph_dir: Path, top_name: str) -> Path | None:
    graph_dir = Path(graph_dir)
    for pattern in IR_SOURCE_PRIORITY:
        candidate = graph_dir / pattern.format(top=top_name)
        if candidate.exists():
            return candidate
    for candidate in sorted(graph_dir.glob("*.ll")) + sorted(graph_dir.glob("*.bc")):
        if candidate.name.startswith("apatb_"):
            continue
        return candidate
    return None


def disassemble_ir_source(ir_source: Path, *, llvm_dis: str = "llvm-dis") -> str:
    ir_source = Path(ir_source)
    if ir_source.suffix == ".ll":
        return ir_source.read_text(encoding="utf-8", errors="replace")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_ll = Path(tmp_dir) / "source.ll"
        result = subprocess.run(
            [llvm_dis, str(ir_source), "-o", str(out_ll)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError("llvm-dis failed for {}: {}".format(ir_source, result.stderr[-2000:]))
        return out_ll.read_text(encoding="utf-8", errors="replace")


def instrument_hls_ir_artifact(
    *,
    graph_dir: Path,
    top_name: str,
    cdfg_node_csv: Path,
    output_ll: Path,
    manifest_path: Path | None = None,
    llvm_dis: str = "llvm-dis",
) -> list[TracePoint]:
    """Instrument a HLS bitcode/text IR artifact and write the resulting .ll."""
    ir_source = find_hls_ir_source(graph_dir, top_name)
    if ir_source is None:
        raise FileNotFoundError("No usable HLS IR source found in {}".format(graph_dir))
    ir_text = disassemble_ir_source(ir_source, llvm_dis=llvm_dis)
    trace_points = build_trace_points(cdfg_node_csv, ir_text)
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    output_ll.write_text(instrument_ir_text(ir_text, trace_points), encoding="utf-8")
    if manifest_path is not None:
        write_manifest(manifest_path, trace_points, ir_source=str(ir_source))
    return trace_points


def build_host_ir_from_source(
    *,
    source_file: Path,
    output_ll: Path,
    clang: str = "clang",
    include_dirs: list[Path] | None = None,
) -> None:
    include_args: list[str] = []
    for include_dir in include_dirs or []:
        include_args.extend(["-I", str(include_dir)])
    cmd = [
        clang,
        "-S",
        "-emit-llvm",
        "-O0",
        "-fno-discard-value-names",
        *include_args,
        "-o",
        str(output_ll),
        str(source_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError("clang IR build failed for {}: {}".format(source_file, result.stderr[-2000:]))


def instrument_host_source_ir(
    *,
    source_file: Path,
    cdfg_node_csv: Path,
    output_ll: Path,
    manifest_path: Path | None = None,
    clang: str = "clang",
    include_dirs: list[Path] | None = None,
) -> list[TracePoint]:
    """Compile raw kernel C to host LLVM IR, instrument it, and write .ll."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_ll = Path(tmp_dir) / "kernel.ll"
        build_host_ir_from_source(
            source_file=source_file,
            output_ll=raw_ll,
            clang=clang,
            include_dirs=include_dirs,
        )
        ir_text = raw_ll.read_text(encoding="utf-8", errors="replace")
    trace_points = build_trace_points(cdfg_node_csv, ir_text)
    output_ll.parent.mkdir(parents=True, exist_ok=True)
    output_ll.write_text(instrument_ir_text(ir_text, trace_points), encoding="utf-8")
    if manifest_path is not None:
        write_manifest(manifest_path, trace_points, ir_source=str(source_file))
    return trace_points
