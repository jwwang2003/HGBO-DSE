import json
from pathlib import Path

from hgp.data_process import gen_dataset_std
from hgp.data_process.ir_trace_instrument import (
    build_trace_points,
    find_hls_ir_source,
    instrument_host_source_ir,
    instrument_ir_text,
    normalize_ir_name,
)
from hgp.data_process.fsmd_binding import annotate_graph_with_fsmd, parse_fsmd_bindings
from hgp.data_process.operator_activity import node_activity_from_operator_merge
from hgp.data_process.switching_activity import compute_sa_ar


def test_activity_mode_none_skips_switching_cache(tmp_path):
    cache_dir = tmp_path / "switching_activity"
    cache_dir.mkdir()
    (cache_dir / "atax_switching_activity.json").write_text(
        '{"by_opcode": {"add": {"sa": 1.0, "ar": 0.5}}}',
        encoding="utf-8",
    )

    loaded = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="none",
        cache_dir=cache_dir,
    )

    assert loaded is None


def test_activity_auto_loads_node_and_opcode_maps(tmp_path):
    cache_dir = tmp_path / "switching_activity"
    cache_dir.mkdir()
    payload = {
        "by_node_id": {"0_1": {"sa": 4.0, "ar": 0.8}},
        "by_opcode": {"add": {"sa": 1.0, "ar": 0.5}},
        "by_op_id": {"0": {"sa": 3.0, "ar": 0.7}},
    }
    (cache_dir / "atax_switching_activity.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    loaded = gen_dataset_std._load_activity_payload(
        "atax",
        activity_mode="auto",
        cache_dir=cache_dir,
    )

    assert loaded == payload


def test_node_activity_from_cdfg_csv_maps_trace_op_ids_to_sample_nodes(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "node_id,opcode\n0_5,load\n0_6,add\n",
        encoding="utf-8",
    )

    mapped = gen_dataset_std._node_activity_from_cdfg_csv(
        {
            "0": {"sa": 0.25, "ar": 1.0},
            "1": {"sa": 0.75, "ar": 0.5},
        },
        csv_path,
    )

    assert mapped == {
        "0_5": {"sa": 0.25, "ar": 1.0},
        "0_6": {"sa": 0.75, "ar": 0.5},
    }


def test_operator_activity_merge_groups_shared_rtl_names(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "\n".join(
            [
                "node_id,rtl_name,opcode,trace_id",
                "0_1,shared_mul,mul,0",
                "0_2,shared_mul,mul,1",
                "0_3,not_exist,add,",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mapped = gen_dataset_std._node_activity_from_operator_merge(
        by_op_id={
            "0": {"sa": 2.0, "ar": 1.0},
            "1": {"sa": 4.0, "ar": 0.5},
        },
        by_opcode={"add": {"sa": 9.0, "ar": 0.25}},
        cdfg_node_csv=csv_path,
    )

    assert mapped == {
        "0_1": {"sa": 3.0, "ar": 0.75},
        "0_2": {"sa": 3.0, "ar": 0.75},
        "0_3": {"sa": 9.0, "ar": 0.25},
    }

    assert node_activity_from_operator_merge(
        by_op_id={
            "0": {"sa": 2.0, "ar": 1.0},
            "1": {"sa": 4.0, "ar": 0.5},
        },
        by_opcode={"add": {"sa": 9.0, "ar": 0.25}},
        cdfg_node_csv=csv_path,
    ) == mapped


def test_operator_activity_merge_uses_opcode_when_trace_ids_are_absent(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "\n".join(
            [
                "node_id,rtl_name,opcode",
                "0_1,shared_mul,mul",
                "0_2,shared_mul,mul",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mapped = gen_dataset_std._node_activity_from_operator_merge(
        by_op_id={
            "0": {"sa": 2.0, "ar": 1.0},
            "1": {"sa": 4.0, "ar": 0.5},
        },
        by_opcode={"mul": {"sa": 5.0, "ar": 0.25}},
        cdfg_node_csv=csv_path,
    )

    assert mapped == {
        "0_1": {"sa": 5.0, "ar": 0.25},
        "0_2": {"sa": 5.0, "ar": 0.25},
    }


def test_operator_activity_merge_prefers_node_id_trace_ids(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "\n".join(
            [
                "node_id,rtl_name,opcode,trace_id",
                "1_36,shared_mul,fmul,legacy",
                "1_41,shared_mul,fmul,",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mapped = node_activity_from_operator_merge(
        by_op_id={
            "1_36": {"sa": 2.0, "ar": 1.0},
            "1_41": {"sa": 6.0, "ar": 0.5},
            "legacy": {"sa": 99.0, "ar": 1.0},
        },
        by_opcode={"fmul": {"sa": 1.0, "ar": 0.25}},
        cdfg_node_csv=csv_path,
    )

    assert mapped == {
        "1_36": {"sa": 4.0, "ar": 0.75},
        "1_41": {"sa": 4.0, "ar": 0.75},
    }


def test_operator_activity_merge_prefers_explicit_node_activity(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "\n".join(
            [
                "node_id,rtl_name,opcode,trace_id",
                "1_36,shared_mul,fmul,legacy",
                "1_41,shared_mul,fmul,",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mapped = node_activity_from_operator_merge(
        by_node_id={
            "1_36": {"sa": 2.0, "ar": 1.0},
            "1_41": {"sa": 6.0, "ar": 0.5},
        },
        by_op_id={
            "1_36": {"sa": 20.0, "ar": 1.0},
            "1_41": {"sa": 60.0, "ar": 0.5},
            "legacy": {"sa": 99.0, "ar": 1.0},
        },
        by_opcode={"fmul": {"sa": 1.0, "ar": 0.25}},
        cdfg_node_csv=csv_path,
    )

    assert mapped == {
        "1_36": {"sa": 4.0, "ar": 0.75},
        "1_41": {"sa": 4.0, "ar": 0.75},
    }


def test_compute_sa_ar_accepts_cdfg_node_string_ids():
    metrics = compute_sa_ar(
        [
            "SA 2_63 32 0",
            "SA 2_63 32 3",
            "SA 2_64 32 1",
        ]
    )

    assert metrics["2_63"] == {"sa": 1.0, "ar": 1.0, "n": 2}
    assert metrics["2_64"] == {"sa": 0.0, "ar": 0.5, "n": 1}


def test_ir_trace_instrumentation_maps_hls_suffixes_to_cdfg_nodes(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "\n".join(
            [
                "node_id,node_name,opcode,bitwidth,rtl_name",
                "1_36,mul,fmul,32,not_exist",
                "1_41,mul_1,fmul,32,not_exist",
                "2_63,mul57_1,fmul,32,fmul_U20",
                "2_64,add58_1,fadd,32,fadd_U18",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ir_text = """
define void @atax(ptr %A) {
entry:
  %mul = fmul float %a, %b
  %mul57 = fmul float %c, %d
  %add58 = fadd float %e, %mul57
  ret void
}
"""

    trace_points = build_trace_points(csv_path, ir_text)

    assert normalize_ir_name("mul57_1") == "mul57"
    assert [point.trace_id for point in trace_points] == ["1_36", "1_41", "2_63", "2_64"]
    assert [point.ir_name for point in trace_points] == ["mul", "mul", "mul57", "add58"]


def test_ir_trace_instrumentation_injects_printf_calls(tmp_path):
    csv_path = tmp_path / "cdfg_node_dict.csv"
    csv_path.write_text(
        "node_id,node_name,opcode,bitwidth,rtl_name\n2_63,mul57_1,fmul,32,fmul_U20\n",
        encoding="utf-8",
    )
    ir_text = """
define void @atax(float %a, float %b) {
entry:
  %mul57 = fmul float %a, %b
  ret void
}
"""
    trace_points = build_trace_points(csv_path, ir_text)

    instrumented = instrument_ir_text(ir_text, trace_points)

    assert '@__sa_trace_id_0 = private unnamed_addr constant [5 x i8] c"2_63\\00"' in instrumented
    assert "%__sa_trace_0_f32_bits = bitcast float %mul57 to i32" in instrumented
    assert "call i32 (" in instrumented
    assert "@printf(" in instrumented


def test_host_source_ir_instrumentation_generates_executable_trace(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    header = source_dir / "mini.h"
    source = source_dir / "mini.c"
    cdfg_csv = tmp_path / "cdfg_node_dict.csv"
    output_ll = tmp_path / "mini_trace.ll"
    manifest = tmp_path / "mini_trace_manifest.json"
    header.write_text(
        "#define N 4\nvoid mini(float a[N], float b[N], float out[N]);\n",
        encoding="utf-8",
    )
    source.write_text(
        "\n".join(
            [
                '#include "mini.h"',
                "void mini(float a[N], float b[N], float out[N]) {",
                "  for (int i = 0; i < N; ++i) {",
                "    out[i] = a[i] * b[i];",
                "  }",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    cdfg_csv.write_text(
        "node_id,node_name,opcode,bitwidth,rtl_name\n0_1,mul,fmul,32,mul_U0\n",
        encoding="utf-8",
    )

    trace_points = instrument_host_source_ir(
        source_file=source,
        cdfg_node_csv=cdfg_csv,
        output_ll=output_ll,
        manifest_path=manifest,
        include_dirs=[source_dir],
    )

    assert [point.trace_id for point in trace_points] == ["0_1"]
    assert "SA %s %d %llu" in output_ll.read_text(encoding="utf-8")
    assert json.loads(manifest.read_text(encoding="utf-8"))["trace_count"] == 1


def test_find_hls_ir_source_prefers_top_bitcode(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "a.g.ld.0.bc").write_text("fallback", encoding="utf-8")
    (graph_dir / "atax.bc").write_text("top", encoding="utf-8")

    assert find_hls_ir_source(graph_dir, "atax") == graph_dir / "atax.bc"


def test_merge_graph_by_hardware_operator_collapses_shared_rtl_nodes():
    nx = __import__("networkx")

    graph = nx.DiGraph()
    graph.add_node(
        "0_1",
        node_type="0",
        rtl_name="shared_add",
        node_name="add_a",
        op_type="add",
        core_name="Adder",
        bitwidth="16",
        opcode="add",
        m_delay="1.0",
        topo_index="2",
        latency=["1", "1"],
        lut="5",
        ff="0",
        dsp="0",
        bram="0",
        uram="0",
    )
    graph.add_node(
        "0_2",
        node_type="0",
        rtl_name="shared_add",
        node_name="add_b",
        op_type="add",
        core_name="Adder",
        bitwidth="32",
        opcode="add",
        m_delay="2.0",
        topo_index="3",
        latency=["3", "4"],
        lut="8",
        ff="1",
        dsp="0",
        bram="0",
        uram="0",
    )
    graph.add_node(
        "0_3",
        node_type="0",
        rtl_name="not_exist",
        node_name="sink",
        op_type="not_exist",
        core_name="not_exist",
        bitwidth="32",
        opcode="store",
        m_delay="1.0",
        topo_index="4",
        latency=["5", "5"],
        lut="0",
        ff="0",
        dsp="0",
        bram="0",
        uram="0",
    )
    graph.add_edge("0_1", "0_3", edge_id="e1", edge_type="1", is_back_edge="0")
    graph.add_edge("0_2", "0_3", edge_id="e2", edge_type="2", is_back_edge="1")

    merged = gen_dataset_std._merge_graph_by_hardware_operator(graph)

    assert set(merged.nodes) == {"0_1", "0_3"}
    assert merged.nodes["0_1"]["merged_node_count"] == 2
    assert merged.nodes["0_1"]["bitwidth"] == "32.0"
    assert merged.nodes["0_1"]["m_delay"] == "2.0"
    assert merged.nodes["0_1"]["latency"] == ["1.0", "4.0"]
    assert merged.edges["0_1", "0_3"]["edge_type"] == "2.0"
    assert merged.edges["0_1", "0_3"]["is_back_edge"] == "1.0"


def test_activity_mode_shuffled_preserves_metrics_but_breaks_mapping(tmp_path):
    cache_dir = tmp_path / "switching_activity"
    cache_dir.mkdir()
    payload = {
        "by_opcode": {
            "add": {"sa": 1.0, "ar": 0.1},
            "mul": {"sa": 2.0, "ar": 0.2},
            "sub": {"sa": 3.0, "ar": 0.3},
        }
    }
    (cache_dir / "atax_switching_activity.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    opcode = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="opcode",
        cache_dir=cache_dir,
    )
    shuffled_a = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="shuffled",
        seed_parts=("bench", "dev", 0),
        cache_dir=cache_dir,
    )
    shuffled_b = gen_dataset_std._load_activity_by_opcode(
        "atax",
        activity_mode="shuffled",
        seed_parts=("bench", "dev", 0),
        cache_dir=cache_dir,
    )

    assert opcode == payload["by_opcode"]
    assert shuffled_a == shuffled_b
    assert sorted(shuffled_a) == sorted(payload["by_opcode"])
    assert sorted(shuffled_a.values(), key=lambda item: item["sa"]) == sorted(
        payload["by_opcode"].values(),
        key=lambda item: item["sa"],
    )


def test_build_tasks_threads_activity_mode(tmp_path):
    bench_path = tmp_path / "bench" / "device"
    script_dir = bench_path / "script"
    script_dir.mkdir(parents=True)
    (script_dir / "ppa_7.json").write_text("{}", encoding="utf-8")

    tasks = gen_dataset_std._build_tasks(
        [("bench", "device", bench_path)],
        device_override=None,
        emit_dot=False,
        timeout_s=5,
        activity_mode="shuffled",
        operator_merge="graph",
        activity_cache_dir=tmp_path / "activity",
    )

    assert tasks == [
        (
            "bench",
            "device",
            7,
            str(bench_path),
            "device",
            False,
            5,
            "shuffled",
            "graph",
            str(tmp_path / "activity"),
        )
    ]


def test_infer_top_name_ignores_pipeline_bind_and_sched_files(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    for name in ["atax_Pipeline_lp1.adb", "atax.bind.adb", "atax.sched.adb", "atax.adb"]:
        Path(graph_dir / name).write_text("", encoding="utf-8")

    assert gen_dataset_std._infer_top_name(graph_dir) == "atax"


def test_feature_encode_prefers_node_activity_over_opcode_fallback():
    nx = __import__("networkx")
    from hgp.data_process.feature_encode import generate_pyg_dot

    graph = nx.DiGraph()
    graph.add_node("0_1", m_delay=1, latency=[0, 1], bitwidth=64, lut=1, ff=1, dsp=0, opcode="add", node_type="0")
    graph.add_node("0_2", m_delay=1, latency=[0, 1], bitwidth=32, lut=1, ff=1, dsp=0, opcode="mul", node_type="0")
    graph.add_edge("0_1", "0_2", edge_type="1", is_back_edge="0")

    encoded = generate_pyg_dot(
        graph,
        None,
        ["m_delay", "latency", "bitwidth", "lut", "ff", "dsp"],
        sa_by_opcode={
            "by_node_id": {"0_1": {"sa": 3.0, "ar": 0.9}},
            "by_opcode": {
                "add": {"sa": 1.0, "ar": 0.1},
                "mul": {"sa": 2.0, "ar": 0.2},
            },
        },
    )

    # src uses node-level SA scaled by 64/32; dst falls back to opcode-level SA.
    assert encoded.edges["0_1", "0_2"]["edge_attr"] == [1.0, 0.0, 6.0, 0.9, 2.0, 0.2, 0.0, 0.0, 0.0]


def test_feature_encode_adds_node_switching_features():
    nx = __import__("networkx")
    from hgp.data_process.feature_encode import generate_pyg_dot

    graph = nx.DiGraph()
    graph.add_node(
        "0_1",
        m_delay=1,
        latency=[0, 1],
        bitwidth=64,
        lut=1,
        ff=1,
        dsp=0,
        opcode="add",
        node_type="0",
        merged_node_count=2,
    )

    encoded = generate_pyg_dot(
        graph,
        None,
        ["m_delay", "latency", "bitwidth", "merged_node_count", "activity_sa", "activity_ar"],
        sa_by_opcode={"by_node_id": {"0_1": {"sa": 3.0, "ar": 0.9}}},
    )

    assert encoded.nodes["0_1"]["x"][:7] == [1.0, 0.0, 1.0, 64.0, 2.0, 6.0, 0.9]


def test_fsmd_binding_parser_skips_bind_sched_prefixes(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "top.bind.adb").write_text(
        """
<root>
  <state_list>
    <state id="2" st_id="2">
      <operation id="48" st_id="2" stage="2" lat="3">
        <core>Cmp</core>
        <MemPortIdVec>0 1</MemPortIdVec>
        <Node id="28" bw="1"><![CDATA[%icmp = icmp_eq i7 %i, 64]]></Node>
        <StgValue><ssdm name="icmp_ln11"/></StgValue>
      </operation>
    </state>
  </state_list>
</root>
""",
        encoding="utf-8",
    )
    (graph_dir / "top.sched.adb").write_text("<ignored/>", encoding="utf-8")
    (graph_dir / "top.adb").write_text(
        """
<root>
  <state_list>
    <state id="2" st_id="2">
      <operation id="48" st_id="2" stage="1" lat="2">
        <core>Cmp</core>
        <MemPortIdVec>0 1</MemPortIdVec>
        <Node id="28" bw="1"><![CDATA[%icmp = icmp_eq i7 %i, 64]]></Node>
        <StgValue><ssdm name="icmp_ln11"/></StgValue>
      </operation>
    </state>
  </state_list>
</root>
""",
        encoding="utf-8",
    )
    (graph_dir / "sub.adb").write_text(
        """
<root>
  <state_list>
    <state id="1" st_id="1">
      <operation id="7" st_id="1" stage="3" lat="1">
        <core>Adder</core>
        <MemPortIdVec></MemPortIdVec>
        <Node id="4" bw="32"><![CDATA[%add = add i32 %a, %b]]></Node>
        <StgValue><ssdm name="add_ln"/></StgValue>
      </operation>
    </state>
  </state_list>
</root>
""",
        encoding="utf-8",
    )

    parsed = parse_fsmd_bindings(graph_dir)

    assert parsed["0_4"]["fsmd_module"] == "sub"
    assert parsed["0_4"]["fsmd_stage"] == 3
    assert parsed["1_28"]["fsmd_module"] == "top"
    assert parsed["1_28"]["fsmd_artifact"] == "bind"
    assert parsed["1_28"]["fsmd_stage"] == 2
    assert parsed["1_28"]["fsmd_mem_port_count"] == 2


def test_fsmd_annotation_adds_operator_metadata(tmp_path):
    nx = __import__("networkx")
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "top.adb").write_text(
        """
<root>
  <state_list>
    <state id="5" st_id="5">
      <operation id="10" st_id="5" stage="2" lat="3">
        <core>RAM</core>
        <MemPortIdVec>0</MemPortIdVec>
        <Node id="7" bw="32"><![CDATA[%load = load i32 %A]]></Node>
        <StgValue><ssdm name="load"/></StgValue>
      </operation>
    </state>
  </state_list>
</root>
""",
        encoding="utf-8",
    )
    graph = nx.DiGraph()
    graph.add_node("0_7", node_type="0", opcode="load", core_name="RAM", rtl_name="not_exist")
    graph.add_node("0_8", node_type="0", opcode="add", core_name="Adder", rtl_name="shared_add")
    graph.add_node("0_9", node_type="0", opcode="add", core_name="Adder", rtl_name="shared_add")

    stats = annotate_graph_with_fsmd(graph, graph_dir)

    assert stats["annotated_nodes"] == 1
    assert graph.nodes["0_7"]["fsmd_state"] == 5
    assert graph.nodes["0_7"]["fsmd_stage"] == 2
    assert graph.nodes["0_7"]["fsmd_mem_port_count"] == 1
    assert graph.nodes["0_7"]["operator_resource_type"] == 2
    assert graph.nodes["0_8"]["operator_shared_count"] == 2
