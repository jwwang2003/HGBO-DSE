import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder

onehotFlag = 1

n_categ_items = ['op_type', 'node_type']

opcode_categ = ['dadd', 'dsub', 'dmul', 'ddiv', 'call',
                'add', 'sub', 'mul', 'div', 'icmp', 'dcmp', 'urem',
                'and', 'or', 'xor', 'lshr', 'shl', 'ashr',
                'select', 'bitselect', 'partselect', 'partset', 'mux', 'switch',
                'load', 'store']

op_type_categ = [['db_op'], ['sp_op'], ['bit_op'], ['ctr_op'], ['mem_op'], ['none']]
node_type_categ = [['0'], ['1']]

# General edge features
e_num_items = ['edge_type', 'is_back']

# Pre-fit encoders (module-level singletons to avoid repeated fitting).
_optype_enc = OneHotEncoder(handle_unknown='ignore')
_optype_enc.fit(op_type_categ)  # 6 bits

_node_type_enc = OneHotEncoder(handle_unknown='ignore')
_node_type_enc.fit(node_type_categ)  # 2 bits


def onehot_enc_gen():
    return _optype_enc, _node_type_enc


_OPCODE_TO_TYPE = {}
for _oc in ('dadd', 'dsub', 'dmul', 'ddiv', 'call'):
    _OPCODE_TO_TYPE[_oc] = 'db_op'
for _oc in ('add', 'sub', 'mul', 'div', 'icmp', 'dcmp', 'urem'):
    _OPCODE_TO_TYPE[_oc] = 'sp_op'
for _oc in ('and', 'or', 'xor', 'lshr', 'shl', 'ashr'):
    _OPCODE_TO_TYPE[_oc] = 'bit_op'
for _oc in ('select', 'bitselect', 'partselect', 'partset', 'mux', 'switch'):
    _OPCODE_TO_TYPE[_oc] = 'ctr_op'
for _oc in ('load', 'store'):
    _OPCODE_TO_TYPE[_oc] = 'mem_op'

_OPTYPE_TO_NUM = {'db_op': 5.0, 'sp_op': 4.0, 'bit_op': 3.0, 'ctr_op': 2.0, 'mem_op': 1.0, 'none': 0.0}


def opcode_type(opcode):
    return _OPCODE_TO_TYPE.get(opcode, 'none')


def opcode_type_numerical(opcode):
    return _OPTYPE_TO_NUM.get(_OPCODE_TO_TYPE.get(opcode, 'none'), 0.0)


def generate_pyg_dot(DG, dot_store_path, n_num_items):
    pyg_DG = DG.__class__()
    pyg_DG.add_nodes_from(DG)
    pyg_DG.add_edges_from(DG.edges)

    node_ids = list(DG.nodes())
    n_nodes = len(node_ids)

    if onehotFlag:
        # Collect all optype and node_type labels for batch encoding.
        optype_labels = []
        node_type_labels = []
        num_feats_per_node = []

        for node_id in node_ids:
            node = DG.nodes[node_id]
            node_feat = []
            for feat_item in n_num_items:
                if feat_item not in node:
                    if feat_item == 'latency':
                        node_feat.extend([0.0, 0.0])
                    else:
                        node_feat.append(0.0)
                else:
                    if feat_item == 'latency':
                        node_feat.extend([float(node[feat_item][0]), float(node[feat_item][1])])
                    elif type(node[feat_item]) == int or type(node[feat_item]) == str:
                        node_feat.append(float(node[feat_item]))
                    else:
                        raise AssertionError(
                            f"feat_item = {feat_item}, unexpected type {type(node[feat_item])}"
                        )

            num_feats_per_node.append(node_feat)
            n_opcode = node.get('opcode', 'none')
            optype_labels.append([opcode_type(n_opcode)])
            node_type_labels.append([node['node_type']])

        # Batch transform — single call to sklearn for all nodes.
        onehot_optype_all = _optype_enc.transform(optype_labels).toarray()   # (N, 6)
        onehot_ntype_all = _node_type_enc.transform(node_type_labels).toarray()  # (N, 2)

        for i, node_id in enumerate(node_ids):
            feat = np.concatenate((
                num_feats_per_node[i],
                onehot_optype_all[i],
                onehot_ntype_all[i],
            ), axis=0)
            pyg_DG.nodes[node_id]['x'] = list(feat)
    else:
        for node_id in node_ids:
            node = DG.nodes[node_id]
            node_feat = []
            for feat_item in n_num_items:
                if feat_item not in node:
                    if feat_item == 'latency':
                        node_feat.extend([0.0, 0.0])
                    else:
                        node_feat.append(0.0)
                else:
                    if feat_item == 'latency':
                        node_feat.extend([float(node[feat_item][0]), float(node[feat_item][1])])
                    elif type(node[feat_item]) == int or type(node[feat_item]) == str:
                        node_feat.append(float(node[feat_item]))
                    else:
                        raise AssertionError(
                            f"feat_item = {feat_item}, unexpected type {type(node[feat_item])}"
                        )
            n_opcode = node.get('opcode', 'none')
            n_optype = opcode_type_numerical(n_opcode)
            node_type_val = float(node['node_type'])
            node_feat.extend([n_optype, node_type_val])
            pyg_DG.nodes[node_id]['x'] = node_feat

    for edge_id in DG.edges():
        edge = DG.edges[edge_id]
        pyg_DG.edges[edge_id]['edge_attr'] = [float(edge['edge_type']), float(edge['is_back_edge'])]

    if dot_store_path:
        nx.nx_pydot.write_dot(pyg_DG, dot_store_path)
    return pyg_DG
