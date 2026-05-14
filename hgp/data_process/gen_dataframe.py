import torch
from torch_geometric.data import Data

try:
    from hgp.board_utils import attach_board_profile, resolve_board_profile
except ImportError:  # pragma: no cover - supports running from hgp/data_process
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from hgp.board_utils import attach_board_profile, resolve_board_profile


def generate_dataframe(DG, metric_list, hls_attr, bench_name, prj_name, df_store_path, board_device=None):

    data = {'x': {}, 'edge_attr': {}}
    for node_id in DG.nodes():
        node = DG.nodes[node_id]
        data['x'] = data['x'] + [node['x']] if data['x'] else [node['x']]
    for edge_id in DG.edges():
        edge = DG.edges[edge_id]
        data['edge_attr'] = data['edge_attr'] + [edge['edge_attr']] if data['edge_attr'] else [edge['edge_attr']]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    edge_index = torch.LongTensor(list(DG.edges)).t().contiguous()
    data['edge_index'] = edge_index.view(2, -1)
    data['hls_attr'] = torch.tensor([hls_attr])
    data['y'] = torch.tensor([metric_list])
    data['bench_name'] = bench_name
    data['prj_name'] = prj_name

    dataframe = Data.from_dict(data)
    if board_device is not None:
        dataframe = attach_board_profile(dataframe, resolve_board_profile(board_device))
    torch.save(dataframe, df_store_path)

    return dataframe
