import os
import torch

from .net import HierNet

def lut_pred(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.to(device)
    model_path = os.path.abspath('../hgp/model/lut_h64_d0_checkpoint_test.pt')
    model = HierNet(in_channels=15, hidden_channels=64, num_layers=3, conv_type='sage',
                    hls_dim=6, drop_out=0.0)
    model.load_state_dict(torch.load(model_path, map_location="cuda")['model'], strict=False)
    model = model.to(device)    

    model.eval()
    with torch.no_grad():
        hls_attr = data['hls_attr']
        num = data.x.shape[0]
        batch = torch.tensor([0 for i in range(num)]).to(device)
        out = model(data.x, data.edge_index, batch, hls_attr)
    lut = out.view(-1).item()
    lut = round(lut)
    return lut
