import torch


def load_pretrained_state_dict(model, state_dict):
    compatible_state_dict = dict(state_dict)
    model_state_dict = model.state_dict()

    for key, value in model_state_dict.items():
        if key.endswith(".select.weight") and key not in compatible_state_dict:
            compatible_state_dict[key] = torch.ones_like(value)

    model.load_state_dict(compatible_state_dict)
