import os
import torch
import torchvision
import torch.nn as nn

# REPO_DIR = '/home/msc-auto/szhao/dinov3'
# CKP_DIR = '/home/msc-auto/szhao/LeveFD'
REPO_DIR = '/home/shuqi/dinov3'
CKP_DIR = '/mnt/ssd1/szhao/LeveFD'

class MLP(torch.nn.Module):
    def __init__(self, units):
        super(MLP, self).__init__()
        self.fc = torch.nn.Linear(units[0], units[1])
        # self.fc1 = torch.nn.Linear(units[1], units[2])
        # self.layer_norm = torch.nn.LayerNorm(units[1])
        self.elu = torch.nn.ELU()

    def forward(self, x):
        x = self.fc(x)
        # x = self.fc1(self.elu(x))
        return x

def get_vision(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if 'resnet' in name:
        if (weights == "r3m") or (weights == "R3M"):
            return get_r3m(name=name, **kwargs)

        func = getattr(torchvision.models, name)
        resnet = func(weights=weights, **kwargs)
        resnet.fc = torch.nn.Identity()
        return resnet
    elif 'dino' in name:
        assert weights is not None
        ckp_path = os.path.join(CKP_DIR, weights + '.pth')
        dinov3_model = torch.hub.load(REPO_DIR, weights, source='local', weights=ckp_path)
        return dinov3_model
    else:
        NotImplementedError()

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def get_linear(units, 
    activation_function = None,
    norm_func_name = None,
    lastnorm_func_name = None):
    in_size = units[0]
    layers = []
    for idx in range(1, len(units)):
        unit = units[idx]
        layer = torch.nn.Linear(in_size, unit)
        layers.append(layer)
        if norm_func_name is not None and idx!=len(units)-1:
            if norm_func_name == 'layer_norm':
                layers.append(torch.nn.LayerNorm(unit))
            elif norm_func_name == 'batch_norm':
                layers.append(torch.nn.BatchNorm1d(unit))
            else:
                raise NotImplementedError()

        if activation_function is not None and idx!=len(units)-1:
            if activation_function == 'elu':
                layers.append(torch.nn.ELU())
            elif activation_function == 'relu':
                layers.append(torch.nn.ReLU())
            # elif activation_function == 'sigmoid':
            #     layers.append(torch.nn.Sigmoid())
            else:
                raise NotImplementedError()
        in_size = unit
    if lastnorm_func_name is not None:
        if lastnorm_func_name == 'tanh':
            layers.append(torch.nn.Tanh())
        elif lastnorm_func_name == 'sigmoid':
            layers.append(torch.nn.Sigmoid())
    return nn.Sequential(*layers)