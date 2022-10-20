"""Load models from the VISSL package.
Code adapted from 
https://github.com/facebookresearch/vissl/blob/main/extra_scripts/convert_vissl_to_torchvision.py
"""

import torch
import torchvision
from torch.hub import load_state_dict_from_url

from .model_utils import _replace_fc


SIMCLR_RN50_URL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/simclr_rn50_800ep_simclr_8node_resnet_16_07_20.7e8feed1/model_final_checkpoint_phase799.torch"
BARLOWTWINS_RN50_URL = "https://dl.fbaipublicfiles.com/vissl/model_zoo/barlow_twins/barlow_twins_32gpus_4node_imagenet1k_1000ep_resnet50.torch"


def replace_module_prefix(state_dict, prefix, replace_with=""):
    state_dict = {
        (key.replace(prefix, replace_with, 1)
         if key.startswith(prefix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def get_torchvision_state_dict(url):
    model = load_state_dict_from_url(url)
    model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]

    return replace_module_prefix(model_trunk, "_feature_blocks.")


def imagenet_resnet50_simclr(output_dim):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    model.load_state_dict(get_torchvision_state_dict(SIMCLR_RN50_URL))
    model.fc.in_features = 2048
    return _replace_fc(model, output_dim)


def imagenet_resnet50_barlowtwins(output_dim):
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    import vissl
    model.load_state_dict(get_torchvision_state_dict(BARLOWTWINS_RN50_URL))
    model.fc.in_features = 2048
    return _replace_fc(model, output_dim)
