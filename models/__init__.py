import sys
import torch
import torch.nn as nn
import torchvision
import types
import timm
#import mae.models_vit as mae

from .preresnet import PreResNet
from .text_models import albert_pretrained
from .text_models import bert_pretrained
from .text_models import bert_pretrained_multilingual
from .text_models import bert_large_pretrained
from .text_models import bert
from .text_models import deberta_pretrained
from .text_models import deberta_large_pretrained
from .model_utils import _replace_fc
from .vissl_models import imagenet_resnet50_simclr
from .vissl_models import imagenet_resnet50_barlowtwins


def domino_preresnet20(output_dim):
    return _replace_fc(PreResNet(domino=True, depth=20), output_dim)


def cifar_preresnet20(output_dim):
    return _replace_fc(PreResNet(domino=False, depth=20), output_dim)


def _base_resnet18_cifar():
    # much wider than cifar_preresnet20
    model = torchvision.models.resnet18(pretrained=False)  # load model from torchvision.models without pretrained weights.
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    return model


def cifar_resnet18(output_dim):
    model = _base_resnet18_cifar()
    return _replace_fc(model, output_dim)


def domino_resnet18(output_dim):
    model = _base_resnet18_cifar()
    # model.fc.in_features *= 2
    return _replace_fc(model, output_dim)


def simclr_cifar_resnet18_twolayerhead(output_dim):
    hidden_dim = 2048
    model = _base_resnet18_cifar()
    d = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(d, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, output_dim))
    model.fc.in_features = d

    return model


def simclr_cifar_resnet18(output_dim):
    # ToDo: consider non-linear projection head?
    # much wider than cifar_preresnet20
    model = torchvision.models.resnet18(pretrained=False)  # load model from torchvision.models without pretrained weights.
    model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = torch.nn.Identity()
    return _replace_fc(model, output_dim)


def imagenet_resnet50(output_dim):
    return _replace_fc(
        torchvision.models.resnet50(pretrained=False), output_dim)


def imagenet_resnet50_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet50(pretrained=True), output_dim)


def imagenet_resnet50_dino(output_dim):
    # workaround to avoid module name collision
    sys.modules.pop("utils")
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model.fc.in_features = 2048
    return _replace_fc(model, output_dim)

def imagenet_resnet50_timm(output_dim):
    return _replace_fc(
        timm.create_model('resnet50', pretrained=True), output_dim)


def imagenet_resnet18_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet18(pretrained=True), output_dim)


def imagenet_resnet34_pretrained(output_dim):
    return _replace_fc(torchvision.models.resnet34(pretrained=True), output_dim)


def imagenet_resnet101_pretrained(output_dim):
    return _replace_fc(
        torchvision.models.resnet101(pretrained=True), output_dim)


def imagenet_resnet152_pretrained(output_dim):
    return _replace_fc(
        torchvision.models.resnet152(pretrained=True), output_dim)


def imagenet_wide_resnet50_2_pretrained(output_dim):
    return _replace_fc(
        torchvision.models.wide_resnet50_2(pretrained=True), output_dim)


def imagenet_resnext50_32x4d_pretrained(output_dim):
    return _replace_fc(
        torchvision.models.resnext50_32x4d(pretrained=True), output_dim)


def _densenet_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.classifier.in_features
    delattr(model, "classifier")

    def classifier(self, x):
        return self.fc(x)
    
    model.classifier = types.MethodType(classifier, model)
    return _replace_fc(model, output_dim)


def imagenet_densenet121_pretrained(output_dim):
    return _densenet_replace_fc(
        torchvision.models.densenet121(pretrained=True), output_dim)


def imagenet_densenet121(output_dim):
    return _densenet_replace_fc(
        torchvision.models.densenet121(pretrained=False), output_dim)


def _vgg_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.classifier[0].in_features
    delattr(model, "classifier")


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    forwardType = types.MethodType
    model.forward = forwardType(forward, model)
    return _replace_fc(model, output_dim)


def imagenet_vgg19_pretrained(output_dim):
    model = torchvision.models.vgg19(pretrained=True)
    return _vgg_replace_fc(model, output_dim)


def imagenet_vgg16_pretrained(output_dim):
    model = torchvision.models.vgg16(pretrained=True)
    return _vgg_replace_fc(model, output_dim)


def imagenet_alexnet_pretrained(output_dim):
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[0].in_features = model.classifier[1].in_features
    return _vgg_replace_fc(model, output_dim)

def load_small_convnext(checkpoint_path, **kwargs):
    from timm.models.convnext import ConvNeXt, checkpoint_filter_fn

    model = ConvNeXt(**kwargs)

    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
    checkpoint = checkpoint_filter_fn(checkpoint, model)

    for k in ['head.fc.weight', 'head.fc.bias']:
        if k in checkpoint:# and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)

    return model

#ConvNeXt models
def _convnext_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.head.fc.in_features
    delattr(model.head, "fc")

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        x = self.fc(x)
        return x

    forwardType = types.MethodType
    model.forward = forwardType(forward, model)
    return _replace_fc(model, output_dim)

def imagenet_convnext_small_pretrained(output_dim):
    model = timm.create_model('convnext_small', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_base_pretrained(output_dim):
    model = timm.create_model('convnext_base', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_large_pretrained(output_dim):
    model = timm.create_model('convnext_large', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_small_in22ft1k_pretrained(output_dim):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=output_dim)
    model = load_small_convnext('/scratch/nvg7279/convnext_models/convnext_small_22k_1k_224.pth', **model_args)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_base_in22ft1k_pretrained(output_dim):
    model = timm.create_model('convnext_base_in22ft1k', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_large_in22ft1k_pretrained(output_dim):
    model = timm.create_model('convnext_large_in22ft1k', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_xlarge_in22ft1k_pretrained(output_dim):
    model = timm.create_model('convnext_xlarge_in22ft1k', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_small_in22k_pretrained(output_dim):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], num_classes=output_dim)
    model = load_small_convnext('/scratch/nvg7279/convnext_models/convnext_small_22k_224.pth', **model_args)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_base_in22k_pretrained(output_dim):
    model = timm.create_model('convnext_base_in22k', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_large_in22k_pretrained(output_dim):
    model = timm.create_model('convnext_large_in22k', pretrained=True)
    return _convnext_replace_fc(model, output_dim)

def imagenet_convnext_xlarge_in22k_pretrained(output_dim):
    model = timm.create_model('convnext_xlarge_in22k', pretrained=True)
    return _convnext_replace_fc(model, output_dim)


def _vit_replace_fc(model, output_dim):
    model.fc = torch.nn.Identity()
    model.fc.in_features = model.head.in_features
    delattr(model, "head")

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x

    forwardType = types.MethodType
    model.forward = forwardType(forward, model)
    return _replace_fc(model, output_dim)


def imagenet_vit_small_pretrained(output_dim):
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_vit_base_pretrained(output_dim):
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_vit_large_pretrained(output_dim):
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_vit_small_in21k_pretrained(output_dim):
    model = timm.create_model('vit_small_patch16_224_in21k', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_vit_base_in21k_pretrained(output_dim):
    model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_vit_large_in21k_pretrained(output_dim):
    model = timm.create_model('vit_large_patch16_224_in21k', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_vit_huge_in21k_pretrained(output_dim):
    model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_beit_base_pretrained(output_dim):
    model = timm.create_model('beit_base_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_beit_large_pretrained(output_dim):
    model = timm.create_model('beit_large_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_beit_base_in22k_pretrained(output_dim):
    model = timm.create_model('beit_base_patch16_224_in22k', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_beit_large_in22k_pretrained(output_dim):
    model = timm.create_model('beit_large_patch16_224_in22k', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_deit_small_pretrained(output_dim):
    model = timm.create_model('deit_small_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_deit_base_pretrained(output_dim):
    model = timm.create_model('deit_base_patch16_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def imagenet_swin_base_pretrained(output_dim):
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    return _vit_replace_fc(model, output_dim)

def load_dino_model(checkpoint_path, **kwargs):
    from timm.models.vision_transformer import VisionTransformer

    model = VisionTransformer(**kwargs)

    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint:# and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)

    return model

def imagenet_dino_small_pretrained(output_dim):
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=output_dim)
    model = load_dino_model('/scratch/nvg7279/dino_models/dino_deitsmall16_pretrain.pth', **model_kwargs)
    return _vit_replace_fc(model, output_dim)

def imagenet_dino_base_pretrained(output_dim):
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=output_dim)
    model = load_dino_model('/scratch/nvg7279/dino_models/dino_vitbase16_pretrain.pth', **model_kwargs)
    return _vit_replace_fc(model, output_dim)

def load_mae_model(model, checkpoint_path, global_pool=True):
    from timm.models.layers import trunc_normal_
    from mae.util.pos_embed import interpolate_pos_embed

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    else:
        checkpoint_model = checkpoint

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:# and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if global_pool:
        for k in ['fc_norm.weight', 'fc_norm.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    trunc_normal_(model.head.weight, std=2e-5)

def imagenet_mae_base_pretrained(output_dim):
    model = mae.vit_base_patch16(num_classes=output_dim)
    load_mae_model(model, '/scratch/nvg7279/mae_models/mae_pretrain_vit_base.pth')
    return _vit_replace_fc(model, output_dim)

def imagenet_mae_large_pretrained(output_dim):
    model = mae.vit_large_patch16(num_classes=output_dim)
    load_mae_model(model, '/scratch/nvg7279/mae_models/mae_pretrain_vit_large.pth')
    return _vit_replace_fc(model, output_dim)

def imagenet_mae_huge_pretrained(output_dim):
    model = mae.vit_huge_patch14(num_classes=output_dim)
    load_mae_model(model, '/scratch/nvg7279/mae_models/mae_pretrain_vit_huge.pth')
    return _vit_replace_fc(model, output_dim)

def imagenet_mae_base_ft1k_pretrained(output_dim):
    model = mae.vit_base_patch16(num_classes=output_dim)
    load_mae_model(model, '/scratch/nvg7279/mae_models/mae_finetuned_vit_base.pth')
    return _vit_replace_fc(model, output_dim)

def imagenet_mae_large_ft1k_pretrained(output_dim):
    model = mae.vit_large_patch16(num_classes=output_dim)
    load_mae_model(model, '/scratch/nvg7279/mae_models/mae_finetuned_vit_large.pth')
    return _vit_replace_fc(model, output_dim)

def imagenet_mae_huge_ft1k_pretrained(output_dim):
    model = mae.vit_huge_patch14(num_classes=output_dim)
    load_mae_model(model, '/scratch/nvg7279/mae_models/mae_finetuned_vit_huge.pth')
    return _vit_replace_fc(model, output_dim)
