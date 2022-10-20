import torchvision.transforms as transforms
import einops
import torch
import math
from transformers import BertTokenizer
from transformers import DebertaV2Tokenizer
from transformers import AlbertTokenizer
from timm.data.random_erasing import RandomErasing

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class RepeatTransform:
    # adapted from https://github.com/sthalles/SimCLR/blob/master/data_aug/view_generator.py
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ColorDistortion(transforms.Compose):
    # adapted from https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/simclr.py
    def __init__(self, s=0.5):
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        self.transforms = [rnd_color_jitter, rnd_gray]


def _add_totensor_normalize(transform_lst, normalize_stats):
    transform_lst.append(transforms.ToTensor())
    if normalize_stats:
        transform_lst.append(transforms.Normalize(*normalize_stats))


def patchify(img, p):
    x = einops.rearrange(img, 'c (h p) (w q) -> (h w) (p q c)', p=p, q=p)
    return x


def unpatchify(x, p, height):
    h = height // p
    x = einops.rearrange(x, '(h w) (p q c) -> c (h p) (w q)', h=h, p=p, q=p)
    return x


class RandomPatchMask:
    # adapted from https://github.com/sthalles/SimCLR/blob/master/data_aug/view_generator.py
    def __init__(self, mask_ratio, patch_size, mask_val=0.):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.mask_val = mask_val

    def __call__(self, x):
        height = x.shape[1]
        x = patchify(x, self.patch_size)
        n_patches = x.shape[0]
        mask = torch.rand(n_patches) < self.mask_ratio
        x[mask] = self.mask_val
        return unpatchify(x, self.patch_size, height)


class BaseDominoTransform(transforms.Compose):
    def __init__(self, augment=True, normalize_stats=None):
        self.transforms = []
        if augment:
            self.transforms = [
                transforms.RandomCrop((64, 32), (4, 4)),
                transforms.RandomHorizontalFlip()]
        _add_totensor_normalize(self.transforms, normalize_stats)


class AugDominoTransform(BaseDominoTransform):
    def __init__(self, train):
        super().__init__(augment=train, normalize_stats=None)


class NoAugDominoTransform(BaseDominoTransform):
    def __init__(self, train):
        super().__init__(augment=False, normalize_stats=None)


class MaskedDominoTransform(BaseDominoTransform):
    def __init__(self, train, mask_ratio=0.75):
        super().__init__(augment=train, normalize_stats=None)
        if train:
            self.transforms.append(
                RandomPatchMask(mask_ratio=mask_ratio, patch_size=4))



class SimCLRDominoTransform(transforms.Compose):
    def __init__(self, train, finetune=False, normalize_stats=None):
        self.transforms = []
        if train or finetune:
            self.transforms = [
                transforms.RandomResizedCrop((64, 32), (4, 4)),
                transforms.RandomHorizontalFlip(p=0.5)]
        if train:
            self.transforms.append(ColorDistortion(s=0.5))
        _add_totensor_normalize(self.transforms, normalize_stats)


class BaseWaterbirdsCelebATransform(transforms.Compose):
    def __init__(self, augment, normalize_stats):
        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        self.transforms = []
        if augment:
            self.transforms = [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip()]
        else:
            self.transforms = [
                transforms.Resize(resize_resolution),
                transforms.CenterCrop(target_resolution)]
        _add_totensor_normalize(self.transforms, normalize_stats)


class AugWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    def __init__(self, train):
        super().__init__(augment=train, normalize_stats=IMAGENET_STATS)


class NoAugWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    def __init__(self, train):
        super().__init__(augment=False, normalize_stats=IMAGENET_STATS)


class NoAugNoNormWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    def __init__(self, train):
        super().__init__(augment=False, normalize_stats=None)


class ImageNetRandomErasingTransform(BaseWaterbirdsCelebATransform):
    def __init__(self, train):
        super().__init__(augment=train, normalize_stats=IMAGENET_STATS)
        if train:
            self.transforms.append(RandomErasing(device="cpu"))


class MaskedWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    def __init__(self, train, mask_ratio=0.75):
        super().__init__(augment=False, normalize_stats=None)
        if train:
            self.transforms.append(
                RandomPatchMask(mask_ratio=mask_ratio, patch_size=14))


class SimCLRWaterbirdsCelebATransform(BaseWaterbirdsCelebATransform):
    def __init__(self, train, finetune=False, normalize_stats=IMAGENET_STATS):
        super().__init__(augment=(train or finetune), normalize_stats=None)
        self.transforms = self.transforms[:-1] # Remove ToTensor
        if train:
            self.transforms.append(ColorDistortion(s=0.5))
        _add_totensor_normalize(self.transforms, normalize_stats)


class SimCLRCifarTransform(transforms.Compose):
    def __init__(self, train, finetune=False, normalize_stats=None):
        self.transforms = []
        if train or finetune:
            self.transforms = [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(p=0.5)]
        if train:
            self.transforms.append(ColorDistortion(s=0.5))
        _add_totensor_normalize(self.transforms, normalize_stats)


class TokenizeTransform:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        return torch.squeeze(torch.stack((
            tokens["input_ids"], tokens["attention_mask"], 
            tokens["token_type_ids"]), dim=2), dim=0)


class BertTokenizeTransform(TokenizeTransform):
    def __init__(self, train):
        super().__init__(
                tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"))
        del train


class BertMultilingualTokenizeTransform(TokenizeTransform):
    def __init__(self, train):
        super().__init__(tokenizer=BertTokenizer.from_pretrained(
                "bert-base-multilingual-uncased"))
        del train


class DebertaTokenizeTransform(TokenizeTransform):
    def __init__(self, train):
        super().__init__(tokenizer=DebertaV2Tokenizer.from_pretrained(
                "microsoft/deberta-v3-base"))
        del train


class AlbertTokenizeTransform(TokenizeTransform):
    def __init__(self, train):
        super().__init__(tokenizer=AlbertTokenizer.from_pretrained(
                "albert-base-v2"))
        del train
