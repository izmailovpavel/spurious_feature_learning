# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base augmentations operators."""

# Adapted from https://github.com/google-research/augmix/blob/master/augmentations.py


import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from .data_transforms import IMAGENET_STATS

IMAGE_SIZE = 224


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.

    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.

    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.

    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                                                     Image.AFFINE, (1, level, 0, 0, 1, 0),
                                                     resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                                                     Image.AFFINE, (1, 0, 0, level, 1, 0),
                                                     resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                                                     Image.AFFINE, (1, 0, level, 0, 1, 0),
                                                     resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                                                     Image.AFFINE, (1, 0, 0, 0, 1, level),
                                                     resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
        level = float_parameter(sample_level(level), 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)


AUGMENTATIONS = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y
]

AUGMENTATIONS_ALL = [
        autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
        translate_x, translate_y, color, contrast, brightness, sharpness
]


class AugmixTransformBase:
    def __init__(
        self, preprocess, aug_prob_coeff=1., mixture_width=3,
        mixture_depth=-1, aug_severity=1, aug_list=AUGMENTATIONS_ALL, 
    ):
        self.aug_list = aug_list
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.preprocess = preprocess

    def __call__(self, image):

        aug_list = AUGMENTATIONS_ALL

        ws = np.float32(
                np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        mix = torch.zeros_like(self.preprocess(image))
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = (
                self.mixture_depth if self.mixture_depth > 0 
                else np.random.randint(1, 4))
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, self.aug_severity)
            # Preprocessing commutes since all coefficients are convex
            mix += ws[i] * self.preprocess(image_aug)

        mixed = (1 - m) * self.preprocess(image) + m * mix
        return mixed


class ImageNetAugmixTransform(transforms.Compose):
    def __init__(self, train):
        preprocess = [
            transforms.ToTensor(), transforms.Normalize(*IMAGENET_STATS)]
        preprocess = transforms.Compose(preprocess)
        if train:
            self.transforms = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                AugmixTransformBase(preprocess=preprocess)]
        else:
            self.transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                preprocess]
