import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
from functools import partial


def mixup_batch(batch, num_classes, alpha=0.2):
    x, y, g, s = batch
    y = torch.nn.functional.one_hot(y, num_classes)
    batch_size = x.size()[0]
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam_ = torch.from_numpy(lam).float()
    lam_x = lam_.float().reshape((-1, 1, 1, 1))
    lam_y = lam_.float().reshape((-1, 1))
    index = torch.randperm(batch_size)
    mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
    mixed_y = lam_y * y + (1 - lam_y) * y[index, :]
    return mixed_x, mixed_y, g, s


def mixup_collate(batch, num_classes):
    collated = default_collate(batch)
    return mixup_batch(collated, num_classes)


def get_collate_fn(mixup, num_classes):
    if mixup:
        return partial(mixup_collate, num_classes=num_classes)
    else:
        return default_collate


def get_sampler_counts(counts, permutation=None):
    weights = sum(counts) / counts
    if permutation is not None:
        try:
            weights = weights[permutation]
        except:
            weights = weights[permutation.long()]
    return WeightedRandomSampler(weights, sum(counts).item(), replacement=True)


def get_sampler(data, args):
    sampler = None
    if args.reweight_groups:
        sampler = get_sampler_counts(data.group_counts, data.group_array)
    elif args.reweight_classes:
        sampler = get_sampler_counts(data.y_counts, data.y_array)
    elif args.reweight_spurious:
        sampler = get_sampler_counts(data.spurious_counts, data.spurious_array)
    return sampler