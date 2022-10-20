import torch
import numpy as np
import tqdm

from . import logging_utils
import data
import argparse
from functools import partial
from torch.utils.data import DataLoader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_s(g, n_spurious):
    y = g // n_spurious
    s = g % n_spurious
    return y, s


def update_dict(acc_groups, y, g, logits):
    preds = torch.argmax(logits, axis=1)
    correct_batch = (preds == y)
    g = g.cpu()
    for g_val in np.unique(g):
        mask = g == g_val
        n = mask.sum().item()
        corr = correct_batch[mask].sum().item()
        acc_groups[g_val].update(corr / n, n)


def get_results(acc_groups, get_ys_func):
    #TODO(izmailovpavel): add mean acc on train group distribution
    groups = acc_groups.keys()
    results = {
        f"accuracy_{get_ys_func(g)[0]}_{get_ys_func(g)[1]}": acc_groups[g].avg
        for g in groups
    }
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    results.update({"worst_accuracy" : min(results.values())})
    return results


def get_model_dataset_args():
    parser = argparse.ArgumentParser()

    # Model args
    parser.add_argument(
        "--model", type=str, required=False,
        default="imagenet_resnet50_pretrained",
        choices=["imagenet_resnet50",
                 "imagenet_resnet50_pretrained",
                 "imagenet_resnet50_dino",
                 "imagenet_resnet50_simclr",
                 "imagenet_resnet50_barlowtwins",
                 "imagenet_resnet50_timm",
                 "imagenet_resnet18_pretrained",
                 "imagenet_resnet34_pretrained",
                 "imagenet_resnet101_pretrained",
                 "imagenet_resnet152_pretrained",
                 "imagenet_wide_resnet50_2_pretrained",
                 "imagenet_resnext50_32x4d_pretrained",
                 "imagenet_densenet121_pretrained",
                 "imagenet_densenet121",
                 "imagenet_convnext_small_pretrained",
                 "imagenet_convnext_base_pretrained",
                 "imagenet_convnext_large_pretrained",
                 "imagenet_convnext_small_in22ft1k_pretrained",
                 "imagenet_convnext_base_in22ft1k_pretrained",
                 "imagenet_convnext_large_in22ft1k_pretrained",
                 "imagenet_convnext_xlarge_in22ft1k_pretrained",
                 "imagenet_convnext_small_in22k_pretrained",
                 "imagenet_convnext_base_in22k_pretrained",
                 "imagenet_convnext_large_in22k_pretrained",
                 "imagenet_convnext_xlarge_in22k_pretrained",
                 "imagenet_vit_small_pretrained",
                 "imagenet_vit_base_pretrained",
                 "imagenet_vit_large_pretrained",
                 "imagenet_vit_small_in21k_pretrained",
                 "imagenet_vit_base_in21k_pretrained",
                 "imagenet_vit_large_in21k_pretrained",
                 "imagenet_vit_huge_in21k_pretrained",
                 "imagenet_dino_small_pretrained",
                 "imagenet_dino_base_pretrained",
                 "imagenet_beit_base_pretrained",
                 "imagenet_beit_large_pretrained",
                 "imagenet_beit_base_in22k_pretrained",
                 "imagenet_beit_large_in22k_pretrained",
                 "imagenet_mae_base_pretrained",
                 "imagenet_mae_large_pretrained",
                 "imagenet_mae_huge_pretrained",
                 "imagenet_mae_base_ft1k_pretrained",
                 "imagenet_mae_large_ft1k_pretrained",
                 "imagenet_mae_huge_ft1k_pretrained",                 
                 "imagenet_deit_small_pretrained",
                 "imagenet_deit_base_pretrained",
                 "imagenet_swin_base_pretrained",
                 "imagenet_vgg19_pretrained",
                 "imagenet_vgg16_pretrained",
                 "imagenet_alexnet_pretrained",
                 "cifar_preresnet20",
                 "cifar_resnet18",
                 "simclr_cifar_resnet18_twolayerhead",
                 "domino_resnet18",
                 "domino_preresnet20",
                 "albert_pretrained",
                 "bert_pretrained",
                 "bert_pretrained_multilingual",
                 "bert_large_pretrained",
                 "bert",
                 "deberta_pretrained",
                 "deberta_large_pretrained"],
        help="Base model")
    # Data args
    parser.add_argument(
        "--data_dir", type=str, required=False,
        default="/data/users/pavel_i/datasets/waterbirds_birds_places/combined",
        help="Train dataset directory")
    parser.add_argument(
        "--test_data_dir", type=str, default=None, required=False,
        help="Test data directory (default: <data_dir>)")
    parser.add_argument(
        "--data_transform", type=str, required=False,
        default="AugWaterbirdsCelebATransform",
        choices=["None",
                 "AugDominoTransform",
                 "NoAugDominoTransform",
                 "SimCLRDominoTransform",
                 "MaskedDominoTransform",
                 "AugWaterbirdsCelebATransform",
                 "SimCLRWaterbirdsCelebATransform",
                 "NoAugWaterbirdsCelebATransform",
                 "NoAugNoNormWaterbirdsCelebATransform",
                 "MaskedWaterbirdsCelebATransform",
                 "ImageNetAugmixTransform",
                 "ImageNetRandomErasingTransform",
                 "SimCLRCifarTransform",
                 "AlbertTokenizeTransform",
                 "BertTokenizeTransform",
                 "BertMultilingualTokenizeTransform",
                 "DebertaTokenizeTransform",
                 ],
        help="Data preprocessing transformation")
    parser.add_argument(
        "--dataset", type=str, required=False,
        default="SpuriousCorrelationDataset",
        choices=["SpuriousCorrelationDataset",
                 "MultiNLIDataset",
                 "FakeSpuriousCIFAR10",
                 "WildsFMOW",
                 "WildsCivilCommentsCoarse",
                 "WildsCivilCommentsCoarseNM",
                 "DeBERTaMultiNLIDataset",
                 "BERTMultilingualMultiNLIDataset"],
        help="Dataset type")
    return parser


def get_default_args():
    parser = get_model_dataset_args()

    parser.add_argument(
        "--output_dir", type=str, help="Output directory")
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--reweight_groups", action='store_true',
        help="Reweight groups")
    parser.add_argument("--reweight_classes", action='store_true',
        help="Reweight classes")
    parser.add_argument("--reweight_spurious", action='store_true',
        help="Reweight based on spurious attribute")

    # Training hypers
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=300)

    # optimizer
    parser.add_argument(
        "--optimizer", type=str, required=False,
        default="sgd_optimizer",
        choices=["sgd_optimizer", "adamw_optimizer", "bert_adamw_optimizer"],
        help="Optimizer name")
    parser.add_argument(
        "--scheduler", type=str, required=False,
        default="constant_lr_scheduler",
        choices=["cosine_lr_scheduler",
                 "constant_lr_scheduler",
                 "bert_lr_scheduler"],
        help="Scheduler name")
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=0.6e-1)
    parser.add_argument("--no_shuffle_train", action='store_true')

    parser.add_argument("--mixup", action='store_true')


    # Understanding exps
    parser.add_argument(
        "--num_minority_groups_remove", type=int, required=False, default=0)
    return parser


def get_data(args, logger=None, contrastive=False, finetune_on_val=False):
    if args.data_transform == "None":
        transform_cls = lambda *args, **kwargs: None
    else:
        transform_cls = getattr(data, args.data_transform)
    train_transform = transform_cls(train=True)
    if contrastive:
        train_transform = data.RepeatTransform(train_transform)
    test_transform = transform_cls(train=False)

    dataset_cls = getattr(data, args.dataset)
    if finetune_on_val:
        # TODO: train or test transform
        trainset = dataset_cls(
            basedir=args.data_dir, split="val", transform=train_transform)
        data.balance_groups(trainset)
    else:
        trainset = dataset_cls(
            basedir=args.data_dir, split="train", transform=train_transform)

    data.remove_minority_groups(trainset, args.num_minority_groups_remove)

    test_data_dir = args.test_data_dir if args.test_data_dir else args.data_dir
    testset_dict = {
        split: dataset_cls(
            basedir=test_data_dir, split=split, transform=test_transform)
        for split in ["test", "val"]}

    collate_fn=data.get_collate_fn(
        mixup=args.mixup, num_classes=trainset.n_classes)

    loader_kwargs = {
        'batch_size': args.batch_size, 'num_workers': 16, 'pin_memory': True}
    sampler = data.get_sampler(trainset, args)
    train_shuffle = False if sampler else not args.no_shuffle_train
    train_loader = DataLoader(
        trainset, shuffle=train_shuffle, sampler=sampler, collate_fn=collate_fn,
        **loader_kwargs)
    test_loader_dict = {
        name: DataLoader(ds, shuffle=False, **loader_kwargs)
        for name, ds in testset_dict.items()
    }

    get_ys_func = partial(get_y_s, n_spurious=testset_dict['test'].n_spurious)
    if logger is not None:
        logging_utils.log_data(
            logger, trainset, testset_dict['test'], testset_dict['val'], get_ys_func=get_ys_func)

    if contrastive:
        finetune_transform = transform_cls(train=True, finetune=True)
        trainset_finetune = dataset_cls(
            basedir=args.data_dir, split="train", transform=finetune_transform)
        finetune_loader = DataLoader(
            trainset_finetune, shuffle=False if sampler else True,
            sampler=sampler,**loader_kwargs)
        return train_loader, finetune_loader, test_loader_dict, get_ys_func

    return train_loader, test_loader_dict, get_ys_func
