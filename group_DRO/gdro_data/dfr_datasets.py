from collections import namedtuple
import sys
sys.path.append("../")
import data as dfr_data
from torch.utils.data import DataLoader
import types


DatasetArgs = namedtuple("DatasetArgs", "data_transform dataset")

DFR_DATA_ARGS = {
    "CelebA": DatasetArgs(
        "AugWaterbirdsCelebATransform", "SpuriousCorrelationDataset"),
    "CUB": DatasetArgs(
        "AugWaterbirdsCelebATransform", "SpuriousCorrelationDataset"),
    "MultiNLI": DatasetArgs(
        "None", "MultiNLIDataset"),
    "CivilComments": DatasetArgs(
        "DebertaTokenizeTransform", "WildsCivilCommentsCoarse"),
        # "BertTokenizeTransform", "WildsCivilCommentsCoarse"),
    "FMOW": DatasetArgs(
        "AugWaterbirdsCelebATransform", "WildsFMOW"),
}


def get_loader(self, train, reweight_groups, **kwargs):
    if not train: # Validation or testing
        assert reweight_groups is None
        shuffle = False
        sampler = None
    elif not reweight_groups: # Training but not reweighting
        shuffle = True
        sampler = None
    else:
        group_weights = len(self)/self._group_counts
        weights = group_weights[self._group_array]

        # Replacement needs to be set to True, otherwise we'll run out of minority samples
        sampler = WeightedRandomSampler(weights, len(self), replacement=True)
        shuffle = False

    loader = DataLoader(
        self,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs)
    return loader


def group_str(self, g):
    y = g // self.n_spurious
    c = g % self.n_spurious
    #y = group_idx // (self.n_groups/self.n_classes)
    #c = group_idx % (self.n_groups//self.n_classes)
    return f"y={y},c={c}"


def rename_group_counts(dataset):
    def group_counts(self):
        return self.group_counts_

    dataset.group_counts_ = dataset.group_counts
    dataset.group_counts = types.MethodType(group_counts, dataset)


def prepare_data(args, train=True):
    data_args = DFR_DATA_ARGS[args.dataset]

    if data_args.data_transform == "None":
        transform_cls = lambda *args, **kwargs: None
    else:
        transform_cls = getattr(dfr_data, data_args.data_transform)
    train_transform = transform_cls(train=True)
    test_transform = transform_cls(train=False)

    dataset_cls = getattr(dfr_data, data_args.dataset)
    trainset = dataset_cls(
        basedir=args.root_dir, split="train", transform=train_transform)
    testset = dataset_cls(
            basedir=args.root_dir, split="test", transform=test_transform)
    valset = dataset_cls(
            basedir=args.root_dir, split="val", transform=test_transform)
    
    trainset.get_loader = types.MethodType(get_loader, trainset)
    testset.get_loader = types.MethodType(get_loader, testset)
    valset.get_loader = types.MethodType(get_loader, valset)

    trainset.group_str = types.MethodType(group_str, trainset)
    testset.group_str = types.MethodType(group_str, testset)
    valset.group_str = types.MethodType(group_str, valset)


    def group_counts(self):
        return self.group_counts_

    rename_group_counts(trainset)
    rename_group_counts(testset)
    rename_group_counts(valset)

    return trainset, valset, testset
