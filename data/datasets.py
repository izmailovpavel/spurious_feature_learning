import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

try:
    import wilds
    from wilds.datasets.wilds_dataset import WILDSSubset
    has_wilds = True
except:
    has_wilds = False

def _get_split(split):
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise(f"Unknown split {split}")

def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError


class SpuriousCorrelationDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)
        
        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        if "spurious" in self.metadata_df:
            self.spurious_array = self.metadata_df["spurious"].values
        else:
            self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        if "group" in self.metadata_df:
            self.group_array = self.metadata_df["group"].values
        else:
            self._get_class_spurious_groups()
        self._count_groups()
        self.text = not "img_filename" in self.metadata_df
        if self.text:
            print("NLP dataset")
            self.text_array = list(pd.read_csv(os.path.join(
                basedir, "text.csv"))["text"])
        else:
            self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(
            self.spurious_array)

    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        # self.n_groups = np.unique(self.group_array).size
        self.n_groups = len(self.group_counts)

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(
            self.y_array * self.n_spurious + self.spurious_array)
        
    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        if self.text:
            x = self._text_getitem(idx)
        else:
            x = self._image_getitem(idx)
        return x, y, g, s

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _text_getitem(self, idx):
        text = self.text_array[idx]
        if self.transform:
            text = self.transform(text)
        return text


class MultiNLIDataset(SpuriousCorrelationDataset):
    """Adapted from https://github.com/kohpangwei/group_DRO/blob/master/data/multinli_dataset.py
    """
    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"
        
        # utils_glue module in basedir is needed to load data
        import sys
        sys.path.append(basedir)


        self.basedir = basedir
        self.metadata_df = pd.read_csv(os.path.join(
            self.basedir, "metadata_random.csv"))
        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli",  
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm"]
        features_array = sum([torch.load(os.path.join(self.basedir, name)) 
                              for name in bert_filenames], start=[])
        all_input_ids = torch.tensor([
            f.input_ids for f in features_array]).long()
        all_input_masks = torch.tensor([
            f.input_mask for f in features_array]).long()
        all_segment_ids = torch.tensor([
            f.segment_ids for f in features_array]).long()
        # all_label_ids = torch.tensor([
        #     f.label_id for f in self.features_array]).long()
        
        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values

        self.x_array = torch.stack((
            all_input_ids,
            all_input_masks,
            all_segment_ids), dim=2)[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df['gold_label'].values
        self.spurious_array = (
            self.metadata_df['sentence2_has_negation'].values)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        x = self.x_array[idx]
        return x, y, g, s


class DeBERTaMultiNLIDataset(MultiNLIDataset):
    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        self.basedir = basedir
        self.metadata_df = pd.read_csv(os.path.join(
            self.basedir, "metadata_random.csv"))
        self.basedir = basedir
        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values
        self.x_array = torch.load(os.path.join(
                self.basedir, "cached_deberta-base_220_mnli"))[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df['gold_label'].values
        self.spurious_array = (
            self.metadata_df['sentence2_has_negation'].values)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()


class BERTMultilingualMultiNLIDataset(MultiNLIDataset):
    def __init__(self, basedir, split="train", transform=None):
        assert transform is None, "transfrom should be None"

        self.basedir = basedir
        self.metadata_df = pd.read_csv(os.path.join(
            self.basedir, "metadata_random.csv"))
        self.basedir = basedir
        split_i = _get_split(split)
        split_mask = (self.metadata_df["split"] == split_i).values
        self.x_array = torch.load(os.path.join(
                self.basedir, "cached_bert-base-multilingual_150_mnli"))[split_mask]
        self.metadata_df = self.metadata_df[split_mask]
        self.y_array = self.metadata_df['gold_label'].values
        self.spurious_array = (
            self.metadata_df['sentence2_has_negation'].values)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()


class BaseWildsDataset(SpuriousCorrelationDataset):
    def __init__(
        self, ds_name, basedir, split, transform, y_name, spurious_name
    ):
        assert has_wilds, "wilds package not found"
        self.basedir = basedir
        self.root_dir = "/".join(self.basedir.split("/")[:-2])
        base_dataset = wilds.get_dataset(
            dataset=ds_name, download=False, root_dir=self.root_dir)
        self.dataset = base_dataset.get_subset(split, transform=transform)

        column_names = self.dataset.metadata_fields
        if y_name:
            y_idx = column_names.index(y_name)
            self.y_array = self.dataset.metadata_array[:, y_idx]
        if spurious_name:
            s_idx = column_names.index(spurious_name)
            self.spurious_idx = s_idx
            self.spurious_array = self.dataset.metadata_array[:, s_idx]
        if y_name and spurious_name:
            self._count_attributes()

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = metadata[self.spurious_idx]
        return x, y, s, s

    def __len__(self):
        return len(self.dataset)


class WildsFMOW(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("fmow", basedir, split, transform, "y", "region")
        self.group_array = self.spurious_array
        self._count_groups()


class WildsPoverty(BaseWildsDataset):
    # TODO(izmailovpavel): test and implement regression training
    def __init__(self, basedir, split="train", transform=None):
        # assert transform is None, "transfrom should be None"
        super().__init__("poverty", basedir, split, transform, "y",
            "urban")
        self.n_classes = None
        self.group_array = self.spurious_array
        self._count_groups()


class WildsCivilCommentsCoarse(BaseWildsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__("civilcomments", basedir, split, transform, "y", None)
        attributes = ["male", "female", "LGBTQ", "black", "white", "christian",
                      "muslim", "other_religions"]
        column_names = self.dataset.metadata_fields
        self.spurious_cols = [column_names.index(a) for a in attributes]
        self.spurious_array = self.get_spurious(self.dataset.metadata_array)
        self._count_attributes()
        self._get_class_spurious_groups()
        self._count_groups()

    def get_spurious(self, metadata):
        if len(metadata.shape) == 1:
            return metadata[self.spurious_cols].sum(-1).clip(max=1)
        else:
            return metadata[:, self.spurious_cols].sum(-1).clip(max=1)

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        s = self.get_spurious(metadata)
        g = y * self.n_spurious + s
        return x, y, g, s


class WildsCivilCommentsCoarseNM(WildsCivilCommentsCoarse):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)
        if split == "train":
            identities_mentioned = self.spurious_array > 0
            toxic = self.y_array == 1
            mask = (identities_mentioned & toxic) | (~identities_mentioned & ~toxic)
            train_idx = self.dataset.indices.copy()[mask]
            self.dataset = WILDSSubset(
                    self.dataset.dataset,
                    indices=train_idx,
                    transform=self.dataset.transform
            )
            self.spurious_array = self.get_spurious(self.dataset.metadata_array)
            self.y_array = self.y_array[mask]
            self._count_attributes()
            self._get_class_spurious_groups()
            self._count_groups()


class FakeSpuriousCIFAR10(SpuriousCorrelationDataset):
    """CIFAR10 with SpuriousCorrelationDataset API.

    Groups are the same as classes.
    """
    def __init__(self, basedir, split, transform=None, val_size=5000):
        split_i = _get_split(split)
        self.ds = CIFAR10(
            root=basedir, train=(split_i != 2),
            download=True, transform=transform)
        if split_i == 0:
            self.ds.data = self.ds.data[:-val_size]
            self.ds.targets = self.ds.targets[:-val_size]
        elif split_i == 1:
            self.ds.data = self.ds.data[-val_size:]
            self.ds.targets = self.ds.targets[-val_size:]

        self.y_array = np.array(self.ds.targets)
        self.n_classes = 10
        self.spurious_array = np.zeros_like(self.y_array)
        self.n_spurious = 1
        self.group_array = self.y_array

        self.n_groups = 10
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(
            self.spurious_array)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        return x, y, y, 0


def remove_minority_groups(trainset, num_remove):
    if num_remove == 0:
        return
    print("Removing minority groups")
    print("Initial groups", np.bincount(trainset.group_array))
    num_groups = np.bincount(trainset.group_array).size
    group_counts = trainset.group_counts
    minority_groups = np.argsort(group_counts.numpy())[:num_remove]
    idx = np.where(np.logical_and.reduce(
        [trainset.group_array != g for g in minority_groups], initial=True))[0]
    trainset.x_array = trainset.x_array[idx]
    trainset.y_array = trainset.y_array[idx]
    trainset.group_array = trainset.group_array[idx]
    trainset.spurious_array = trainset.spurious_array[idx]
    if hasattr(trainset, 'filename_array'):
        trainset.filename_array = trainset.filename_array[idx]
    trainset.metadata_df = trainset.metadata_df.iloc[idx]
    trainset.group_counts = torch.from_numpy(
            np.bincount(trainset.group_array, minlength=num_groups))
    print("Final groups", np.bincount(trainset.group_array))


def balance_groups(ds):
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0]
        for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[:min_group] for idx in group_idx]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)
