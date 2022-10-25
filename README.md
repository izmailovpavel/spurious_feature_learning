# On Feature Learning in the Presence of Spurious Correlations

This repository contains experiments for the NeurIPS 2022 paper _On Feature Learning in the Presence of Spurious Correlations_ by [Pavel Izmailov](https://izmailovpavel.github.io/), [Polina Kirichenko](https://polkirichenko.github.io/), [Nate Gruver](https://ngruver.github.io/) and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

## Introduction

Deep classifiers are known to rely on spurious features — patterns which are correlated with the target on the training data but not inherently relevant to the learning problem, such as the image backgrounds when classifying the foregrounds.
In this paper we evaluate the amount of information about the core (non-spurious) features that can be decoded from the representations learned by standard empirical risk minimization (ERM) and specialized group robustness training. 
Following recent work on Deep Feature Reweighting (DFR), we evaluate the feature representations by re-training the last layer of the model on a held-out set where the spurious correlation is broken.
On multiple vision and NLP problems, we show that the features learned by simple ERM are highly competitive with the features learned by specialized group robustness methods targeted at reducing the effect of spurious correlations.
Moreover, we show that the quality of learned feature representations is greatly affected by the design decisions beyond the training method, such as the model architecture and pre-training strategy.
On the other hand, we find that strong regularization is not necessary for learning high quality feature representations.
Finally, using insights from our analysis, we significantly improve upon the best results reported in the literature on the popular Waterbirds, CelebA hair color prediction and WILDS-FMOW problems, achieving 97\%, 92\% and 50\% worst-group accuracies, respectively.

![image](https://user-images.githubusercontent.com/14368801/196981506-e25fbc3f-8d56-4bde-92ef-3456eae9f300.png)

Please cite our paper if you find it helpful in your work:

```bibtex
@article{izmailov2022feature,
  title={On Feature Learning in the Presence of Spurious Correlations},
  author={Izmailov, Pavel and Kirichenko, Polina and Gruver, Nate and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2210.11369},
  year={2022}
}
```

## File Structure

```
.
+-- data/
|   +-- __init__.py
|   +-- augmix_transforms.py (AugMix augmentation)
|   +-- data_transforms.py (Preprocessing and data augmentation)
|   +-- dataloaders.py (RWY and RWG samplers and MixUp)
|   +-- datasets.py (Dataset classes)
+-- dataset_files/utils_glue.py (File to copy into the MultiNLI dataset directory)
+-- group_DRO/ (Group DRO codebase)
+-- models/
|   +-- __init__.py (Most of the vision models)
|   +-- preresnet.py (Preactivation ResNet; not used in the experiments)
|   +-- text_models.py (BERT classifier model)
|   +-- vissl_models.py (Contrastive models from vissl)
+-- optimizers/
|   +-- __init__.py (SGD, AdamW and LR schedulers)
+-- utils/
|   +-- __init__.py
|   +-- common_utils.py (Common utilities used in different scripts)
|   +-- logging_utils.py (Logging-related utilities)
|   +-- supervised_utils.py (Utilities for supervised training)
+-- train_supervised.py (Train base models)
+-- dfr_evaluate_spurious.py (Tune and evaluate DFR for a given base model)
+-- dfr_evaluate_auroc.py (Tune and evaluate DFR on the CXR dataset)
```

## Requirements

- [`torch`](https://pytorch.org/get-started/locally/)
- [`torchvision`](https://pytorch.org/get-started/locally/)
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`transformers`](https://huggingface.co/docs/transformers/installation)
- [`vissl`](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md)
- [`scikit-learn`](https://scikit-learn.org/stable/install.html)
- [`numpy`](https://numpy.org/install/)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`wilds`](https://wilds.stanford.edu/get_started/)


## Data access

### Waterbirds and CelebA

Please follow the instructions in the [DFR repo](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access) to prepare the Waterbirds and CelebA datasets.

### Civil Comments and MultiNLI

The Civil Comments dataset should be downloaded automatically when you run experiments, no manual preparation needed.

To run experiments on the MultiNLI dataset, please manually download and unzip the dataset from [this link](https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz).
Further, please copy the `dataset_files/utils_glue.py` to the root directory of the dataset.

### WILDS-FMOW

To run experiments on the FMOW dataset, you first need to run `wilds.get_dataset(dataset="fmow", download=False, root_dir=<ROOT DIR>)` from python console or in a jupyter notebook.

### CXR

The chest drain labels for the CXR dataset are not publically available, so we cannot share the code for preparing this dataset.


## Example commands

Waterbirds:
```bash
python3 train_supervised.py --output_dir=logs/waterbirds/erm_seed1 \
	--num_epochs=100 --eval_freq=1 --save_freq=100 --seed=1 \
	--weight_decay=1e-4 --batch_size=32 --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained


python3 dfr_evaluate_spurious.py --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained \
	--ckpt_path=logs/waterbirds/erm_seed1/final_checkpoint.pt \
	--result_path=wb_erm_seed1_dfr.pkl --predict_spurious
```

CelebA:
```bash
python3 train_supervised.py --output_dir=logs/celeba/erm_seed1 \
	--num_epochs=20 --eval_freq=1 --save_freq=100 --seed=1 \
	--weight_decay=1e-4 --batch_size=100 --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained

python3 dfr_evaluate_spurious.py --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset --model=imagenet_resnet50_pretrained \
	--ckpt_path=logs/celeba/erm_seed1/final_checkpoint.pt \
	--result_path=celeba_erm_seed1_dfr.pkl --predict_spurious
```

FMOW
```bash
python3 train_supervised.py --output_dir=logs/fmow/erm_seed1 \
	--num_epochs=20 --eval_freq=5 --save_freq=100 --seed=1 \
	--weight_decay=1e-4 --batch_size=100 --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=WildsFMOW --model=imagenet_resnet50_pretrained

python3 dfr_evaluate_spurious.py --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=WildsFMOW  --model=imagenet_resnet50_pretrained \
	--ckpt_path=logs/fmow/erm_seed1/final_checkpoint.pt \
	--result_path=fmow_erm_seed1_dfr.pkl --predict_spurious
```

CXR
```bash
python3 train_supervised.py --output_dir=logs/cxr/erm_seed1 \
	--num_epochs=20 --eval_freq=5 --save_freq=100 --seed=1 \
	--weight_decay=1e-4 --batch_size=100 --init_lr=3e-3 \
	--scheduler=cosine_lr_scheduler --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=SpuriousCorrelationDataset \
	--model=imagenet_densenet121_pretrained

python3 dfr_evaluate_auroc.py --data_dir=<DATA_DIR> \
	--data_transform=AugWaterbirdsCelebATransform \
	--dataset=WildsFMOW  --model=imagenet_resnet50_pretrained \
	--ckpt_path=logs/cxr/erm_seed1/final_checkpoint.pt \
	--result_path=cxr_erm_seed1_dfr.pkl
```


Civil Comments
```bash
python3 train_supervised.py --output_dir=logs/civilcomments/erm_seed1 \
	--num_epochs=10 --eval_freq=1 --save_freq=10 --seed=1 \
	--weight_decay=1.e-4 --batch_size=16 --init_lr=1e-5 \
	--scheduler=bert_lr_scheduler --data_dir=<DATA_DIR> \
	--data_transform=BertTokenizeTransform \
	--dataset=WildsCivilCommentsCoarse --model=bert_pretrained \
	--optimizer=bert_adamw_optimizer

python3 dfr_evaluate_spurious.py --data_dir=<DATA_DIR> \
	--data_transform=BertTokenizeTransform \
	--dataset=WildsCivilCommentsCoarse --model=bert_pretrained \
	--ckpt_path=logs/civilcomments/methods/erm_seed1/final_checkpoint.pt \
	--result_path=civilcomments_methods_erm_seed1_dfr.pkl --predict_spurious
```

MultiNLI
```bash
python3 train_supervised.py --output_dir=logs/multinli/erm_seed1/ \
	--num_epochs=10 --eval_freq=1 --save_freq=10 --seed=1 \
	--weight_decay=1.e-4 --batch_size=16 --init_lr=1e-5 \
	--scheduler=bert_lr_scheduler --data_dir=<DATA_DIR> \
	--data_transform=None --dataset=MultiNLIDataset --model=bert_pretrained \
	--optimizer=bert_adamw_optimizer

python3 dfr_evaluate_spurious.py --data_dir=<DATA_DIR> \
	--data_transform=None --dataset=MultiNLIDataset --model=bert_pretrained \
	--ckpt_path=logs/multinli/erm_seed1/final_checkpoint.pt \
	--result_path=multinli_erm_seed1_dfr.pkl --predict_spurious

```

`<DATA_DIR>` should be the root directory of the dataset, e.g. `/datasets/fmow_v1.1/`. We provide example `--output_dir` and `--result_path` argument values, but you can change them to your convenience;
these arguments define the location of the logs and checkpoints, and the dfr results file respectively.

### Other architectures and augmentations

You can specify the base model with the `--model` flag.
For example: 
- ResNet-50 pretrained on ImageNet: `--model=imagenet_resnet50_pretrained`
- ResNet-50 initialized randomly: `--model=imagenet_resnet50`
- ConvNext XLarge pretrained on ImageNet22k `--model=imagenet_convnext_xlarge_in22k_pretrained`
- DenseNet-121 pretrained on ImageNet: `--model=imagenet_densenet121_pretrained`

Note that for some of the models (some ConvNext models, MAE, DINO), you need to manually download the checkpoints and put them in the `/ckpts/` directory, or, alternatively, change the paths in the `models/__init__.py` file.

You can specify the data augmentation policy with the `--data_transform` flag:
- No augmentation: `--data_transform=NoAugWaterbirdsCelebATransform`
- Default augmentation: `--data_transform=AugWaterbirdsCelebATransform`
- Random Erase augmentation: `--data_transform=ImageNetRandomErasingTransform`
- AugMix augmentation: `--data_transform=ImageNetAugmixTransform`

You can apply MixUp with any augmentation policy by using the `--mixup` flag.

For a full list of models and augmentations available, run:
```bash
python3 train_supervised.py -h
```

### RWY and RWG

To run the RWY method, add `--reweight_classes`;
to run the RWG method, add `--reweight_groups`.
You can then evaluate the saved model weights with DFR analogously to the commands for ERM training above.

### Group-DRO

We modified the [group-DRO code](https://github.com/PolinaKirichenko/deep_feature_reweighting) to be compatible with our model and dataset implementations.
You can run experiments with group-DRO with the standard group-DRO comands, as described in the [group-DRO codebase](https://github.com/PolinaKirichenko/deep_feature_reweighting), but adding the `--dfr_data --dfr_model` flags.
With these flags, you can use all the models and datasets implemented in our repo.
For example, to run on the FMOW dataset, you can use the following commang:

```bash
python3 run_expt.py -s confounder -d FMOW --model imagenet_resnet50_pretrained \
	-t target -c confounder \  # the values of these flags are irrelevant 
	--root_dir <DATA_DIR> --robust --save_best --save_last --save_step 200 \
	--batch_size 100 --n_epochs 20 --gamma 0.1 --augment_data --lr 0.001 \
	--weight_decay 0.001 --generalization_adjustment 0 --seed 1 \
	--log_dir logs/fmow/gdro_seed1 --dfr_data --dfr_model
```

This command should be run from the `group_DRO` folder.
You can then evaluate the saved model weights with DFR analogously to the commands for ERM training above.

## Code References

- We used the [DFR codebase](https://github.com/PolinaKirichenko/deep_feature_reweighting) as the basis for our code.
The DFR codebase in turn is based on the [group-DRO codebase](https://github.com/PolinaKirichenko/deep_feature_reweighting).

- We also include a modified version of the Group-DRO code in the `group_DRO` folder;
this folder is ported from the [group-DRO codebase](https://github.com/PolinaKirichenko/deep_feature_reweighting) with minor modifications.

- Our model implementations are based on the `torchvision`, `timm` and `transformers` packages.

- Our implementation of AugMix is ported from [the AugMix repo](https://github.com/google-research/augmix/blob/master/augmentations.py).
