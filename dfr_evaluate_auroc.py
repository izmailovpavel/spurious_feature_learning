"""Evaluate DFR on spurious correlations datasets."""

import torch

import numpy as np
import os
import sys
import tqdm
import json
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import models
import utils
from utils import supervised_utils

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

# WaterBirds
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
REG = "l1"


def get_args():
    parser = utils.get_model_dataset_args()

    parser.add_argument(
        "--result_path", type=str, default="logs/",
        help="Path to save results")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument(
        "--save_embeddings", action='store_true',
        help="Save embeddings on disc")
    parser.add_argument(
        "--predict_spurious", action='store_true',
        help="Predict spurious attribute instead of class label")
    parser.add_argument(
        "--log_dir", type=str, default="", help="For loading wandb results")

    args = parser.parse_args()
    args.num_minority_groups_remove = 0
    args.reweight_groups = False
    args.reweight_spurious = False
    args.reweight_classes = False
    args.no_shuffle_train = True
    args.mixup = False
    return args


def get_results_cxr(acc_groups, get_ys_func):
    groups = acc_groups.keys()
    results = {
        f"accuracy_{get_ys_func(g)[0]}_{get_ys_func(g)[1]}": acc_groups[g].avg
        for g in groups
    }
    # We don't have 0_1 group in the data (healthy, with chest drain)
    # so the acc for this group will always be 0
    # and we need to eliminate it from computing worst group acc
    results.update({"worst_accuracy": min(results["accuracy_0_0"], results["accuracy_1_0"], results["accuracy_1_1"])})
    all_correct = sum([acc_groups[g].sum for g in groups])
    all_total = sum([acc_groups[g].count for g in groups])
    results.update({"mean_accuracy" : all_correct / all_total})
    return results


def compute_aucs(all_y, all_scores, all_g):
    masks = [((all_g == 0) | (all_g == 1) | (all_g == 2)), ((all_g == 0) | (all_g == 1) | (all_g == 3))]
    group_aucs = np.array(
        [roc_auc_score(all_y[mask], all_scores[mask]) for mask in masks])
    total_auc = roc_auc_score(all_y, all_scores)
    return group_aucs, total_auc


def eval_with_auc(model, test_loader_dict):
    model.eval()
    results_dict = {}
    aucs_dict = {}
    with torch.no_grad():
        # Currently test_loader_dict has "test" and "val"
        for test_name, test_loader in test_loader_dict.items():

            acc_groups = {g_idx: utils.AverageMeter() for g_idx in range(test_loader.dataset.n_groups)}
            all_y = []
            all_g = []
            all_scores = []

            for x, y, g, p in tqdm.tqdm(test_loader):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = model(x)
                utils.update_dict(acc_groups, y, g, logits)

                scores = logits[:, 1]
                all_y.append(y.detach().cpu().numpy())
                all_g.append(g.numpy())
                all_scores.append(scores.detach().cpu().numpy())

            results_dict[test_name] = acc_groups

            all_y = np.concatenate(all_y)
            all_scores = np.concatenate(all_scores)
            all_g = np.concatenate(all_g)
            group_aucs, total_auc = compute_aucs(all_y, all_scores, all_g)
            aucs_dict[test_name + "_group_aucs"] = group_aucs
            aucs_dict[test_name + "_total_auc"] = total_auc

    return results_dict, aucs_dict


def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True):
    # We are tuning hypers using worst group AUC

    worst_aucs = {}
    x_val = all_embeddings["val"]
    y_val = all_y["val"]
    g_val = all_g["val"]

    n_val = len(x_val) // 2
    idx = np.arange(len(x_val))
    np.random.shuffle(idx)

    x_train = x_val[idx[n_val:]]
    y_train = y_val[idx[n_val:]]
    g_train = g_val[idx[n_val:]]

    # We are not balancing groups for CXR

    # n_groups = np.max(g_train) + 1
    # g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
    # min_g = np.min([len(g) for g in g_idx]) + 10000
    # for g in g_idx:
    #     np.random.shuffle(g)
    # x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
    # y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx]).astype(int)
    # g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])

    x_val = x_val[idx[:n_val]]
    y_val = y_val[idx[:n_val]].astype(int)
    g_val = g_val[idx[:n_val]]

    # Uncomment this to only use "no drain" positive examples in training linear model
    # In practice it's been better to use all data

    # mask = (g_train == 0) | (g_train == 2)
    # x_train = x_train[mask]
    # y_train = y_train[mask]
    # g_train = g_train[mask]

    print("Val tuning:", np.bincount(g_train))
    if preprocess:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)

    for c in C_OPTIONS:
        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")#, class_weight={0: 1., 1: 1000})
        logreg.fit(x_train, y_train)
        preds_val = logreg.predict_proba(x_val)[:, 1]
        # The first mask is classifying negative against positive without drain,
        # the second mask is classifying negative against positive with drain
        masks = [((g_val == 0) | (g_val == 1) | (g_val == 2)), ((g_val == 0) | (g_val == 1) | (g_val == 3))]
        group_aucs = np.array(
            [roc_auc_score(y_val[mask], preds_val[mask]) for mask in masks])
        print(c, group_aucs)
        worst_auc = np.min(group_aucs)
        worst_aucs[c] = worst_auc

        masks = [(g_val == g) for g in [0, 2, 3]]
        preds_val = (preds_val > 0.5).astype(int)
        group_accs = np.array(
            [(y_val[mask] == preds_val[mask]).mean() for mask in masks])
        print(c, group_accs)
        print()

    ks, vs = list(worst_aucs.keys()), list(worst_aucs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        c, all_embeddings, all_y, all_g, num_retrains=1,
        preprocess=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["val"])
    for i in range(num_retrains):
        x_train = all_embeddings["val"]
        y_train = all_y["val"]
        g_train = all_g["val"]

        # We are not balancing groups for CXR
        # n_groups = np.max(g_val) + 1
        # g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
        # min_g = np.min([len(g) for g in g_idx]) + 10000
        # for g in g_idx:
        #     np.random.shuffle(g)
        # x_train = np.concatenate([x_val[g[:min_g]] for g in g_idx])
        # y_train = np.concatenate([y_val[g[:min_g]] for g in g_idx])
        # g_train = np.concatenate([g_val[g[:min_g]] for g in g_idx])

        # We could train the linear model only on positive data without drains
        # mask = (g_train == 0) | (g_train == 2)
        # x_train = x_train[mask]
        # y_train = y_train[mask]
        # g_train = g_train[mask]

        print(np.bincount(g_train), c)
        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")#, class_weight={0: 1., 1: 1000})
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)

    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    preds_test = logreg.predict_proba(x_test)[:, 1]
    # The first mask is classifying healthy against positive without drain,
    # the second mask is classifying healthy against positive with drain
    masks = [((g_test == 0) | (g_test == 1) | (g_test == 2)), ((g_test == 0) | (g_test == 1) | (g_test == 3))]
    group_aucs = np.array(
        [roc_auc_score(y_test[mask], preds_test[mask]) for mask in masks])
    total_auc = roc_auc_score(y_test, preds_test)

    masks = [(g_test == g) for g in [0, 2, 3]]
    preds_test = (preds_test > 0.5).astype(int)
    group_accs = np.array(
        [(y_test[mask] == preds_test[mask]).mean() for mask in masks])
    total_acc = (y_test == preds_test).mean()

    print(group_accs)
    print(group_aucs)

    return group_aucs, group_accs, total_auc, total_acc


def main(args):
    if has_wandb:
        wandb.init(project='ssl_robustness')
        args.__dict__.update(wandb.config)

        if len(args.log_dir) > 0:
            config = os.path.join(args.log_dir, "args.json")
            with open(config) as fd:
                model = json.load(fd)['model']

            args.model = model
            args.ckpt_path = os.path.join(args.log_dir, "final_checkpoint.pt")
            if not os.path.exists(args.ckpt_path):
                args.ckpt_path = os.path.join(args.log_dir, "tmp_checkpoint.pt")

            args.result_path = os.path.join(args.log_dir, "dfr_test.pkl")

    print(args)

    # Load data
    logger = utils.Logger() if not has_wandb else None
    train_loader, test_loader_dict, get_ys_func = (
        utils.get_data(args, logger, contrastive=False))
    n_classes = train_loader.dataset.n_classes

    # Model
    model_cls = getattr(models, args.model)
    model = model_cls(n_classes)
    if args.ckpt_path:
        print(f"Loading weights {args.ckpt_path}")
        try:
            model.load_state_dict(torch.load(args.ckpt_path))
        except:
            model.load_state_dict(torch.load(args.ckpt_path)["model"])
    else:
        print("Using initial weights")
    model.cuda()
    model.eval()

    # Evaluate model
    print("Base Model")
    # base_model_results = supervised_utils.eval(model, test_loader_dict)
    base_model_results, base_auc_dict = eval_with_auc(model, test_loader_dict)
    base_model_results = {
        name: get_results_cxr(accs, get_ys_func) for name, accs in base_model_results.items()}
    print(base_model_results)
    print()

    # Compute embeddings
    model.fc = torch.nn.Identity()
    splits = ["test", "val"]
    if os.path.exists(f"{args.result_path[:-4]}.npz"):
        arr_z = np.load(f"{args.result_path[:-4]}.npz")

        all_embeddings = {split: arr_z[f"embeddings_{split}"] for split in splits}
        all_y = {split: arr_z[f"y_{split}"] for split in splits}
        all_p = {split: arr_z[f"p_{split}"] for split in splits}
        all_g = {split: arr_z[f"g_{split}"] for split in splits}
    else:
        all_embeddings = {}
        all_y, all_p, all_g = {}, {}, {}
        for name, loader in [("test", test_loader_dict["test"]),
                             ("val", test_loader_dict["val"])]:
            all_embeddings[name] = []
            all_y[name], all_p[name], all_g[name] = [], [], []
            for x, y, g, p in tqdm.tqdm(loader):
                with torch.no_grad():
                    all_embeddings[name].append(model(x.cuda()).detach().cpu().numpy())
                    all_y[name].append(y.detach().cpu().numpy())
                    all_g[name].append(g.detach().cpu().numpy())
                    all_p[name].append(p.detach().cpu().numpy())
            all_embeddings[name] = np.vstack(all_embeddings[name])
            all_y[name] = np.concatenate(all_y[name])
            all_g[name] = np.concatenate(all_g[name])
            all_p[name] = np.concatenate(all_p[name])

        if args.save_embeddings:
            np.savez(f"{args.result_path[:-4]}.npz",
                     embeddings_test=all_embeddings["test"],
                     embeddings_val=all_embeddings["val"],
                     y_test=all_y["test"],
                     y_val=all_y["val"],
                     g_test=all_g["test"],
                     g_val=all_g["val"],
                     p_test=all_p["test"],
                     p_val=all_p["val"],
                     )

    # Add fake spurious labels to healthy group
    # in case of group balancing for training linear model,
    # however, best results so far were on the whole validation
    # mask = all_g["val"] == 0
    # all_g["val"][mask] = np.random.randint(0, 2, size=sum(mask))

    # DFR on validation
    print("DFR")
    dfr_results = {}
    c = dfr_on_validation_tune(all_embeddings, all_y, all_g)
    dfr_results["best_hypers"] = c
    print("Hypers:", (c))
    group_aucs, group_accs, total_auc, total_acc = dfr_on_validation_eval(
        c, all_embeddings, all_y, all_g)
    dfr_results["test_accs"] = group_accs
    dfr_results["test_worst_acc"] = np.min(group_accs)
    dfr_results["test_mean_acc"] = total_acc
    dfr_results["test_aucs"] = group_aucs
    dfr_results["test_worst_auc"] = np.min(group_aucs)
    dfr_results["test_mean_auc"] = total_auc
    print(dfr_results)
    print()

    all_results = {}
    all_results["base_model_results"] = base_model_results
    all_results["base_model_aucs"] = base_auc_dict
    all_results["dfr_val_results"] = dfr_results

    # if args.predict_spurious:
    #     print("Predicting spurious attribute")
    #     all_y = all_p
    #
    #     # DFR on validation
    #     print("DFR (spurious)")
    #     dfr_spurious_results = {}
    #     c = dfr_on_validation_tune(
    #         all_embeddings, all_y, all_g)
    #     dfr_spurious_results["best_hypers"] = c
    #     print("Hypers:", (c))
    #     test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
    #         c, all_embeddings, all_y, all_g)
    #     dfr_spurious_results["test_accs"] = test_accs
    #     dfr_spurious_results["train_accs"] = train_accs
    #     dfr_spurious_results["test_worst_acc"] = np.min(test_accs)
    #     dfr_spurious_results["test_mean_acc"] = test_mean_acc
    #     print(dfr_spurious_results)
    #     print()
    #
    #     all_results["dfr_val_spurious_results"] = dfr_spurious_results

    print(all_results)

    command = " ".join(sys.argv)
    all_results["command"] = command
    all_results["model"] = args.model
    with open(args.result_path, 'wb') as f:
        pickle.dump(all_results, f)

    if has_wandb:
        wandb.log(all_results)


if __name__ == '__main__':
    args = get_args()
    main(args)