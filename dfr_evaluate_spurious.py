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
        "--drop_group", type=int, default=None, required=False,
        help="Drop group from evaluation")
    parser.add_argument(
        "--log_dir", type=str, default="", help="For loading wandb results")
    parser.add_argument(
        "--save_linear_model", action='store_true', help="Save linear model weights")
    parser.add_argument(
        "--save_best_epoch", action='store_true', help="Save best epoch num to pkl")
    # DFR TR
    parser.add_argument(
        "--dfr_train", action='store_true', help="Use train data for reweighting")

    args = parser.parse_args()
    args.num_minority_groups_remove = 0
    args.reweight_groups = False
    args.reweight_spurious = False
    args.reweight_classes = False
    args.no_shuffle_train = True
    args.mixup = False
    args.load_from_checkpoint = True
    return args


def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True, num_retrains=1):

    worst_accs = {}
    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1

        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_train = x_val[idx[n_val:]]
        y_train = y_val[idx[n_val:]]
        g_train = g_val[idx[n_val:]]

        n_groups = np.max(g_train) + 1
        g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
        y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
        g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        print("Val tuning:", np.bincount(g_train))
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
            logreg.fit(x_train, y_train)
            preds_val = logreg.predict(x_val)
            group_accs = np.array(
                [(preds_val == y_val)[g_val == g].mean()
                 for g in range(n_groups)])
            worst_acc = np.min(group_accs)
            if i == 0:
                worst_accs[c] = worst_acc
            else:
                worst_accs[c] += worst_acc
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        args, c, all_embeddings, all_y, all_g, target_type="target", num_retrains=20,
        preprocess=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["val"])
    for i in range(num_retrains):
        for _ in range(20):
            x_val = all_embeddings["val"]
            y_val = all_y["val"]
            g_val = all_g["val"]
            n_groups = np.max(g_val) + 1
            g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
            min_g = np.min([len(g) for g in g_idx])
            for g in g_idx:
                np.random.shuffle(g)
            x_train = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_train = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_train = np.concatenate([g_val[g[:min_g]] for g in g_idx])

            if np.any(np.unique(y_train) != np.unique(all_y["val"])):
                # do we need the same thing in tuning?
                print("missing classes, reshuffling...")
                continue
            else:
                break

        print(np.bincount(g_train))
        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
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

    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]

    if args.save_linear_model:
        linear_model = {
            'coef': logreg.coef_,
            'intercept': logreg.intercept_,
            'scaler': scaler
        }
        dir_linear_model = os.path.join(os.path.dirname(args.result_path), 'dfr_linear_models')
        if not os.path.isdir(dir_linear_model):
            os.makedirs(dir_linear_model)
        linear_model_path = os.path.join(dir_linear_model,
                                         os.path.basename(args.result_path)[:-4] + f'_linear_model_{target_type}.pkl')
        with open(linear_model_path, 'wb') as f:
            pickle.dump(linear_model, f)

    return test_accs, test_mean_acc, train_accs


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
    if args.ckpt_path and args.load_from_checkpoint:
        print(f"Loading weights {args.ckpt_path}")
        ckpt_dict = torch.load(args.ckpt_path)
        try:
            model.load_state_dict(ckpt_dict)
        except:
            print("Loading one-output Checkpoint")
            w = ckpt_dict["fc.weight"]
            w_ = torch.zeros((2, w.shape[1]))
            w_[1, :] = w
            b = ckpt_dict["fc.bias"]
            b_ = torch.zeros((2,))
            b_[1] = b
            ckpt_dict["fc.weight"] = w_
            ckpt_dict["fc.bias"] = b_
            model.load_state_dict(ckpt_dict)
    else:
        print("Using initial weights")
    model.cuda()
    model.eval()

    # Evaluate model
    print("Base Model")
    base_model_results = supervised_utils.eval(model, test_loader_dict)
    base_model_results = {
        name: utils.get_results(accs, get_ys_func) for name, accs in base_model_results.items()}
    print(base_model_results)
    print()
    
    model.fc = torch.nn.Identity()
    #splits = ["test", "val"]
    splits = {
        "test": test_loader_dict["test"],
        "val": test_loader_dict["val"]
    }
    if args.dfr_train:
        splits["train"] = train_loader
    print(splits.keys())
    if os.path.exists(f"{args.result_path[:-4]}.npz"):
        arr_z = np.load(f"{args.result_path[:-4]}.npz")

        all_embeddings = {split: arr_z[f"embeddings_{split}"] for split in splits}
        all_y = {split: arr_z[f"y_{split}"] for split in splits}
        all_p = {split: arr_z[f"p_{split}"] for split in splits}
        all_g = {split: arr_z[f"g_{split}"] for split in splits}
    else:
        all_embeddings = {}
        all_y, all_p, all_g = {}, {}, {}
        for name, loader in splits.items():
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

    if args.drop_group is not None:
        print("Dropping group", args.drop_group)
        all_masks = {name: all_g[name] != args.drop_group for name in splits}
        for name in splits:
            all_y[name] = all_y[name][all_masks[name]]
            all_g[name] = all_g[name][all_masks[name]]
            all_p[name] = all_p[name][all_masks[name]]
            all_embeddings[name] = all_embeddings[name][all_masks[name]]
    
    if args.dfr_train:
        print("Reweighting on training data")
        all_y["val"] = all_y["train"]
        all_g["val"] = all_g["train"]
        all_p["val"] = all_p["train"]
        all_embeddings["val"] = all_embeddings["train"]



    # DFR on validation
    print("DFR")
    dfr_results = {}
    c = dfr_on_validation_tune(
        all_embeddings, all_y, all_g)
    dfr_results["best_hypers"] = c
    print("Hypers:", (c))
    test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
        args, c, all_embeddings, all_y, all_g, target_type="target")
    dfr_results["test_accs"] = test_accs
    dfr_results["train_accs"] = train_accs
    dfr_results["test_worst_acc"] = np.min(test_accs)
    dfr_results["test_mean_acc"] = test_mean_acc
    print(dfr_results)
    print()

    all_results = {}
    all_results["base_model_results"] = base_model_results
    all_results["dfr_val_results"] = dfr_results

    if args.predict_spurious:
        print("Predicting spurious attribute")
        all_y = all_p

        # DFR on validation
        print("DFR (spurious)")
        dfr_spurious_results = {}
        c = dfr_on_validation_tune(
            all_embeddings, all_y, all_g)
        dfr_spurious_results["best_hypers"] = c
        print("Hypers:", (c))
        test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
            args, c, all_embeddings, all_y, all_g, target_type="spurious")
        dfr_spurious_results["test_accs"] = test_accs
        dfr_spurious_results["train_accs"] = train_accs
        dfr_spurious_results["test_worst_acc"] = np.min(test_accs)
        dfr_spurious_results["test_mean_acc"] = test_mean_acc
        print(dfr_spurious_results)
        print()

        all_results["dfr_val_spurious_results"] = dfr_spurious_results
    
    print(all_results)


    command = " ".join(sys.argv)
    all_results["command"] = command
    all_results["model"] = args.model

    if args.ckpt_path:
        if os.path.exists(os.path.join(os.path.dirname(args.ckpt_path), 'args.json')):
            base_model_args_file = os.path.join(os.path.dirname(args.ckpt_path), 'args.json')
            with open(base_model_args_file) as fargs:
                base_model_args = json.load(fargs)
                all_results["base_args"] = base_model_args
        if args.save_best_epoch:
            if os.path.exists(os.path.join(os.path.dirname(args.ckpt_path), 'best_epoch_num.npy')):
                base_epoch_file = os.path.join(os.path.dirname(args.ckpt_path), 'best_epoch_num.npy')
                best_epoch_num = np.load(base_epoch_file)[0]
                all_results["base_model_best_epoch"] = best_epoch_num

    with open(args.result_path, 'wb') as f:
        pickle.dump(all_results, f)

    if has_wandb:
        wandb.log(all_results)


if __name__ == '__main__':
    args = get_args()
    main(args)
