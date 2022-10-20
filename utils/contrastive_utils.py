"""SimCLR implementation.

Adapted from https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/simclr_lin.py
"""
import tqdm
import torch
import torch.nn.functional as F

import utils


def train_epoch(model, loader, optimizer, temp):
    # TODO(izmailovpavel): add temperature
    n_groups = loader.dataset.n_groups
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    loss_meter = utils.AverageMeter()
    acc_groups = {
        g_idx: utils.AverageMeter() for g_idx in range(n_groups)}

    for batch in tqdm.tqdm(loader):
        xs, _, g, _ = batch
        n_aug, n = len(xs), len(xs[0])
        assert n_aug == 2
        x = torch.cat(xs, axis=0)
        x = x.cuda()

        optimizer.zero_grad()
        z = model(x.cuda())
        z = F.normalize(z, dim=1)
        sim_mat = z @ z.T / temp
        
        ids = torch.cat([torch.arange(n) for i in range(n_aug)])
        positives_mask = ids[:, None] == ids[None, :]

        negatives = sim_mat[~positives_mask].reshape(
            (n * n_aug, (n - 1) * n_aug))

        positives_mask_noid = (
            positives_mask.float() - torch.eye(n * n_aug)).bool()
        positives = sim_mat[positives_mask_noid].reshape((n * n_aug, n_aug - 1))
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros((n * n_aug,)).long().cuda()
        groups = torch.cat([g for _ in range(n_aug)], dim=0)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss, x.size(0))
        utils.update_dict(acc_groups, labels, groups, logits)
    example_images = x[[0, n, 1, n+1]]
    return loss_meter, acc_groups, example_images


def train_contrastive_supervised_epoch(model, loader, optimizer, temp):
    # TODO(izmailovpavel): add temperature
    n_groups = loader.dataset.n_groups
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    loss_meter = utils.AverageMeter()
    acc_groups = {
        g_idx: utils.AverageMeter() for g_idx in range(n_groups)}
    acc_groups_sup = {
        g_idx: utils.AverageMeter() for g_idx in range(n_groups)}

    for batch in tqdm.tqdm(loader):
        xs, y, g, _ = batch
        n_aug, n = len(xs), len(xs[0])
        assert n_aug == 2
        x = torch.cat(xs, axis=0)
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        e = model.embed(x.cuda())
        z = model.proj_head(e)
        z = F.normalize(z, dim=1)
        sim_mat = z @ z.T / temp
        
        ids = torch.cat([torch.arange(n) for i in range(n_aug)])
        positives_mask = ids[:, None] == ids[None, :]

        negatives = sim_mat[~positives_mask].reshape(
            (n * n_aug, (n - 1) * n_aug))

        positives_mask_noid = (
            positives_mask.float() - torch.eye(n * n_aug)).bool()
        positives = sim_mat[positives_mask_noid].reshape((n * n_aug, n_aug - 1))
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros((n * n_aug,)).long().cuda()
        groups = torch.cat([g for _ in range(n_aug)], dim=0)

        loss = criterion(logits, labels)

        # supervised
        logits_sup = model.classifier(e)
        labels_sup = torch.cat([y for _ in range(n_aug)], dim=0)
        loss += criterion(logits_sup, labels_sup)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss, x.size(0))
        utils.update_dict(acc_groups, labels, groups, logits)
        utils.update_dict(acc_groups_sup, labels_sup, groups, logits_sup)
    example_images = x[[0, n, 1, n+1]]
    return loss_meter, acc_groups, acc_groups_sup, example_images


def eval(
        model, finetune_loader, test_loader_dict, get_ys_func,
        finetune_epochs=10
    ):
    model.eval()
    results_dict = {}
    ds = finetune_loader.dataset
    n_classes, n_groups = ds.n_classes, ds.n_groups

    original_fc = model.fc
    model.fc = torch.nn.Identity()
    classifier = torch.nn.Linear(
        original_fc.in_features, n_classes).cuda()
    
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        finetune_loader.batch_size * 0.1 / 256,
        momentum=0.9,
        weight_decay=0.,
        nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_epochs*len(finetune_loader))
    
    print("Linear probing")
    for epoch in range(finetune_epochs):
        loss_meter = utils.AverageMeter()
        acc_meter = utils.AverageMeter()
        acc_groups = {
            g_idx: utils.AverageMeter() for g_idx in range(n_groups)}

        bar = tqdm.tqdm(finetune_loader)
        for batch in bar:
            x, y, g, _ = batch
            x, y, g = x.cuda(), y.cuda(), g.cuda()
            with torch.no_grad():
                embedding = model(x).view(x.size(0), -1)
            logits = classifier(embedding)
            loss = F.cross_entropy(logits, y)

            acc = (logits.argmax(dim=1) == y).float().mean()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))
            utils.update_dict(acc_groups, y, g, logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            bar.set_description(
                "Finetune epoch {}, loss: {:.4f}, acc: {:.4f}".format(
                    epoch, loss_meter.avg, acc_meter.avg))

    train_results = utils.get_results(acc_groups, get_ys_func)
    results_dict["train_loss"] = loss_meter.avg
    results_dict.update({
        f"train_{key}": value for key, value in train_results.items()})

    with torch.no_grad():
        for name, loader in test_loader_dict.items():
            loss_meter = utils.AverageMeter()
            acc_meter = utils.AverageMeter()
            acc_groups = {
                g_idx: utils.AverageMeter() for g_idx in range(n_groups)}
            bar = tqdm.tqdm(loader)
            for batch in bar:
                x, y, g, _ = batch
                x, y, g = x.cuda(), y.cuda(), g.cuda()
                logits = classifier(model(x).view(x.size(0), -1))
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                loss_meter.update(loss.item(), x.size(0))
                acc_meter.update(acc.item(), x.size(0))
                utils.update_dict(acc_groups, y, g, logits)
                bar.set_description(
                    "Eval {} loss: {:.4f}, acc: {:.4f}".format(
                        name, loss_meter.avg, acc_meter.avg))
            results_dict[f"{name}_loss"] = loss_meter.avg
            eval_results = utils.get_results(acc_groups, get_ys_func)
            results_dict.update({
                f"{name}_{key}": value for key, value in eval_results.items()})

    model.fc = original_fc

    return results_dict