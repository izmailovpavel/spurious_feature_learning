import torch
import transformers


def sgd_optimizer_fromparams(params, lr, momentum, weight_decay):
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=momentum,
        weight_decay=weight_decay)
    return optimizer



def sgd_optimizer(model, lr, momentum, weight_decay):
    return sgd_optimizer_fromparams(
        model.parameters(), lr, momentum, weight_decay)


def adamw_optimizer(model, lr, momentum, weight_decay):
    del momentum
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=weight_decay)
    return optimizer


def bert_adamw_optimizer(model, lr, momentum, weight_decay):
    # Adapted from https://github.com/facebookresearch/BalancingGroups/blob/main/models.py
    del momentum
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = []
    nodecay_params = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in no_decay):
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]
    optimizer = transformers.AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=1e-8)
    return optimizer


def cosine_lr_scheduler(optimizer, num_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)


def constant_lr_scheduler(optimizer, num_steps):
    return None


def bert_lr_scheduler(optimizer, num_steps):
    return transformers.get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=num_steps)
