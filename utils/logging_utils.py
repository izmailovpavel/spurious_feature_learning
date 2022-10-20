import os
import torch
import numpy as np
import tqdm
import sys
import json
import torchvision
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

import utils


class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def write_dict_to_tb(writer, dict, prefix, step):
    if prefix[-1] != "/":
        prefix += "/"
    for key, value in dict.items():
        writer.add_scalar(f"{prefix}{key}", value, step)


def prepare_logging(output_dir, args):
    print('Preparing directory %s' % output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        args_json = json.dumps(vars(args))
        f.write(args_json)

    writer = SummaryWriter(log_dir=output_dir)
    logger = Logger(os.path.join(output_dir, 'log.txt'))
    return writer, logger


def log_after_epoch(
        logger, writer, epoch, loss_meter, acc_groups, get_ys_func, tag,
        images=None
    ):
    logger.write(f"Epoch {epoch}\t Loss: {loss_meter.avg}\n")
    results = utils.get_results(acc_groups, get_ys_func)
    logger.write(f"Train results \n")
    logger.write(str(results) + "\n")
    
    write_dict_to_tb(writer, results, tag, epoch)

    if images is not None:
        images = images[:4]
        images_concat = torchvision.utils.make_grid(
            images, nrow=2, padding=2, pad_value=1.)
        writer.add_image("data/", images_concat, epoch)

    if has_wandb:
        wandb.log({f"train_{k}": results[k] for k in results}, step=epoch)


def log_test_results(logger, writer, epoch, acc_groups, get_ys_func, tag):
    results = utils.get_results(acc_groups, get_ys_func)
    utils.write_dict_to_tb(writer, results, tag, epoch)
    logger.write(f"Test results {tag} \n")
    logger.write(str(results))
    if has_wandb:
        wandb.log({f"test_{k}": results[k] for k in results}, step=epoch)


def log_data(logger, train_data, test_data, val_data=None, get_ys_func=None):
    for data, name in [(train_data, "Train"), (test_data, "Test"), 
                       (val_data, "Val")]:
        if data:
            logger.write(f'{name} Data (total {len(data)})\n')
            print("N groups ", data.n_groups)
            for group_idx in range(data.n_groups):
                y_idx, s_idx = get_ys_func(group_idx)
                logger.write(
                    f'    Group {group_idx} (y={y_idx}, s={s_idx}):'
                    f' n = {data.group_counts[group_idx]:.0f}\n')

def log_optimizer(writer, optimizer, epoch):
    group = optimizer.param_groups[0]
    hypers = {name: group[name] for name in ["lr", "weight_decay"]}
    write_dict_to_tb(writer, hypers, "optimizer_hypers", epoch)
