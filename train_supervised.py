import os
import torch
import models
import optimizers
import utils
from utils import supervised_utils
try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False

def get_args_parser():
    parser = utils.get_default_args()
    # TODO: add new supervised specific params?
    return parser

def main(args):
    if has_wandb:
        wandb.init(dir=args.output_dir)
        args.__dict__.update(wandb.config)
        print(args)

    utils.set_seed(args.seed)
    writer, logger = utils.prepare_logging(args.output_dir, args)

    # Data
    train_loader, test_loader_dict, get_ys_func = (
        utils.get_data(args, logger, contrastive=False))
    n_classes = train_loader.dataset.n_classes

    # Model
    model_cls = getattr(models, args.model)
    model = model_cls(n_classes)
    model.cuda()

    # Optimizer
    optimizer = getattr(optimizers, args.optimizer)(
        model, lr=args.init_lr, momentum=args.momentum_decay, 
        weight_decay=args.weight_decay)
    scheduler = getattr(optimizers, args.scheduler)(optimizer, args.num_epochs)
    start_epoch = 0
    
    if args.resume is not None:
        print('Resuming from checkpoint at {}...'.format(args.resume), end=" ")
        checkpoint = torch.load(args.resume)
        if "model" in checkpoint:
            print("Doing proper resume")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler:
                scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            print("Strating at epoch", start_epoch)
        else:
            print("Loading weights as init")
            model.load_state_dict(checkpoint)

    logger.flush()
    train_tag = "train/"

    if args.mixup:
        base_criterion = torch.nn.KLDivLoss()
        def criterion(logits, y):
            return base_criterion(logits.log_softmax(-1), y)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    # Train loop
    best_val_wga = None
    for epoch in range(start_epoch, args.num_epochs):
        loss_meter, acc_groups, example_images = supervised_utils.train_epoch(
            model, train_loader, optimizer, criterion)
        if scheduler: scheduler.step()

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }
        if scheduler:
            state_dict["scheduler"] = scheduler.state_dict()

        torch.save(state_dict, os.path.join(args.output_dir, f'resumable_checkpoint.pt'))

        utils.log_after_epoch(
            logger, writer, epoch, loss_meter, acc_groups, get_ys_func,
            images=example_images, tag=train_tag)
        utils.log_optimizer(writer, optimizer, epoch)

        if epoch % args.eval_freq == 0:
            results_dict = supervised_utils.eval(model, test_loader_dict)
            for ds_name, acc_groups in results_dict.items():
                utils.log_test_results(
                    logger, writer, epoch, acc_groups, get_ys_func, ds_name)
            val_wga = utils.get_results(results_dict["val"], get_ys_func)["worst_accuracy"]
            if (best_val_wga is None or val_wga > best_val_wga):
                logger.write('\n')
                logger.write(f"New best validation WGA: {val_wga}")
                best_val_wga = val_wga
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, f'best_checkpoint.pt'))

        if epoch % args.save_freq == 0:# and epoch > 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f'checkpoint_{epoch}.pt'))
        logger.write('\n')

    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'final_checkpoint.pt'))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    assert args.reweight_groups + args.reweight_classes <= 1
    main(args)
