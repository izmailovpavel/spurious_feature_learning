import utils
from utils import AverageMeter
import tqdm
import torch



def train_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_meter = AverageMeter()
    n_groups = loader.dataset.n_groups
    acc_groups = {g_idx: AverageMeter() for g_idx in range(n_groups)}
    
    for batch in (pbar := tqdm.tqdm(loader)):
        x, y, g, s = batch
        x, y, s = x.cuda(), y.cuda(), s.cuda()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss, x.size(0))
        
        preds = torch.argmax(logits, dim=1)
        if len(y.shape) > 1:
            # mixup
            y = torch.argmax(y, dim=1)

        utils.update_dict(acc_groups, y, g, logits)
        acc = (preds == y).float().mean()

        pbar.set_description("Loss: {:.3f} ({:3f}); Acc: {:3f}".format(
            loss.item(), loss_meter.avg, acc))

    return loss_meter, acc_groups, x


def eval(model, test_loader_dict):
    model.eval()
    results_dict = {}
    with torch.no_grad():
        # Currently test_loader_dict has "test" and "val"
        for test_name, test_loader in test_loader_dict.items():
            acc_groups = {g_idx: AverageMeter() for g_idx in range(test_loader.dataset.n_groups)}
            for x, y, g, p in tqdm.tqdm(test_loader):
                x, y, p = x.cuda(), y.cuda(), p.cuda()
                logits = model(x)
                utils.update_dict(acc_groups, y, g, logits)
            results_dict[test_name] = acc_groups
    return results_dict
