import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("../../")

def compute_loss(loss_fun, pred, true, task_type=None):
    """
    Compute loss and prediction score

    Args:
        pred (torch.tensor): Unnormalized prediction
        true (torch.tensor): Ground truth labels
        task_type (str): User specified task type

    Returns: Loss, normalized prediction score

    """
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    if task_type is None:
        if loss_fun == 'cross_entropy':
            # multiclass
            if pred.ndim > 1 and true.ndim == 1:
                pred = F.log_softmax(pred, dim=-1)
                return F.nll_loss(pred, true), pred
            # binary or multilabel
            else:
                true = true.float()
                return bce_loss(pred, true), torch.sigmoid(pred)
        elif loss_fun == 'mse':
            true = true.float()
            return mse_loss(pred, true), pred
        else:
            raise ValueError('Loss func {} not supported'.format(
                loss_fun))
    else:
        if task_type == 'classification_multi':
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        elif 'classification' in task_type and 'binary' in task_type:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
        elif task_type == 'regression':
            true = true.float()
            # return mse_loss(torch.exp(pred), torch.exp(true)), pred
            return mse_loss(pred, true), pred
        else:
            raise ValueError('Task type {} not supported'.format(task_type))



# TODO: some parameters could be further refactored
def create_optimizer(cfg, params):
    r"""Creates a config-driven optimizer."""
    params = filter(lambda p: p.requires_grad, params)

    if cfg.optim.optimizer == 'adam':
        optimizer = optim.Adam(params,
                               lr=cfg.optim.base_lr,
                               weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer == 'sgd':
        optimizer = optim.SGD(params,
                              lr=cfg.optim.base_lr,
                              momentum=cfg.optim.momentum,
                              weight_decay=cfg.optim.weight_decay)
    else:
        raise ValueError('Optimizer {} not supported'.format(
            cfg.optim.optimizer))

    return optimizer


def create_scheduler(cfg, optimizer):
    r"""Creates a config-driven learning rate scheduler."""
    if cfg.optim.scheduler == 'none':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=cfg.optim.max_epoch +
                                              1)
    elif cfg.optim.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=cfg.optim.steps,
                                                   gamma=cfg.optim.lr_decay)
    elif cfg.optim.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.optim.max_epoch)
    else:
        raise ValueError('Scheduler {} not supported'.format(
            cfg.optim.scheduler))
    return scheduler

