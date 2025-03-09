import os
import torch
import numpy as np
from typing import Tuple, List
from src.training.metrics import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    

def save_model(model, optimizer, model_dir) -> None:
    """
    Saves the model state and optimizer state on the dict
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(model_dir, "checkpoint.pt"),
    )


def train(train_loader, model, optimizer, device) -> Tuple[float, float]:
    """Implements the main training loop."""
    model.train()

    train_loss_meter = AverageMeter("train loss")
    train_acc_meter = AverageMeter("train accuracy")

    # loop over each minibatch
    for (x, y, _, _) in train_loader:
        x = x.to(device)
        y = y.to(device)

        n = x.shape[0]
        logits = model(x)
        batch_acc = compute_accuracy(logits, y)
        train_acc_meter.update(val=batch_acc, n=n)

        batch_loss = compute_loss(model, logits, y, is_normalize=True)
        train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Return data to cpu
        x = x.cpu()
        y = y.cpu()

    return train_loss_meter.avg, train_acc_meter.avg


def validate(val_loader, model, optimizer, device) -> Tuple[float, float]:
    """Evaluate on held-out split (either val or test)"""
    model.eval()

    val_loss_meter = AverageMeter("val loss")
    val_acc_meter = AverageMeter("val accuracy")

    # loop over whole val set
    for (x, y, _, _) in val_loader:
        x = x.to(device)
        y = y.to(device)

        n = x.shape[0]
        logits = model(x)

        batch_acc = compute_accuracy(logits, y)
        val_acc_meter.update(val=batch_acc, n=n)

        batch_loss = compute_loss(model, logits, y, is_normalize=True)
        val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

        # Return data to cpu
        x = x.cpu()
        y = y.cpu()

    return val_loss_meter.avg, val_acc_meter.avg


def predict(test_loader, model, device):
    """Uses the best model on validation to predict the labels and store them"""
    model = model.to(device)
    indexTracker = 0
    for batchCount, (x, y, imageDir, annDir) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        n = x.shape[0]
        logits = model(x)
        # Return data to cpu
        x = x.cpu()
        y = y.cpu()

        if batchCount == 0:
            true_labels = y.numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            valImageDir = list(imageDir)
            valAnnDir = list(annDir)
        else:
            true_labels = np.concatenate((true_labels, y.numpy()))
            predictions = np.concatenate((predictions, torch.argmax(logits, dim=1).cpu().numpy()))
            valImageDir += list(imageDir)
            valAnnDir += list(annDir)
        indexTracker += len(x)
    model.cpu()

    return true_labels, predictions, valImageDir, valAnnDir