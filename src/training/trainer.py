import os
from typing import Tuple, Union
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.data.data_loader import ImageLoader
from src.resnet import MyResNet18
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.training.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
        self,
        data_dir: str,
        model: MyResNet18,
        optimizer: Optimizer,
        model_dir: str,
        train_data_transforms: transforms.Compose,
        val_data_transforms: transforms.Compose,
        batch_size: int = 100,
        inp_size=(64, 64),
        load_from_disk: bool = True,
    ) -> None:
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = model.to(device)
        dataloader_args = {"num_workers": 1, "pin_memory": True} if device=="cuda" else {}

        self.train_dataset = ImageLoader(
            data_dir, split="train", transform=train_data_transforms
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.val_dataset = ImageLoader(
            data_dir, split="validation", transform=val_data_transforms
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        )

        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []
        self.best_accuracy = 0

        # self.val_images = np.zeros((len(self.val_dataset), inp_size[0], inp_size[1]))
        self.true_labels = np.zeros(len(self.val_dataset), dtype=int)
        self.predictions = np.zeros(len(self.val_dataset), dtype=int)
        self.valImageDir = []
        self.valAnnDir = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()

    def save_model(self) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_dir, "checkpoint.pt"),
        )

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):

            train_loss, train_acc = self.train_epoch()

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)
            
            # save_im = epoch_idx == (num_epochs - 1) # Save validation images and predictions at the last epoch
            val_loss, val_acc = self.validate()
            self.validation_loss_history.append(val_loss)
            self.validation_accuracy_history.append(val_acc)
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                self.save_model()
                

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train Accuracy: {train_acc:.4f}"
                + f" Validation Accuracy: {val_acc:.4f}"
            )

    def train_epoch(self) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        train_loss_meter = AverageMeter("train loss")
        train_acc_meter = AverageMeter("train accuracy")

        # loop over each minibatch
        for (x, y, _, _) in self.train_loader:
            x = x.to(device)
            y = y.to(device)

            n = x.shape[0]
            logits = self.model(x)
            batch_acc = compute_accuracy(logits, y)
            train_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Return data to cpu
            x = x.cpu()
            y = y.cpu()

        return train_loss_meter.avg, train_acc_meter.avg

    def validate(self) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter("val loss")
        val_acc_meter = AverageMeter("val accuracy")

        # loop over whole val set
        for (x, y, _, _) in self.val_loader:
            x = x.to(device)
            y = y.to(device)

            n = x.shape[0]
            logits = self.model(x)

            batch_acc = compute_accuracy(logits, y)
            val_acc_meter.update(val=batch_acc, n=n)

            batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
            val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            # Return data to cpu
            x = x.cpu()
            y = y.cpu()

        return val_loss_meter.avg, val_acc_meter.avg
    
    def predict(self, saved_model):
        """Uses the best model on validation to predict the labels and store them"""
        indexTracker = 0
        for (x, y, imageDir, annDir) in self.val_loader:
            x = x.to(device)
            y = y.to(device)

            n = x.shape[0]
            logits = saved_model(x)
            # Return data to cpu
            x = x.cpu()
            y = y.cpu()

            # self.val_images[indexTracker:indexTracker+len(x)] = x.squeeze(1).numpy()
            self.true_labels[indexTracker:indexTracker+len(x)] = y.numpy()
            self.predictions[indexTracker:indexTracker+len(x)] = torch.argmax(logits, dim=1).cpu().numpy()
            self.valImageDir += list(imageDir)
            self.valAnnDir += list(annDir)
            indexTracker += len(x)
        self.model.cpu()

    def generate_confusion_matrix(self):
        confusionMatrix = np.zeros((6, 6), dtype=int)
        for true_label, prediction in zip(self.true_labels, self.predictions):
            confusionMatrix[true_label, prediction] += 1

        return confusionMatrix

    def plot_loss_history(self) -> None:
        """Plots the loss history"""
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))
        plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
        plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()

    def plot_accuracy(self) -> None:
        """Plots the accuracy history"""
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))
        plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
        plt.plot(epoch_idxs, self.train_accuracy_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_accuracy_history, "-r", label="validation")
        plt.title("Accuracy history")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.show()

def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_labels
) -> None:
    """Plots the confusion matrix

    Args:
    -   confusion_matrix: a (num_classes, num_classes) numpy array
                          representing the confusion matrix
    -   class_labels: A list containing the class labels at the index of their label_number
                      e.g. if the labels are {"Cat": 0, "Monkey": 2, "Dog": 1},
                           the return value should be ["Cat", "Dog", "Monkey"]
                      The length of class_labels should be num_classes
    """
    fig, ax = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(5)

    num_classes = len(class_labels)

    ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-Truth label")
    ax.set_title("Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(num_classes):
        for j in range(num_classes):
            _ = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.show()