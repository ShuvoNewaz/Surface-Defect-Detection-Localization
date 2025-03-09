import os
from typing import Tuple, Union
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.data.data_loader import ImageLoader
from src.resnet import MyResNet18
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.training.train_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):

            train_loss, train_acc = train(self.train_loader, self.model, self.optimizer, device)

            self.train_loss_history.append(train_loss)
            self.train_accuracy_history.append(train_acc)
            
            # save_im = epoch_idx == (num_epochs - 1) # Save validation images and predictions at the last epoch
            val_loss, val_acc = validate(self.val_loader, self.model, self.optimizer, device)
            self.validation_loss_history.append(val_loss)
            self.validation_accuracy_history.append(val_acc)
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                # self.save_model()
                save_model(self.model, self.optimizer, self.model_dir)
                

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train Accuracy: {train_acc:.4f}"
                + f" Validation Accuracy: {val_acc:.4f}"
            )