import logging

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import neps

NUM_GPU = 8  # Number of GPUs to use for DDP


class ToyModel(nn.Module):
    """ Taken from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html """
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

class LightningModel(L.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.model = ToyModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

def evaluate_pipeline(lr=0.1, epoch=20):
    L.seed_everything(42)
    # Model
    model = LightningModel(lr=lr)

    # Generate random tensors for data and labels
    data = torch.rand((1000, 10))
    labels = torch.rand((1000, 5))

    dataset = list(zip(data, labels))

    train_dataset, val_dataset, test_dataset = random_split(dataset, [600, 200, 200])

    # Define simple data loaders using tensors and slicing
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Trainer with DDP Strategy
    trainer = L.Trainer(gradient_clip_val=0.25,
                        max_epochs=epoch,
                        fast_dev_run=False,
                        strategy='ddp',
                        devices=NUM_GPU
                        )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.validate(model, test_dataloader)
    return trainer.logged_metrics["val_loss"].item()


pipeline_space = dict(
    lr=neps.Float(
        lower=0.001,
        upper=0.1,
        log=True,
        prior=0.01
        ),
    epoch=neps.Integer(
        lower=1,
        upper=3,
        is_fidelity=True
        )
    )

logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/pytorch_lightning_ddp",
    fidelities_to_spend=5
    )
