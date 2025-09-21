# Example pipeline used from; https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning

import os

import torch
from torchvision import transforms, datasets, models
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.lr_scheduler import PolynomialLR


class LitSegmentation(L.LightningModule):
    def __init__(self, iters_per_epoch, lr, momentum, weight_decay):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=21, aux_loss=True)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.iters_per_epoch = iters_per_epoch
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)["out"]
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch):
        images, targets = batch
        outputs = self.model(images)["out"]
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = PolynomialLR(
            optimizer,
            total_iters=self.iters_per_epoch * self.trainer.max_epochs,
            power=0.9,
        )
        return [optimizer], [scheduler]


class SegmentationData(L.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        dataset_path = ".data/VOC/VOCtrainval_11-May-2012.tar"
        if not os.path.exists(dataset_path):
            datasets.VOCSegmentation(root=".data/VOC", download=True)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        target_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256, 256), antialias=True)]
        )
        train_dataset = datasets.VOCSegmentation(
            root=".data/VOC", transform=transform, target_transform=target_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
        )

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        target_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((256, 256), antialias=True)]
        )
        val_dataset = datasets.VOCSegmentation(
            root=".data/VOC",
            year="2012",
            image_set="val",
            transform=transform,
            target_transform=target_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
        )


def evaluate_pipeline(**kwargs):
    data = SegmentationData(kwargs.get("batch_size", 4))
    data.prepare_data()
    iters_per_epoch = len(data.train_dataloader())
    model = LitSegmentation(
        iters_per_epoch,
        kwargs.get("lr", 0.02),
        kwargs.get("momentum", 0.9),
        kwargs.get("weight_decay", 1e-4),
    )
    trainer = L.Trainer(
        max_epochs=kwargs.get("epoch", 30),
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_checkpointing=False,
    )
    trainer.fit(model, data)
    val_loss = trainer.logged_metrics["val_loss"].detach().item()
    return val_loss


if __name__ == "__main__":
    import neps
    import logging

    logging.basicConfig(level=logging.INFO)

    # Search space for hyperparameters
    pipeline_space = dict(
        lr=neps.HPOFloat(lower=0.0001, upper=0.1, log=True, prior=0.02),
        momentum=neps.HPOFloat(lower=0.1, upper=0.9, prior=0.5),
        weight_decay=neps.HPOFloat(lower=1e-5, upper=1e-3, log=True, prior=1e-4),
        epoch=neps.HPOInteger(lower=10, upper=30, is_fidelity=True),
        batch_size=neps.HPOInteger(lower=4, upper=12, prior=4),
    )

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/hpo_image_segmentation",
        fidelities_to_spend=500
    )
