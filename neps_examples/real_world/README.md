# Real World Examples

1. **Image Segmentation Pipeline Hyperparameter Optimization**

This example demonstrates how to perform hyperparameter optimization (HPO) for an image segmentation pipeline using NePS. The pipeline consists of a ResNet-50 model to segment images model trained on PASCAL Visual Object Classes (VOC) Dataset (http://host.robots.ox.ac.uk/pascal/VOC/).

We compare the performance of the optimized hyperparameters with the default hyperparameters. using the validation loss achieved on the dataset after training the model with the respective hyperparameters.

```python3

# Example pipeline used from; https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning

import torch
from torchvision import transforms, datasets, models
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
import os
from lightning.pytorch.utilities.rank_zero import rank_zero_only
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
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = PolynomialLR(
            optimizer, total_iters=self.iters_per_epoch * self.trainer.max_epochs, power=0.9
        )
        return [optimizer], [scheduler]
    
    
    
class SegmentationData(L.LightningDataModule):
    def __init__(self, batch_size=4):
        super().__init__()
        self.batch_size = batch_size
        
    def prepare_data(self):
        dataset_path = "data/VOCtrainval_11-May-2012.tar"
        if not os.path.exists(dataset_path):
            datasets.VOCSegmentation(root="data", download=True)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256), antialias=True)])
        train_dataset = datasets.VOCSegmentation(root="data", transform=transform, target_transform=target_transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=63)
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256), antialias=True)])
        val_dataset = datasets.VOCSegmentation(root="data", year='2012', image_set='val', transform=transform, target_transform=target_transform)
        return torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=63)

    

def main(**kwargs):
    data = SegmentationData(kwargs.get("batch_size", 4))
    iters_per_epoch = len(data.train_dataloader())
    model = LitSegmentation(iters_per_epoch, kwargs.get("lr", 0.02), kwargs.get("momentum", 0.9), kwargs.get("weight_decay", 1e-4)) 
    trainer = L.Trainer(max_epochs=kwargs.get("epoch", 30), strategy=DDPStrategy(find_unused_parameters=True), enable_checkpointing=False)
    trainer.fit(model, data)
    val_loss = trainer.logged_metrics["val_loss"].detach().item()
    return val_loss


if __name__ == "__main__":
    import neps
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Search space for hyperparameters
    pipeline_space = dict(
        lr=neps.Float(
            lower=0.0001, 
            upper=0.1, 
            log=True, 
            prior=0.02
            ),
        momentum=neps.Float(
            lower=0.1, 
            upper=0.9, 
            prior=0.5
            ),
        weight_decay=neps.Float(
            lower=1e-5, 
            upper=1e-3, 
            log=True, 
            prior=1e-4
            ),
        epoch=neps.Integer(
            lower=10,
            upper=30,
            is_fidelity=True
            ),
        batch_size=neps.Integer(
            lower=4,
            upper=12,
            prior=4
        ),
    )

    neps.run(
        evaluate_pipeline=main, 
        pipeline_space=pipeline_space, 
        root_directory="hpo_image_segmentation", 
        max_evaluations_total=500
    )

```

The search space has been set with the priors set to the hyperparameters found in this base example: https://lightning.ai/lightning-ai/studios/image-segmentation-with-pytorch-lightning

We run the HPO process for 188 trials and obtain new set of hyperpamereters that outperform the default hyperparameters.

| Hyperparameter | Prior | Optimized Value |
|----------------|-------|-----------------|
| learning_rate  | 0.02 | 0.006745150778442621 |
| batch_size     | 4 | 5 |
| momentum       | 0.5 | 0.5844767093658447 |
| weight_decay   | 0.0001 | 0.00012664785026572645 |


![Validation Loss Curves](val_loss_image_segmentation.png "Validation Loss Curves")

The validation loss achieved on the dataset after training the model with the newly sampled hyperparameters is shown in the figure above.

We compare the validation loss values when the model is trained with the default hyperparameters and the optimized hyperparameters:

Validation Loss with Default Hyperparameters: 0.114094577729702

Validation Loss with Optimized Hyperparameters: 0.0997161939740181

The optimized hyperparameters outperform the default hyperparameters by 12.61%.