"""Based on: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html

Mind that this example does not run on Windows at the moment."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2


class LanguageModel(L.LightningModule):
    def __init__(self, vocab_size, lr):
        super().__init__()
        self.model = Transformer(  # 1B parameters
            vocab_size=vocab_size,
            nlayers=32,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )
        self.lr = lr

    def training_step(self, batch):
        input, target = batch
        output = self.model(input, target)
        loss = F.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def evaluate_pipeline(lr=0.1, epoch=20):
    L.seed_everything(42)

    # Data
    dataset = WikiText2()
    train_dataloader = DataLoader(dataset)

    # Model
    model = LanguageModel(vocab_size=dataset.vocab_size, lr=lr)

    # Trainer
    trainer = L.Trainer(accelerator="cuda", strategy=FSDPStrategy())
    trainer.fit(model, train_dataloader, max_epochs=epoch)
    return trainer.logged_metrics["train_loss"].detach().item()


if __name__ == "__main__":
    import neps
    import logging

    logging.basicConfig(level=logging.INFO)

    class HPOSpace(neps.PipelineSpace):
        lr = neps.Float(min_value=0.001, max_value=0.1, log=True, prior=0.01)
        epoch = neps.Fidelity(neps.Integer(min_value=1, max_value=3))

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=HPOSpace(),
        root_directory="results/pytorch_lightning_fsdp",
        max_evaluations_total=5,
    )
