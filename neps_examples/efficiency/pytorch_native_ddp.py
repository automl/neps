"""Some parts of this code are taken from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

Mind that this example does not run on Windows at the moment."""

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

import neps
import logging

NUM_GPU = 8  # Number of GPUs to use for DDP

# On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
# For TcpStore, same way as on Linux.


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):
    """Taken from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html"""

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(rank, world_size, loss_dict, learning_rate, epochs):
    """Taken from https://pytorch.org/tutorials/intermediate/ddp_tutorial.html (modified)"""
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=learning_rate)

    total_loss = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10))
        labels = torch.randn(20, 5).to(rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch} complete")

    loss_dict[rank] = total_loss

    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")


def evaluate_pipeline(learning_rate, epochs):
    from torch.multiprocessing import Manager

    world_size = NUM_GPU  # Number of GPUs

    manager = Manager()
    loss_dict = manager.dict()

    mp.spawn(
        demo_basic,
        args=(world_size, loss_dict, learning_rate, epochs),
        nprocs=world_size,
        join=True,
    )

    loss = sum(loss_dict.values()) // world_size
    return {"loss": loss}


class HPOSpace(neps.PipelineSpace):
    learning_rate = neps.Float(min_value=10e-7, max_value=10e-3, log=True)
    epochs = neps.Integer(min_value=1, max_value=3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=HPOSpace(),
        root_directory="results/pytorch_ddp",
        max_evaluations_total=25,
    )
