from __future__ import annotations

import random

import numpy as np
import torch


def has_instance(collection, *types):
    return any([isinstance(el, typ) for el in collection for typ in types])


def filter_instances(collection, *types):
    return [el for el in collection if any([isinstance(el, typ) for typ in types])]


def get_rnd_state() -> dict:
    np_state = list(np.random.get_state())
    np_state[1] = np_state[1].tolist()
    state = {
        "random_state": random.getstate(),
        "np_seed_state": np_state,
        "torch_seed_state": torch.random.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_seed_state"] = [
            dev.tolist() for dev in torch.cuda.get_rng_state_all()
        ]
    return state


def set_rnd_state(state: dict):
    # rnd_s1, rnd_s2, rnd_s3 = state["random_state"]
    random.setstate(
        tuple(
            tuple(rnd_s) if isinstance(rnd_s, list) else rnd_s
            for rnd_s in state["random_state"]
        )
    )
    np.random.set_state(tuple(state["np_seed_state"]))
    torch.random.set_rng_state(torch.ByteTensor(state["torch_seed_state"]))
    if torch.cuda.is_available() and "torch_cuda_seed_state" in state:
        torch.cuda.set_rng_state_all(
            [torch.ByteTensor(dev) for dev in state["torch_cuda_seed_state"]]
        )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
