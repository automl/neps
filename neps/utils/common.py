from __future__ import annotations

import inspect
import random
from typing import Callable

import numpy as np
import torch

# Inspecting functions


def get_fun_args_and_defaults(function: Callable):
    signature = inspect.signature(function)
    return list(signature.parameters.keys()), {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def has_instance(collection, *types):
    return any([isinstance(el, typ) for el in collection for typ in types])


# Synchronizing state


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
    rnd_s1, rnd_s2, rnd_s3 = state["random_state"]
    random.setstate((rnd_s1, tuple(rnd_s2), rnd_s3))
    np.random.set_state(tuple(state["np_seed_state"]))
    torch.random.set_rng_state(torch.ByteTensor(state["torch_seed_state"]))
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.ByteTensor(dev) for dev in state["torch_cuda_seed_state"]]
        )
