import inspect
from typing import Callable

# Inspecting functions


def get_fun_args_and_defaults(function: Callable):
    signature = inspect.signature(function)
    return list(signature.parameters.keys()), {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


# Types


def has_instance(collection, *types):
    return any([isinstance(el, typ) for el in collection for typ in types])
