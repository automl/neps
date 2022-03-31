import logging
from functools import partial

import torch

cat_channels = partial(torch.cat, dim=1)

logger = logging.getLogger(__name__)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def iter_flatten(iterable):
    """
    Flatten a potentially deeply nested python list
    """
    # taken from https://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            yield from iter_flatten(e)
        else:
            yield e
