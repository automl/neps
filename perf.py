from __future__ import annotations

from functools import partial

import numpy as np
from graph import Grammar, Identity, ReLUConvBN, sample_grammar
from torch import nn

structure = {
    "S": (
        Grammar.NonTerminal(
            ["C", "reluconvbn", "S", "S C", "O O O"],
            nn.Sequential,
        )
    ),
    "C": (["O", "O S reluconvbn", "O S", "S"], nn.Sequential),
    "O": ["3", "1", "id"],
    "reluconvbn": partial(
        ReLUConvBN, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
    ),
    "id": Identity,
    "3": partial(
        nn.Conv2d, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
    ),
    "1": partial(
        nn.Conv2d, in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0
    ),
}


if __name__ == "__main__":
    sample = sample_grammar("S", grammar=grammar, rng=np.random.default_rng())
