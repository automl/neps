from __future__ import annotations

from functools import partial

import numpy as np
from graph import (
    Grammar,
    Identity,
    Node,
    ReLUConvBN,
    parse,
    sample_grammar,
    to_nxgraph,
    to_string,
)
from torch import nn

structure = {
    "S": (
        Grammar.NonTerminal(
            ["C", "reluconvbn", "S", "S C", "O O O", "S S O O O O O O"],
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
    import time

    import rich

    grammar = Grammar.from_dict(structure)
    rng = np.random.default_rng()
    sample: Node = sample_grammar("S", grammar=grammar, rng=rng)
    graph = to_nxgraph(sample)
    # model = to_model(sample)

    t0 = time.perf_counter()
    samples = 10000

    for _ in range(samples):
        sample: Node = sample_grammar("S", grammar=grammar, rng=rng)
        string = to_string(sample)
        parse(string=string, grammar=grammar)
        # graph = to_nxgraph(sample)
        # mutate_leaf_parents(root=sample, grammar=grammar, rng=rng)
        # model = to_model(sample)

    t1 = time.perf_counter()
    rich.print(f"sampling takes {(t1 - t0) / samples}s on average over {samples} samples")
    rich.print(f"duration for {samples} samples: {t1 - t0}s ")
