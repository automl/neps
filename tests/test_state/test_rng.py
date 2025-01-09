from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from neps.state.seed_snapshot import SeedSnapshot


@pytest.mark.parametrize(
    "make_ints",
    [
        lambda: [random.randint(0, 100) for _ in range(10)],
        lambda: list(np.random.randint(0, 100, (10,))),
        lambda: list(torch.randint(0, 100, (10,))),
    ],
)
def test_randomstate_consistent(
    tmp_path: Path, make_ints: Callable[[], list[int]]
) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    seed_dir = tmp_path / "seed_dir"
    seed_dir.mkdir(exist_ok=True, parents=True)

    seed_state = SeedSnapshot.new_capture()
    integers_1 = make_ints()

    seed_state.set_as_global_seed_state()

    integers_2 = make_ints()
    assert integers_1 == integers_2
