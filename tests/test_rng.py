from __future__ import annotations

from pathlib import Path
import random
from typing import Callable
import numpy as np
import torch
import pytest

from neps.utils._rng import SeedState

@pytest.mark.parametrize(
    "make_ints", (
        lambda: [random.randint(0, 100) for _ in range(10)],
        lambda: list(np.random.randint(0, 100, (10,))),
        lambda: list(torch.randint(0, 100, (10,))),
    )
)
def test_randomstate_consistent(tmp_path: Path, make_ints: Callable[[], list[int]]) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    seed_dir = tmp_path / "seed_dir"

    seed_state = SeedState.get()
    integers_1 = make_ints()

    seed_state.set_as_global_state()
    integers_2 = make_ints()

    assert integers_1 == integers_2

    SeedState.get().dump(seed_dir)
    integers_3 = make_ints()

    assert integers_3 != integers_2, "Ensure we have actually changed random state"

    SeedState.load(seed_dir).set_as_global_state()
    integers_4 = make_ints()

    assert integers_3 == integers_4
