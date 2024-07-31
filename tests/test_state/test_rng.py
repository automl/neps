from __future__ import annotations

from pathlib import Path
import random
from typing import Callable
import numpy as np
import torch
import pytest

from neps.state.seed_snapshot import SeedSnapshot
from neps.state.filebased import ReaderWriterSeedSnapshot


@pytest.mark.parametrize(
    "make_ints",
    (
        lambda: [random.randint(0, 100) for _ in range(10)],
        lambda: list(np.random.randint(0, 100, (10,))),
        lambda: list(torch.randint(0, 100, (10,))),
    ),
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

    ReaderWriterSeedSnapshot.write(SeedSnapshot.new_capture(), seed_dir)

    integers_3 = make_ints()
    assert integers_3 != integers_2, "Ensure we have actually changed random state"

    ReaderWriterSeedSnapshot.read(seed_dir).set_as_global_seed_state()
    integers_4 = make_ints()

    assert integers_3 == integers_4

    before = SeedSnapshot.new_capture()
    after = SeedSnapshot.new_capture()

    _ = make_ints()

    after.recapture()
    assert before != after
