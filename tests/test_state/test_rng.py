from __future__ import annotations

import pickle
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from neps.state.seed_snapshot import RNGStateManager, use_generator_globally


@pytest.mark.parametrize(
    ("make_ints", "rng_factory"),
    [
        # Python RNG
        (
            lambda rng: [rng.randint(0, 100) for _ in range(10)],
            lambda seed: random.Random(seed),
        ),
        # NumPy Generator
        (
            lambda rng: list(rng.integers(0, 100, size=10)),
            lambda seed: np.random.default_rng(seed),
        ),
        # PyTorch Generator
        (
            lambda rng: list(torch.randint(0, 100, (10,), generator=rng)),
            lambda seed: torch.Generator().manual_seed(seed),
        ),
    ],
)
def test_randomstate_consistent(
    tmp_path: Path,
    make_ints: Callable[[Any], list[int]],
    rng_factory: Callable[[int], Any],
):
    seed = 230

    rng1 = rng_factory(seed)
    integers_1 = make_ints(rng1)

    rng2 = rng_factory(seed)
    integers_2 = make_ints(rng2)

    assert integers_1 == integers_2

    rng3 = rng_factory(1111)
    integers_3 = make_ints(rng3)

    assert integers_1 != integers_3


@pytest.mark.parametrize("seed", [0, 42, 999])
def test_capture_reproducibility(seed):
    """Ensure new_capture gives reproducible RNG states."""
    mgr1 = RNGStateManager.new_capture(
        seed, torch_rng=torch.Generator().manual_seed(seed)
    )
    mgr2 = RNGStateManager.new_capture(
        seed, torch_rng=torch.Generator().manual_seed(seed)
    )

    # Check that the Python RNG produces the same values
    py_vals1 = [mgr1.py_rng.randint(0, 100) for _ in range(10)]
    py_vals2 = [mgr2.py_rng.randint(0, 100) for _ in range(10)]
    assert py_vals1 == py_vals2

    # Check NumPy RNG
    np_vals1 = mgr1.np_rng.integers(0, 100, size=10).tolist()
    np_vals2 = mgr2.np_rng.integers(0, 100, size=10).tolist()
    assert np_vals1 == np_vals2

    # Check Torch CPU RNG
    t_vals1 = torch.randint(0, 100, (10,), generator=mgr1.torch_manual_rng)
    t_vals2 = torch.randint(0, 100, (10,), generator=mgr2.torch_manual_rng)
    assert torch.equal(t_vals1, t_vals2)


@pytest.mark.parametrize("seed", [123])
def test_getsetstate_pickling(seed):
    """Ensure pickling and unpickling preserves RNG states."""
    mgr = RNGStateManager.new_capture(seed)
    dumped = pickle.dumps(mgr)
    loaded = pickle.loads(dumped)  # noqa: S301

    assert mgr == loaded

    # Check reproducibility after restoring
    val_before = mgr.py_rng.randint(0, 100)
    val_after = loaded.py_rng.randint(0, 100)
    assert val_before == val_after


@pytest.mark.parametrize("seed", [7])
def test_context_manager_sets_global_rng(seed):
    """Ensure use_generator_globally temporarily sets global RNG."""
    gen = torch.Generator().manual_seed(seed)

    torch.manual_seed(999)
    torch.randint(0, 100, (1,)).item()

    with use_generator_globally(gen):
        val_inside = torch.randint(0, 100, (1,)).item()
    val_after = torch.randint(0, 100, (1,)).item()

    # Inside the context, the value should match gen's first draw
    gen_check = torch.randint(0, 100, (1,), generator=gen).item()
    assert val_inside == gen_check

    # Outside, the global RNG is restored
    assert val_after != val_inside


@pytest.mark.parametrize("seed", [42])
def test_cuda_generator_if_available(seed):
    """Ensure CUDA generator captures work if CUDA is available."""
    if torch.cuda.is_available():
        mgr = RNGStateManager.new_capture(seed)
        assert mgr.torch_cuda_rng is not None
        assert len(mgr.torch_cuda_rng) == torch.cuda.device_count()
        # Generate a tensor on GPU with the captured generator
        t_gpu = torch.randint(
            0, 100, (10,), generator=mgr.torch_cuda_rng[0], device="cuda"
        )
        assert t_gpu.device.type == "cuda"
