"""Snapshot of the global rng state."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Tuple, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    import torch

    NP_RNG_STATE: TypeAlias = Tuple[str, np.ndarray, int, int, float]
    PY_RNG_STATE: TypeAlias = Tuple[int, Tuple[int, ...], Union[int, None]]
    TORCH_RNG_STATE: TypeAlias = torch.Tensor
    TORCH_CUDA_RNG_STATE: TypeAlias = List[torch.Tensor]


@dataclass
class SeedSnapshot:
    """State of the global rng.

    Primarly enables storing of the rng state to disk using a binary format
    native to each library, allowing for potential version mistmatches between
    processes loading the state, as long as they can read the binary format.
    """

    np_rng: NP_RNG_STATE
    py_rng: PY_RNG_STATE
    torch_rng: TORCH_RNG_STATE | None
    torch_cuda_rng: TORCH_CUDA_RNG_STATE | None

    @classmethod
    def new_capture(cls) -> SeedSnapshot:
        """Current state of the global rng.

        Takes a snapshot, including cloning or copying any arrays, tensors, etc.
        """
        self = cls(None, None, None, None)  # type: ignore
        self.recapture()
        return self

    def recapture(self) -> None:
        """Reread the state of the global rng into this snapshot."""
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html

        self.py_rng = random.getstate()

        np_keys = np.random.get_state(legacy=True)
        assert np_keys[0] == "MT19937"  # type: ignore
        self.np_rng = (np_keys[0], np_keys[1].copy(), *np_keys[2:])  # type: ignore

        with contextlib.suppress(Exception):
            import torch

            self.torch_rng = torch.random.get_rng_state().clone()
            torch_cuda_keys: list[torch.Tensor] | None = None
            if torch.cuda.is_available():
                torch_cuda_keys = [c.clone() for c in torch.cuda.get_rng_state_all()]
            self.torch_cuda_rng = torch_cuda_keys

    def set_as_global_seed_state(self) -> None:
        """Set the global rng to the given state."""
        np.random.set_state(self.np_rng)
        random.setstate(self.py_rng)

        if self.torch_rng is not None or self.torch_cuda_rng is not None:
            import torch

            if self.torch_rng is not None:
                torch.random.set_rng_state(self.torch_rng)

            if self.torch_cuda_rng is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.torch_cuda_rng)

    def __eq__(self, other: Any, /) -> bool:  # noqa: PLR0911
        if not isinstance(other, SeedSnapshot):
            return False

        if not (self.py_rng == other.py_rng):
            return False

        if not (
            self.np_rng[0] == other.np_rng[0]
            and self.np_rng[2] == other.np_rng[2]
            and self.np_rng[3] == other.np_rng[3]
            and self.np_rng[4] == other.np_rng[4]
        ):
            return False

        if not np.array_equal(self.np_rng[1], other.np_rng[1]):
            return False

        if self.torch_rng is not None and other.torch_rng is not None:
            import torch

            if not torch.equal(self.torch_rng, other.torch_rng):
                return False

        if self.torch_cuda_rng is not None and other.torch_cuda_rng is not None:
            import torch

            if not all(
                torch.equal(a, b)
                for a, b in zip(self.torch_cuda_rng, other.torch_cuda_rng)
            ):
                return False

        if not isinstance(self.torch_rng, type(other.torch_rng)):
            return False

        return isinstance(self.torch_cuda_rng, type(other.torch_cuda_rng))
