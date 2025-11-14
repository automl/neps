"""Snapshot of the global rng state."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch

if TYPE_CHECKING:
    NP_RNG_STATE: TypeAlias = tuple[str, np.ndarray, int, int, float]
    PY_RNG_STATE: TypeAlias = tuple[int, tuple[int, ...], int | None]
    TORCH_RNG_STATE: TypeAlias = torch.Tensor
    TORCH_CUDA_RNG_STATE: TypeAlias = list[torch.Tensor]


@dataclass
class RNGStateManager:
    """State of the global rng.

    Primarly used as a seed manager, having all the RNGs needed and enables storing of
    the rng state to disk using a binary format native to each library,
    allowing for potential version mistmatches between
    processes loading the state, as long as they can read the binary format.
    """

    np_rng_state: NP_RNG_STATE
    py_rng_state: PY_RNG_STATE
    torch_rng_state: TORCH_RNG_STATE
    torch_cuda_rng_state: TORCH_CUDA_RNG_STATE | None

    # Not appear in the dump
    py_rng: random.Random
    np_rng: np.random.Generator
    torch_manual_rng: torch.Generator
    torch_cuda_rng: list[torch.Generator] | None = None

    @classmethod
    def new_capture(
        cls,
        seed: int | None = None,
        np_rng: np.random.Generator | None = None,
        torch_rng: torch.Generator | None = None,
    ) -> RNGStateManager:
        """Current state of the global rng.

        Takes a snapshot, including cloning or copying any arrays, tensors, etc.
        """
        self = cls(
            np_rng_state=None,  # type: ignore
            py_rng_state=None,  # type: ignore
            torch_rng_state=None,  # type: ignore
            torch_cuda_rng_state=None,
            py_rng=random.Random(seed),
            np_rng=np_rng or np.random.default_rng(seed),
            torch_manual_rng=torch_rng
            or torch.Generator().manual_seed(seed or torch.seed()),
            torch_cuda_rng=None,
        )
        if torch.cuda.is_available():
            self.torch_cuda_rng = [
                torch.Generator(device=f"cuda:{i}").manual_seed(
                    (seed or seed or torch.seed()) + i
                )
                for i in range(torch.cuda.device_count())
            ]

        self.capture_local()
        return self

    def capture_local(self) -> None:
        """Capture the current state of the local rngs."""
        # Capture Python RNG state
        self.py_rng_state = self.py_rng.getstate()

        # Capture NumPy RNG state
        self.np_rng_state = self.np_rng.bit_generator.state

        # Capture PyTorch CPU generator state
        self.torch_rng_state = self.torch_manual_rng.get_state().clone()

        # Capture PyTorch CUDA generators state
        if self.torch_cuda_rng is not None:
            self.torch_cuda_rng_state = [
                g.get_state().clone() for g in self.torch_cuda_rng
            ]

    def __getstate__(self) -> dict[str, Any]:
        return {
            "np_rng_state": self.np_rng_state,
            "py_rng_state": self.py_rng_state,
            "torch_rng_state": self.torch_rng_state,
            "torch_cuda_rng_state": self.torch_cuda_rng_state,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.np_rng_state = state["np_rng_state"]
        self.py_rng_state = state["py_rng_state"]
        self.torch_rng_state = state["torch_rng_state"]
        self.torch_cuda_rng_state = state.get("torch_cuda_rng_state")

        self.py_rng = random.Random()  # or whatever Random object you want
        self.py_rng.setstate(self.py_rng_state)

        # Restore NumPy RNG
        self.np_rng = np.random.default_rng()  # create a new Generator
        self.np_rng.bit_generator.state = self.np_rng_state

        # Restore PyTorch CPU generator
        self.torch_manual_rng = torch.Generator()
        self.torch_manual_rng.set_state(self.torch_rng_state)

        # Restore PyTorch CUDA generators
        if self.torch_cuda_rng_state is not None:
            self.torch_cuda_rng = [
                torch.Generator(device=f"cuda:{i}").set_state(state)
                for i, state in enumerate(self.torch_cuda_rng_state)
            ]

    def __eq__(self, other: Any, /) -> bool:
        if not isinstance(other, RNGStateManager):
            return False

        if not (self.py_rng_state == other.py_rng_state):
            return False

        if not self.np_rng_state == other.np_rng_state:
            return False

        if self.torch_rng_state is not None and other.torch_rng_state is not None:
            import torch

            if not torch.equal(self.torch_rng_state, other.torch_rng_state):
                return False

        if (
            self.torch_cuda_rng_state is not None
            and other.torch_cuda_rng_state is not None
        ):
            import torch

            if not all(
                torch.equal(a, b)
                for a, b in zip(
                    self.torch_cuda_rng_state, other.torch_cuda_rng_state, strict=False
                )
            ):
                return False

        return True


@contextlib.contextmanager
def use_generator_globally(  # noqa: D103
    generator: torch.Generator,
    device: str = "cpu",
) -> Generator[Any, Any, Any]:
    if device == "cpu":
        old_state = torch.get_rng_state()
        torch.set_rng_state(generator.get_state())
    else:
        old_state = torch.cuda.get_rng_state_all()
        torch.cuda.set_rng_state_all(
            [generator.get_state() for _ in range(torch.cuda.device_count())]
        )
    try:
        yield
    finally:
        if device == "cpu":
            torch.set_rng_state(old_state)
        else:
            torch.cuda.set_rng_state_all(old_state)
