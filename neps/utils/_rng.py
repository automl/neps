from __future__ import annotations

import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple, Union
from typing_extensions import TypeAlias

import numpy as np
import torch

NP_RNG_STATE: TypeAlias = Tuple[str, np.ndarray, int, int, float]
PY_RNG_STATE: TypeAlias = Tuple[int, Tuple[int, ...], Union[int, None]]
TORCH_RNG_STATE: TypeAlias = torch.Tensor
TORCH_CUDA_RNG_STATE: TypeAlias = List[torch.Tensor]


@dataclass
class SeedState:
    """State of the global rng.

    Primarly enables storing of the rng state to disk using a binary format
    native to each library, allowing for potential version mistmatches between
    processes loading the state, as long as they can read the binary format.
    """

    # It seems like they're all uint32 but I can't be sure.
    PY_RNG_STATE_DTYPE = np.int64

    np_rng: NP_RNG_STATE
    py_rng: PY_RNG_STATE
    torch_rng: TORCH_RNG_STATE
    torch_cuda_rng: TORCH_CUDA_RNG_STATE | None

    @classmethod
    def get(cls) -> SeedState:
        """Current state of the global rng.

        Takes a snapshot, including cloning or copying any arrays, tensors, etc.
        """
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html
        np_keys = np.random.get_state(legacy=True)
        assert np_keys[0] == "MT19937"  # type: ignore
        np_keys = (np_keys[0], np_keys[1].copy(), *np_keys[2:])  # type: ignore

        py_rng = random.getstate()
        torch_rng = torch.random.get_rng_state().clone()
        torch_cuda_keys: list[torch.Tensor] | None = None
        if torch.cuda.is_available():
            torch_cuda_keys = [c.clone() for c in torch.cuda.get_rng_state_all()]

        return cls(
            np_rng=np_keys,  # type: ignore
            py_rng=py_rng,
            torch_rng=torch_rng,
            torch_cuda_rng=torch_cuda_keys,
        )

    def set_as_global_state(self) -> None:
        """Set the global rng to the given state."""
        np.random.set_state(self.np_rng)
        random.setstate(self.py_rng)
        torch.random.set_rng_state(self.torch_rng)
        if self.torch_cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_rng)

    def dump(self, path: Path) -> None:
        """Save the state to a directory."""
        if path.exists():
            assert path.is_dir()
        else:
            path.mkdir(parents=True)

        py_rng_version, py_rng_state, py_guass_next = self.py_rng
        np_rng_kind, np_rng_state, np_pos, np_has_gauss, np_cached_gauss = self.np_rng

        seed_info = {
            "np_rng_kind": np_rng_kind,
            "np_pos": np_pos,
            "np_has_gauss": np_has_gauss,
            "np_cached_gauss": np_cached_gauss,
            "py_rng_version": py_rng_version,
            "py_guass_next": py_guass_next,
        }

        # NOTE(eddiebergman): Chose JSON since it's fast and non-injectable
        with (path / "seed_info.json").open("w") as f:
            json.dump(seed_info, f)

        py_rng_state_arr = np.array(py_rng_state, dtype=self.PY_RNG_STATE_DTYPE)
        with (path / "py_rng.npy").open("wb") as f:
            py_rng_state_arr.tofile(f)

        with (path / "np_rng_state.npy").open("wb") as f:
            np_rng_state.tofile(f)

        torch.save(self.torch_rng, path / "torch_rng_state.pt")

        if self.torch_cuda_rng:
            torch.save(self.torch_cuda_rng, path / "torch_cuda_rng_state.pt")

    @classmethod
    def load(cls, path: Path) -> SeedState:
        assert path.is_dir()

        with (path / "seed_info.json").open("r") as f:
            seed_info = json.load(f)

        # Load and set pythons rng
        py_rng_state = tuple(
            int(x) for x in np.fromfile(path / "py_rng.npy", dtype=cls.PY_RNG_STATE_DTYPE)
        )
        np_rng_state = np.fromfile(path / "np_rng_state.npy", dtype=np.uint32)

        # By specifying `weights_only=True`, it disables arbitrary object loading
        torch_rng_state = torch.load(path / "torch_rng_state.pt", weights_only=True)

        torch_cuda_rng = None
        torch_cuda_rng_path = path / "torch_cuda_rng_state.pt"
        if torch_cuda_rng_path.exists():
            # By specifying `weights_only=True`, it disables arbitrary object loading
            torch_cuda_rng = torch.load(
                path / "torch_cuda_rng_state.pt",
                weights_only=True,
            )

        return cls(
            np_rng=(
                seed_info["np_rng_kind"],
                np_rng_state,
                seed_info["np_pos"],
                seed_info["np_has_gauss"],
                seed_info["np_cached_gauss"],
            ),
            py_rng=(
                seed_info["py_rng_version"],
                py_rng_state,
                seed_info["py_guass_next"],
            ),
            torch_rng=torch_rng_state,
            torch_cuda_rng=torch_cuda_rng,
        )

    @classmethod
    @contextmanager
    def use(
        cls,
        path: Path,
        *,
        update_on_exit: bool = True,
    ) -> Iterator[SeedState]:
        """Context manager to use a seed state.

        If the path exists, load the seed state from the path and set it as the
        global state. Otherwise, use the current global state.

        Args:
            path: Path to the seed state.
            update_on_exit: If True, get the seed state after the context manager returns
                and save it to the path.

        Yields:
            SeedState: The seed state in use.
        """
        if path.exists():
            seed_state = cls.load(path)
            seed_state.set_as_global_state()
        else:
            seed_state = cls.get()

        yield seed_state

        if update_on_exit:
            cls.get().dump(path)
