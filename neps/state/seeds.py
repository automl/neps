from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple, Union
from typing_extensions import TypeAlias

import numpy as np
import torch

from neps.state.shared import Shared

if TYPE_CHECKING:
    NP_RNG_STATE: TypeAlias = Tuple[str, np.ndarray, int, int, float]
    PY_RNG_STATE: TypeAlias = Tuple[int, Tuple[int, ...], Union[int, None]]
    TORCH_RNG_STATE: TypeAlias = torch.Tensor
    TORCH_CUDA_RNG_STATE: TypeAlias = List[torch.Tensor]

# It seems like they're all uint32 but I can't be sure.
PY_RNG_STATE_DTYPE = np.int64


def _deserialize_from_directory(directory: Path) -> SeedSnapshot:
    import torch

    with (directory / "seed_info.json").open("r") as f:
        seed_info = json.load(f)

    # Load and set pythons rng
    py_rng_state = tuple(
        int(x) for x in np.fromfile(directory / "py_rng.npy", dtype=PY_RNG_STATE_DTYPE)
    )

    np_rng_state = np.fromfile(directory / "np_rng_state.npy", dtype=np.uint32)

    # By specifying `weights_only=True`, it disables arbitrary object loading
    torch_rng_state = torch.load(directory / "torch_rng_state.pt", weights_only=True)

    torch_cuda_rng = None
    torch_cuda_rng_path = directory / "torch_cuda_rng_state.pt"
    if torch_cuda_rng_path.exists():
        # By specifying `weights_only=True`, it disables arbitrary object loading
        torch_cuda_rng = torch.load(
            directory / "torch_cuda_rng_state.pt",
            weights_only=True,
        )

    return SeedSnapshot(
        np_rng=(
            seed_info["np_rng_kind"],
            np_rng_state,
            seed_info["np_pos"],
            seed_info["np_has_gauss"],
            seed_info["np_cached_gauss"],
        ),
        py_rng=(seed_info["py_rng_version"], py_rng_state, seed_info["py_guass_next"]),
        torch_rng=torch_rng_state,
        torch_cuda_rng=torch_cuda_rng,
    )


def _serialize_to_directory(seeds: SeedSnapshot, directory: Path) -> None:
    """Save the state to a directory."""
    import torch

    if directory.exists():
        assert directory.is_dir()
    else:
        directory.mkdir(parents=True)

    py_rng_version, py_rng_state, py_guass_next = seeds.py_rng
    np_rng_kind, np_rng_state, np_pos, np_has_gauss, np_cached_gauss = seeds.np_rng

    seed_info = {
        "np_rng_kind": np_rng_kind,
        "np_pos": np_pos,
        "np_has_gauss": np_has_gauss,
        "np_cached_gauss": np_cached_gauss,
        "py_rng_version": py_rng_version,
        "py_guass_next": py_guass_next,
    }

    # NOTE(eddiebergman): Chose JSON since it's fast and non-injectable
    with (directory / "seed_info.json").open("w") as f:
        json.dump(seed_info, f)

    py_rng_state_arr = np.array(py_rng_state, dtype=PY_RNG_STATE_DTYPE)
    with (directory / "py_rng.npy").open("wb") as f:
        py_rng_state_arr.tofile(f)

    with (directory / "np_rng_state.npy").open("wb") as f:
        np_rng_state.tofile(f)

    torch.save(seeds.torch_rng, directory / "torch_rng_state.pt")

    if seeds.torch_cuda_rng:
        torch.save(seeds.torch_cuda_rng, directory / "torch_cuda_rng_state.pt")


@dataclass
class SeedSnapshot:
    """State of the global rng.

    Primarly enables storing of the rng state to disk using a binary format
    native to each library, allowing for potential version mistmatches between
    processes loading the state, as long as they can read the binary format.
    """

    np_rng: NP_RNG_STATE
    py_rng: PY_RNG_STATE
    torch_rng: TORCH_RNG_STATE
    torch_cuda_rng: TORCH_CUDA_RNG_STATE | None

    @classmethod
    def new_capture(cls) -> SeedSnapshot:
        """Current state of the global rng.

        Takes a snapshot, including cloning or copying any arrays, tensors, etc.
        """
        self = cls(None, None, None, None)  # type: ignore
        self.capture()
        return self

    def capture(self) -> None:
        """Reread the state of the global rng into this snapshot."""
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html
        import torch

        np_keys = np.random.get_state(legacy=True)
        assert np_keys[0] == "MT19937"  # type: ignore
        np_keys = (np_keys[0], np_keys[1].copy(), *np_keys[2:])  # type: ignore

        py_rng = random.getstate()
        torch_rng = torch.random.get_rng_state().clone()
        torch_cuda_keys: list[torch.Tensor] | None = None
        if torch.cuda.is_available():
            torch_cuda_keys = [c.clone() for c in torch.cuda.get_rng_state_all()]

        self.np_rng = np_keys  # type: ignore
        self.py_rng = py_rng
        self.torch_rng = torch_rng
        self.torch_cuda_rng = torch_cuda_keys

    def set_as_global_seed_state(self) -> None:
        """Set the global rng to the given state."""
        import torch

        np.random.set_state(self.np_rng)
        random.setstate(self.py_rng)
        torch.random.set_rng_state(self.torch_rng)
        if self.torch_cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_rng)

    def as_filesystem_shared(self, directory: Path) -> Shared[SeedSnapshot, Path]:
        """Return the trial as a shared object."""
        return Shared.using_directory(
            self,
            directory,
            serialize=_serialize_to_directory,
            deserialize=_deserialize_from_directory,
            lockname=".seed_lock",
            version_filename=".seed_sha",
        )

    def __eq__(self, other: Any, /) -> bool:
        if not isinstance(other, SeedSnapshot):
            return False

        if (
            self.py_rng == other.py_rng
            and self.np_rng[0] == other.np_rng[0]
            and self.np_rng[2] == other.np_rng[2]
            and self.np_rng[3] == other.np_rng[3]
            and self.np_rng[4] == other.np_rng[4]
            and np.array_equal(self.np_rng[1], other.np_rng[1])
            and torch.equal(self.torch_rng, other.torch_rng)
        ):
            if self.torch_cuda_rng is None:
                return other.torch_cuda_rng is None

            if other.torch_cuda_rng is None:
                return False

            return all(
                torch.equal(a, b)
                for a, b in zip(
                    self.torch_cuda_rng,
                    other.torch_cuda_rng,
                    strict=True,
                )
            )
        return False
