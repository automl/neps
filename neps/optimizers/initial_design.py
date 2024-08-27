from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from neps.search_spaces.encoding import DataEncoder


class InitialDesign(Protocol):
    def sample(self, n: int) -> list[dict[str, Any]]: ...


@dataclass
class Sobol(InitialDesign):
    seed: int
    """The seed for the Sobol sequence."""

    encoder: DataEncoder
    """The encoding used to encode the samples."""

    scramble: bool = True
    """Whether to scramble the Sobol sequence."""

    buffer_sample_multiplier: int = 2
    """How many samples to generate in the buffer before checking for uniqueness."""

    allow_undersampling: bool = False
    """If True, will allow undersampling if we can't generate `n` unique samples."""

    def sample(self, n: int) -> list[dict[str, Any]]:
        """Sample `n` points from the Sobol sequence.

        !!! warning

            If `self.allow_undersampling` is False, this method will raise a ValueError if
            it cannot generate `n` unique samples.

        Args:
            n: The number of points to sample.

        Returns:
            A list of `n` points sampled from the Sobol sequence.
        """
        assert self.encoder.tensors is not None

        if self.encoder.has_graphs():
            # TODO: Won't work on graphs
            raise NotImplementedError("Graphs are not yet supported.")

        if self.encoder.n_numerical == 0 and self.encoder.n_categorical > 0:
            # TODO: We need to do something else if we have only categoricals
            # as we are going to get a lot of duplicates
            raise NotImplementedError("Only categorical variables are not yet supported.")

        ndim = self.encoder.n_numerical + self.encoder.n_categorical
        sobol = torch.quasirandom.SobolEngine(dimension=ndim, scramble=True, seed=5)

        SAMPLE_SIZE = self.buffer_sample_multiplier * n
        unit_x = sobol.draw(SAMPLE_SIZE, dtype=torch.float64)

        x = self.encoder.tensors.from_unit_tensor(unit_x)

        # NOTE: We have to check uniqueness after conversion from unit cube space
        # as we could have multiple unit floats mapping to the same categories or integers
        unique_x = torch.unique(x, dim=0)
        if len(unique_x) < n and not self.allow_undersampling:
            raise ValueError(
                f"Could not generate {n} unique samples, got {len(unique_x)}\n{self=}"
            )

        return self.encoder.decode_dicts(unique_x[:n])
