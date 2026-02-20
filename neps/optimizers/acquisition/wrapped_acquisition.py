"""Module to wrap the existing acquisition function to account for mixed search spaces.

For mixed search spaces, we first keep the categorical dimensions fixed to some randomly
chosen values and perform optimization over the continuous dimensions.
Next, we select the numerical dimensions from the returned best candidate and keep them
fixed, while we use `optimize_acqf_discrete_local_search` over the categorical dimensions.

For this, we need to wrap the existing acquisition function to accept tensors containing
only the categorical dimensions since BoTorch does not natively support keeping numerical
dimensions fixed in `optimize_acqf_discrete_local_search`.

Inside `WrappedAcquisition`, we concatenate the fixed numerical dimensions to the tensor
containing only the categoricals before passing it to the original acquisition function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import t_batch_mode_transform
from botorch.acquisition.monte_carlo import concatenate_pending_points

if TYPE_CHECKING:
    from torch import Tensor

    from neps.space.encoding import ConfigEncoder


class WrappedAcquisition(AcquisitionFunction):
    """Acquisition function wrapper for mixed search spaces."""

    def __init__(
        self,
        acq: AcquisitionFunction,
        encoder: ConfigEncoder,
        fixed_numericals: dict[int, float],
    ) -> None:
        """Initialize the wrapped acquisition function.

        Args:
            acq: The base acquisition function.
            fixed_numericals: A dictionary mapping numerical dimension indices to their
                fixed values.
        """
        super().__init__(model=acq.model)
        # NOTE: Remove X_pending from the base acquisition function.
        # See similar note in WeightedAcquisition.
        if (X_pending := getattr(acq, "X_pending", None)) is not None:
            acq.set_X_pending(None)
            self.set_X_pending(X_pending)
        else:
            acq.set_X_pending(None)
            self.set_X_pending(None)

        self.acq = acq
        self.encoder = encoder
        self.fixed_numericals = fixed_numericals

    @concatenate_pending_points  # type: ignore
    @t_batch_mode_transform()  # type: ignore
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the wrapped acquisition function on the candidate set X
        after concatenating the fixed numerical dimensions.

        Args:
            X: A `batch_shape x q x d_categorical`-dim tensor of candidates, where
                `d_categorical` is the number of categorical dimensions.

        Returns:
            A `batch_shape`-dim tensor of acquisition function values at the input
            candidates.
        """
        batch, q, c_dims = X.shape
        n_dims = len(self.fixed_numericals)
        new_X_shape = (batch, q, c_dims + n_dims)

        # Create a new tensor to hold the concatenated dimensions
        x_full: torch.Tensor = torch.empty(new_X_shape, dtype=X.dtype, device=X.device)

        # Create a mask to identify positions of categorical and numerical dimensions
        mask = torch.ones(c_dims + n_dims, dtype=torch.bool, device=X.device)
        insert_idxs = torch.tensor(list(self.fixed_numericals.keys()), device=X.device)
        mask[insert_idxs] = False

        # Fill in the fixed numerical values and the input categorical values
        for idx, val in self.fixed_numericals.items():
            x_full[:, :, idx] = val
        x_full[:, :, mask] = X

        # Pass the concatenated tensor to the original acquisition function
        return self.acq(x_full)
