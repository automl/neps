"""This module provides most of the functionality we require in NePS for now,
i.e., we need the ability to apply an arbitrary weight to an acquisition function.

I spent some time understanding the meaning of the various dimensions of botorch/gpytorch.

The two primary dimensions to consider are:

* `d` - The dimensionality of the design space, i.e. how many hyperparameters.
* `batch` - The number of independent evaluations to make, i.e. how many times to
    evaluate the acquisition function.

There are two extra dimensions which are special cases and need to be accounted for.

* `q` - Comes from the `qXXX` variants of acquisition, these will add an extra dimension
    `q` to each `batch`, where instead of a `batch` representing a single config to get
    the acquisition of, we might instead be getting the acquisition of 5 configs together,
    representing the joint utility of evaluating these 5 configs, relative to other sets
    of 5 configs. This dimension is _reduced_ away in the final step of the acquisition
    when suggesting which set of group of 5 configs to suggest.

* `mc_samples` - Comes from the `SampleReducdingXXX` variants of acquisition, will add an
    extra dimension `mc_samples` which represent the amount of Monte Carlo samples used
    to estimate the acquisition. These will eventually be _reduced_ away but are present
    in the intermediate steps. These variants also seem to have `q` variants implicitly
    and so you are likely to see the `q` dimension whever you see the `mc_samples`
    dimension, even if it is just `q=1`.

* `m` - The number of objectives in the multi-objective case. We will
    specifically ignore this for now, however it exists as the last dimension (after `d`)
    and is the first to be reduced away. They are also used in _constrainted_ settings
    which we will also ignore for now.

The most expanded tensor shape is the following, with the usual order of reduction being
the following below. If you are not using a SamplingReducing variant, you will not see
`mc_samples` and if you are not using a `q` variant, you will not see `q`. The simplest
case then being `acq(tensor: batch x d)`.

* `batch x q x d`.
        reduce(..., d) = Config -> Single number  (!!!Acq applies here!!!)
* `batch x q`.
        expand(mc_samples , ...) = MC Sampling from posterior (I think)
* `mc_samples x batch x q`.
        reduce(..., q) = Joint-Config-Group -> Single number.
* `mc_samples x batch`
        reduce(mc_samples, ...) = MC-samples -> statistical estimate
* `batch`

Finally we get out a batch of values we can argmax over, used to index into either a
single configuration or a single index into a joint-group of `q` configurations.

!!! tip

    The `mc_samples` is not of concern to the `WeightedAcquisition` below, and
    broadcasting can be used, as a result, the `apply_weight` function only needs
    to be able to handle:

    * (X: batch x q x d, acq_values: batch x q, acq: A) -> batch x q

    If utilizing the configurations `X` for weighting, you effectively will want
    to reduce the `d` dimension.

As a result of this, acquisition functions need to be able to handle arbitrary dimensions
and act accordingly.

This module mostly follows the structure of the
`PriorGuidedAcquisitionFunction` which weights the acquisition function by a prior.

* https://botorch.org/api/_modules/botorch/acquisition/prior_guided.html#PriorGuidedAcquisitionFunction

We use this to create a more generic `WeightedAcquisition` which follows the required
structure to make new weightings easier to implement, but also to serve as an educational
reference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from botorch.acquisition import SampleReducingMCAcquisitionFunction
from botorch.acquisition.analytic import AcquisitionFunction, t_batch_mode_transform
from botorch.acquisition.monte_carlo import concatenate_pending_points

if TYPE_CHECKING:
    from torch import Tensor

A = TypeVar("A", bound=AcquisitionFunction)


class WeightedAcquisition(AcquisitionFunction):
    """Class for weighting acquisition functions.

    Please see module docstring for more information.
    """

    def __init__(
        self,
        acq: A,
        apply_weight: Callable[[Tensor, Tensor, A], Tensor],
    ) -> None:
        """Initialize the weighted acquisition function.

        Args:
            acq: The base acquisition function.
            apply_weight: A function that takes the acquisition function values, the
                design points and the acquisition function itself and returns the
                weighted acquisition function values.

                Please see the module docstring for more information on the dimensions
                and how to handle them.
        """
        super().__init__(model=acq.model)
        # NOTE: We remove the X_pending from the base acquisition function as we will get
        # it in our own forward with `@concatenate_pending_points` and pass that forward.
        # This avoids possible duplicates. Also important to explicitly set it to None
        # even if it does not exist as otherwise the attribute does not exists -_-
        if (X_pending := getattr(acq, "X_pending", None)) is not None:
            acq.set_X_pending(None)
            self.set_X_pending(X_pending)
        else:
            acq.set_X_pending(None)
            self.set_X_pending(None)

        self.apply_weight = apply_weight
        self.acq = acq
        self._log = acq._log

    # Taken from PiBO implementation in botorch (PriorGuidedAcquisitionFunction).
    @concatenate_pending_points
    @t_batch_mode_transform()  # type: ignore
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate a weighted acquisition function on the candidate set X.

        Args:
            X: A tensor of size `batch_shape x q x d`-dim tensor of `q` `d`-dim
                design points.

        Returns:
            A tensor with the `d` dimension reduced away, representing the
            weighted acquisition function values at the given design points `X`.
        """
        if isinstance(self.acq, SampleReducingMCAcquisitionFunction):
            # shape: mc_samples x batch x q-candidates
            acq_values = self.acq._non_reduced_forward(X)
            weighted_acq_values = self.apply_weight(acq_values, X, self.acq)
            q_reduced_acq = self.acq._q_reduction(weighted_acq_values)
            sample_reduced_acq = self.acq._sample_reduction(q_reduced_acq)
            return sample_reduced_acq.squeeze(-1)

        # shape: batch x q-candidates
        acq_values = self.acq(X).unsqueeze(-1)
        weighted_acq_values = self.apply_weight(acq_values, X, self.acq)
        return weighted_acq_values.squeeze(-1)
