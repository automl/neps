from __future__ import annotations

from typing import TYPE_CHECKING

from botorch.models import SingleTaskGP
from gpytorch.kernels import AdditiveKernel, Kernel
from grakel_replace.torch_wl_kernel import GraphDataset, TorchWLKernel

if TYPE_CHECKING:
    import networkx as nx
    from torch import Tensor


class MixedSingleTaskGP(SingleTaskGP):
    """A Gaussian Process model for mixed input spaces containing numerical, categorical,
    and graph features.

    This class extends BoTorch's SingleTaskGP to support hybrid inputs by combining:
    - Standard kernels for numerical and categorical features.
    - Weisfeiler-Lehman kernel for graph structures.

    The kernels are combined using an additive kernel structure.

    Attributes:
        _wl_kernel (TorchWLKernel): Instance of the Weisfeiler-Lehman kernel.
        _K_train (Tensor): Precomputed graph kernel matrix for training graphs.
        _train_inputs (tuple[Tensor, list[nx.Graph]]): Tuple of training inputs.
        num_cat_kernel (Module | None): Kernel for numerical/categorical features.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_graphs: list[nx.Graph],
        train_Y: Tensor,
        num_cat_kernel: Kernel,
        wl_kernel: TorchWLKernel,
        train_Yvar: Tensor | None = None,
        **kwargs,
    ) -> None:
        """Initialize the mixed-input Gaussian Process model.

        Args:
            train_X (Tensor): Training tensor for numerical and categorical features.
            train_graphs (list[nx.Graph]): List of training graph instances.
            train_Y (Tensor): Target values for training data.
            train_Yvar (Tensor | None): Observation noise variance (optional).
            num_cat_kernel (Module | None): Kernel for numerical/categorical features
                (optional).
            wl_kernel (TorchWLKernel | None): Weisfeiler-Lehman kernel instance
                (optional).
            **kwargs: Additional arguments for SingleTaskGP initialization.
        """
        if train_X.size(0) == 0 or len(train_graphs) == 0:
            raise ValueError("Training inputs (features and graphs) cannot be empty.")

        # Initialize the base SingleTaskGP with a num/cat kernel (if provided)
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=num_cat_kernel,
            **kwargs,
        )
        # Initialize the Weisfeiler-Lehman kernel or use a default one
        self._wl_kernel = wl_kernel

        # Preprocess the training graphs into a compatible format and compute the graph
        # kernel matrix
        self._train_graph_dataset = GraphDataset.from_networkx(train_graphs)
        self._K_train = self._wl_kernel(self._train_graph_dataset)

        # Store graphs as part of train_inputs for using them in the __call__ method
        self._train_inputs = (train_X, train_graphs)

        # If a kernel for numerical/categorical features is provided, combine it with
        # the WL kernel
        if num_cat_kernel is not None:
            self.covar_module = AdditiveKernel(
                num_cat_kernel,
                WLKernel(self._K_train, self._wl_kernel, self._train_graph_dataset),
            )

        self.num_cat_kernel = num_cat_kernel

    def __call__(self, X: Tensor, graphs: list[nx.Graph] | None = None, **kwargs):
        """Custom __call__ method that retrieves train graphs if not explicitly passed."""
        print("__call__", X.shape, len(graphs) if graphs is not None else None)  # noqa: T201
        if graphs is None:  # Use stored graphs from train_inputs if not provided
            graphs = self._train_inputs[1]
        return self.forward(X)
