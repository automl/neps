from __future__ import annotations

from typing import TYPE_CHECKING

from botorch.models import SingleTaskGP
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import AdditiveKernel
from gpytorch.module import Module
from grakel_replace.torch_wl_kernel import GraphDataset, TorchWLKernel

if TYPE_CHECKING:
    import networkx as nx
    from torch import Tensor


class MixedSingleTaskGP(SingleTaskGP):
    """A Gaussian Process model that handles numerical, categorical, and graph inputs.

    This class extends BoTorch's SingleTaskGP to work with hybrid input spaces containing:
    - Numerical features
    - Categorical features
    - Graph structures

    It uses the Weisfeiler-Lehman (WL) kernel for graph inputs and combines it with
    standard kernels for numerical/categorical features using an additive kernel structure

    Attributes:
        _wl_kernel (TorchWLKernel): The Weisfeiler-Lehman kernel for graph similarity
        _train_graphs (List[nx.Graph]): Training set graph instances
        _K_graph (Tensor): Pre-computed graph kernel matrix for training data
        num_cat_kernel (Optional[Module]): Kernel for numerical/categorical features
    """

    def __init__(
        self,
        train_X: Tensor,  # Shape: (n_samples, n_numerical_categorical_features)
        train_graphs: list[nx.Graph],  # List of n_samples graph instances
        train_Y: Tensor,  # Shape: (n_samples, n_outputs)
        train_Yvar: Tensor | None = None,  # Shape: (n_samples, n_outputs) or None
        num_cat_kernel: Module | None = None,
        wl_kernel: TorchWLKernel | None = None,
        **kwargs  # Additional arguments passed to SingleTaskGP
    ) -> None:
        """Initialize the mixed input Gaussian Process model.

        Args:
            train_X: Training data tensor for numerical and categorical features
            train_graphs: List of training graphs
            train_Y: Target values
            train_Yvar: Observation noise variance (optional)
            num_cat_kernel: Kernel for numerical/categorical features (optional)
            wl_kernel: Custom Weisfeiler-Lehman kernel instance (optional)
            **kwargs: Additional arguments for SingleTaskGP initialization
        """
        # Initialize parent class with initial covar_module
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=num_cat_kernel or self._graph_kernel_wrapper(),
            **kwargs
        )

        # Initialize WL kernel with default parameters if not provided
        self._wl_kernel = wl_kernel or TorchWLKernel(n_iter=5, normalize=True)
        self._train_graphs = train_graphs

        # Convert graphs to required format and compute kernel matrix
        self._train_graph_dataset = GraphDataset.from_networkx(train_graphs)
        self._K_train = self._wl_kernel(self._train_graph_dataset)

        if num_cat_kernel is not None:
            # Create additive kernel combining numerical/categorical and graph kernels
            combined_kernel = AdditiveKernel(
                num_cat_kernel,
                self._graph_kernel_wrapper()
            )
            self.covar_module = combined_kernel

        self.num_cat_kernel = num_cat_kernel

    def _graph_kernel_wrapper(self) -> Module:
        """Creates a GPyTorch-compatible kernel module wrapping the WL kernel.

        This wrapper allows the WL kernel to be used within the GPyTorch framework
        by providing a forward method that returns the pre-computed kernel matrix.

        Returns:
            Module: A GPyTorch kernel module wrapping the WL kernel computation
        """

        class WLKernelWrapper(Module):
            def __init__(self, parent: MixedSingleTaskGP):
                super().__init__()
                self.parent = parent

            def forward(
                self,
                x1: Tensor,
                x2: Tensor | None = None,
                diag: bool = False,
                last_dim_is_batch: bool = False
            ) -> Tensor:
                """Compute the kernel matrix for the graph inputs.

                Args:
                    x1: First input tensor (unused, required for interface compatibility)
                    x2: Second input tensor (must be None)
                    diag: Whether to return only diagonal elements
                    last_dim_is_batch: Whether the last dimension is a batch dimension

                Returns:
                    Tensor: Pre-computed graph kernel matrix

                Raises:
                    NotImplementedError: If x2 is not None (cross-covariance not implemented)
                """
                if x2 is None:
                    return self.parent._K_train

                # Compute cross-covariance between train and test graphs
                test_dataset = GraphDataset.from_networkx(self.parent._test_graphs)
                return self.parent._wl_kernel(
                    self.parent._train_graph_dataset,
                    test_dataset
                )

        return WLKernelWrapper(self)

    def forward(self, X: Tensor, graphs: list[nx.Graph]) -> MultivariateNormal:
        """Forward pass computing the GP distribution for given inputs.

        Computes the kernel matrix for both numerical/categorical features and graphs,
        combines them if both are present, and returns the resulting GP distribution.

        Args:
            X: Input tensor for numerical and categorical features
            graphs: List of input graphs

        Returns:
            MultivariateNormal: GP distribution for the given inputs
        """
        if len(X) != len(graphs):
            raise ValueError(
                f"Number of feature vectors ({len(X)}) must match "
                f"number of graphs ({len(graphs)})"
            )

        # Process new graphs and compute kernel matrix
        proc_graphs = GraphDataset.from_networkx(graphs)
        K_new = self._wl_kernel(proc_graphs)  # Shape: (n_samples, n_samples)

        # If we have both numerical/categorical and graph features
        if self.num_cat_kernel is not None:
            # Compute kernel for numerical/categorical features
            K_num_cat = self.num_cat_kernel(X)
            # Add the kernels (element-wise addition)
            K_combined = K_num_cat + K_new
        else:
            K_combined = K_new

        # Compute mean using the mean module
        mean_x = self.mean_module(X)

        return MultivariateNormal(mean_x, K_combined)
