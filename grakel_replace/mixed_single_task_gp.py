from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import torch
from botorch.models import SingleTaskGP
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import AdditiveKernel, Kernel
from grakel_replace.torch_wl_kernel import GraphDataset, TorchWLKernel

if TYPE_CHECKING:
    from gpytorch.module import Module
    from torch import Tensor


class WLKernel(Kernel):
    """Weisfeiler-Lehman Kernel for graph similarity
    integrated into the GPyTorch framework.

    This kernel encapsulates the precomputed Weisfeiler-Lehman graph kernel matrix
    and provides it in a GPyTorch-compatible format.
    It computes either the training kernel
    or the cross-kernel between training and test graphs as needed.
    """

    def __init__(
        self,
        K_train: Tensor,
        wl_kernel: TorchWLKernel,
        train_graph_dataset: GraphDataset
    ) -> None:
        super().__init__()
        self._K_train = K_train
        self._wl_kernel = wl_kernel
        self._train_graph_dataset = train_graph_dataset

    def forward(
        self, x1: Tensor,
        x2: Tensor | None = None,
        diag: bool = False,
        last_dim_is_batch: bool = False
    ) -> Tensor:
        """Forward method to compute the kernel matrix for the graph inputs.

        Args:
            x1 (Tensor): First input tensor
                (unused, required for interface compatibility).
            x2 (Tensor | None): Second input tensor.
                If None, computes the training kernel matrix.
            diag (bool): Whether to return only the diagonal of the kernel matrix.
            last_dim_is_batch (bool): Whether the last dimension is a batch dimension.

        Returns:
            Tensor: The computed kernel matrix.
        """
        if x2 is None:
            K = self._K_train
            # Handle batch dimension if present in x1
            if x1.dim() > 2:  # We have a batch dimension
                batch_size = x1.size(0)
                target_size = x1.size(1)  # This should be 11 in our case
                # Resize K to match the expected dimensions
                K = K.unsqueeze(0)  # Add batch dimension
                # Pad or interpolate K to match target size
                if K.size(1) != target_size:
                    K_resized = torch.zeros(1, target_size, target_size, dtype=K.dtype,
                                            device=K.device)
                    K_resized[:, :K.size(1), :K.size(2)] = K
                    K = K_resized
                K = K.expand(batch_size, target_size, target_size)
            return K.to(dtype=x1.dtype)

        # Similar logic for cross-kernel case
        test_dataset = GraphDataset.from_networkx(x2)
        K = self._wl_kernel(self._train_graph_dataset, test_dataset)
        if x1.dim() > 2:
            batch_size = x1.size(0)
            target_size = x1.size(1)
            if K.size(0) != target_size:
                K_resized = torch.zeros(target_size, target_size, dtype=K.dtype,
                                        device=K.device)
                K_resized[:K.size(0), :K.size(1)] = K
                K = K_resized
            K = K.unsqueeze(0).expand(batch_size, target_size, target_size)
        return K.to(dtype=x1.dtype)


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
        train_inputs (tuple[Tensor, list[nx.Graph]]): Tuple of training inputs.
        num_cat_kernel (Module | None): Kernel for numerical/categorical features.
    """

    def __init__(
        self,
        train_X: Tensor,
        train_graphs: list[nx.Graph],
        train_Y: Tensor,
        train_Yvar: Tensor | None = None,
        num_cat_kernel: Module | None = None,
        wl_kernel: TorchWLKernel | None = None,
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
        # Store graphs as part of train_inputs for using them in the __call__ method
        self.train_inputs = (train_X, train_graphs)

        # Initialize the Weisfeiler-Lehman kernel or use a default one
        self._wl_kernel = wl_kernel or TorchWLKernel(n_iter=5, normalize=True)

        # Preprocess the training graphs into a compatible format and compute the graph
        # kernel matrix
        self._train_graph_dataset = GraphDataset.from_networkx(train_graphs)
        self._K_train = self._wl_kernel(self._train_graph_dataset)

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
        if graphs is None:  # Use stored graphs from train_inputs if not provided
            graphs = self.train_inputs[1]
        return self.forward(X, graphs)

    def forward(self, X: Tensor, graphs: list[nx.Graph]) -> MultivariateNormal:
        """Forward pass to compute the Gaussian Process distribution for given inputs.

        This combines the numerical/categorical kernel with the graph kernel
        to compute the joint covariance matrix.

        Args:
            X (Tensor): Input tensor for numerical and categorical features.
            graphs (list[nx.Graph]): List of input graphs.

        Returns:
            MultivariateNormal: The Gaussian Process distribution for the inputs.
        """
        if len(X) != len(graphs):
            raise ValueError(
                f"Number of feature vectors ({len(X)}) must match "
                f"number of graphs ({len(graphs)})"
            )
        if not all(isinstance(g, nx.Graph) for g in graphs):
            raise TypeError("Expected input type is a list of NetworkX graphs.")

            # Process the new graph inputs into a compatible dataset
        proc_graphs = GraphDataset.from_networkx(graphs)

        # Compute the kernel matrix for the new graphs
        K_new = self._wl_kernel(proc_graphs)
        K_new = K_new.to(dtype=X.dtype)

        # Combine the graph kernel with the numerical/categorical kernel (if present)
        if self.num_cat_kernel is not None:
            K_num_cat = self.num_cat_kernel(X)

            # Ensure K_new matches K_num_cat dimensions
            if K_num_cat.dim() > 2:
                batch_size = K_num_cat.size(0)
                target_size = K_num_cat.size(1)

                # Resize K_new if needed
                if K_new.size(-1) != target_size:
                    K_new_resized = torch.zeros(
                        *K_new.shape[:-2], target_size, target_size,
                        dtype=K_new.dtype,
                        device=K_new.device
                    )
                    K_new_resized[..., :K_new.size(-2), :K_new.size(-1)] = K_new
                    K_new = K_new_resized

                if K_new.dim() < K_num_cat.dim():
                    K_new = K_new.unsqueeze(0).expand(batch_size, target_size,
                                                      target_size)

            # Convert to dense tensor if needed
            if hasattr(K_num_cat, "to_dense"):
                K_num_cat = K_num_cat.to_dense()

            K_combined = K_num_cat + K_new
        else:
            K_combined = K_new

        # Compute the mean using the mean module and construct the GP distribution
        mean_x = self.mean_module(X)
        return MultivariateNormal(mean_x, K_combined)
