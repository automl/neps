from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from botorch.models.gp_regression_mixed import Kernel
from grakel_replace.utils import graphs_to_tensors
from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    import networkx as nx


class BoTorchWLKernel(Kernel):
    has_lengthscale = False

    def __init__(
        self,
        graph_lookup: list[nx.Graph],
        n_iter: int = 5,
        *,
        normalize: bool = True,
        active_dims: tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(active_dims=active_dims, **kwargs)
        self.graph_lookup = graph_lookup
        self.n_iter = n_iter
        self.normalize = normalize

        self._init_caches()
        self._precompute_graph_data()

    def _init_caches(self) -> None:
        """Initialize cache dictionaries."""
        self.adjacency_cache = {}
        self.label_cache = {}
        self.cache = {}

    def _precompute_graph_data(self) -> None:
        """Precompute and cache adjacency matrices and initial node labels."""
        self._init_caches()
        self.adjacency_cache, self.label_cache = graphs_to_tensors(
            self.graph_lookup, device=self.device
        )

    def set_graph_lookup(self, graph_lookup: list[nx.Graph]) -> None:
        """Update the graph lookup and refresh the cached data."""
        self.graph_lookup = graph_lookup
        self._precompute_graph_data()

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        *,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        x1_is_x2 = torch.equal(x1, x2)
        indices = tuple(x1.flatten().tolist()) if x1_is_x2 else (
            tuple(x1.flatten().tolist()), tuple(x2.flatten().tolist()))
        if indices in self.cache:
            return self.cache[indices]

        # Compute kernel matrix if not cached
        K = self._compute_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch)
        self.cache[indices] = K
        return K

    def _compute_kernel(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Compute the kernel matrix.

        Args:
            x1 (Tensor): Tensor of indices for the first set of graphs
            x2 (Tensor): Tensor of indices for the second set of graphs
            diag (bool): If True, only computes the diagonal of the kernel matrix
            last_dim_is_batch (bool): Whether the last dimension represents batch size

        Returns:
            Tensor: Kernel matrix containing pairwise similarities between graphs
        """
        if last_dim_is_batch:
            raise NotImplementedError("Batch dimension handling is not implemented.")

        # Handle batched input if present
        if x1.ndim == 3:
            return self._handle_batched_input(x1, x2, diag)

        # Convert indices to integer lists and handle special cases
        indices1, indices2 = self._prepare_indices(x1, x2)

        # Check if we're computing self-similarity or cross-similarity
        if torch.equal(x1, x2):
            return self._compute_self_kernel(indices1, diag)
        else:
            return self._compute_cross_kernel(indices1, indices2, diag)

    def _handle_batched_input(self, x1: Tensor, x2: Tensor, diag: bool) -> Tensor:
        """Handle computation for batched input tensors."""
        q_dim_size = x1.shape[0]
        assert x2.shape[0] == q_dim_size

        out = torch.empty((q_dim_size, x1.shape[1], x2.shape[1]), device=x1.device)
        for q in range(q_dim_size):
            out[q] = self._compute_kernel(x1[q], x2[q], diag=diag)
        return out

    def _prepare_indices(self, x1: Tensor, x2: Tensor) -> tuple[list[int], list[int]]:
        """Convert tensor indices to integer lists and handle special cases."""
        indices1 = x1.flatten().round().to(torch.int64).tolist()
        indices2 = x2.flatten().round().to(torch.int64).tolist()

        # Handle special case for -1 index
        if -1 in indices1 or -1 in indices2:
            self._handle_negative_one_index()

        return indices1, indices2

    def _handle_negative_one_index(self) -> None:
        """Handle the special case where -1 index is present."""
        # Check if -1 is a valid index in the list
        if -1 < 0 or -1 >= len(self.adjacency_cache):
            last_graph_idx = len(self.graph_lookup) - 1
            # Append the last graph's adjacency matrix and labels to the cache
            self.adjacency_cache.append(self.adjacency_cache[last_graph_idx])
            self.label_cache.append(self.label_cache[last_graph_idx])

    def _compute_self_kernel(self, indices: list[int], diag: bool) -> Tensor:
        """Compute kernel matrix for a self-similarity case."""
        indices_tuple = tuple(indices)
        if indices_tuple in self.cache:
            return self.cache[indices_tuple]

        # Get precomputed data for the indices
        adj_matrices = [self.adjacency_cache[i] for i in indices]
        label_tensors = [self.label_cache[i] for i in indices]

        # Compute kernel matrix
        K = self._compute_base_kernel(adj_matrices, label_tensors)
        if diag:
            K = torch.diag(K)

        self.cache[indices_tuple] = K
        return K

    def _compute_cross_kernel(
        self,
        indices1: list[int],
        indices2: list[int],
        diag: bool
    ) -> Tensor:
        """Compute kernel matrix for cross-similarity case."""
        cache_key = (tuple(indices1), tuple(indices2))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute unique set of graphs needed
        all_graphs = list(set(indices1 + indices2))
        adj_matrices = [self.adjacency_cache[i] for i in all_graphs]
        label_tensors = [self.label_cache[i] for i in all_graphs]

        # Compute full kernel matrix
        K_full = self._compute_base_kernel(adj_matrices, label_tensors)

        # Extract relevant submatrix
        idx1 = [all_graphs.index(i) for i in indices1]
        idx2 = [all_graphs.index(i) for i in indices2]
        K = K_full[idx1][:, idx2]
        if diag:
            K = torch.diag(K)

        self.cache[cache_key] = K
        return K

    def _compute_base_kernel(self, adj_matrices: list[Tensor],
                             label_tensors: list[Tensor]) -> Tensor:
        """Compute the base kernel matrix using WL algorithm."""
        _kernel = TorchWLKernel(n_iter=self.n_iter, normalize=self.normalize)
        return _kernel(adj_matrices, label_tensors)


class TorchWLKernel(Module):
    """A custom implementation of Weisfeiler-Lehman (WL) Kernel in PyTorch.

    The WL Kernel is a graph kernel that measures similarity between graphs based on
    their structural properties. It works by iteratively updating node labels based on
    their neighborhoods and computing feature vectors from label distributions.

    Args:
        n_iter: Number of WL iterations to perform
        normalize: bool, optional. Whether to normalize the kernel matrix

    Attributes:
        device: torch.device for computation (CPU/GPU)
        label_dict: Mapping from node labels to numerical indices
        label_counter: Counter for generating new label indices
    """

    def __init__(self, n_iter: int = 5, *, normalize: bool = True) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.normalize = normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Keep track of labels across iterations
        self.label_dict = {}
        self.label_counter = 0

    def _get_node_neighbors(self, adj: Tensor) -> list[list[int]]:
        """Extract neighborhood information from adjacency matrix."""
        if adj.layout == torch.sparse_csr:
            adj = adj.to_sparse_coo()

        adj = adj.coalesce()
        rows, cols = adj.indices()
        num_nodes = adj.size(0)

        neighbors = [[] for _ in range(num_nodes)]
        for row, col in zip(rows.tolist(), cols.tolist(), strict=False):
            neighbors[row].append(col)

        return neighbors

    def _wl_iteration(self, adj: Tensor, labels: Tensor) -> Tensor:
        """Perform one WL iteration."""
        if not self.label_dict:
            # Start new labels after initial ones
            self.label_counter = labels.max().item() + 1

        num_nodes = labels.size(0)
        new_labels = []
        neighbors = self._get_node_neighbors(adj)

        for node_idx in range(num_nodes):
            # Get current node label
            node_label = labels[node_idx].item()
            neighbor_labels = sorted([labels[n].item() for n in neighbors[node_idx]])

            credential = f"{node_label},{neighbor_labels}"

            # Update label dictionary
            if credential not in self.label_dict:
                self.label_dict[credential] = self.label_counter
                self.label_counter += 1

            new_labels.append(self.label_dict[credential])

        return torch.tensor(new_labels, dtype=torch.long, device=self.device)

    def _compute_feature_vector(self, all_labels: list[list[Tensor]]) -> Tensor:
        """Compute the histogram feature vector for all graphs."""
        batch_size = len(all_labels[0])
        features = []

        for iteration_labels in all_labels:
            # Find maximum label value across all graphs in this iteration
            max_label = max(label.max().item() for label in iteration_labels) + 1

            iter_features = torch.zeros((batch_size, max_label),
                                        dtype=torch.float32,
                                        device=self.device)

            # Compute label frequencies
            for graph_idx, labels in enumerate(iteration_labels):
                counts = torch.bincount(labels, minlength=max_label).float()
                iter_features[graph_idx] = counts

            features.append(iter_features)

        return torch.cat(features, dim=1)

    def forward(self, adj_matrices: list[Tensor], label_tensors: list[Tensor]) -> Tensor:
        """Compute WL kernel matrix for a list of graphs.

        Args:
            adj_matrices: Precomputed sparse adjacency matrices for graphs.
            label_tensors: Precomputed node label tensors for graphs.

        Returns:
            Kernel matrix containing pairwise graph similarities.
        """
        if len(adj_matrices) != len(label_tensors):
            raise ValueError("Mismatch between adjacency matrices and label tensors.")

        # Reset label dictionary for new computation
        self.label_dict = {}
        # Store all label iterations
        all_labels = [label_tensors]

        # Perform WL iterations
        for _ in range(self.n_iter):
            new_labels = [
                self._wl_iteration(adj, labels)
                for adj, labels in zip(adj_matrices, all_labels[-1], strict=False)
            ]
            all_labels.append(new_labels)

        # Compute feature vectors and kernel matrix (similarity matrix)
        final_features = self._compute_feature_vector(all_labels)
        kernel_matrix = torch.mm(final_features, final_features.t())

        if self.normalize:
            diag = torch.sqrt(torch.diag(kernel_matrix))
            kernel_matrix /= torch.outer(diag, diag)

        return kernel_matrix
