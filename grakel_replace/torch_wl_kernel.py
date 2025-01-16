from __future__ import annotations

from typing import Any

import networkx as nx
import torch
from botorch.models.gp_regression_mixed import Kernel
from torch import Tensor
from torch.nn import Module


class TorchWLKernel(Kernel):
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

        # Cache adjacency matrices and initial node labels
        self.adjacency_cache = {}
        self.label_cache = {}
        self.cache = {}

        self._precompute_graph_data()

    def _precompute_graph_data(self) -> None:
        """Precompute adjacency matrices and initial node labels for all graphs."""
        self.adjacency_cache = {}
        self.label_cache = {}

        for idx, graph in enumerate(self.graph_lookup):
            self.adjacency_cache[idx] = self._get_sparse_adj(graph)
            self.label_cache[idx] = self._init_node_labels(graph)

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
        """
        Compute the kernel matrix efficiently with caching and minimal redundant computations.

        Args:
            x1 (Tensor): Tensor of indices for the first set of graphs.
            x2 (Tensor): Tensor of indices for the second set of graphs.
            diag (bool, optional): If True, only computes the diagonal of the kernel matrix. Defaults to False.
            last_dim_is_batch (bool, optional): Whether the last dimension represents batch size. Defaults to False.

        Returns:
            Tensor: Kernel matrix containing pairwise similarities between graphs.
        """
        if last_dim_is_batch:
            raise NotImplementedError("Batch dimension handling is not implemented.")

        x1_is_x2 = torch.equal(x1, x2)

        if x1.ndim == 3:
            # Handle batched input
            q_dim_size = x1.shape[0]
            assert x2.shape[0] == q_dim_size

            out = torch.empty((q_dim_size, x1.shape[1], x2.shape[1]), device=x1.device)
            for q in range(q_dim_size):
                out[q] = self._compute_kernel(x1[q], x2[q], diag=diag)
            return out

        # Ensure indices are integers
        indices1 = x1.flatten().round().to(torch.int64).tolist()
        indices2 = x2.flatten().round().to(torch.int64).tolist()

        # Handle the special case for -1
        if -1 in indices1 or -1 in indices2:
            if -1 not in self.adjacency_cache:
                # Map -1 to the last graph in the graph lookup
                last_graph_idx = len(self.graph_lookup) - 1
                self.adjacency_cache[-1] = self.adjacency_cache[last_graph_idx]
                self.label_cache[-1] = self.label_cache[last_graph_idx]

        if x1_is_x2:
            indices = tuple(indices1)
            if indices in self.cache:
                return self.cache[indices]

            # Compute pairwise kernel
            adj_matrices = [self.adjacency_cache[i] for i in indices]
            label_tensors = [self.label_cache[i] for i in indices]

            _kernel = _TorchWLKernel(n_iter=self.n_iter, normalize=self.normalize)
            K = _kernel(adj_matrices, label_tensors)
            if diag:
                K = torch.diag(K)

            # Cache the result
            self.cache[indices] = K
            return K
        else:
            cache_key = (tuple(indices1), tuple(indices2))
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Compute cross-kernel
            all_graphs = list(set(indices1 + indices2))
            adj_matrices = [self.adjacency_cache[i] for i in all_graphs]
            label_tensors = [self.label_cache[i] for i in all_graphs]

            _kernel = _TorchWLKernel(n_iter=self.n_iter, normalize=self.normalize)
            K_full = _kernel(adj_matrices, label_tensors)

            # Extract the relevant submatrix
            idx1 = [all_graphs.index(i) for i in indices1]
            idx2 = [all_graphs.index(i) for i in indices2]
            K = K_full[idx1][:, idx2]
            if diag:
                K = torch.diag(K)

            # Cache the result
            self.cache[cache_key] = K
            return K

    def _get_sparse_adj(self, graph: nx.Graph) -> Tensor:
        """Convert a NetworkX graph to a sparse adjacency tensor."""
        edges = list(graph.edges())
        num_nodes = graph.number_of_nodes()

        if not edges:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty(0),
                size=(num_nodes, num_nodes),
                device=self.device,
            )

        edge_indices: list[tuple[int, int]] = edges + [(v, u) for u, v in edges]
        rows, cols = zip(*edge_indices, strict=False)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(edge_indices), dtype=torch.float)

        return torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), device=self.device
        ).to_sparse_csr()  # Convert to CSR for efficient operations

    def _init_node_labels(self, graph: nx.Graph) -> Tensor:
        """Initialize node label tensor from graph attributes."""
        labels: list[int] = []
        label_dict: dict[str, int] = {}
        label_counter = 0

        for node in range(graph.number_of_nodes()):
            if "label" in graph.nodes[node]:
                label = graph.nodes[node]["label"]
            else:
                label = str(node)
            if label not in label_dict:
                label_dict[label] = label_counter
                label_counter += 1
            labels.append(label_dict[label])

        return torch.tensor(labels, dtype=torch.long, device=self.device)


class _TorchWLKernel(Module):
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
        self.label_dict: dict[tuple, int] = {}
        self.label_counter: int = 0
        self.cache = {}
        self.hash_module = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.normal_(self.hash_module.weight)

    def _wl_iteration(self, adj: Tensor, labels: Tensor) -> Tensor:
        """Perform one iteration of the WL algorithm to update node labels."""
        # Ensure the adjacency matrix is in COO format before coalescing
        if adj.layout == torch.sparse_csr:
            adj = adj.to_sparse_coo()

        adj = adj.coalesce()
        indices = adj.indices()
        rows, cols = indices
        num_nodes = labels.size(0)

        # Create a unique key for caching
        cache_key = (
            adj.indices().cpu().numpy().tobytes(), labels.cpu().numpy().tobytes())

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Create a mask for each node's neighbors
        neighbor_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool,
                                    device=self.device)
        neighbor_mask[rows, cols] = True

        # Get neighbor labels for each node
        neighbor_labels = labels.unsqueeze(0).expand(num_nodes, -1)
        neighbor_labels = neighbor_labels.masked_fill(~neighbor_mask, -1)

        # Sort neighbor labels for each node
        sorted_neighbor_labels, _ = torch.sort(neighbor_labels, dim=1, descending=True)

        # Remove padding (-1 values) from sorted labels
        valid_neighbors_mask = sorted_neighbor_labels != -1
        max_neighbors = valid_neighbors_mask.sum(1).max().item()
        sorted_neighbor_labels = sorted_neighbor_labels[:, :max_neighbors]

        # Combine node labels with neighbor labels
        node_labels_expanded = labels.unsqueeze(1).expand(-1, max_neighbors)

        # Create feature vectors
        features = torch.cat([
            node_labels_expanded.unsqueeze(-1).float(),
            sorted_neighbor_labels.unsqueeze(-1).float()
        ], dim=-1)

        # Hash the combined features
        hashed_features = self.hash_module(features).squeeze(-1)
        hashed_labels = hashed_features.sum(dim=1)

        # Convert to discrete labels
        _, new_labels = torch.unique(hashed_labels, sorted=True, return_inverse=True)

        # Cache the result
        self.cache[cache_key] = new_labels
        return new_labels

    def _compute_feature_vector(self, all_labels: list[list[Tensor]]) -> Tensor:
        """Compute feature vectors for all graphs in a batch."""
        max_label = 0
        for iteration_labels in all_labels:
            for labels in iteration_labels:
                max_label = max(max_label, labels.max().item())

        batch_size = len(all_labels[0])
        features = torch.zeros((batch_size, max_label + 1),
                               dtype=torch.float32, device=self.device)

        # Accumulate label counts for each graph
        for graph_idx in range(batch_size):
            # Sum contributions from all WL iterations
            for iteration_labels in all_labels:
                graph_labels = iteration_labels[graph_idx]
                label_counts = torch.bincount(
                    graph_labels,
                    minlength=max_label + 1
                ).float()
                features[graph_idx] += label_counts

        return features

    def forward(
        self,
        adj_matrices: list[Tensor],
        label_tensors: list[Tensor],
    ) -> Tensor:
        """Compute WL kernel matrix for a list of graphs.

        Args:
            adj_matrices: Precomputed sparse adjacency matrices for graphs.
            label_tensors: Precomputed node label tensors for graphs.

        Returns:
            Kernel matrix containing pairwise graph similarities.
        """
        if len(adj_matrices) != len(label_tensors):
            raise ValueError("Mismatch between adjacency matrices and label tensors.")

        # Perform WL iterations to update the node labels
        all_labels = [label_tensors]
        for _ in range(self.n_iter):
            new_labels = [
                self._wl_iteration(adj, labels)
                for adj, labels in zip(adj_matrices, all_labels[-1], strict=False)
            ]
            all_labels.append(new_labels)

        # Compute feature vectors for each graph in the batch
        final_features = self._compute_feature_vector(all_labels)

        # Compute kernel matrix (similarity matrix)
        kernel_matrix = torch.mm(final_features, final_features.t())

        # Apply normalization if requested
        if self.normalize:
            diag = torch.sqrt(torch.diag(kernel_matrix))
            kernel_matrix /= (diag.unsqueeze(0) * diag.unsqueeze(1))

        return kernel_matrix


class GraphDataset:
    """Utility class to convert NetworkX graphs for WL kernel."""

    @staticmethod
    def from_networkx(
        graphs: list[nx.Graph], node_labels_tag: str = "label"
    ) -> list[nx.Graph]:
        if not all(isinstance(g, nx.Graph) for g in graphs):
            raise TypeError("Expected input type is a list of NetworkX graphs.")

        """Convert NetworkX graphs ensuring proper node labeling."""
        processed_graphs = []
        for g in graphs:
            g = g.copy()
            # Add default labels if not present
            for node in g.nodes():
                if node_labels_tag not in g.nodes[node]:
                    g.nodes[node][node_labels_tag] = str(node)
            processed_graphs.append(g)
        return processed_graphs
