from __future__ import annotations

from typing import Any

import networkx as nx
import torch
from botorch.models.gp_regression_mixed import Kernel


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
        x1: torch.Tensor,
        x2: torch.Tensor,
        *,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ):
        if last_dim_is_batch:
            raise NotImplementedError("TODO: Figure this out")

        assert x1.shape[-1] == 1, "Last dimension must be the graph index"
        assert x2.shape[-1] == 1, "Last dimension must be the graph index"

        x1_is_x2 = torch.equal(x1, x2)

        if x1.ndim == 3:
            q_dim_size = x1.shape[0]
            assert x2.shape[0] == q_dim_size

            out = torch.empty((q_dim_size, x1.shape[1], x2.shape[1]), device=x1.device)
            for q in range(q_dim_size):
                out[q] = self.forward(x1[q], x2[q], diag=diag)
            return out

        if x1_is_x2:
            indices = x1.flatten().to(torch.int64).tolist()
            all_graphs = indices
            select = None
        else:
            indices1 = x1.flatten().to(torch.int64).tolist()
            indices2 = x2.flatten().to(torch.int64).tolist()
            all_graphs = indices1 + indices2
            select = lambda K: K[: len(indices1), len(indices1):]

        # Handle the special case for -1
        all_graphs = [
            len(self.graph_lookup) - 1 if i == -1 else i for i in all_graphs
        ]

        # Use cached adjacency matrices and labels
        adj_matrices = [self.adjacency_cache[i] for i in all_graphs]
        label_tensors = [self.label_cache[i] for i in all_graphs]

        _kernel = _TorchWLKernel(n_iter=self.n_iter, normalize=self.normalize)
        K = _kernel(adj_matrices, label_tensors)
        K_selected = K if select is None else select(K)
        if diag:
            return torch.diag(K_selected)
        return K_selected

    def _get_sparse_adj(self, graph: nx.Graph) -> torch.sparse.Tensor:
        """Convert a NetworkX graph to a sparse adjacency tensor."""
        edges = list(graph.edges())
        num_nodes = graph.number_of_nodes()

        if not edges:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty(0),
                size=(num_nodes, num_nodes),
                device=torch.device("cpu"),
            )

        edge_indices: list[tuple[int, int]] = edges + [(v, u) for u, v in edges]
        rows, cols = zip(*edge_indices, strict=False)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.ones(len(edge_indices), dtype=torch.float)

        return torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), device=torch.device("cpu")
        )

    def _init_node_labels(self, graph: nx.Graph) -> torch.Tensor:
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

        return torch.tensor(labels, dtype=torch.long)


class _TorchWLKernel(torch.nn.Module):
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
        self.device: torch.device = torch.device("cpu")
        self.label_dict: dict[tuple, int] = {}
        self.label_counter: int = 0

    def _wl_iteration(self, adj: torch.sparse.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Perform one WL iteration to update node labels.
        Concatenate own label with sorted neighbor labels.

        Args:
            adj: Sparse adjacency matrix
            labels: Current node label tensor

        Returns:
            Updated node label tensor
        """
        # Ensure the adjacency matrix is in COO format
        adj = adj.coalesce()
        indices = adj.indices()
        adj.values()

        # Get the neighbors for each node
        rows, cols = indices
        neighbors = cols[rows]

        # Create a list of combined labels for each node
        combined_labels = []
        for node in range(labels.size(0)):
            node_neighbors = neighbors[rows == node]
            node_neighbor_labels = labels[node_neighbors]
            combined_label = (labels[node].item(), tuple(node_neighbor_labels.tolist()))
            combined_labels.append(combined_label)

        # Update the label dictionary and counter
        new_labels = torch.empty_like(labels)
        for i, label in enumerate(combined_labels):
            if label not in self.label_dict:
                self.label_dict[label] = self.label_counter
                self.label_counter += 1
            new_labels[i] = self.label_dict[label]

        return new_labels

    def _compute_feature_vector(self, labels: torch.Tensor, size: int) -> torch.Tensor:
        """Compute histogram feature vector from node labels.

        Args:
            labels: Node label tensor
            size: Size of the feature vector

        Returns:
            Feature vector representing label distribution
        """
        feature = torch.zeros(size, device=self.device, dtype=torch.float32)
        unique, counts = torch.unique(labels, return_counts=True)
        feature[unique] = counts.to(dtype=feature.dtype)
        return feature

    def forward(
        self,
        adj_matrices: list[torch.sparse.Tensor],
        label_tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute WL kernel matrix for a list of graphs.

        Args:
            adj_matrices: Precomputed sparse adjacency matrices for graphs.
            label_tensors: Precomputed node label tensors for graphs.

        Returns:
            Kernel matrix containing pairwise graph similarities.
        """
        # Validate inputs
        if len(adj_matrices) != len(label_tensors):
            raise ValueError("Mismatch between adjacency matrices and label tensors.")

        # Perform WL iterations to update node labels
        all_labels = [label_tensors]
        for _ in range(self.n_iter):
            new_labels = [
                self._wl_iteration(adj, labels)
                for adj, labels in zip(adj_matrices, all_labels[-1], strict=False)
            ]
            all_labels.append(new_labels)

        # Compute feature vectors for each graph at each iteration
        feature_vectors = []
        label_counter = max(
            max(labels.max().item() for labels in label_set)
            for label_set in all_labels
        ) + 1
        for iteration_labels in all_labels:
            feature_vectors.append(
                torch.stack(
                    [
                        self._compute_feature_vector(labels, label_counter)
                        for labels in iteration_labels
                    ]
                )
            )

        # Combine features from all iterations
        final_features = torch.stack(feature_vectors).sum(dim=0)

        # Compute kernel matrix (similarity matrix)
        kernel_matrix = torch.mm(final_features, final_features.t())

        # Apply normalization if requested
        if self.normalize:
            diag = torch.sqrt(torch.diag(kernel_matrix))
            kernel_matrix /= diag.unsqueeze(0) * diag.unsqueeze(1)

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
