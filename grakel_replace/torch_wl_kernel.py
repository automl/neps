from __future__ import annotations

from collections import Counter

import networkx as nx
import torch
from torch import nn


class TorchWLKernel(nn.Module):
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

    def __init__(self, n_iter: int = 5, normalize: bool = True) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.normalize = normalize
        self.device: torch.device = torch.device("cpu")
        self.label_dict: dict[str, int] = {}
        self.label_counter: int = 0

    def _get_sparse_adj(self, graph: nx.Graph) -> torch.sparse.Tensor:
        """Convert a NetworkX graph to a sparse adjacency tensor.

        Args:
            graph: Input NetworkX graph

        Returns:
            Sparse tensor representation of the graph's adjacency matrix
        """
        edges = list(graph.edges())
        num_nodes = graph.number_of_nodes()

        if not edges:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty(0),
                size=(num_nodes, num_nodes),
                device=self.device
            )

        # Create bidirectional edge indices for undirected graph
        edge_indices: list[tuple[int, int]] = edges + [(v, u) for u, v in edges]
        rows, cols = zip(*edge_indices, strict=False)

        indices = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
        values = torch.ones(len(edge_indices), dtype=torch.float, device=self.device)

        return torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes),
            device=self.device
        )

    def _init_node_labels(self, graph: nx.Graph) -> torch.Tensor:
        """Initialize node label tensor from graph attributes.

        Args:
            graph: Input NetworkX graph

        Returns:
            Tensor of numerical node label indices
        """
        labels: list[int] = []

        for node in range(graph.number_of_nodes()):
            if "label" in graph.nodes[node]:
                label = graph.nodes[node]["label"]
            else:
                label = str(node)
            if label not in self.label_dict:
                self.label_dict[label] = self.label_counter
                self.label_counter += 1
            labels.append(self.label_dict[label])

        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def _wl_iteration(
        self,
        adj: torch.sparse.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Perform one WL iteration to update node labels.
        Concatenate own label with sorted neighbor labels.

        Args:
            adj: Sparse adjacency matrix
            labels: Current node label tensor

        Returns:
            Updated node label tensor
        """
        new_labels: list[int] = []
        indices = adj.coalesce().indices()

        for node in range(adj.size(0)):
            node_label = labels[node].item()
            # Get indices of neighbors for current node
            neighbors = indices[1][indices[0] == node]
            neighbor_labels = sorted([labels[n].item() for n in neighbors])

            # Check if all neighbors have the same label as the current node
            if all(labels[n] == labels[node] for n in neighbors):
                new_labels.append(node_label)
            else:
                # Create new label from node and neighbor information
                combined_label = f"{node_label}_{neighbor_labels}"
                if combined_label not in self.label_dict:
                    self.label_dict[combined_label] = self.label_counter
                    self.label_counter += 1
                new_labels.append(self.label_dict[combined_label])

        return torch.tensor(new_labels, dtype=torch.long, device=self.device)

    def _compute_feature_vector(
        self,
        labels: torch.Tensor,
        size: int
    ) -> torch.Tensor:
        """Compute histogram feature vector from node labels.

        Args:
            labels: Node label tensor
            size: Size of the feature vector

        Returns:
            Feature vector representing label distribution
        """
        # Handle the case where all node labels are the same
        unique_labels = set(labels.cpu().numpy())
        if len(unique_labels) == 1:
            feature = torch.zeros(size, device=self.device)
            feature[labels[0].item()] = len(labels)
            return feature

        label_counts = Counter(labels.cpu().numpy())
        feature = torch.zeros(size, device=self.device)

        for label, count in label_counts.items():
            if label < size:  # Safety check
                feature[label] = count

        return feature

    def forward(self, graphs: list[nx.Graph]) -> torch.Tensor:
        """Compute WL kernel matrix for a list of graphs.

        Args:
            graphs: List of NetworkX graphs to compare

        Returns:
            Kernel matrix containing pairwise graph similarities

        Raises:
            TypeError: If input is not a list of NetworkX graphs
        """
        # Validate input
        if (not isinstance(graphs, list) or
            not all(isinstance(g, nx.Graph) for g in graphs)):
            raise TypeError("Expected input type is a list of NetworkX graphs.")

        # Setup computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_dict = {}
        self.label_counter = 0

        # Handle a case of empty graphs list or empty individual graphs
        if not graphs or all(g.number_of_nodes() == 0 for g in graphs):
            return torch.zeros((len(graphs), len(graphs)), device=self.device)

        # Convert graphs to sparse adjacency matrices and initialize labels
        adj_matrices = [self._get_sparse_adj(g) for g in graphs]
        label_tensors = [self._init_node_labels(g) for g in graphs]

        # Collect label tensors from all iterations
        all_label_tensors: list[list[torch.Tensor]] = [label_tensors]
        for _ in range(self.n_iter):
            new_labels = [
                self._wl_iteration(adj, labels)
                for adj, labels in zip(adj_matrices, all_label_tensors[-1], strict=False)
            ]
            all_label_tensors.append(new_labels)

        # Compute feature matrices using final label count
        feature_matrices = [
            torch.stack([
                self._compute_feature_vector(labels, self.label_counter)
                for labels in iteration_labels
            ])
            for iteration_labels in all_label_tensors
        ]

        # Combine features from all iterations
        final_features = torch.stack(feature_matrices).sum(dim=0)

        # Compute kernel matrix
        kernel_matrix = torch.mm(final_features, final_features.t())

        # Apply normalization if requested
        if self.normalize:
            diag = torch.sqrt(torch.diag(kernel_matrix))
            kernel_matrix = kernel_matrix / (diag.unsqueeze(0) * diag.unsqueeze(1))

        return kernel_matrix


class GraphDataset:
    """Utility class to convert NetworkX graphs for WL kernel."""

    @staticmethod
    def from_networkx(graphs: list[nx.Graph], node_labels_tag: str = "label") -> list[
        nx.Graph]:
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
