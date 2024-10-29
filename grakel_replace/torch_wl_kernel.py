from __future__ import annotations

from collections import Counter

import networkx as nx
import torch
from torch import nn


class TorchWLKernel(nn.Module):
    """Custom PyTorch implementation of Weisfeiler-Lehman Kernel.

    Args:
        n_iter: Number of WL iterations
        normalize: Whether to normalize the kernel matrix
    """

    def __init__(self, n_iter: int = 5, normalize: bool = True):
        super().__init__()
        self.n_iter = n_iter
        self.normalize = normalize
        self.label_dict = {}
        self.label_counter = 0

    def _get_sparse_adj(self, graph: nx.Graph) -> torch.sparse.Tensor:
        """Convert NetworkX graph to sparse adjacency tensor."""
        edges = list(graph.edges())
        if not edges:
            num_nodes = graph.number_of_nodes()
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty(0),
                size=(num_nodes, num_nodes),
                device=self.device
            )

        # Create COO format indices
        row = torch.tensor([e[0] for e in edges], dtype=torch.long)
        col = torch.tensor([e[1] for e in edges], dtype=torch.long)
        edges = torch.stack([
            torch.cat([row, col]),  # Add both directions for undirected graph
            torch.cat([col, row])
        ])

        values = torch.ones(edges.size(1), dtype=torch.float)
        N = graph.number_of_nodes()

        return torch.sparse_coo_tensor(
            edges, values, (N, N),
            device=self.device
        )

    def _init_node_labels(self, graph: nx.Graph) -> torch.Tensor:
        """Initialize node label tensor from graph."""
        # Get node labels and convert to indices
        labels = []
        for node in range(graph.number_of_nodes()):
            label = graph.nodes[node].get("label", str(node))
            if label not in self.label_dict:
                self.label_dict[label] = self.label_counter
                self.label_counter += 1
            labels.append(self.label_dict[label])

        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def _wl_iteration(self, adj: torch.sparse.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """Perform one WL iteration."""
        # Concatenate own label with sorted neighbor labels
        new_labels = []
        for node in range(adj.size(0)):
            node_label = labels[node].item()
            neighbors = adj.coalesce().indices()[1][adj.coalesce().indices()[0] == node]
            neighbor_label_list = sorted([labels[n].item() for n in neighbors])
            combined = f"{node_label}_{neighbor_label_list}"

            if combined not in self.label_dict:
                self.label_dict[combined] = self.label_counter
                self.label_counter += 1
            new_labels.append(self.label_dict[combined])

        return torch.tensor(new_labels, dtype=torch.long, device=self.device)

    def _compute_feature_vector(self, labels: torch.Tensor, size: int) -> torch.Tensor:
        """Compute histogram feature vector from labels with fixed size."""
        counts = Counter(labels.cpu().numpy())
        feature = torch.zeros(size, device=self.device)
        for label, count in counts.items():
            if label < size:  # Safety check
                feature[label] = count
        return feature

    def forward(self, graphs: list[nx.Graph]) -> torch.Tensor:
        """Compute WL kernel matrix for a list of graphs.

        Args:
            graphs: List of NetworkX graphs

        Returns:
            Kernel matrix as a torch.Tensor
        """
        # Validate input
        if (not isinstance(graphs, list) or
            not all(isinstance(g, nx.Graph) for g in graphs)):
            raise TypeError("Expected input type is a list of NetworkX graphs.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_dict = {}
        self.label_counter = 0

        # Handle case of empty graphs list or empty individual graphs
        if not graphs or all(g.number_of_nodes() == 0 for g in graphs):
            return torch.zeros((len(graphs), len(graphs)), device=self.device)

        # Convert graphs to sparse adjacency matrices and initialize labels
        adj_matrices = [self._get_sparse_adj(g) for g in graphs]
        label_tensors = [self._init_node_labels(g) for g in graphs]

        # Pre-allocate feature matrices list
        feature_matrices = []

        # First, run all iterations to compute maximum label count
        all_label_tensors = [label_tensors]
        for _ in range(self.n_iter):
            new_label_tensors = [
                self._wl_iteration(adj, labels)
                for adj, labels in zip(adj_matrices, all_label_tensors[-1])
            ]
            all_label_tensors.append(new_label_tensors)

        max_label_count = self.label_counter

        # Now compute feature vectors for all iterations with fixed size
        for labels_list in all_label_tensors:
            features = torch.stack([
                self._compute_feature_vector(labels, max_label_count)
                for labels in labels_list
            ])
            feature_matrices.append(features)

        # Sum up feature matrices from all iterations
        final_features = torch.stack(feature_matrices).sum(dim=0)

        # Compute kernel matrix
        K = torch.mm(final_features, final_features.t())

        # Normalize if requested
        if self.normalize:
            diag = torch.sqrt(torch.diag(K))
            K = K / (diag.unsqueeze(0) * diag.unsqueeze(1))

        return K


class GraphDataset:
    """Utility class to convert NetworkX graphs for WL kernel."""

    @staticmethod
    def from_networkx(graphs: list[nx.Graph], node_labels_tag: str = "label") -> list[
        nx.Graph]:
        """Convert NetworkX graphs ensuring proper node labeling."""
        processed_graphs = []
        for g in graphs:
            # Ensure nodes are numbered from 0 to n-1
            g = nx.convert_node_labels_to_integers(g)
            # Add default labels if not present
            for node in g.nodes():
                if node_labels_tag not in g.nodes[node]:
                    g.nodes[node][node_labels_tag] = str(node)
            processed_graphs.append(g)
        return processed_graphs
