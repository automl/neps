from __future__ import annotations

import random

import networkx as nx
import numpy as np
import torch


def seed_all(seed: int = 100):
    """Seed all random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility with CuDNN (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def min_max_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Scale the input tensor to the range [0, 1]."""
    min_vals = tensor.min(dim=0, keepdim=True).values
    max_vals = tensor.max(dim=0, keepdim=True).values
    return (tensor - min_vals) / (max_vals - min_vals)


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
