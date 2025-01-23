from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import networkx as nx


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


def graphs_to_tensors(
    graphs: list[nx.Graph],
    device: torch.device | None = None
) -> tuple[list[torch.sparse.Tensor], list[torch.Tensor]]:
    """Convert a list of NetworkX graphs into sparse adjacency matrices and label tensors.

    Args:
        graphs (List[nx.Graph]): A list of NetworkX graphs.
        device (torch.device | None): The device to place the tensors on.
        Default is CPU.

    Returns:
        Tuple[List[torch.sparse.Tensor], List[torch.Tensor]]:
            A tuple containing:
            - A list of sparse adjacency matrices.
            - A list of label tensors.
    """
    if device is None:
        device = torch.device("cpu")

    adjacency_matrices = []
    label_tensors = []

    # Create a consistent label mapping across all graphs
    label_dict: dict[str, int] = {}
    label_counter: int = 0

    for graph in graphs:
        # Create adjacency matrix
        edges = list(graph.edges())
        num_nodes = graph.number_of_nodes()

        if not edges:
            adj = torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long),
                values=torch.empty(0),
                size=(num_nodes, num_nodes),
                device=device,
            ).to_sparse_csr()
        else:
            edge_indices = edges + [(v, u) for u, v in edges]
            rows, cols = zip(*edge_indices, strict=False)
            indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
            values = torch.ones(len(edge_indices), dtype=torch.float, device=device)
            adj = torch.sparse_coo_tensor(
                indices, values, (num_nodes, num_nodes), device=device
            ).to_sparse_csr()

        adjacency_matrices.append(adj)

        # Create label tensor
        node_labels: list[int] = []
        for node in range(graph.number_of_nodes()):
            if "label" in graph.nodes[node]:
                label = graph.nodes[node]["label"]
                if label not in label_dict:
                    label_dict[label] = label_counter
                    label_counter += 1
                node_labels.append(label_dict[label])
            else:
                node_labels.append(node)

        label_tensors.append(torch.tensor(node_labels, dtype=torch.long, device=device))

    return adjacency_matrices, label_tensors
