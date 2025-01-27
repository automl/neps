from __future__ import annotations

import random

import networkx as nx
import numpy as np
import torch


def seed_all(seed: int = 100) -> None:
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
    graphs: list[nx.Graph], device: torch.device | None = None
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


def sample_graphs(graphs: list[nx.Graph], num_samples: int) -> list[nx.Graph]:
    """Sample graphs using random walks or edge modifications.

    Args:
        graphs (list[nx.Graph]): Existing training graphs.
        num_samples (int): Number of graph samples to generate.

    Returns:
        list[nx.Graph]: Sampled graphs.
    """
    sampled_graphs = []
    for _ in range(num_samples):
        base_graph = random.choice(graphs)
        sampled_graph = base_graph.copy()

        # More aggressive modifications
        num_modifications = random.randint(2, 5)  # Increase minimum modifications
        for _ in range(num_modifications):
            if random.random() > 0.3:  # 70% chance to add edge
                nodes = list(sampled_graph.nodes)
                if len(nodes) >= 2:
                    u, v = random.sample(nodes, 2)
                    if not sampled_graph.has_edge(u, v):
                        sampled_graph.add_edge(u, v)
            elif sampled_graph.edges:  # 30% chance to remove edge
                u, v = random.choice(list(sampled_graph.edges))
                sampled_graph.remove_edge(u, v)

        # Ensure the graph stays connected
        if not nx.is_connected(sampled_graph):
            components = list(nx.connected_components(sampled_graph))
            for i in range(len(components) - 1):
                u = random.choice(list(components[i]))
                v = random.choice(list(components[i + 1]))
                sampled_graph.add_edge(u, v)

        sampled_graphs.append(sampled_graph)

    return sampled_graphs
