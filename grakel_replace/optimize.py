from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
from botorch.optim import optimize_acqf_mixed

if TYPE_CHECKING:
    import networkx as nx
    from botorch.acquisition import AcquisitionFunction


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
        sampled_graph = base_graph.copy()  # Copy base graph
        # Modify the graph with edge additions or removals
        for _ in range(random.randint(1, 3)):
            if len(sampled_graph.edges) > 0:
                # Randomly remove or add edges
                if random.random() > 0.5:
                    u, v = random.choice(list(sampled_graph.edges))
                    sampled_graph.remove_edge(u, v)
                else:
                    u = random.choice(list(sampled_graph.nodes))
                    v = random.choice(list(sampled_graph.nodes))
                    sampled_graph.add_edge(u, v)
        sampled_graphs.append(sampled_graph)
    return sampled_graphs


def optimize_acqf_graph(
    acq_function: AcquisitionFunction,
    bounds: torch.Tensor,
    fixed_features_list: list[dict[int, float]] | None = None,
    num_graph_samples: int = 10,
    train_graphs: list[nx.Graph] | None = None,
    num_restarts: int = 10,
    raw_samples: int = 1024,
    q: int = 1,
) -> tuple[torch.Tensor, float]:
    """Optimize acquisition function with graph sampling.

    Args:
        acq_function: Acquisition function to optimize
        bounds: Bounds for numerical/categorical features
        fixed_features_list: Fixed categorical feature configurations
        num_graph_samples: Number of graphs to sample
        train_graphs: Original training graphs
        num_restarts: Number of optimization restarts
        raw_samples: Number of raw samples to generate
        q: Number of candidates to generate

    Returns:
        tuple: Best candidate and acquisition score.
    """
    if train_graphs is None:
        raise ValueError("train_graphs cannot be None.")

    sampled_graphs = sample_graphs(train_graphs, num_samples=num_graph_samples)

    best_candidates, best_scores = [], []

    for _graph in sampled_graphs:
        for fixed_features in fixed_features_list or [{}]:
            candidates, scores = optimize_acqf_mixed(
                acq_function=acq_function,
                bounds=bounds,
                fixed_features_list=[fixed_features],
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                q=q,
            )
            best_candidates.append(candidates)
            best_scores.append(scores)

    best_scores_tensor = torch.tensor(best_scores)
    best_idx = torch.argmax(best_scores_tensor)
    return best_candidates[best_idx], best_scores_tensor[best_idx].item()
