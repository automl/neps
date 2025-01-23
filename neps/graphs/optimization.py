from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from botorch.optim import optimize_acqf_mixed

from neps.graphs.context_managers import set_graph_lookup
from neps.graphs.utils import sample_graphs

if TYPE_CHECKING:
    import networkx as nx
    from botorch.acquisition import AcquisitionFunction


def optimize_acqf_graph(
    acq_function: AcquisitionFunction,
    bounds: torch.Tensor,
    fixed_features_list: list[dict[int, int]] | None = None,
    num_graph_samples: int = 10,
    train_graphs: list[nx.Graph] | None = None,
    num_restarts: int = 10,
    raw_samples: int = 1024,
    q: int = 1,
) -> tuple[torch.Tensor, float]:
    """Optimize an acquisition function with graph sampling.

    This function optimizes the acquisition function by sampling graphs from the training
    set, temporarily updating the kernel's graph lookup, and evaluating the acquisition
    function for each sampled graph. The best candidate and its corresponding acquisition
    score are returned.

    Args:
        acq_function (AcquisitionFunction): The acquisition function to optimize.
        bounds (torch.Tensor): A 2 x d tensor of bounds for numerical and categorical
            features, where d is the number of features.
        fixed_features_list (list[dict[int, float]] | None): A list of dictionaries
            specifying fixed categorical feature configurations. Each dictionary maps
            feature indices to their fixed values. Defaults to None.
        num_graph_samples (int): The number of graphs to sample from the training set.
            Defaults to 10.
        train_graphs (list[nx.Graph] | None): The original training graphs. If None, a
            ValueError is raised.
        num_restarts (int): The number of optimization restarts. Defaults to 10.
        raw_samples (int): The number of raw samples to generate for optimization.
            Defaults to 1024.
        q (int): The number of candidates to generate. Defaults to 1.

    Returns:
        tuple[torch.Tensor, float]: A tuple containing the best candidate (as a tensor)
            and its corresponding acquisition score.

    Raises:
        ValueError: If `train_graphs` is None.
    """
    if train_graphs is None:
        raise ValueError("train_graphs cannot be None.")

    # Sample graphs from the training set
    sampled_graphs = sample_graphs(train_graphs, num_samples=num_graph_samples)

    # Initialize lists to store the best candidates and their scores
    best_candidates, best_scores = [], []

    # Get the index of the graph feature in the bounds
    graph_idx = bounds.shape[1] - 1

    # Iterate through each sampled graph
    for graph in sampled_graphs:
        # Temporarily set the graph lookup for the kernel
        with set_graph_lookup(acq_function.model.covar_module, [graph], append=True):
            # Iterate through each fixed feature configuration (if provided)
            for fixed_features in fixed_features_list or [{}]:
                # Add the graph index to the fixed features, indicating that the last
                # graphin the lookup should be used
                updated_fixed_features = {**fixed_features, graph_idx: -1.0}

                # Optimize the acquisition function with the updated fixed features
                candidates, scores = optimize_acqf_mixed(
                    acq_function=acq_function,
                    bounds=bounds,
                    fixed_features_list=[updated_fixed_features],
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    q=q,
                )

                # Store the candidates and their scores
                best_candidates.append(candidates)
                best_scores.append(scores)

    # Find the index of the best score
    best_idx = torch.argmax(torch.tensor(best_scores))

    # Return the best candidate and its score
    return best_candidates[best_idx], best_scores[best_idx].item()
