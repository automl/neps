from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import networkx as nx
import torch
from botorch.optim import optimize_acqf_mixed
from grakel_replace.torch_wl_kernel import BoTorchWLKernel

if TYPE_CHECKING:
    from botorch.acquisition import AcquisitionFunction
    from botorch.models.gp_regression_mixed import Kernel


@contextmanager
def set_graph_lookup(
    kernel: Kernel,
    new_graphs: list[nx.Graph],
    *,
    append: bool = True,
) -> Iterator[None]:
    """Context manager to temporarily set the graph lookup for a kernel.

    Args:
        kernel (Kernel): The kernel whose graph lookup is to be set.
        new_graphs (list[nx.Graph]): The new graphs to set in the graph lookup.
        append (bool, optional): Whether to append the new graphs to the existing graph
            lookup. Defaults to True.
    """
    kernel_prev_graphs: list[tuple[Kernel, list[nx.Graph]]] = []

    # Determine the modules to update based on the kernel type
    if isinstance(kernel, BoTorchWLKernel):
        modules = [kernel]
    else:
        assert hasattr(
            kernel, "sub_kernels"
        ), "Kernel module must have sub_kernels method."
        modules = [k for k in kernel.sub_kernels() if isinstance(k, BoTorchWLKernel)]

    # Save the current graph lookup and set the new graph lookup
    for kern in modules:
        kernel_prev_graphs.append((kern, kern.graph_lookup))
        if append:
            kern.set_graph_lookup([*kern.graph_lookup, *new_graphs])
        else:
            kern.set_graph_lookup(new_graphs)

    yield

    # Restore the original graph lookup after the context manager exits
    for kern, prev_graphs in kernel_prev_graphs:
        kern.set_graph_lookup(prev_graphs)


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
    graph_idx = bounds.shape[1] - 1
    # Iterate through all the kernels and include the sampled graph.
    for _graph in sampled_graphs:
        with set_graph_lookup(acq_function.model.covar_module, [_graph], append=True):
            for fixed_features in fixed_features_list or [{}]:
                # We then consider this graph as a fixed feature, i.e. in the X's
                # generated during acquisition, the graph column will just be full
                # of `-1` indicating to select the very last graph in the lookup
                # they used.
                candidates, scores = optimize_acqf_mixed(
                    acq_function=acq_function,
                    bounds=bounds,
                    fixed_features_list=[{**fixed_features, graph_idx: -1.0}],
                    num_restarts=num_restarts,
                    raw_samples=raw_samples, q=q)
                best_candidates.append(candidates)
                best_scores.append(scores)
    best_idx = torch.argmax(torch.tensor(best_scores))
    return best_candidates[best_idx], best_scores[best_idx].item()
