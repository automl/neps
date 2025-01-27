from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from botorch.models import SingleTaskGP

from neps.optimizers.models.graphs.kernels import BoTorchWLKernel, compute_kernel

if TYPE_CHECKING:
    import networkx as nx
    from botorch.models.gp_regression_mixed import Kernel


@contextmanager
def set_graph_lookup(
    kernel_or_gp: Kernel | SingleTaskGP,
    new_graphs: list[nx.Graph],
    *,
    append: bool = True,
) -> Iterator[None]:
    """Context manager to temporarily set the graph lookup for a kernel or GP model.

    Args:
        kernel_or_gp (Kernel | SingleTaskGP): The kernel or GP model whose graph lookup is
            to be set.
        new_graphs (list[nx.Graph]): The new graphs to set in the graph lookup.
        append (bool, optional): Whether to append the new graphs to the existing graph
            lookup. Defaults to True.
    """
    kernel_prev_graphs: list[tuple[Kernel, list[nx.Graph]]] = []

    # Determine the modules to update based on the input type
    if isinstance(kernel_or_gp, SingleTaskGP):
        modules = [
            k
            for k in kernel_or_gp.covar_module.sub_kernels()
            if isinstance(k, BoTorchWLKernel)
        ]
    elif isinstance(kernel_or_gp, BoTorchWLKernel):
        modules = [kernel_or_gp]
    else:
        assert hasattr(kernel_or_gp, "sub_kernels"), (
            "Kernel module must have sub_kernels method."
        )
        modules = [
            k for k in kernel_or_gp.sub_kernels() if isinstance(k, BoTorchWLKernel)
        ]

    # Save the current graph lookup and set the new graph lookup
    for kern in modules:
        compute_kernel.cache_clear()

        kernel_prev_graphs.append((kern, kern.graph_lookup))
        if append:
            kern.set_graph_lookup([*kern.graph_lookup, *new_graphs])
        else:
            kern.set_graph_lookup(new_graphs)

    yield

    # Restore the original graph lookup after the context manager exits
    for kern, prev_graphs in kernel_prev_graphs:
        kern.set_graph_lookup(prev_graphs)
