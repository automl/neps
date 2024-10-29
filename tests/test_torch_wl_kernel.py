from __future__ import annotations

import networkx as nx
import pytest
import torch
from grakel import WeisfeilerLehman, graph_from_networkx
from grakel_replace.torch_wl_kernel import TorchWLKernel, GraphDataset


@pytest.mark.parametrize("n_iter", [1, 2, 3, 4, 5, 6, 7])
@pytest.mark.parametrize("normalize", [True, False])
def test_wl_kernel_against_grakel(n_iter, normalize):
    """Test the custom WL kernel against Grakel's implementation.

    Args:
        n_iter: Number of iterations for the WL kernel.
        normalize: Whether to normalize the kernel matrix.
    """
    # Create example graphs for testing
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (2, 3)])
    for node in G1.nodes():
        G1.nodes[node]["label"] = str(node)

    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    for node in G2.nodes():
        G2.nodes[node]["label"] = str(node)

    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (1, 3), (3, 2)])
    for node in G3.nodes():
        G3.nodes[node]["label"] = str(node)

    graphs = GraphDataset.from_networkx([G1, G2, G3])

    # Initialize and compute kernel matrix using custom WLKernel
    wl_kernel = TorchWLKernel(n_iter=n_iter, normalize=normalize)
    torch_kernel_matrix = wl_kernel(graphs).cpu().detach().numpy()

    # Convert to Grakel-compatible format
    grakel_graphs = graph_from_networkx([G1, G2, G3], node_labels_tag="label")

    # Initialize and compute kernel matrix using Grakel's Weisfeiler-Lehman kernel
    grakel_wl = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)
    grakel_kernel_matrix = grakel_wl.fit_transform(grakel_graphs)

    # Assert that the kernel matrices are similar within a reasonable tolerance
    assert torch.allclose(
        torch.tensor(torch_kernel_matrix, dtype=torch.float64),
        torch.tensor(grakel_kernel_matrix, dtype=torch.float64),
        atol=1e-100
    ), f"Mismatch found in kernel matrices with n_iter={n_iter} and normalize={normalize}"


def test_kernel_symmetry():
    """Test if the kernel matrix is symmetric."""
    # Create example graphs for testing
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    for node in G.nodes():
        G.nodes[node]["label"] = str(node)

    graphs = GraphDataset.from_networkx([G, G])
    wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
    K = wl_kernel(graphs)

    # Check if the kernel matrix is symmetric
    assert torch.allclose(K, K.T, atol=1e-100), "Kernel matrix is not symmetric"


def test_empty_graph():
    """Test the kernel computation for an empty graph."""
    # Test with an empty graph
    G_empty = nx.Graph()
    graphs = GraphDataset.from_networkx([G_empty])
    wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

    # Check if kernel returns a valid matrix (1x1 zero matrix expected)
    K = wl_kernel(graphs)
    assert K.shape == (1, 1), "Kernel matrix shape for empty graph is incorrect"
    assert K.item() == 0.0, "Kernel matrix value for empty graph should be zero"


def test_invalid_input():
    """Test that invalid inputs raise the appropriate TypeError."""
    # Test with invalid input types
    wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

    with pytest.raises(TypeError, match="Expected input type is a list of NetworkX graphs"):
        wl_kernel("invalid_input")  # Passing a string instead of a list of graphs

    with pytest.raises(TypeError, match="Expected input type is a list of NetworkX graphs"):
        wl_kernel([1, 2, 3])  # Passing a list of integers instead of graphs


def test_kernel_on_single_node_graph():
    """Test the kernel computation for single-node graphs."""
    # Test with a single-node graph
    G_single = nx.Graph()
    G_single.add_node(0)
    G_single.nodes[0]["label"] = "0"

    graphs = GraphDataset.from_networkx([G_single, G_single])
    wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
    K = wl_kernel(graphs)

    # Check if a kernel matrix for identical single-node graphs is valid and symmetric
    assert K.shape == (2, 2), "Kernel matrix shape for single-node graphs is incorrect"
    assert K[0, 0] == K[1, 1], "Self-similarity for single-node graph should be the same"
    assert torch.allclose(K, K.T, atol=1e-100), "Kernel matrix is not symmetric for single-node graph"
