import networkx as nx
import pytest
import torch
from grakel import WeisfeilerLehman, graph_from_networkx
from grakel_replace.torch_wl_kernel import TorchWLKernel, GraphDataset


class TestTorchWLKernel:
    @pytest.fixture
    def example_graphs(self):
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

        return [G1, G2, G3]

    @pytest.mark.parametrize("n_iter", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_wl_kernel_against_grakel(self, n_iter, normalize, example_graphs):
        """Test the custom WL kernel against Grakel's implementation."""
        graphs = GraphDataset.from_networkx(example_graphs)

        # Initialize and compute kernel matrix using custom WLKernel
        wl_kernel = TorchWLKernel(n_iter=n_iter, normalize=normalize)
        torch_kernel_matrix = wl_kernel(graphs).cpu().detach().numpy()

        # Convert to Grakel-compatible format
        grakel_graphs = graph_from_networkx(example_graphs, node_labels_tag="label")

        # Initialize and compute kernel matrix using Grakel's Weisfeiler-Lehman kernel
        grakel_wl = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)
        grakel_kernel_matrix = grakel_wl.fit_transform(grakel_graphs)

        # Assert that the kernel matrices are similar within a reasonable tolerance
        assert torch.allclose(
            torch.tensor(torch_kernel_matrix, dtype=torch.float64),
            torch.tensor(grakel_kernel_matrix, dtype=torch.float64),
            atol=1e-100
        ), (f"Mismatch found in kernel matrices with n_iter={n_iter} and "
            f"normalize={normalize}")

    def test_kernel_symmetry(self, example_graphs):
        """Test if the kernel matrix is symmetric."""
        graphs = GraphDataset.from_networkx([example_graphs[0], example_graphs[0]])
        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(graphs)

        # Check if the kernel matrix is symmetric
        assert torch.allclose(K, K.T, atol=1e-100), "Kernel matrix is not symmetric"

    def test_empty_graph(self):
        """Test the kernel computation for an empty graph."""
        # Test with an empty graph
        G_empty = nx.Graph()
        graphs = GraphDataset.from_networkx([G_empty])
        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

        # Check if kernel returns a valid matrix (1x1 zero matrix expected)
        K = wl_kernel(graphs)
        assert K.shape == (1, 1), "Kernel matrix shape for empty graph is incorrect"
        assert K.item() == 0.0, "Kernel matrix value for empty graph should be zero"

    def test_invalid_input(self):
        """Test that invalid inputs raise the appropriate TypeError."""
        # Test with invalid input types
        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

        with pytest.raises(TypeError, match="Expected input type is a list of NetworkX graphs"):
            wl_kernel("invalid_input")  # Passing a string instead of a list of graphs

        with pytest.raises(TypeError, match="Expected input type is a list of NetworkX graphs"):
            wl_kernel([1, 2, 3])  # Passing a list of integers instead of graphs

    def test_kernel_on_single_node_graph(self, example_graphs):
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

    def test_wl_kernel_with_empty_graph_and_reordered_edges(self, example_graphs):
        """Test the TorchWLKernel with an empty graph and a graph with reordered edges."""
        # Create example graphs for testing
        G_empty = nx.Graph()
        G = example_graphs[0]
        G_reordered = nx.Graph()
        G_reordered.add_edges_from([(1, 4), (2, 3), (1, 2), (0, 1), (1, 3)])
        for node in G_reordered.nodes():
            G_reordered.nodes[node]["label"] = str(node)

        graphs = GraphDataset.from_networkx([G_empty, G, G_reordered])
        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(graphs)

        # Check if the kernel matrix is valid and the values
        # are the same for the original and reordered graphs
        assert K.shape == (3, 3), "Kernel matrix shape is incorrect"
        assert K[1, 1] == K[2, 2], "Kernel value for original and reordered graphs should be the same"

    @pytest.mark.parametrize("n_iter", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_wl_kernel_with_different_node_labels(self, n_iter, normalize, example_graphs):
        """Test the TorchWLKernel with graphs having different node labels."""
        # Create example graphs with different node labels
        G1 = example_graphs[0]
        for node in G1.nodes():
            G1.nodes[node]["label"] = f"node_{node}"

        G2 = example_graphs[1]
        for node in G2.nodes():
            G2.nodes[node]["label"] = f"vertex_{node}"

        G3 = example_graphs[2]
        for node in G3.nodes():
            G3.nodes[node]["label"] = f"n{node}"

        graphs = GraphDataset.from_networkx([G1, G2, G3])

        # Initialize and compute kernel matrix using custom TorchWLKernel
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
        ), (f"Mismatch found in kernel matrices with n_iter={n_iter} and "
            f"normalize={normalize} for graphs with different node labels")

    def test_wl_kernel_with_same_node_labels(self, example_graphs):
        """Test the TorchWLKernel with graphs having the same node labels."""
        # Create example graphs with the same node labels
        G1 = example_graphs[0]
        for node in G1.nodes():
            G1.nodes[node]["label"] = "A"

        G2 = example_graphs[1]
        for node in G2.nodes():
            G2.nodes[node]["label"] = "A"

        G3 = example_graphs[2]
        for node in G3.nodes():
            G3.nodes[node]["label"] = "A"

        graphs = GraphDataset.from_networkx([G1, G2, G3])
        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(graphs)

        # Check if the kernel matrix is valid and the values are the same for the graphs with the same node labels
        assert K.shape == (3, 3), "Kernel matrix shape is incorrect"
        assert torch.allclose(K, K.T, atol=1e-100), "Kernel matrix is not symmetric"
        assert torch.all(K == K[0, 0]), ("Kernel values should be the same for "
                                         "graphs with the same node labels")
