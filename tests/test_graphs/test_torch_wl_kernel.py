from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import torch
from grakel import WeisfeilerLehman, graph_from_networkx
from neps.graphs.kernels import TorchWLKernel
from neps.graphs.utils import graphs_to_tensors


class TestTorchWLKernel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture()
    def example_graphs_set(self):
        # Create example graphs for testing
        G1 = nx.Graph()
        G1.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 3), (3, 4)])
        for node in G1.nodes():
            G1.nodes[node]["label"] = str(node)

        G2 = nx.Graph()
        G2.add_edges_from([(0, 1), (1, 2), (3, 4), (4, 0)])
        for node in G2.nodes():
            G2.nodes[node]["label"] = str(node)

        G3 = nx.Graph()
        G3.add_edges_from([(0, 1), (1, 3), (3, 2), (2, 4), (4, 0), (1, 2)])
        for node in G3.nodes():
            G3.nodes[node]["label"] = str(node)

        return [G1, G2, G3]

    @pytest.fixture()
    def random_graphs_sets(self):
        # Set a seed for reproducibility
        seed = 100
        np.random.seed(seed)
        torch.manual_seed(seed)
        random_graph_sets = []

        # Generate 10 random sets of graphs
        for _ in range(10):
            # Number of graphs in the set (2 to 10)
            num_graphs = np.random.randint(2, 11)
            graph_set = []

            for _ in range(num_graphs):
                # Number of nodes in the graph (3 to 50)
                num_nodes = np.random.randint(3, 51)
                G = nx.Graph()

                # Add nodes with labels
                for node in range(num_nodes):
                    G.add_node(node, label=str(node))

                # Add random edges
                for u in range(num_nodes):
                    for v in range(u + 1, num_nodes):
                        if np.random.rand() > 0.5:  # 50% chance to add an edge
                            G.add_edge(u, v)

                graph_set.append(G)

            random_graph_sets.append(graph_set)

        return random_graph_sets

    @pytest.mark.parametrize("n_iter", [1, 2, 3, 5, 10])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_wl_kernel_against_grakel(self, n_iter, normalize, random_graphs_sets):
        for graph_set in random_graphs_sets:
            adjacency_matrices, label_tensors = graphs_to_tensors(
                graph_set, device=self.device)

            # Initialize Torch WL Kernel
            torch_kernel = TorchWLKernel(n_iter=n_iter, normalize=normalize)
            torch_kernel_matrix = torch_kernel(adjacency_matrices,
                                               label_tensors).cpu().numpy()

            # Initialize GraKel WL Kernel
            grakel_graphs = list(
                graph_from_networkx(graph_set, node_labels_tag="label", as_Graph=True))
            grakel_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)
            grakel_kernel_matrix = grakel_kernel.fit_transform(grakel_graphs)

            # Compare the kernel matrices
            np.testing.assert_allclose(
                torch_kernel_matrix,
                grakel_kernel_matrix,
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Kernel matrices differ for graph={graph_set}, n_iter={n_iter}"
            )

    def test_empty_graph(self):
        G_empty = nx.Graph()
        G_empty.add_node(0)
        G_empty.nodes[0]["label"] = "0"

        adjacency_matrices, label_tensors = graphs_to_tensors([G_empty],
                                                              device=self.device)

        # Initialize kernel and compute
        kernel = TorchWLKernel(n_iter=3, normalize=True)
        kernel_matrix = kernel(adjacency_matrices, label_tensors)

        # For a single graph, should get a 1x1 matrix with value 1.0
        expected = torch.ones(1, 1, device=self.device)
        torch.testing.assert_close(kernel_matrix, expected)

    def test_invalid_input(self):
        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

        with pytest.raises(ValueError,
                           match="Mismatch between adjacency matrices and label tensors"):
            wl_kernel([], [torch.tensor([0])])

    def test_kernel_on_single_node_graph(self):
        G_single = nx.Graph()
        G_single.add_node(0)
        G_single.nodes[0]["label"] = "0"

        adjacency_matrices, label_tensors = graphs_to_tensors([G_single],
                                                              device=self.device)

        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(adjacency_matrices, label_tensors)

        expected = torch.ones(1, 1, device=self.device)
        torch.testing.assert_close(K, expected)

    def test_wl_kernel_with_empty_graph_and_reordered_edges(self, random_graphs_sets):
        """Test the TorchWLKernel with an empty graph and a graph with reordered edges."""
        for graph_set in random_graphs_sets:
            # Create an empty graph
            G_empty = nx.Graph()
            G_empty.add_node(0)
            G_empty.nodes[0]["label"] = "0"

            # Select the first graph from the set to reorder its edges
            G = graph_set[0]
            G_reordered = nx.Graph()

            # Add all nodes from the original graph to G_reordered
            for node in G.nodes():
                G_reordered.add_node(node, label=G.nodes[node]["label"])

            # Reorder edges randomly
            edges = list(G.edges())
            np.random.shuffle(edges)  # Randomly shuffle the edges
            G_reordered.add_edges_from(edges)

            # Combine the empty graph, original graph, and reordered graph
            graphs = [G_empty, G, G_reordered]
            adjacency_matrices, label_tensors = graphs_to_tensors(
                graphs, device=self.device
            )

            # Initialize and compute the kernel
            wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
            K = wl_kernel(adjacency_matrices, label_tensors)

            assert K.shape == (3, 3), "Kernel matrix shape is incorrect"
            assert torch.allclose(K[1, 1], K[2, 2]), \
                "Kernel value for original and reordered graphs should be the same"

    @pytest.mark.parametrize("n_iter", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_wl_kernel_with_different_node_labels(self, n_iter, normalize,
                                                  example_graphs_set):
        graphs = []
        for i, G in enumerate(example_graphs_set):
            G_copy = G.copy()
            prefix = ["node_", "vertex_", "n"][i]
            for node in G_copy.nodes():
                G_copy.nodes[node]["label"] = f"{prefix}{node}"
            graphs.append(G_copy)

        adjacency_matrices, label_tensors = graphs_to_tensors(graphs,
                                                              device=self.device)

        wl_kernel = TorchWLKernel(n_iter=n_iter, normalize=normalize)
        torch_kernel_matrix = wl_kernel(adjacency_matrices, label_tensors).cpu().numpy()

        grakel_graphs = graph_from_networkx(graphs, node_labels_tag="label")
        grakel_wl = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)
        grakel_kernel_matrix = grakel_wl.fit_transform(grakel_graphs)

        np.testing.assert_allclose(
            torch_kernel_matrix,
            grakel_kernel_matrix,
            rtol=1e-5,
            atol=1e-8,
            err_msg=f"Kernel matrices differ for n_iter={n_iter}, normalize={normalize}"
        )

    def test_wl_kernel_with_same_node_labels(self, example_graphs_set):
        """Test WL kernel behavior with same node labels but different structures.

        Even when all nodes have the same label, the WL kernel should:
        1. Produce a symmetric matrix
        2. Have 1.0 on the diagonal (self-similarity)
        3. Have off-diagonal values less than 1.0 (different structures)
        4. Maintain non-negative values (it's a valid kernel)
        """
        graphs = []
        for G in example_graphs_set:
            G_copy = G.copy()
            for node in G_copy.nodes():
                G_copy.nodes[node]["label"] = "A"
            graphs.append(G_copy)

        adjacency_matrices, label_tensors = graphs_to_tensors(
            graphs, device=self.device)

        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(adjacency_matrices, label_tensors)

        # Check basic properties
        assert K.shape == (3, 3), "Kernel matrix shape is incorrect"
        assert torch.allclose(K, K.T, atol=1e-4), "Kernel matrix is not symmetric"

        # Check diagonal elements are 1 (normalized self-similarity)
        assert torch.allclose(torch.diag(K), torch.ones_like(torch.diag(K)), atol=1e-4), \
            "Diagonal elements should be 1.0"

        # Check off-diagonal elements are less than 1 (different structures)
        off_diag_mask = ~torch.eye(K.shape[0], dtype=torch.bool, device=self.device)
        assert torch.all(K[off_diag_mask] < 1.0), \
            "Off-diagonal elements should be less than 1.0 for different structures"

        # Check all elements are non-negative (valid kernel)
        assert torch.all(K >= 0), "Kernel values should be non-negative"
