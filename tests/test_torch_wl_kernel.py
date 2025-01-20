import pytest
import torch
import numpy as np
import networkx as nx
from grakel import WeisfeilerLehman, graph_from_networkx
from grakel_replace.torch_wl_kernel import TorchWLKernel


class TestTorchWLKernel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def create_adjacency_and_labels(self, graphs):
        """Helper method to create adjacency matrices and label tensors from graphs."""
        adjacency_matrices = []
        label_tensors = []

        # Create a consistent label mapping
        label_dict = {}
        current_label = 0

        for graph in graphs:
            # Create adjacency matrix
            adj = nx.adjacency_matrix(graph).tocoo()
            indices = torch.LongTensor([adj.row, adj.col])
            values = torch.FloatTensor(adj.data)
            size = torch.Size([graph.number_of_nodes(), graph.number_of_nodes()])
            sparse_adj = torch.sparse_coo_tensor(
                indices, values, size, device=self.device
            ).to_sparse_csr()
            adjacency_matrices.append(sparse_adj)

            # Create label tensor with controlled mapping
            node_labels = []
            for node in range(graph.number_of_nodes()):
                if "label" in graph.nodes[node]:
                    label = graph.nodes[node]["label"]
                    if label not in label_dict:
                        label_dict[label] = current_label
                        current_label += 1
                    node_labels.append(label_dict[label])
                else:
                    node_labels.append(node)

            label_tensors.append(
                torch.tensor(node_labels, dtype=torch.long, device=self.device)
            )

        return adjacency_matrices, label_tensors

    @pytest.mark.parametrize("n_iter", [1, 2, 3, 5, 10])
    @pytest.mark.parametrize("normalize", [False])
    def test_wl_kernel_against_grakel(self, n_iter, normalize, example_graphs):
        adjacency_matrices, label_tensors = self.create_adjacency_and_labels(
            example_graphs)

        # Initialize Torch WL Kernel
        torch_kernel = TorchWLKernel(n_iter=n_iter, normalize=normalize)
        torch_kernel_matrix = torch_kernel(adjacency_matrices,
                                           label_tensors).cpu().numpy()

        # Initialize GraKel WL Kernel
        grakel_graphs = list(
            graph_from_networkx(example_graphs, node_labels_tag="label", as_Graph=True))
        grakel_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)
        grakel_kernel_matrix = grakel_kernel.fit_transform(grakel_graphs)

        # Define tolerances based on normalization
        rtol = 1e-5 if normalize else 1e-4
        atol = 1e-8 if normalize else 1e-7

        # Compare the kernel matrices
        np.testing.assert_allclose(
            torch_kernel_matrix,
            grakel_kernel_matrix,
            rtol=rtol,
            atol=atol,
            err_msg=f"Kernel matrices differ for n_iter={n_iter}, normalize={normalize}"
        )

    def test_empty_graph(self):
        G_empty = nx.Graph()
        G_empty.add_node(0)
        G_empty.nodes[0]["label"] = "0"

        adjacency_matrices, label_tensors = self.create_adjacency_and_labels([G_empty])

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

        adjacency_matrices, label_tensors = self.create_adjacency_and_labels([G_single])

        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(adjacency_matrices, label_tensors)

        expected = torch.ones(1, 1, device=self.device)
        torch.testing.assert_close(K, expected)

    def test_wl_kernel_with_empty_graph_and_reordered_edges(self, example_graphs):
        """Test the TorchWLKernel with an empty graph and a graph with reordered edges."""
        # Create example graphs for testing
        G_empty = nx.Graph()
        G_empty.add_node(0)
        G_empty.nodes[0]["label"] = "0"

        G = example_graphs[0]
        G_reordered = nx.Graph()
        G_reordered.add_edges_from([(1, 4), (2, 3), (1, 2), (0, 1), (1, 3)])
        for node in G_reordered.nodes():
            G_reordered.nodes[node]["label"] = str(node)

        graphs = [G_empty, G, G_reordered]
        adjacency_matrices, label_tensors = self.create_adjacency_and_labels(graphs)

        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(adjacency_matrices, label_tensors)

        assert K.shape == (3, 3), "Kernel matrix shape is incorrect"
        assert K[1, 1] == K[
            2, 2], "Kernel value for original and reordered graphs should be the same"

    @pytest.mark.parametrize("n_iter", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("normalize", [True, False])
    def test_wl_kernel_with_different_node_labels(self, n_iter, normalize,
                                                  example_graphs):
        graphs = []
        for i, G in enumerate(example_graphs):
            G_copy = G.copy()
            prefix = ["node_", "vertex_", "n"][i]
            for node in G_copy.nodes():
                G_copy.nodes[node]["label"] = f"{prefix}{node}"
            graphs.append(G_copy)

        adjacency_matrices, label_tensors = self.create_adjacency_and_labels(graphs)

        wl_kernel = TorchWLKernel(n_iter=n_iter, normalize=normalize)
        torch_kernel_matrix = wl_kernel(adjacency_matrices, label_tensors).cpu().numpy()

        grakel_graphs = graph_from_networkx(graphs, node_labels_tag="label")
        grakel_wl = WeisfeilerLehman(n_iter=n_iter, normalize=normalize)
        grakel_kernel_matrix = grakel_wl.fit_transform(grakel_graphs)

        # Define tolerances based on normalization, matching the main test
        rtol = 1e-5 if normalize else 1e-4
        atol = 1e-8 if normalize else 1e-7

        # Updated assertion with both rtol and atol
        np.testing.assert_allclose(
            torch_kernel_matrix,
            grakel_kernel_matrix,
            rtol=rtol,
            atol=atol,
            err_msg=f"Kernel matrices differ for n_iter={n_iter}, normalize={normalize}"
        )

    def test_wl_kernel_with_same_node_labels(self, example_graphs):
        """Test WL kernel behavior with same node labels but different structures.

        Even when all nodes have the same label, the WL kernel should:
        1. Produce a symmetric matrix
        2. Have 1.0 on the diagonal (self-similarity)
        3. Have off-diagonal values less than 1.0 (different structures)
        4. Maintain non-negative values (it's a valid kernel)
        """
        graphs = []
        for G in example_graphs:
            G_copy = G.copy()
            for node in G_copy.nodes():
                G_copy.nodes[node]["label"] = "A"
            graphs.append(G_copy)

        adjacency_matrices, label_tensors = self.create_adjacency_and_labels(graphs)

        wl_kernel = TorchWLKernel(n_iter=3, normalize=True)
        K = wl_kernel(adjacency_matrices, label_tensors)

        # Check basic properties
        assert K.shape == (3, 3), "Kernel matrix shape is incorrect"
        assert torch.allclose(K, K.T, atol=1e-4), "Kernel matrix is not symmetric"

        # Check diagonal elements are 1 (normalized self-similarity)
        assert torch.allclose(torch.diag(K), torch.ones_like(torch.diag(K)), atol=1e-4), \
            "Diagonal elements should be 1.0"

        # Check off-diagonal elements are less than 1 (different structures)
        off_diag_mask = ~torch.eye(K.shape[0], dtype=bool)
        assert torch.all(K[off_diag_mask] < 1.0), \
            "Off-diagonal elements should be less than 1.0 for different structures"

        # Check all elements are non-negative (valid kernel)
        assert torch.all(K >= 0), "Kernel values should be non-negative"
