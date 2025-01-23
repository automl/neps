import pytest
import torch
import networkx as nx
from botorch.models.gp_regression_mixed import Kernel

from grakel_replace.kernels import BoTorchWLKernel


def create_simple_graphs(num_graphs: int) -> list[nx.Graph]:
    """Helper function to create a list of graphs."""
    graphs = []
    for i in range(num_graphs):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edges_from([(0, 1), (1, 2)])
        graphs.append(G)
    return graphs


class TestBoTorchWLKernel:
    @pytest.fixture
    def simple_graphs(self):
        return create_simple_graphs(3)

    @pytest.fixture
    def wl_kernel(self, simple_graphs):
        return BoTorchWLKernel(
            graph_lookup=simple_graphs,
            n_iter=2,
            normalize=True,
            active_dims=(0,),
        )

    def test_initialization(self, wl_kernel, simple_graphs):
        """Test that the kernel is initialized correctly."""
        assert isinstance(wl_kernel, Kernel)
        assert len(wl_kernel.graph_lookup) == len(simple_graphs)
        assert wl_kernel.n_iter == 2
        assert wl_kernel.normalize is True
        assert torch.equal(wl_kernel.active_dims, torch.tensor([0]))

    def test_precompute_graph_data(self, wl_kernel):
        """Test that graph data is precomputed correctly."""
        assert hasattr(wl_kernel, "adjacency_cache")
        assert hasattr(wl_kernel, "label_cache")
        assert len(wl_kernel.adjacency_cache) == len(wl_kernel.graph_lookup)
        assert len(wl_kernel.label_cache) == len(wl_kernel.graph_lookup)

    def test_set_graph_lookup(self, wl_kernel):
        """Test that the graph lookup can be updated."""
        new_graphs = create_simple_graphs(2)
        wl_kernel.set_graph_lookup(new_graphs)
        assert len(wl_kernel.graph_lookup) == 2
        assert len(wl_kernel.adjacency_cache) == 2
        assert len(wl_kernel.label_cache) == 2

    def test_forward_self_kernel(self, wl_kernel):
        """Test the kernel computation for self-similarity."""
        x = torch.tensor([[0], [1], [2]], dtype=torch.float64)
        K = wl_kernel.forward(x, x)
        assert K.shape == (3, 3)  # Kernel matrix should be 3x3
        assert torch.allclose(K, K.T)  # Kernel matrix should be symmetric

    def test_forward_cross_kernel(self, wl_kernel):
        """Test the kernel computation for cross-similarity."""
        x1 = torch.tensor([[0], [1]], dtype=torch.float64)
        x2 = torch.tensor([[1], [2]], dtype=torch.float64)
        K = wl_kernel.forward(x1, x2)
        assert K.shape == (2, 2)  # Kernel matrix should be 2x2

    def test_forward_diagonal(self, wl_kernel):
        """Test the kernel computation for diagonal only."""
        x = torch.tensor([[0], [1], [2]], dtype=torch.float64)
        K = wl_kernel.forward(x, x, diag=True)
        assert K.shape == (3,)  # Diagonal should be a vector of length 3

    def test_handle_negative_one_index(self, wl_kernel):
        """Test the handling of the -1 index."""
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float64)
        K = wl_kernel.forward(x, x)
        assert K.shape == (3, 3)  # Kernel matrix should be 3x3
        # Ensure that -1 refers to the last graph
        last_graph_idx = len(wl_kernel.graph_lookup) - 1
        assert torch.allclose(K[0, 0], K[last_graph_idx, last_graph_idx])

    def test_forward_batched_input(self, wl_kernel):
        """Test the kernel computation for batched input."""
        x1 = torch.tensor([[[0], [1]], [[1], [2]]], dtype=torch.float64)
        x2 = torch.tensor([[[1], [2]], [[0], [1]]], dtype=torch.float64)
        K = wl_kernel.forward(x1, x2)
        assert K.shape == (2, 2, 2)  # Batched kernel matrix should be 2x2x2

    def test_forward_invalid_input(self, wl_kernel):
        """Test that invalid input raises an error."""
        x1 = torch.tensor([[0], [1], [2]], dtype=torch.float64)
        x2 = torch.tensor([[0], [1]], dtype=torch.float64)
        with pytest.raises(NotImplementedError):
            wl_kernel.forward(x1, x2, last_dim_is_batch=True)
