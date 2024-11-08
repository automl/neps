import pytest
import torch
import networkx as nx
from gpytorch.kernels import MaternKernel, AdditiveKernel
from botorch.models.gp_regression_mixed import CategoricalKernel, ScaleKernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal

from grakel_replace.mixed_single_task_gp import MixedSingleTaskGP
from grakel_replace.torch_wl_kernel import TorchWLKernel


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    n_samples = 5
    n_numerical = 2
    n_categorical = 2

    # Create numerical and categorical features
    X = torch.empty(size=(n_samples, n_numerical + n_categorical), dtype=torch.float64)
    X[:, :n_numerical] = torch.rand(size=(n_samples, n_numerical), dtype=torch.float64)
    X[:, n_numerical:] = torch.randint(0, 3, size=(n_samples, n_categorical),
                                       dtype=torch.float64)

    # Create sample graphs
    graphs = [nx.erdos_renyi_graph(n=4, p=0.5) for _ in range(n_samples)]

    # Create target values
    y = torch.rand(size=(n_samples, 1), dtype=torch.float64)

    return X, graphs, y


@pytest.fixture
def sample_kernels():
    """Create sample kernels for testing."""
    n_numerical = 2
    n_categorical = 2

    matern = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=n_numerical,
            active_dims=tuple(range(n_numerical)),
        ),
    )

    hamming = ScaleKernel(
        CategoricalKernel(
            ard_num_dims=n_categorical,
            active_dims=tuple(range(n_numerical, n_numerical + n_categorical)),
        ),
    )

    combined_kernel = AdditiveKernel(matern, hamming)
    wl_kernel = TorchWLKernel(n_iter=3, normalize=True)

    return combined_kernel, wl_kernel


def test_initialization(sample_data, sample_kernels):
    """Test that MixedSingleTaskGP initializes correctly."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    # Test initialization with all parameters
    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    assert gp._train_graphs == graphs
    assert isinstance(gp._wl_kernel, TorchWLKernel)
    assert gp.num_cat_kernel == combined_kernel


def test_forward_shape(sample_data, sample_kernels):
    """Test that forward pass returns correct shapes."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # Test forward pass with training data
    output = gp.forward(X, graphs)
    assert isinstance(output, MultivariateNormal)
    assert output.mean.shape == (len(X),)
    assert output.covariance_matrix.shape == (len(X), len(X))

    # Test forward pass with different sized test data
    n_test = 3
    test_X = torch.rand(size=(n_test, X.shape[1]), dtype=torch.float64)
    test_graphs = [nx.erdos_renyi_graph(n=4, p=0.5) for _ in range(n_test)]

    output = gp.forward(test_X, test_graphs)
    assert output.mean.shape == (n_test,)
    assert output.covariance_matrix.shape == (n_test, n_test)


def test_input_validation(sample_data, sample_kernels):
    """Test that appropriate errors are raised for invalid inputs."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # Test mismatched number of features and graphs
    test_X = torch.rand(size=(3, X.shape[1]), dtype=torch.float64)
    test_graphs = [nx.erdos_renyi_graph(n=4, p=0.5) for _ in range(4)]  # Different length

    with pytest.raises(ValueError,
                       match="Number of feature vectors.*must match.*number of graphs"):
        gp.forward(test_X, test_graphs)


def test_kernel_combination(sample_data):
    """Test that numerical/categorical and graph kernels are properly combined."""
    X, graphs, y = sample_data

    # Create GP with only graph kernel
    gp_graph_only = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
    )

    output_graph = gp_graph_only.forward(X, graphs)
    graph_var = output_graph.variance

    # Create GP with both kernels
    n_numerical = 2
    matern = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=n_numerical,
            active_dims=tuple(range(n_numerical)),
        ),
    )

    gp_combined = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=matern,
    )

    output_combined = gp_combined.forward(X, graphs)
    combined_var = output_combined.variance

    # Combined kernel should have larger variance due to addition
    assert torch.all(combined_var > graph_var)


def test_prediction_consistency(sample_data, sample_kernels):
    """Test that predictions are consistent between multiple forward passes."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # Multiple forward passes should give same result
    output1 = gp.forward(X, graphs)
    output2 = gp.forward(X, graphs)

    assert torch.allclose(output1.mean, output2.mean)
    assert torch.allclose(output1.variance, output2.variance)


def test_graph_kernel_caching(sample_data, sample_kernels):
    """Test that graph kernel matrices are properly cached."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # First forward pass
    _ = gp.forward(X, graphs)
    K_train_1 = gp._K_train.clone()

    # Second forward pass
    _ = gp.forward(X, graphs)
    K_train_2 = gp._K_train.clone()

    # Cached kernel matrices should be identical
    assert torch.allclose(K_train_1, K_train_2)


def test_mean_predictions(sample_data, sample_kernels):
    """Test that mean predictions are reasonable."""
    X, graphs, y = sample_data
    combined_kernel, wl_kernel = sample_kernels

    gp = MixedSingleTaskGP(
        train_X=X,
        train_graphs=graphs,
        train_Y=y,
        num_cat_kernel=combined_kernel,
        wl_kernel=wl_kernel,
    )

    # Test predictions
    with torch.no_grad():
        output = gp.forward(X, graphs)
        predictions = output.mean
        uncertainties = output.variance.sqrt()

        # Mean predictions should be within reasonable bounds
        assert torch.all(predictions >= y.min() - 2 * uncertainties)
        assert torch.all(predictions <= y.max() + 2 * uncertainties)
