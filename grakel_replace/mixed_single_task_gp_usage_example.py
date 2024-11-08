import networkx as nx
import torch
from botorch.models.gp_regression_mixed import CategoricalKernel, ScaleKernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import AdditiveKernel, MaternKernel
from grakel_replace.mixed_single_task_gp import MixedSingleTaskGP
from grakel_replace.torch_wl_kernel import TorchWLKernel

TRAIN_CONFIGS = 10
TEST_CONFIGS = 10
TOTAL_CONFIGS = TRAIN_CONFIGS + TEST_CONFIGS

N_NUMERICAL = 2
N_CATEGORICAL = 2
N_CATEGORICAL_VALUES_PER_CATEGORY = 3
N_GRAPH = 2

kernels = []

# Create numerical and categorical features
X = torch.empty(size=(TOTAL_CONFIGS, N_NUMERICAL + N_CATEGORICAL), dtype=torch.float64)
if N_NUMERICAL > 0:
    X[:, :N_NUMERICAL] = torch.rand(
        size=(TOTAL_CONFIGS, N_NUMERICAL),
        dtype=torch.float64,
    )

if N_CATEGORICAL > 0:
    X[:, N_NUMERICAL:] = torch.randint(
        0,
        N_CATEGORICAL_VALUES_PER_CATEGORY,
        size=(TOTAL_CONFIGS, N_CATEGORICAL),
        dtype=torch.float64,
    )

# Create random graph architectures
graphs = []
for _ in range(TOTAL_CONFIGS):
    G = nx.erdos_renyi_graph(n=5, p=0.5)  # Random graph with 5 nodes
    graphs.append(G)

# Create random target values
y = torch.rand(size=(TOTAL_CONFIGS,), dtype=torch.float64)

# Setup kernels for numerical and categorical features
if N_NUMERICAL > 0:
    matern = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=N_NUMERICAL,
            active_dims=tuple(range(N_NUMERICAL)),
        ),
    )
    kernels.append(matern)

if N_CATEGORICAL > 0:
    hamming = ScaleKernel(
        CategoricalKernel(
            ard_num_dims=N_CATEGORICAL,
            active_dims=tuple(range(N_NUMERICAL, N_NUMERICAL + N_CATEGORICAL)),
        ),
    )
    kernels.append(hamming)

# Combine numerical and categorical kernels
combined_num_cat_kernel = AdditiveKernel(*kernels) if kernels else None

# Create WL kernel for graphs
wl_kernel = TorchWLKernel(n_iter=5, normalize=True)

# Split into train and test sets
train_x = X[:TRAIN_CONFIGS]
train_graphs = graphs[:TRAIN_CONFIGS]
train_y = y[:TRAIN_CONFIGS].unsqueeze(-1)  # Add dimension for botorch

test_x = X[TRAIN_CONFIGS:]
test_graphs = graphs[TRAIN_CONFIGS:]
test_y = y[TRAIN_CONFIGS:].unsqueeze(-1)

# Initialize the mixed GP
gp = MixedSingleTaskGP(
    train_X=train_x,
    train_graphs=train_graphs,
    train_Y=train_y,
    num_cat_kernel=combined_num_cat_kernel,
    wl_kernel=wl_kernel,
)

# Compute the posterior distribution
multivariate_normal: MultivariateNormal = gp.forward(train_x, train_graphs)
print("Posterior distribution:", multivariate_normal)

# Making predictions on test data
with torch.no_grad():
    posterior = gp.forward(test_x, test_graphs)
    predictions = posterior.mean
    uncertainties = posterior.variance.sqrt()
    covar = posterior.covariance_matrix

print("\nMean:", predictions)
print("Variance:", uncertainties)
print("Covariance matrix:", covar)
