from itertools import product

import networkx as nx
import torch
from botorch.acquisition import LinearMCObjective, qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression_mixed import CategoricalKernel, ScaleKernel
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import AdditiveKernel, MaternKernel
from grakel_replace.mixed_single_task_gp import MixedSingleTaskGP
from grakel_replace.optimize import optimize_acqf_graph
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

# =============== Fitting the GP using botorch ===============

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Define the acquisition function
acq_function = qLogNoisyExpectedImprovement(
    model=gp,
    X_baseline=X,
    objective=LinearMCObjective(weights=torch.tensor([-1.0])),
    prune_baseline=True,
)

# Define bounds
bounds = torch.tensor(
    [
        [0.0, 1.0] * N_NUMERICAL
        + [0.0, N_CATEGORICAL_VALUES_PER_CATEGORY - 1] * N_CATEGORICAL
    ]
).view(2, -1)

cats_per_column: dict[int, list[float]] = {
    column_ix: [float(i) for i in range(N_CATEGORICAL_VALUES_PER_CATEGORY)]
    for column_ix in range(N_NUMERICAL, N_NUMERICAL + N_CATEGORICAL)
}

# Generate fixed categorical features
fixed_cats: list[dict[int, float]]
if len(cats_per_column) == 1:
    col, choice_indices = next(iter(cats_per_column.items()))
    fixed_cats = [{col: i} for i in choice_indices]
else:
    fixed_cats = [
        dict(zip(cats_per_column.keys(), combo))
        for combo in product(*cats_per_column.values())
    ]

# Use the graph-optimized acquisition function
best_candidate, best_score = optimize_acqf_graph(
    acq_function=acq_function,
    bounds=bounds,
    fixed_features_list=fixed_cats,
    train_graphs=graphs,
    num_graph_samples=10,  # Number of graphs to sample
    num_restarts=3,
    raw_samples=250,
    q=1,
)

print("Best candidate:", best_candidate)
print("Acquisition score:", best_score)
