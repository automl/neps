from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from typing import TYPE_CHECKING

import networkx as nx
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import LinearMCObjective, qLogNoisyExpectedImprovement
from botorch.models.gp_regression_mixed import CategoricalKernel, Kernel, ScaleKernel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel, MaternKernel
from grakel_replace.optimize import optimize_acqf_graph
from grakel_replace.torch_wl_kernel import TorchWLKernel

if TYPE_CHECKING:
    from gpytorch.distributions.multivariate_normal import MultivariateNormal

TRAIN_CONFIGS = 50
TEST_CONFIGS = 10
TOTAL_CONFIGS = TRAIN_CONFIGS + TEST_CONFIGS

N_NUMERICAL = 2
N_CATEGORICAL = 1
N_CATEGORICAL_VALUES_PER_CATEGORY = 2
N_GRAPH = 1
assert N_GRAPH == 1, "This example only supports a single graph feature"

kernels = []

# Create numerical and categorical features
X = torch.empty(
    size=(TOTAL_CONFIGS, N_NUMERICAL + N_CATEGORICAL + N_GRAPH),
    dtype=torch.float64,
)
if N_NUMERICAL > 0:
    X[:, :N_NUMERICAL] = torch.rand(
        size=(TOTAL_CONFIGS, N_NUMERICAL),
        dtype=torch.float64,
    )

if N_CATEGORICAL > 0:
    X[:, N_NUMERICAL : N_NUMERICAL + N_CATEGORICAL] = torch.randint(
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

# Assign a new index column to the graphs
X[:, -1] = torch.arange(TOTAL_CONFIGS, dtype=torch.float64)

# Create random target values
y = torch.rand(size=(TOTAL_CONFIGS,), dtype=torch.float64) + 0.5

# Split into train and test sets
train_x = X[:TRAIN_CONFIGS]
train_graphs = graphs[:TRAIN_CONFIGS]
train_y = y[:TRAIN_CONFIGS].unsqueeze(-1)  # Add dimension for botorch

test_x = X[TRAIN_CONFIGS:]
test_graphs = graphs[TRAIN_CONFIGS:]
test_y = y[TRAIN_CONFIGS:].unsqueeze(-1)


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

if N_GRAPH > 0:
    wl_kernel = ScaleKernel(
        TorchWLKernel(
            graph_lookup=train_graphs,
            n_iter=5,
            normalize=True,
            active_dims=(X.shape[1] - 1,),  # Last column
        )
    )
    kernels.append(wl_kernel)


# Combine numerical and categorical kernels
kernel = AdditiveKernel(*kernels)

from botorch.models import SingleTaskGP

# Initialize the mixed GP
gp = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=kernel)

# Compute the posterior distribution
# The wl_kernel will use the indices to index into the training graphs it is holding
# on to...
multivariate_normal: MultivariateNormal = gp.forward(train_x)


# Making predictions on test data
# No the wl_kernel needs to be aware of the test graphs
@contextmanager
def set_graph_lookup(_gp: SingleTaskGP, new_graphs: list[nx.Graph]) -> Iterator[None]:
    kernel_prev_graphs: list[tuple[Kernel, list[nx.Graph]]] = []
    for kern in _gp.covar_module.sub_kernels():
        if isinstance(kern, TorchWLKernel):
            kernel_prev_graphs.append((kern, kern.graph_lookup))
            kern.set_graph_lookup(new_graphs)

    yield

    for _kern, _prev_graphs in kernel_prev_graphs:
        _kern.set_graph_lookup(_prev_graphs)


with torch.no_grad(), set_graph_lookup(gp, train_graphs + test_graphs):
    posterior = gp.forward(test_x)
    predictions = posterior.mean
    uncertainties = posterior.variance.sqrt()
    covar = posterior.covariance_matrix

# =============== Fitting the GP using botorch ===============


mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

# Define the acquisition function
acq_function = qLogNoisyExpectedImprovement(
    model=gp,
    X_baseline=train_x,
    objective=LinearMCObjective(weights=torch.tensor([-1.0])),
    prune_baseline=True,
)

# Define bounds
bounds = torch.tensor(
    [
        [0.0] * N_NUMERICAL + [0.0] * N_CATEGORICAL + [-1.0] * N_GRAPH,
        [1.0] * N_NUMERICAL
        + [float(N_CATEGORICAL_VALUES_PER_CATEGORY - 1)] * N_CATEGORICAL
        + [len(X) - 1] * N_GRAPH,
    ]
)

# Setup categorical feature optimization
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
        dict(zip(cats_per_column.keys(), combo, strict=False))
        for combo in product(*cats_per_column.values())
    ]


print("------------------")  # noqa: T201
# Use the graph-optimized acquisition function
best_candidate, best_score = optimize_acqf_graph(
    acq_function=acq_function,
    bounds=bounds,
    fixed_features_list=fixed_cats,
    train_graphs=train_graphs,
    num_graph_samples=2,
    num_restarts=2,
    raw_samples=16,
    q=1,
)
