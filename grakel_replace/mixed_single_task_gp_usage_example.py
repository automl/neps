from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import product
from typing import TYPE_CHECKING

import networkx as nx
import torch
from botorch import fit_gpytorch_mll, settings
from botorch.acquisition import LinearMCObjective, qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import CategoricalKernel, Kernel, ScaleKernel
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel, MaternKernel
from grakel_replace.optimize import optimize_acqf_graph
from grakel_replace.torch_wl_kernel import TorchWLKernel
from grakel_replace.utils import min_max_scale, seed_all

if TYPE_CHECKING:
    from gpytorch.distributions.multivariate_normal import MultivariateNormal

start_time = time.time()
settings.debug._set_state(True)
seed_all()

TRAIN_CONFIGS = 50
TEST_CONFIGS = 10
TOTAL_CONFIGS = TRAIN_CONFIGS + TEST_CONFIGS

N_NUMERICAL = 2
N_CATEGORICAL = 1
N_CATEGORICAL_VALUES_PER_CATEGORY = 2
N_GRAPH = 1

assert N_GRAPH == 1, "This example only supports a single graph feature"

# Generate random data
X = torch.cat([
    torch.rand((TOTAL_CONFIGS, N_NUMERICAL), dtype=torch.float64),
    torch.randint(0, N_CATEGORICAL_VALUES_PER_CATEGORY, (TOTAL_CONFIGS, N_CATEGORICAL),
                  dtype=torch.float64),
    torch.arange(TOTAL_CONFIGS, dtype=torch.float64).unsqueeze(1)
], dim=1)

# Generate random graphs
graphs = [nx.erdos_renyi_graph(5, 0.5) for _ in range(TOTAL_CONFIGS)]

# Generate random target values
y = torch.rand(TOTAL_CONFIGS, dtype=torch.float64) + 0.5

# Split into train and test sets
train_x, test_x = X[:TRAIN_CONFIGS], X[TRAIN_CONFIGS:]
train_graphs, test_graphs = graphs[:TRAIN_CONFIGS], graphs[TRAIN_CONFIGS:]
train_y, test_y = y[:TRAIN_CONFIGS].unsqueeze(-1), y[TRAIN_CONFIGS:].unsqueeze(-1)

train_x, test_x = min_max_scale(train_x), min_max_scale(test_x)

kernels = [
    ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=N_NUMERICAL, active_dims=range(N_NUMERICAL))),
    ScaleKernel(CategoricalKernel(
        ard_num_dims=N_CATEGORICAL,
        active_dims=range(N_NUMERICAL, N_NUMERICAL + N_CATEGORICAL))),
    ScaleKernel(TorchWLKernel(
        graph_lookup=train_graphs, n_iter=5, normalize=True,
        active_dims=(X.shape[1] - 1,)))
]

gp = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=AdditiveKernel(*kernels))

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

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

acq_function = qLogNoisyExpectedImprovement(
    model=gp,
    X_baseline=train_x,
    objective=LinearMCObjective(weights=torch.tensor([-1.0])),
    prune_baseline=True,
)

bounds = torch.tensor([
    [0.0] * N_NUMERICAL + [0.0] * N_CATEGORICAL + [-1.0] * N_GRAPH,
    [1.0] * N_NUMERICAL + [
        float(N_CATEGORICAL_VALUES_PER_CATEGORY - 1)] * N_CATEGORICAL + [
        len(X) - 1] * N_GRAPH,
])

cats_per_column = {i: list(range(N_CATEGORICAL_VALUES_PER_CATEGORY)) for i in
                   range(N_NUMERICAL, N_NUMERICAL + N_CATEGORICAL)}
fixed_cats = [dict(zip(cats_per_column.keys(), combo, strict=False)) for combo in
              product(*cats_per_column.values())]

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

print(f"Best candidate: {best_candidate}")
print(f"Best score: {best_score}")
print(f"Elapsed time: {time.time() - start_time} seconds")
