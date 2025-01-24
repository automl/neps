from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import LinearMCObjective, qLogNoisyExpectedImprovement
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import CategoricalKernel, ScaleKernel
from botorch.optim import optimize_acqf_mixed
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import AdditiveKernel, MaternKernel

if TYPE_CHECKING:
    from gpytorch.distributions.multivariate_normal import MultivariateNormal

TRAIN_CONFIGS = 10
TEST_CONFIGS = 10
TOTAL_CONFIGS = TRAIN_CONFIGS + TEST_CONFIGS

N_NUMERICAL = 2
N_CATEGORICAL = 2
N_CATEGORICAL_VALUES_PER_CATEGORY = 3

kernels = []

# Create some random encoded hyperparameter configurations
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

y = torch.rand(size=(TOTAL_CONFIGS,), dtype=torch.float64)

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

combined_num_cat_kernel = AdditiveKernel(*kernels)

train_x = X[:TRAIN_CONFIGS]
train_y = y[:TRAIN_CONFIGS]

test_x = X[TRAIN_CONFIGS:]
test_y = y[TRAIN_CONFIGS:]

K_matrix = combined_num_cat_kernel.forward(train_x, train_x)

train_y = train_y.unsqueeze(-1)
test_y = test_y.unsqueeze(-1)

gp = SingleTaskGP(
    train_X=train_x,
    train_Y=train_y,
    covar_module=combined_num_cat_kernel,
)

multivariate_normal: MultivariateNormal = gp.forward(train_x)

# =============== Fitting the GP using botorch ===============


mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

acq_function = qLogNoisyExpectedImprovement(
    model=gp,
    X_baseline=train_x,
    objective=LinearMCObjective(weights=torch.tensor([-1.0])),
    prune_baseline=True,
)

# Define bounds
bounds = torch.tensor(
    [
        [0.0] * N_NUMERICAL + [0.0] * N_CATEGORICAL,
        [1.0] * N_NUMERICAL + [
            float(N_CATEGORICAL_VALUES_PER_CATEGORY - 1)] * N_CATEGORICAL
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

best_candidate, best_score = optimize_acqf_mixed(
    acq_function=acq_function,
    bounds=bounds,
    fixed_features_list=fixed_cats,
    num_restarts=10,
    raw_samples=10,
    q=1,
)

