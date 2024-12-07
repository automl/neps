import torch
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import CategoricalKernel, ScaleKernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import AdditiveKernel, MaternKernel

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
print(
    "K_matrix: ", K_matrix.to_dense()
)

train_y = train_y.unsqueeze(-1)
test_y = test_y.unsqueeze(-1)

gp = SingleTaskGP(
    train_X=train_x,
    train_Y=train_y,
    covar_module=combined_num_cat_kernel,
)

multivariate_normal: MultivariateNormal = gp.forward(train_x)
print("Mean:", multivariate_normal.mean)
print("Variance:", multivariate_normal.variance)
print("Covariance matrix:", multivariate_normal.covariance_matrix)
