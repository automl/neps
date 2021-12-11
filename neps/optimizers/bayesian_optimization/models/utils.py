import numpy as np
import torch


def normalize_y(y: torch.Tensor):
    y_mean = torch.mean(y) if isinstance(y, torch.Tensor) else np.mean(y)
    y_std = torch.std(y) if isinstance(y, torch.Tensor) else np.std(y)
    if y_std == 0:
        y_std = 1
    y = (y - y_mean) / y_std
    return y, y_mean, y_std


def unnormalize_y(y, y_mean, y_std, scale_std=False):
    """Similar to the undoing of the pre-processing step above, but on the output predictions"""
    if not scale_std:
        y = y * y_std + y_mean
    else:
        y *= y_std
    return y


def standardize_x(
    x: torch.Tensor, x_min: torch.Tensor = None, x_max: torch.Tensor = None
):
    """Standardize the vectorial input into a d-dimensional hypercube [0, 1]^d, where d is the number of features.
    if x_min ond x_max are supplied, x2 will be standardised using these instead. This is used when standardising the
    validation/test inputs.
    """
    if (x_min is not None and x_max is None) or (x_min is None and x_max is not None):
        raise ValueError(
            "Either *both* or *neither* of x_min, x_max need to be supplied!"
        )
    if x_min is None:
        x_min = torch.min(x, 0)[0]
        x_max = torch.max(x, 0)[0]
    x = (x - x_min) / (x_max - x_min)
    return x, x_min, x_max


def compute_log_marginal_likelihood(
    K_i: torch.Tensor,
    logDetK: torch.Tensor,
    y: torch.Tensor,
    normalize: bool = True,
    log_prior_dist=None,
):
    """Compute the zero mean Gaussian process log marginal likelihood given the inverse of Gram matrix K(x2,x2), its
    log determinant, and the training label vector y.
    Option:

    normalize: normalize the log marginal likelihood by the length of the label vector, as per the gpytorch
    routine.

    prior: A pytorch distribution object. If specified, the hyperparameter prior will be taken into consideration and
    we use Type-II MAP instead of Type-II MLE (compute log_posterior instead of log_evidence)
    """
    lml = (
        -0.5 * y.t() @ K_i @ y
        + 0.5 * logDetK
        - y.shape[0]
        / 2.0
        * torch.log(
            2
            * torch.tensor(
                np.pi,
            )
        )
    )
    if log_prior_dist is not None:
        lml -= log_prior_dist
    return lml / y.shape[0] if normalize else lml


def compute_pd_inverse(K: torch.tensor, jitter: float = 1e-5):
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion."""
    n = K.shape[0]
    assert (
        isinstance(jitter, float) or jitter.ndim == 0
    ), "only homoscedastic noise variance is allowed here!"
    is_successful = False
    fail_count = 0
    max_fail = 3
    while fail_count < max_fail and not is_successful:
        try:
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10 ** fail_count
            K_ = K + jitter_diag
            Kc = torch.linalg.cholesky(K_)
            is_successful = True
        except RuntimeError:
            fail_count += 1
    if not is_successful:
        print(K)
        raise RuntimeError("Gram matrix not positive definite despite of jitter")
    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.float(), logDetK.float()
