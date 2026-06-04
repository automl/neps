

import numpy as np



def _expected_improvement(
    pred_mean: float,
    pred_var: float,
    best_y: float,
) -> float:
    """
    Compute Expected Improvement acquisition function.

    For minimization, EI = (best_y - pred_mean) * Phi(Z) + sqrt(pred_var) * phi(Z)
    where Z = (best_y - pred_mean) / sqrt(pred_var)

    Args:
        pred_mean: Predicted objective value
        pred_var: Predicted variance
        best_y: Best observed objective so far

    Returns:
        Expected Improvement score
    """
    from scipy.stats import norm

    # Standardized improvement
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = best_y - pred_mean
        Z = imp / np.sqrt(pred_var)

        # EI = improvement * CDF(Z) + std * PDF(Z)
        ei = imp * norm.cdf(Z) + np.sqrt(pred_var) * norm.pdf(Z)

        # Handle edge cases
        ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

    return float(ei)


def _greedy_batch_acquisition(
    pred_means: np.ndarray,
    pred_vars: np.ndarray,
    best_y: float,
    kernel_fn: callable,
    candidate_configs: list[dict],
    acqu_func: callable = _expected_improvement,
) -> np.ndarray:
    """
    Compute batch-aware acquisition using greedy selection (approximates qLogNoisyEI).

    This greedy batch acquisition:
    1. Selects the highest-acqu candidate first (using acqu_func)
    2. Updates predictions for remaining candidates (reducing uncertainty for similar ones)
    3. Repeats to select the next best candidate

    This accounts for the correlation between batch members through the kernel.
    Works with any acquisition function passed via acqu_func parameter.

    Args:
        pred_means: Array of predicted means for all candidates
        pred_vars: Array of predicted variances for all candidates
        best_y: Best observed value
        kernel_fn: Kernel function to measure candidate similarity
        candidate_configs: List of hierarchical config dicts for candidates
        acqu_func: Acquisition function to use (EI or LogEI). Default: _expected_improvement

    Returns:
        Greedy batch acquisition scores (same length as inputs)
    """
    n = len(pred_means)
    batch_scores = np.zeros(n)
    selected_indices = []

    # Greedy selection: iteratively pick best remaining candidate
    for step in range(n):
        # Compute acquisition for all remaining candidates
        acqu_vals = np.array([
            acqu_func(pred_means[i], pred_vars[i], best_y)
            if i not in selected_indices else -np.inf
            for i in range(n)
        ])

        # Select highest acquisition value
        best_idx = np.argmax(acqu_vals)
        selected_indices.append(best_idx)
        batch_scores[best_idx] = acqu_vals[best_idx]

        # Update predictions for unselected candidates based on selected one
        # The idea: once we pick a candidate, uncertainty decreases for similar candidates
        if step < n - 1:
            selected_config = candidate_configs[best_idx]
            for i in range(n):
                if i not in selected_indices:
                    # Similarity to the just-selected candidate
                    similarity = kernel_fn(candidate_configs[i], selected_config)
                    # Reduce variance for similar candidates (they provide similar info)
                    pred_vars[i] *= (1.0 - similarity ** 2)

    return batch_scores


def _log_expected_improvement(
    pred_mean: float,
    pred_var: float,
    best_y: float,
) -> float:
    """
    Compute Expected Improvement in log space (more numerically stable for large ranges).

    Uses log-transformed EI to avoid numerical issues when improvement is very large/small.

    Args:
        pred_mean: Predicted objective value
        pred_var: Predicted variance
        best_y: Best observed objective so far

    Returns:
        Log of Expected Improvement score
    """
    ei = _expected_improvement(pred_mean, pred_var, best_y)
    # Avoid log(0)
    return float(np.log(ei + 1e-8))


def _hamming_kernel(config_i: dict, config_j: dict, length_scale: float = 1.0) -> float:
    """
    Compute similarity between two hierarchical configs using Hamming distance kernel.

    Args:
        config_i, config_j: Dicts mapping decision paths to choice indices
        length_scale: Controls how quickly similarity decays with distance

    Returns:
        Similarity score in [0, 1], where 1 = identical configs
    """
    # Get all keys from both configs
    all_keys = set(config_i.keys()) | set(config_j.keys())

    if not all_keys:
        return 1.0

    # Count differing decisions
    differences = 0
    for key in all_keys:
        val_i = config_i.get(key)
        val_j = config_j.get(key)
        # If one config doesn't have the key (different branches), count as maximum difference
        if val_i is None or val_j is None:
            differences += 1
        elif val_i != val_j:
            differences += 1

    # Normalize Hamming distance
    normalized_distance = differences / len(all_keys)

    # Convert to similarity using RBF-like kernel
    return float(np.exp(-normalized_distance / length_scale))


def _compute_gram_matrix(
    configs: list[dict],
    kernel_fn: callable = _hamming_kernel,
) -> np.ndarray:
    """
    Build gram matrix K[i,j] = kernel(config_i, config_j).

    Args:
        configs: List of hierarchical config dicts
        kernel_fn: Kernel function to use

    Returns:
        Symmetric positive semi-definite gram matrix of shape (n, n)
    """
    n = len(configs)
    K = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            k_val = kernel_fn(configs[i], configs[j])
            K[i, j] = k_val
            K[j, i] = k_val

    return K


def _kernel_ridge_regression(
    K: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-6,
) -> np.ndarray:
    """
    Fit Kernel Ridge Regression to training data.

    Solves: (K + alpha*I) @ coefficients = y

    Args:
        K: Gram matrix of shape (n, n)
        y: Target values of shape (n,)
        alpha: Regularization parameter (lambda in standard KRR)

    Returns:
        Coefficient vector for making predictions
    """
    n = len(y)
    regularized_K = K + alpha * np.eye(n)
    coefficients = np.linalg.solve(regularized_K, y)
    return coefficients


class KernelSurrogateModel:
    """Kernel-based surrogate model for hierarchical configs."""

    def __init__(
        self,
        training_configs: list[dict],
        training_y: np.ndarray,
        kernel_fn: callable = _hamming_kernel,
        krr_alpha: float = 1e-6,
    ):
        """
        Initialize the surrogate model.

        Args:
            training_configs: List of completed config dicts
            training_y: Array of objectives
            kernel_fn: Kernel function
            krr_alpha: KRR regularization parameter
        """
        self.training_configs = training_configs
        self.training_y = training_y
        self.kernel_fn = kernel_fn

        # Precompute gram matrix and fit KRR
        self.K_train = _compute_gram_matrix(training_configs, kernel_fn)
        self.coefficients = _kernel_ridge_regression(self.K_train, training_y, alpha=krr_alpha)

        # Store for prediction
        self.K_inv = np.linalg.inv(self.K_train + krr_alpha * np.eye(len(training_configs)))
        self.krr_alpha = krr_alpha

    def predict(self, config: dict) -> tuple[float, float]:
        """
        Predict mean and variance at a new config.

        Args:
            config: Hierarchical config dict

        Returns:
            (predicted_mean, predicted_variance)
        """
        # Compute kernel vector: k_test[i] = kernel(config, training_config_i)
        k_test = np.array([
            self.kernel_fn(config, train_config)
            for train_config in self.training_configs
        ])

        # Predicted mean: y_pred = k_test^T @ coefficients
        pred_mean = float(k_test @ self.coefficients)

        # Predicted variance: var = k(x,x) - k_test^T @ K_inv @ k_test
        k_diag = self.kernel_fn(config, config)
        pred_var = float(k_diag - k_test @ self.K_inv @ k_test)

        # Ensure variance is non-negative (numerical stability)
        pred_var = max(pred_var, 1e-8)

        return pred_mean, pred_var
