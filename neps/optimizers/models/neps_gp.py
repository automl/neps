

from typing import Any, Callable, Literal

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
        print(f"Batch acquisition step {step+1}/{n}.",end="\r" if step < n - 1 else "\n", flush=True)
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


# ---------------------------------------------------------------------------
# Tuned surrogates over arbitrary (dict, dict) -> float kernel functions.
#
# Both operate directly on the config-dict kernel (no vector encoding) and
# expose the same .predict(config) -> (mean, var) interface as
# KernelSurrogateModel so the existing acquisition code works unchanged.
#
#   TunedKernelSurrogateModel — KRR with the kernel's hyperparameters tuned
#       by closed-form leave-one-out CV each fit (**hp injection).
#   GPSurrogateModel          — exact GP with noise + kernel hyperparameters
#       set by "none" / "loo_cv" / "mll" / "mll_noise_only" each fit.
# ---------------------------------------------------------------------------


class GPSurrogateModel:
    """Exact GP surrogate over an arbitrary (dict, dict) -> float kernel.

    y = f(x) + eps, eps ~ N(0, noise). The kernel's hyperparameters are
    injected via **hp (kernel_fn must accept them as keyword arguments);
    the observation noise enters the Gram matrix as K + noise * I.

    hp_tuning selects how (noise, kernel hyperparameters) are set each fit:
      * "none"           — fixed defaults (noise=1e-6, kernel defaults)
      * "loo_cv"         — coordinate-wise grid over kernel hyperparameters
                           and noise, scored by closed-form LOO MSE
      * "mll"            — scipy L-BFGS-B on the log marginal likelihood
                           over noise + kernel hyperparameters jointly
      * "mll_noise_only" — MLL over noise only, kernel hyperparameters at
                           their defaults (cheap: the Gram matrix is fixed
                           during the optimization)
    """

    def __init__(
        self,
        training_configs: list[dict],
        training_y: np.ndarray,
        kernel_fn: Callable,
        tunable_params: dict[str, tuple[float, float]] | None = None,
        hp_tuning: Literal["none", "loo_cv", "mll", "mll_noise_only"] = "none",
        noise: float = 1e-6,
        n_samples: int = 5,
        seed: int = 0,
    ):
        self.kernel_fn = kernel_fn
        self.tunable_params = dict(tunable_params or {})
        self.hp_tuning = hp_tuning
        self.noise = noise
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self.y = np.asarray(training_y, dtype=np.float64)
        self._fit(list(training_configs))

    def _noise_bounds(self) -> tuple[float, float]:
        y_var = float(np.var(self.y)) or 1.0
        return (1e-10, y_var + 1.0)

    def _gram(self, configs: list[dict], **hp) -> np.ndarray:
        n = len(configs)
        K = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                k = float(self.kernel_fn(configs[i], configs[j], **hp))
                K[i, j] = K[j, i] = k
        return K

    def _loo_mse(self, K: np.ndarray, noise: float) -> float:
        A = K + noise * np.eye(len(self.y))
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return np.inf
        loo = self.y - (A_inv @ self.y) / np.diag(A_inv)
        return float(np.mean((loo - self.y) ** 2))

    def _tune_loo(self, configs: list[dict]) -> tuple[dict, float]:
        """Coordinate-wise grid over kernel hyperparameters + noise."""
        grids: dict[str, list[float]] = {}
        for name, (lo, hi) in {
            **self.tunable_params, "noise": self._noise_bounds()
        }.items():
            lo, hi = float(lo), float(hi)
            if lo <= 0 or hi <= lo:
                grids[name] = [lo]
            else:
                grids[name] = list(
                    np.exp(np.linspace(np.log(lo), np.log(hi), self.n_samples))
                )
        best_hp: dict = {}
        best_noise = self.noise
        best = self._loo_mse(self._gram(configs, **best_hp), best_noise)
        for name, values in grids.items():
            for v in values:
                if name == "noise":
                    hp, noise = best_hp, float(v)
                else:
                    hp, noise = {**best_hp, name: float(v)}, best_noise
                mse = self._loo_mse(self._gram(configs, **hp), noise)
                if mse < best:
                    best, best_hp, best_noise = mse, hp, noise
        return best_hp, best_noise

    def _tune_mll(self, configs: list[dict]) -> tuple[dict, float]:
        """MLL optimization. "mll": noise + kernel hyperparameters jointly;
        "mll_noise_only": noise only (kernel hyperparameters at defaults)."""
        from scipy.optimize import minimize

        n = len(self.y)
        names = list(self.tunable_params) if self.hp_tuning == "mll" else []
        # theta = [log noise, log hp_1, ..., log hp_k]; init at the fixed
        # default noise / geometric mean of each hyperparameter's bounds.
        theta0 = [np.log(self.noise)]
        bounds = [tuple(np.log(self._noise_bounds()))]
        for name in names:
            lo, hi = self.tunable_params[name]
            bounds.append((np.log(lo), np.log(hi)))
            theta0.append(0.5 * (np.log(lo) + np.log(hi)))

        def neg_mll(theta: np.ndarray) -> float:
            noise = float(np.exp(theta[0]))
            hp = {name: float(np.exp(t)) for name, t in zip(names, theta[1:])}
            K = self._gram(configs, **hp) + noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e10
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            return float(
                0.5 * self.y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi)
            )

        res = minimize(neg_mll, np.array(theta0), method="L-BFGS-B", bounds=bounds)
        noise = float(np.exp(res.x[0]))
        hp = {name: float(np.exp(t)) for name, t in zip(names, res.x[1:])}
        return hp, noise

    def _fit(self, training_configs: list[dict]) -> None:
        if self.hp_tuning == "none":
            hp, noise = {}, self.noise
        elif self.hp_tuning == "loo_cv":
            hp, noise = self._tune_loo(training_configs)
        elif self.hp_tuning in ("mll", "mll_noise_only"):
            hp, noise = self._tune_mll(training_configs)
        else:
            raise ValueError(f"Unknown hp_tuning: {self.hp_tuning}")
        self.hp, self.noise = hp, noise

        K = self._gram(training_configs, **hp) + noise * np.eye(len(self.y))
        # Cholesky with jitter fallback for near-singular Gram matrices.
        jitter = 0.0
        for _ in range(10):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(len(self.y)))
                break
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10.0, 1e-12)
        else:
            raise np.linalg.LinAlgError("GP Gram matrix not PSD even with jitter")
        Li_y = np.linalg.solve(L, self.y)
        self.alpha = np.linalg.solve(L.T, Li_y)
        self.K_inv = np.linalg.inv(K)
        self._train_configs = training_configs

    def predict(self, config: dict) -> tuple[float, float]:
        k_test = np.array(
            [float(self.kernel_fn(config, c, **self.hp)) for c in self._train_configs]
        )
        pred_mean = float(k_test @ self.alpha)
        k_diag = float(self.kernel_fn(config, config, **self.hp))
        pred_var = float(k_diag - k_test @ self.K_inv @ k_test)
        return pred_mean, max(pred_var, 1e-8)


class TunedKernelSurrogateModel:
    """KRR surrogate over an arbitrary (dict, dict) -> float kernel function,
    tuning the kernel's hyperparameters by closed-form leave-one-out CV.

    The kernel function must accept the tunable hyperparameters as keyword
    arguments:

        kernel_fn(config_i: dict, config_j: dict, **hp) -> float

    Calling it without **hp must yield the default (untuned) kernel, so the
    same closure works for both tuned and untuned surrogates.

    tunable_params maps hyperparameter name -> (low, high) bounds. Each fit,
    a coordinate-wise grid search is run: for each hyperparameter in turn,
    `n_samples` log-spaced values within its bounds are evaluated (holding
    the other hyperparameters at their current best values), scored by the
    closed-form LOO MSE, and the best setting kept. The empty default
    setting is always evaluated first, so with tune=False or no
    tunable_params the model reduces to the plain untuned KRR:

        A = K + krr_alpha * I
        loo_i = y_i - [A^-1 y]_i / [A^-1]_ii
    """

    def __init__(
        self,
        training_configs: list[dict],
        training_y: np.ndarray,
        kernel_fn: Callable,
        tunable_params: dict[str, tuple[float, float]] | None = None,
        krr_alpha: float = 1e-6,
        tune: bool = True,
        n_samples: int = 5,
        seed: int = 0,
    ):
        self.kernel_fn = kernel_fn
        self.tunable_params = dict(tunable_params or {})
        self.krr_alpha = krr_alpha
        self.tune = tune
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)
        self.y = np.asarray(training_y, dtype=np.float64)
        self._fit(list(training_configs))

    def _sample_hp(self) -> list[dict]:
        """Coordinate-wise grid candidates: the empty default, then for each
        tunable hyperparameter in turn, log-spaced values within its bounds
        (the other hyperparameters stay at their defaults/current best)."""
        settings = [{}]
        if self.tune and self.tunable_params:
            for name, (lo, hi) in self.tunable_params.items():
                lo, hi = float(lo), float(hi)
                if lo <= 0 or hi <= lo:
                    values = [lo]
                else:
                    values = np.exp(
                        np.linspace(np.log(lo), np.log(hi), self.n_samples)
                    )
                for v in values:
                    settings.append({name: float(v)})
        return settings

    def _gram(self, configs: list[dict], **hp) -> np.ndarray:
        n = len(configs)
        K = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                k = float(self.kernel_fn(configs[i], configs[j], **hp))
                K[i, j] = K[j, i] = k
        return K

    def _fit(self, training_configs: list[dict]) -> None:
        best = (np.inf, {})
        for hp in self._sample_hp():
            K = self._gram(training_configs, **hp)
            A = K + self.krr_alpha * np.eye(len(self.y))
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                continue
            A_inv_y = A_inv @ self.y
            loo = self.y - A_inv_y / np.diag(A_inv)
            mse = float(np.mean((loo - self.y) ** 2))
            if mse < best[0]:
                best = (mse, hp)
        self.hp = best[1]
        K = self._gram(training_configs, **self.hp)
        A = K + self.krr_alpha * np.eye(len(self.y))
        self.A_inv = np.linalg.inv(A)
        self.coefficients = self.A_inv @ self.y
        self._train_configs = training_configs

    def predict(self, config: dict) -> tuple[float, float]:
        k_test = np.array(
            [float(self.kernel_fn(config, c, **self.hp)) for c in self._train_configs]
        )
        pred_mean = float(k_test @ self.coefficients)
        pred_var = float(1.0 - k_test @ self.A_inv @ k_test)
        return pred_mean, max(pred_var, 1e-8)
