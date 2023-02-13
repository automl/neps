import itertools
import logging
import warnings
from copy import deepcopy
from typing import Iterable, Union

import numpy as np
import torch

from ..kernels.combine_kernels_hierarchy import ProductKernel, SumKernel

# GP model as a weighted average between the vanilla vectorial GP and the graph GP
from ..kernels.graph_kernel import GraphKernels
from ..kernels.utils import extract_configs_hierarchy
from ..kernels.vectorial_kernels import Stationary
from ..kernels.weisfilerlehman import WeisfilerLehman


# Code for psd_safe_cholesky from gypytorch
class _value_context:
    _global_value = None

    @classmethod
    def value(cls):
        return cls._global_value

    @classmethod
    def _set_value(cls, value):
        cls._global_value = value

    def __init__(self, value):
        self._orig_value = self.__class__.value()
        self._instance_value = value

    def __enter__(
        self,
    ):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


class _dtype_value_context:
    _global_float_value = None
    _global_double_value = None
    _global_half_value = None

    @classmethod
    def value(cls, dtype):
        if torch.is_tensor(dtype):
            dtype = dtype.dtype
        if dtype == torch.float:
            return cls._global_float_value
        elif dtype == torch.double:
            return cls._global_double_value
        elif dtype == torch.half:
            return cls._global_half_value
        else:
            raise RuntimeError(f"Unsupported dtype for {cls.__name__}.")

    @classmethod
    def _set_value(cls, float_value, double_value, half_value):
        if float_value is not None:
            cls._global_float_value = float_value
        if double_value is not None:
            cls._global_double_value = double_value
        if half_value is not None:
            cls._global_half_value = half_value

    def __init__(
        self, float=None, double=None, half=None  # pylint: disable=redefined-builtin
    ):
        self._orig_float_value = (
            self.__class__.value()  # pylint: disable=no-value-for-parameter
        )
        self._instance_float_value = float
        self._orig_double_value = (
            self.__class__.value()  # pylint: disable=no-value-for-parameter
        )
        self._instance_double_value = double
        self._orig_half_value = (
            self.__class__.value()  # pylint: disable=no-value-for-parameter
        )
        self._instance_half_value = half

    def __enter__(
        self,
    ):
        self.__class__._set_value(
            self._instance_float_value,
            self._instance_double_value,
            self._instance_half_value,
        )

    def __exit__(self, *args):
        self.__class__._set_value(
            self._orig_float_value, self._orig_double_value, self._orig_half_value
        )
        return False


class cholesky_jitter(_dtype_value_context):
    """
    The jitter value used by `psd_safe_cholesky` when using cholesky solves.
    - Default for `float`: 1e-6
    - Default for `double`: 1e-8
    """

    _global_float_value = 1e-6  # type: ignore[assignment]
    _global_double_value = 1e-8  # type: ignore[assignment]

    @classmethod
    def value(cls, dtype=None):
        if dtype is None:
            # Deprecated in 1.4: remove in 1.5
            warnings.warn(
                "cholesky_jitter is now a _dtype_value_context and should be called with a dtype argument",
                DeprecationWarning,
            )
            return cls._global_float_value
        return super().value(dtype=dtype)


class _feature_flag:
    r"""Base class for feature flag settings with global scope.
    The default is set via the `_default` class attribute.
    """

    _default = False
    _state = None

    @classmethod
    def is_default(cls):
        return cls._state is None

    @classmethod
    def on(cls):
        if cls.is_default():
            return cls._default
        return cls._state

    @classmethod
    def off(cls):
        return not cls.on()

    @classmethod
    def _set_state(cls, state):
        cls._state = state

    def __init__(self, state=True):
        self.prev = self.__class__._state
        self.state = state

    def __enter__(self):
        self.__class__._set_state(self.state)

    def __exit__(self, *args):
        self.__class__._set_state(self.prev)
        return False


class verbose_linalg(_feature_flag):
    """
    Print out information whenever running an expensive linear algebra routine (e.g. Cholesky, CG, Lanczos, CIQ, etc.)
    (Default: False)
    """

    _default = False

    # Create a global logger
    logger = logging.getLogger("LinAlg (Verbose)")
    logger.setLevel(logging.DEBUG)

    # Output logging results to the stdout stream
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


class cholesky_max_tries(_value_context):
    """
    The max_tries value used by `psd_safe_cholesky` when using cholesky solves.
    (Default: 3)
    """

    _global_value = 3  # type: ignore[assignment]


class NumericalWarning(RuntimeWarning):
    """
    Warning thrown when convergence criteria are not met, or when comptuations require extra stability.
    """

    pass  # pylint: disable=unnecessary-pass


class NanError(RuntimeError):
    pass


class NotPSDError(RuntimeError):
    pass


def _psd_safe_cholesky(A, out=None, jitter=None, max_tries=None):
    # Maybe log
    if verbose_linalg.on():
        verbose_linalg.logger.debug(f"Running Cholesky on a matrix of size {A.shape}.")

    if out is not None:
        out = (out, torch.empty(A.shape[:-2], dtype=torch.int32, device=out.device))

    L, info = torch.linalg.cholesky_ex(A, out=out)
    if not torch.any(info):
        return L

    isnan = torch.isnan(A)
    if isnan.any():
        raise NanError(
            f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
        )

    if jitter is None:
        jitter = cholesky_jitter.value(A.dtype)
    if max_tries is None:
        max_tries = cholesky_max_tries.value()
    Aprime = A.clone()
    jitter_prev = 0
    for i in range(max_tries):
        jitter_new = jitter * (10**i)
        # add jitter only where needed
        diag_add = (
            ((info > 0) * (jitter_new - jitter_prev))
            .unsqueeze(-1)
            .expand(*Aprime.shape[:-1])
        )
        Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
        jitter_prev = jitter_new
        warnings.warn(
            f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal",
            NumericalWarning,
        )
        L, info = torch.linalg.cholesky_ex(Aprime, out=out)
        if not torch.any(info):
            return L
    raise NotPSDError(
        f"Matrix not positive definite after repeatedly adding jitter up to {jitter_new:.1e}."
    )


def psd_safe_cholesky(A, upper=False, out=None, jitter=None, max_tries=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        A (Tensor):
            The tensor to compute the Cholesky decomposition of
        upper (bool, optional):
            See torch.cholesky
        out (Tensor, optional):
            See torch.cholesky
        jitter (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
            uses settings.cholesky_jitter.value()
        max_tries (int, optional):
            Number of attempts (with successively increasing jitter) to make before raising an error.
    """
    L = _psd_safe_cholesky(A, out=out, jitter=jitter, max_tries=max_tries)
    if upper:
        if out is not None:
            out = out.transpose_(-1, -2)
        else:
            L = L.transpose(-1, -2)
    return L


# Code for psd_safe_cholesky from gypytorch


class ComprehensiveGPHierarchy:
    def __init__(
        self,
        graph_kernels: Iterable,
        hp_kernels: Iterable,
        likelihood: float = 1e-3,
        weights=None,
        learn_all_h=False,
        graph_feature_ard=True,
        d_graph_features: int = 0,
        normalize_combined_kernel=True,
        hierarchy_consider: list = None,  # or a list of integers e.g. [0,1,2,3]
        vectorial_features: list = None,
        combined_kernel: str = "sum",
        verbose: bool = False,
        surrogate_model_fit_args: dict = None,
        gpytorch_kinv: bool = False,
    ):
        self.likelihood = likelihood
        self.surrogate_model_fit_args = surrogate_model_fit_args or {}
        self.learn_all_h = learn_all_h
        self.hierarchy_consider = hierarchy_consider
        self.normalize_combined_kernel = normalize_combined_kernel
        if self.hierarchy_consider is None:
            self.learn_all_h = False
        self.domain_kernels: list = []
        if bool(graph_kernels):
            self.domain_kernels += list(graph_kernels)
        if bool(hp_kernels):
            self.domain_kernels += list(hp_kernels)

        self.hp_kernels = hp_kernels  # impose on scalar graph features
        self.n_kernels: int = len(self.domain_kernels)
        self.n_graph_kernels: int = len(
            [i for i in self.domain_kernels if isinstance(i, GraphKernels)]
        )
        self.n_vector_kernels: int = self.n_kernels - self.n_graph_kernels
        self.graph_feature_ard = graph_feature_ard
        self.vectorial_features = vectorial_features
        self.d_graph_features = d_graph_features

        if weights is not None:
            self.fixed_weights = True
            if weights is not None:
                assert len(weights) == self.n_kernels, (
                    "the weights vector, if supplied, needs to have the same length as "
                    "the number of kernel_operators!"
                )
            self.init_weights = (
                weights
                if isinstance(weights, torch.Tensor)
                else torch.tensor(weights).flatten()
            )
        else:
            self.fixed_weights = False
            # Initialise the domain kernel weights to uniform
            self.init_weights = torch.tensor(
                [1.0 / self.n_kernels] * self.n_kernels,
            )

        self.weights = self.init_weights.clone()

        if combined_kernel == "product":
            self.combined_kernel = ProductKernel(
                *self.domain_kernels,
                weights=self.weights,
                hierarchy_consider=self.hierarchy_consider,
                d_graph_features=self.d_graph_features,
            )
        elif combined_kernel == "sum":
            self.combined_kernel = SumKernel(
                *self.domain_kernels,
                weights=self.weights,
                hierarchy_consider=self.hierarchy_consider,
                d_graph_features=self.d_graph_features,
            )
        else:
            raise NotImplementedError(
                f'Combining kernel {combined_kernel} is not yet implemented! Only "sum" '
                f'or "product" are currently supported. '
            )
        # Verbose mode
        self.verbose = verbose
        # Cache the Gram matrix inverse and its log-determinant
        self.K, self.K_i, self.logDetK = [None] * 3
        self.layer_weights = None
        self.nlml = None

        self.x_configs: list = None  # type: ignore[assignment]
        self.y: torch.Tensor = None
        self.y_: torch.Tensor = None
        self.y_mean: torch.Tensor = None
        self.y_std: torch.Tensor = None
        self.n: int = None  # type: ignore[assignment]

        self.gpytorch_kinv = gpytorch_kinv

    def _optimize_graph_kernels(self, h_: int, lengthscale_):
        weights = self.init_weights.clone()
        if self.hierarchy_consider is None:
            graphs, _ = extract_configs_hierarchy(
                self.x_configs,
                d_graph_features=self.d_graph_features,
                hierarchy_consider=self.hierarchy_consider,
            )
            for i, k in enumerate(self.combined_kernel.kernels):
                if not isinstance(k, GraphKernels):
                    continue
                elif isinstance(k, WeisfilerLehman):
                    _grid_search_wl_kernel(
                        k,
                        h_,
                        [x[i] for x in graphs]
                        if isinstance(graphs[0], list)
                        else [c for c in graphs],
                        self.y,
                        self.likelihood,
                        lengthscales=lengthscale_,
                        gpytorch_kinv=self.gpytorch_kinv,
                    )
                else:
                    logging.warning(
                        "(Graph) kernel optimisation for "
                        + type(k).__name__
                        + " not implemented yet."
                    )
        else:
            if self.learn_all_h:
                best_nlml = torch.tensor(np.inf)
                best_subtree_depth_combo = None
                best_K = None
                train_y = self.y
                h_combo_candidates = generate_h_combo_candidates(self.hierarchy_consider)

                for h_combo in h_combo_candidates:
                    for i, k in enumerate(self.combined_kernel.kernels):
                        if isinstance(k, WeisfilerLehman):
                            k.change_kernel_params({"h": h_combo[i]})
                    K = self.combined_kernel.fit_transform(
                        weights,
                        self.x_configs,
                        normalize=self.normalize_combined_kernel,
                        layer_weights=None,
                        rebuild_model=True,
                        save_gram_matrix=True,
                    )
                    K_i, logDetK = compute_pd_inverse(
                        K, self.likelihood, self.gpytorch_kinv
                    )
                    nlml = -compute_log_marginal_likelihood(K_i, logDetK, train_y)
                    if nlml < best_nlml:
                        best_nlml = nlml
                        best_subtree_depth_combo = h_combo
                        best_K = torch.clone(K)
                for i, k in enumerate(self.combined_kernel.kernels):
                    if isinstance(k, WeisfilerLehman):
                        k.change_kernel_params({"h": best_subtree_depth_combo[i]})  # type: ignore[index]
                self.combined_kernel._gram = best_K  # pylint: disable=protected-access
            else:
                best_nlml = torch.tensor(np.inf)
                best_subtree_depth = None
                best_K = None
                train_y = self.y

                for h_i in list(h_):  # type: ignore[call-overload]
                    # only optimize h in wl kernel
                    if isinstance(self.combined_kernel.kernels[0], WeisfilerLehman):
                        self.combined_kernel.kernels[0].change_kernel_params({"h": h_i})
                        K = self.combined_kernel.fit_transform(
                            weights,
                            self.x_configs,
                            normalize=self.normalize_combined_kernel,
                            layer_weights=None,
                            rebuild_model=True,
                            save_gram_matrix=True,
                        )
                        K_i, logDetK = compute_pd_inverse(
                            K, self.likelihood, self.gpytorch_kinv
                        )
                        nlml = -compute_log_marginal_likelihood(K_i, logDetK, train_y)
                        # print(i, nlml)
                        if nlml < best_nlml:
                            best_nlml = nlml
                            best_subtree_depth = h_i
                            best_K = torch.clone(K)
                if isinstance(self.combined_kernel.kernels[0], WeisfilerLehman):
                    self.combined_kernel.kernels[0].change_kernel_params(
                        {"h": best_subtree_depth}
                    )
                    self.combined_kernel._gram = (  # pylint: disable=protected-access
                        best_K
                    )

    def fit(self, train_x: Iterable, train_y: Union[Iterable, torch.Tensor]):
        self._fit(train_x, train_y, **self.surrogate_model_fit_args)

    def _fit(
        self,
        train_x: Iterable,
        train_y: Union[Iterable, torch.Tensor],
        iters: int = 20,
        optimizer: str = "adam",
        wl_subtree_candidates: tuple = tuple(range(5)),
        wl_lengthscales: tuple = tuple(
            np.e**i for i in range(-2, 3)  # type: ignore[name-defined]
        ),
        optimize_lik: bool = True,
        max_lik: float = 0.5,  # pylint: disable=unused-argument
        optimize_wl_layer_weights: bool = False,
        optimizer_kwargs: dict = None,
    ):
        # Called by self._fit
        self._reset_XY(train_x, train_y)

        # Get the node weights, if needed
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.1}
        if len(wl_subtree_candidates) > 0:
            self._optimize_graph_kernels(
                wl_subtree_candidates,  # type: ignore[arg-type]
                wl_lengthscales,
            )

        weights = self.init_weights.clone()

        if (not self.fixed_weights) and len(self.domain_kernels) > 1:
            weights.requires_grad_(True)

        # set the prior values for the lengthscales of the two global features of the final architecture graph
        if self.graph_feature_ard:
            theta_vector = torch.log(torch.tensor([0.6, 0.6]))
        else:
            theta_vector = torch.log(torch.tensor([0.6]))

        # if use continuous graph properties and we set to use stationary kernels
        if self.d_graph_features > 0 and len(self.hp_kernels) > 0:  # type: ignore[arg-type]
            # TODO modify the code on theta_vector betlow to be compatibale with HPO
            # theta in this case are the lengthscales for the two global property of
            # the final architecture graph
            # theta_vector = get_theta_vector(vectorial_features=self.vectorial_features)
            theta_vector.requires_grad_(True)

        # Whether to include the likelihood (jitter or noise variance) as a hyperparameter
        likelihood = torch.tensor(
            self.likelihood,
        )
        if optimize_lik:
            likelihood.requires_grad_(True)

        layer_weights = None
        if optimize_wl_layer_weights:
            for k in self.domain_kernels:
                if isinstance(k, WeisfilerLehman):
                    layer_weights = torch.ones(k.h + 1).requires_grad_(True)
                    if layer_weights.shape[0] <= 1:
                        layer_weights = None
                    else:
                        break

        # Linking the optimizer variables to the sum kernel
        optim_vars = []
        # if theta_vector is not None: # TODO used for HPO
        #     for a in theta_vector.values():
        #         if a is not None and a.requires_grad:
        #             optim_vars.append(a)
        # if we use graph features, we will optimize the corresponding stationary kernel lengthscales
        if self.d_graph_features > 0 and theta_vector.requires_grad:
            optim_vars.append(theta_vector)

        for a in [weights, likelihood, layer_weights]:
            if a is not None and a.is_leaf and a.requires_grad:
                optim_vars.append(a)

        nlml = None
        if len(optim_vars) == 0:  # Skip optimisation
            K = self.combined_kernel.fit_transform(
                weights,
                self.x_configs,
                normalize=self.normalize_combined_kernel,
                feature_lengthscale=torch.exp(theta_vector),
                layer_weights=layer_weights,
                rebuild_model=True,
            )
            K_i, logDetK = compute_pd_inverse(K, likelihood, self.gpytorch_kinv)
        else:
            # Select the optimizer
            assert optimizer.lower() in ["adam", "sgd"]
            if optimizer.lower() == "adam":
                optim = torch.optim.Adam(optim_vars, **optimizer_kwargs)
            else:
                optim = torch.optim.SGD(optim_vars, **optimizer_kwargs)

            K = None
            optim_vars_list = []
            nlml_list = []
            for i in range(iters):
                optim.zero_grad()
                K = self.combined_kernel.fit_transform(
                    weights,
                    self.x_configs,
                    normalize=self.normalize_combined_kernel,
                    feature_lengthscale=torch.exp(theta_vector),
                    layer_weights=layer_weights,
                    rebuild_model=True,
                    save_gram_matrix=True,
                )
                K_i, logDetK = compute_pd_inverse(K, likelihood, self.gpytorch_kinv)
                nlml = -compute_log_marginal_likelihood(K_i, logDetK, self.y)
                nlml.backward(create_graph=True)
                if self.verbose and i % 10 == 0:
                    print(
                        "Iteration:",
                        i,
                        "/",
                        iters,
                        "Negative log-marginal likelihood:",
                        nlml.item(),
                        theta_vector,
                        weights,
                        likelihood,
                    )
                optim.step()

                with torch.no_grad():
                    likelihood.clamp_(  # pylint: disable=expression-not-assigned
                        1e-5, max_lik
                    ) if likelihood is not None and likelihood.is_leaf else None

                optim_vars_list.append(
                    [
                        theta_vector.clone().detach(),
                        weights.clone().detach(),
                        likelihood.clone().detach(),
                    ]
                )
                nlml_list.append(nlml.item())

                optim.zero_grad(set_to_none=True)

            theta_vector, weights, likelihood = optim_vars_list[np.argmin(nlml_list)]
            K = self.combined_kernel.fit_transform(
                weights,
                self.x_configs,
                normalize=self.normalize_combined_kernel,
                feature_lengthscale=torch.exp(theta_vector),
                layer_weights=layer_weights,
                rebuild_model=True,
                save_gram_matrix=True,
            )
            K_i, logDetK = compute_pd_inverse(K, likelihood, self.gpytorch_kinv)

        # Apply the optimal hyperparameters
        # transform the weights in the combine_kernel function
        self.weights = weights
        self.K_i = K_i.clone()
        self.K = K.clone()
        self.logDetK = logDetK.clone()
        self.likelihood = likelihood.item()
        self.theta_vector = theta_vector  # pylint: disable=attribute-defined-outside-init
        self.layer_weights = layer_weights
        self.nlml = nlml.detach().cpu() if nlml is not None else None

        for k in self.combined_kernel.kernels:
            if isinstance(k, Stationary):
                k.update_hyperparameters(lengthscale=torch.exp(theta_vector))

        self.combined_kernel.weights = weights.clone()
        if self.verbose:
            print("Optimisation summary: ")
            print("Optimal NLML: ", nlml)
            print("Lengthscales: ", torch.exp(theta_vector))
            try:
                print(
                    "Optimal h: ",
                    self.domain_kernels[0]._h,  # pylint: disable=protected-access
                )
            except AttributeError:
                pass
            print("Weights: ", self.weights)
            print("Lik:", self.likelihood)
            print("Optimal layer weights", layer_weights)

    def predict(self, x_configs, preserve_comp_graph: bool = False):
        """Kriging predictions"""

        if not isinstance(x_configs, list):
            # Convert a single input X_s to a singleton list
            x_configs = [x_configs]

        if self.K_i is None or self.logDetK is None:
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize "
                "function to fit on the training data first!"
            )

        # Concatenate the full list
        X_configs_all = self.x_configs + x_configs

        # Make a copy of the sum_kernels for this step, to avoid breaking the autodiff
        # if grad guided mutation is used
        if preserve_comp_graph:
            combined_kernel_copy = deepcopy(self.combined_kernel)
        else:
            combined_kernel_copy = self.combined_kernel

        K_full = combined_kernel_copy.fit_transform(
            self.weights,
            X_configs_all,
            layer_weights=self.layer_weights,
            normalize=self.normalize_combined_kernel,
            feature_lengthscale=torch.exp(self.theta_vector),
            rebuild_model=True,
            save_gram_matrix=False,
            gp_fit=False,
        )

        K_s = K_full[: self.n :, self.n :]

        K_ss = K_full[self.n :, self.n :] + self.likelihood * torch.eye(
            len(x_configs),
        )

        mu_s = K_s.t() @ self.K_i @ self.y
        cov_s = K_ss - K_s.t() @ self.K_i @ K_s
        # TODO not taking the diag?
        cov_s = torch.clamp(cov_s, self.likelihood, np.inf)
        mu_s = unnormalize_y(mu_s, self.y_mean, self.y_std)
        std_s = torch.sqrt(cov_s)
        std_s = unnormalize_y(std_s, None, self.y_std, True)
        cov_s = std_s**2
        if preserve_comp_graph:
            del combined_kernel_copy
        return mu_s, cov_s

    def predict_single_hierarchy(
        self, x_configs, hierarchy_id=0, preserve_comp_graph: bool = False
    ):
        """Kriging predictions"""

        if not isinstance(x_configs, list):
            # Convert a single input X_s to a singleton list
            x_configs = [x_configs]

        if self.K_i is None or self.logDetK is None:
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize function to "
                "fit on the training data first!"
            )

        # Concatenate the full list
        X_configs_all = self.x_configs + x_configs

        # Make a copy of the sum_kernels for this step, to avoid breaking the autodiff if grad guided mutation is used
        if preserve_comp_graph:
            combined_kernel_copy = deepcopy(self.combined_kernel)
        else:
            combined_kernel_copy = self.combined_kernel

        K_sub_full = combined_kernel_copy.fit_transform_single_hierarchy(
            self.weights,
            X_configs_all,
            normalize=self.normalize_combined_kernel,
            hierarchy_id=hierarchy_id,
            feature_lengthscale=torch.exp(self.theta_vector),
            layer_weights=self.layer_weights,
            rebuild_model=True,
            save_gram_matrix=False,
            gp_fit=False,
        )

        K_s = K_sub_full[: self.n :, self.n :]
        K_ss = K_sub_full[self.n :, self.n :]
        mu_s = K_s.t() @ self.K_i @ self.y
        cov_s_full = K_ss - K_s.t() @ self.K_i @ K_s
        cov_s = torch.clamp(cov_s_full, self.likelihood, np.inf)
        mu_s = unnormalize_y(mu_s, self.y_mean, self.y_std)
        std_s = torch.sqrt(cov_s)
        std_s = unnormalize_y(std_s, None, self.y_std, True)
        cov_s = std_s**2
        if preserve_comp_graph:
            del combined_kernel_copy
        return mu_s, cov_s

    @property
    def x(self):
        return self.x_configs

    def _reset_XY(self, train_x: Iterable, train_y: Union[Iterable, torch.Tensor]):
        self.x_configs = train_x  # type: ignore[assignment]
        self.n = len(self.x_configs)
        train_y_tensor = (
            train_y
            if isinstance(train_y, torch.Tensor)
            else torch.tensor(train_y, dtype=torch.get_default_dtype())
        )
        self.y_ = train_y_tensor
        self.y, self.y_mean, self.y_std = normalize_y(train_y_tensor)
        # The Gram matrix of the training data
        self.K_i, self.logDetK = None, None

    def dmu_dphi(
        self,
        X_s=None,
        # compute_grad_var=False,
        average_across_features=True,
        average_across_occurrences=False,
    ):
        r"""
        Compute the derivative of the GP posterior mean at the specified input location with respect to the
        *vector embedding* of the graph (e.g., if using WL-subtree, this function computes the gradient wrt
        each subtree pattern)

        The derivative is given by
        $
        \frac{\partial \mu^*}{\partial \phi ^*} = \frac{\partial K(\phi, \phi^*)}{\partial \phi ^ *}K(\phi, \phi)^{-1}
        \mathbf{y}
        $

        which derives directly from the GP posterior mean formula, and since the term $K(\phi, \phi)^{-1} and \mathbf{y}
        are both independent of the testing points (X_s, or \phi^*}, the posterior gradient is simply the matrix
        produce of the kernel gradient with the inverse Gram and the training label vector.

        Parameters
        ----------
        X_s: The locations on which the GP posterior mean derivatives should be evaluated. If left blank, the
        derivatives will be evaluated at the training points.

        compute_grad_var: bool. If true, also compute the gradient variance.

        The derivative of GP is also a GP, and thus the predictive distribution of the posterior gradient is Gaussian.
        The posterior mean is given above, and the posterior variance is:
        $
        \mathbb{V}[\frac{\partial f^*}{\partial \phi^*}]= \frac{\partial^2k(\phi^*, \phi^*)}{\partial \phi^*^2} -
        \frac{\partial k(\phi^*, \Phi)}{\partial \phi^*}K(X, X)^{-1}\frac{\partial k{(\Phi, \phi^*)}}{\partial \phi^*}
        $

        Returns
        -------
        list of K torch.Tensor of the shape N x2 D, where N is the length of the X_s list (each element of which is a
        networkx graph), K is the number of kernel_operators in the combined kernel and D is the dimensionality of the
        feature vector (this is determined by the specific graph kernel.

        OR

        list of K torch.Tensor of shape D, if averaged_over_samples flag is enabled.
        """
        if self.K_i is None or self.logDetK is None:
            raise ValueError(
                "Inverse of Gram matrix is not instantiated. Please call the optimize "
                "function to fit on the training data first!"
            )
        if self.n_vector_kernels:
            if X_s is not None:
                V_s = self._get_vectorial_features(X_s, self.vectorial_feactures)
                V_s, _, _ = standardize_x(V_s, self.x_features_min, self.x_features_max)
            else:
                V_s = self.x_features
                X_s = self.x[:]
        else:
            V_s = None
            X_s = X_s if X_s is not None else self.x[:]

        alpha = (self.K_i @ self.y).double().reshape(1, -1)
        dmu_dphi = []
        # dmu_dphi_var = [] if compute_grad_var else None

        Ks_handles = []
        feature_matrix = []
        for j, x_s in enumerate(X_s):
            jacob_vecs = []
            if V_s is None:
                handles = self.combined_kernel.forward_t(
                    self.weights,
                    [x_s],
                )
            else:
                handles = self.combined_kernel.forward_t(self.weights, [x_s], V_s[j])
            Ks_handles.append(handles)
            # Each handle is a 2-tuple. first element is the Gram matrix, second element is the leaf variable
            feature_vectors = []
            for handle in handles:
                k_s, y, _ = handle
                # k_s is output, leaf is input, alpha is the K_i @ y term which is constant.
                # When compute_grad_var is not required, computational graphs do not need to be saved.
                jacob_vecs.append(
                    torch.autograd.grad(
                        outputs=k_s, inputs=y, grad_outputs=alpha, retain_graph=False
                    )[0]
                )
                feature_vectors.append(y)
            feature_matrix.append(feature_vectors)
            jacob_vecs = torch.cat(jacob_vecs)
            dmu_dphi.append(jacob_vecs)

        feature_matrix = torch.cat([f[0] for f in feature_matrix])
        if average_across_features:
            dmu_dphi = torch.cat(dmu_dphi)
            # compute the weighted average of the gradient across N_t.
            # feature matrix is of shape N_t x K x D
            avg_mu, avg_var, incidences = get_grad(
                dmu_dphi, feature_matrix, average_across_occurrences
            )
            return avg_mu, avg_var, incidences
        return (
            dmu_dphi,
            None,
            feature_matrix.sum(dim=0) if average_across_occurrences else feature_matrix,
        )


def get_grad(grad_matrix, feature_matrix, average_occurrences=False):
    r"""
    Average across the samples via a Monte Carlo sampling scheme. Also estimates the
    empirical variance. :param average_occurrences: if True, do a weighted summation
    based on the frequency distribution of the occurrence to compute a gradient *per
    each feature*. Otherwise, each different occurrence (\phi_i = k) will get a
    different gradient estimate.
    """
    assert grad_matrix.shape == feature_matrix.shape
    # Prune out the all-zero columns that pop up sometimes
    valid_cols = []
    for col_idx in range(feature_matrix.size(1)):
        if not torch.all(feature_matrix[:, col_idx] == 0):
            valid_cols.append(col_idx)
    feature_matrix = feature_matrix[:, valid_cols]
    grad_matrix = grad_matrix[:, valid_cols]

    _, D = feature_matrix.shape
    if average_occurrences:
        avg_grad = torch.zeros(D)
        avg_grad_var = torch.zeros(D)
        for d in range(D):
            current_feature = feature_matrix[:, d].clone().detach()
            instances, indices, counts = torch.unique(
                current_feature, return_inverse=True, return_counts=True
            )
            weight_vector = torch.tensor([counts[i] for i in indices]).type(torch.float)
            weight_vector /= weight_vector.sum()
            mean = torch.sum(weight_vector * grad_matrix[:, d])
            # Compute the empirical variance of gradients
            variance = torch.sum(weight_vector * grad_matrix[:, d] ** 2) - mean**2
            avg_grad[d] = mean
            avg_grad_var[d] = variance
        return avg_grad, avg_grad_var, feature_matrix.sum(dim=0)
    else:
        # The maximum number possible occurrences -- 7 is an example, if problem occurs, maybe we can increase this
        # number. But for now, for both NAS-Bench datasets, this should be more than enough!
        max_occur = 7
        avg_grad = torch.zeros(D, max_occur)
        avg_grad_var = torch.zeros(D, max_occur)
        incidences = torch.zeros(D, max_occur)
        for d in range(D):
            current_feature = feature_matrix[:, d].clone().detach()
            instances, indices, counts = torch.unique(
                current_feature, return_inverse=True, return_counts=True
            )
            for i, val in enumerate(instances):
                # Find index of all feature counts that are equal to the current val
                feature_at_val = grad_matrix[current_feature == val]
                avg_grad[d, int(val)] = torch.mean(feature_at_val)
                avg_grad_var[d, int(val)] = torch.var(feature_at_val)
                incidences[d, int(val)] = counts[i]
        return avg_grad, avg_grad_var, incidences


# Optimize Graph kernel
def getBack(var_grad_fn, logger):
    logger.debug(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                logger.debug(n[0])
                logger.debug(f"Tensor with grad found: {tensor}")
                logger.debug(f" - gradient: {tensor.grad}")
            except AttributeError:
                getBack(n[0], logger)


def _grid_search_wl_kernel(
    k: WeisfilerLehman,
    subtree_candidates,
    train_x: list,
    train_y: torch.Tensor,
    lik: float,
    subtree_prior=None,  # pylint: disable=unused-argument
    lengthscales=None,
    lengthscales_prior=None,  # pylint: disable=unused-argument
    gpytorch_kinv: bool = False,
):
    """Optimize the *discrete hyperparameters* of Weisfeiler Lehman kernel.
    k: a Weisfeiler-Lehman kernel instance
    hyperparameter_candidate: list of candidate hyperparameter to try
    train_x: the train data
    train_y: the train label
    lik: likelihood
    lengthscale: if using RBF kernel for successive embedding, the list of lengthscale to be grid searched over
    """
    # lik = 1e-6
    assert len(train_x) == len(train_y)
    best_nlml = torch.tensor(np.inf)
    best_subtree_depth = None
    best_lengthscale = None
    best_K = None
    if lengthscales is not None and k.se is not None:
        candidates = [(h_, l_) for h_ in subtree_candidates for l_ in lengthscales]
    else:
        candidates = [(h_, None) for h_ in subtree_candidates]

    for i in candidates:
        if k.se is not None:
            k.change_se_params({"lengthscale": i[1]})
        k.change_kernel_params({"h": i[0]})
        K = k.fit_transform(train_x, rebuild_model=True, save_gram_matrix=True)
        # print(K)
        K_i, logDetK = compute_pd_inverse(K, lik, gpytorch_kinv)
        # print(train_y)
        nlml = -compute_log_marginal_likelihood(K_i, logDetK, train_y)
        # print(i, nlml)
        if nlml < best_nlml:
            best_nlml = nlml
            best_subtree_depth, best_lengthscale = i
            best_K = torch.clone(K)
    # print("h: ", best_subtree_depth, "theta: ", best_lengthscale)
    # print(best_subtree_depth)
    k.change_kernel_params({"h": best_subtree_depth})
    if k.se is not None:
        k.change_se_params({"lengthscale": best_lengthscale})
    k._gram = best_K  # pylint: disable=protected-access


def get_theta_vector(vectorial_features):
    if vectorial_features is None:
        return None
    theta_vector = {}
    for key, dim in vectorial_features.items():
        t = torch.ones(dim)
        if t.shape[0] > 1:
            t.requires_grad_(True)
        theta_vector[key] = t
    return theta_vector


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


def generate_h_combo_candidates(hierarchy_consider):
    h_range_all_hierarchy = [range(min(hier + 2, 4)) for hier in hierarchy_consider]
    h_range_all_hierarchy = [range(5)] + h_range_all_hierarchy
    h_combo_all = list(itertools.product(*h_range_all_hierarchy))
    h_combo_sub = []
    for h_combo in h_combo_all:
        sorted_h_combo = sorted(h_combo)
        if sorted_h_combo not in h_combo_sub:
            h_combo_sub.append(sorted_h_combo)
    return h_combo_sub


def compute_pd_inverse(
    K: torch.tensor, jitter: float = 1e-5, gpytorch_kinv: bool = False
):
    """Compute the inverse of a postive-(semi)definite matrix K using Cholesky inversion."""
    if gpytorch_kinv:
        Kc = psd_safe_cholesky(K)
        try:
            Kc.required_grad = True
        except Exception:
            Kc = torch.Tensor(Kc)
    else:
        n = K.shape[0]
        assert (
            isinstance(jitter, float) or jitter.ndim == 0
        ), "only homoscedastic noise variance is allowed here!"
        is_successful = False
        fail_count = 0
        max_fail = 3
        while fail_count < max_fail and not is_successful:
            try:
                jitter_diag = jitter * torch.eye(n, device=K.device) * 10**fail_count
                K_ = K + jitter_diag
                Kc = torch.linalg.cholesky(K_)
                is_successful = True
            except RuntimeError:
                fail_count += 1
        if not is_successful:
            raise RuntimeError(
                f"Gram matrix not positive definite despite of jitter:\n{K}"
            )

    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.to(torch.get_default_dtype()), logDetK.to(torch.get_default_dtype())
