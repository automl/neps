import logging
from copy import deepcopy
from typing import Iterable, Union

import gpytorch
import numpy as np

try:
    import torch
except ModuleNotFoundError:
    from install_dev_utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from ..kernels.combine_kernels import ProductKernel, SumKernel

# GP model as a weighted average between the vanilla vectorial GP and the graph GP
from ..kernels.graph_kernel import GraphKernels
from ..kernels.vectorial_kernels import Stationary
from ..kernels.weisfilerlehman import WeisfilerLehman
from ..utils.nasbowl_utils import extract_configs
from .utils import (
    compute_log_marginal_likelihood,
    compute_pd_inverse,
    normalize_y,
    standardize_x,
    unnormalize_y,
)


# A vanilla GP with RBF kernel
class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        kernel: gpytorch.kernels,
    ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# handle normal and reduce graphs as a list
class ComprehensiveGP:
    def __init__(
        self,
        graph_kernels: Iterable,
        hp_kernels: Iterable,
        likelihood: float = 1e-3,
        weights=None,
        vector_theta_bounds: tuple = (1e-5, 0.1),
        graph_theta_bounds: tuple = (1e-1, 1.0e1),
        combined_kernel: str = "sum",
        verbose: bool = False,
    ):
        self.likelihood = likelihood

        self.domain_kernels: list = []
        if bool(graph_kernels):
            self.domain_kernels += list(graph_kernels)
        if bool(hp_kernels):
            self.domain_kernels += list(hp_kernels)

        self.n_kernels: int = len(self.domain_kernels)
        self.n_graph_kernels: int = len(
            [i for i in self.domain_kernels if isinstance(i, GraphKernels)]
        )
        self.n_vector_kernels: int = self.n_kernels - self.n_graph_kernels

        self.feature_d = None

        if weights is not None:
            self.fixed_weights = True
            if weights is not None:
                assert len(weights) == len(graph_kernels), (
                    "the weights vector, if supplied, needs to have the same length as "
                    "the number of kernel_operators!"
                )
            self.weights = (
                weights
                if isinstance(weights, torch.Tensor)
                else torch.tensor(weights).flatten()
            )
        else:
            self.fixed_weights: bool = False
            # Initialise the domain kernel weights to uniform
            self.weights = torch.tensor(
                [1.0 / self.n_kernels] * self.n_kernels,
            )

        if combined_kernel == "product":
            self.combined_kernel = ProductKernel(
                *self.domain_kernels, weights=self.weights
            )
        elif combined_kernel == "sum":
            self.combined_kernel = SumKernel(*self.domain_kernels, weights=self.weights)
        else:
            raise NotImplementedError(
                f'Combining kernel {combined_kernel} is not yet implemented! Only "sum" or "product" are currently supported.'
            )
        self.vector_theta_bounds = vector_theta_bounds
        self.graph_theta_bounds = graph_theta_bounds
        # Verbose mode
        self.verbose = verbose
        # Cache the Gram matrix inverse and its log-determinant
        self.K, self.K_i, self.logDetK = [None] * 3
        self.theta_vector = None
        self.layer_weights = None
        self.nlml = None

        self.x_configs: list = None
        self.y: torch.Tensor = None
        self.y_: torch.Tensor = None
        self.y_mean: torch.Tensor = None
        self.y_std: torch.Tensor = None
        self.n: int = None

    def _optimize_graph_kernels(self, h_: int, lengthscale_):
        graphs, _ = extract_configs(self.x_configs)
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
                )
            else:
                logging.warning(
                    "(Graph) kernel optimisation for "
                    + type(k).__name__
                    + " not implemented yet."
                )

    def fit(
        self,
        iters: int = 20,
        optimizer: str = "adam",
        wl_subtree_candidates: tuple = tuple(range(5)),
        wl_lengthscales=tuple([np.e ** i for i in range(-2, 3)]),
        optimize_lik: bool = True,
        max_lik: float = 0.01,
        optimize_wl_layer_weights: bool = False,
        optimizer_kwargs: dict = None,
    ):
        """

        Parameters
        ----------
        iters
        optimizer
        wl_subtree_candidates
        wl_lengthscales
        optimize_lik
        max_lik
        optimize_wl_layer_weights
        optimizer_kwargs

        Returns
        -------

        """
        # Get the node weights, if needed

        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 0.1}
        if len(wl_subtree_candidates):
            self._optimize_graph_kernels(
                wl_subtree_candidates,
                wl_lengthscales,
            )

        weights = self.weights.clone()

        if (not self.fixed_weights) and len(self.domain_kernels) > 1:
            weights.requires_grad_(True)
        # Initialise the lengthscale to be the geometric mean of the theta_vector bounds.
        theta_vector = torch.sqrt(
            torch.tensor(
                [self.vector_theta_bounds[0] * self.vector_theta_bounds[1]],
            )
        )  # if self.feature_d else None
        # theta_graph = torch.sqrt(
        #     torch.tensor([self.graph_theta_bounds[0] * self.graph_theta_bounds[1]])
        # ).requires_grad_(True)
        # Only requires gradient of lengthscale if there is any vectorial input
        # if self.feature_d:
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
        for a in [theta_vector, weights, likelihood, layer_weights]:
            if a is not None and a.is_leaf and a.requires_grad:
                optim_vars.append(a)
        nlml = None
        if len(optim_vars) == 0:  # Skip optimisation
            K = self.combined_kernel.fit_transform(
                weights,
                self.x_configs,
                feature_lengthscale=theta_vector,
                layer_weights=layer_weights,
                rebuild_model=True,
            )
            K_i, logDetK = compute_pd_inverse(K, likelihood)
        else:
            # Select the optimizer
            assert optimizer.lower() in ["adam", "sgd"]
            if optimizer.lower() == "adam":
                optim = torch.optim.Adam(optim_vars, **optimizer_kwargs)
            else:
                optim = torch.optim.SGD(optim_vars, **optimizer_kwargs)

            K = None
            for i in range(iters):
                optim.zero_grad()
                K = self.combined_kernel.fit_transform(
                    weights,
                    self.x_configs,
                    feature_lengthscale=theta_vector,
                    layer_weights=layer_weights,
                    rebuild_model=True,
                    save_gram_matrix=True,
                )
                K_i, logDetK = compute_pd_inverse(K, likelihood)
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
                        weights,  # theta_graph,
                        likelihood,
                    )
                optim.step()  # TODO
                with torch.no_grad():
                    # pylint: disable=expression-not-assigned
                    weights.clamp_(
                        0.0, 1.0
                    ) if weights is not None and weights.is_leaf else None
                    theta_vector.clamp_(
                        self.vector_theta_bounds[0], self.vector_theta_bounds[1]
                    ) if theta_vector is not None and theta_vector.is_leaf else None
                    likelihood.clamp_(
                        1e-5, max_lik
                    ) if likelihood is not None and likelihood.is_leaf else None
                    layer_weights.clamp_(
                        0.0, 1.0
                    ) if layer_weights is not None and layer_weights.is_leaf else None
                    # pylint: enable=expression-not-assigned
                # print('grad,', theta_graph)
            K_i, logDetK = compute_pd_inverse(K, likelihood)

        # Apply the optimal hyperparameters
        self.weights = weights.clone() / torch.sum(weights)
        self.K_i = K_i.clone()
        self.K = K.clone()
        self.logDetK = logDetK.clone()
        self.likelihood = likelihood.item()
        self.theta_vector = theta_vector
        self.layer_weights = layer_weights
        self.nlml = nlml.detach().cpu() if nlml is not None else None

        for k in self.combined_kernel.kernels:
            if isinstance(k, Stationary):
                k.lengthscale = theta_vector.clamp(
                    self.vector_theta_bounds[0], self.vector_theta_bounds[1]
                ).item()
            # elif isinstance(k, GraphKernels) and k.lengthscale_ is not None:
            #     k.lengthscale_ = theta_graph.clamp(self.graph_theta_bounds[0], self.graph_theta_bounds[1])
        self.combined_kernel.weights = weights.clone()
        if self.verbose:
            print("Optimisation summary: ")
            print("Optimal NLML: ", nlml)
            print("Lengthscales: ", theta_vector)
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
        # print('Graph Lengthscale', theta_graph)

    def predict(self, x_configs, preserve_comp_graph: bool = False):
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

        K_full = combined_kernel_copy.fit_transform(
            self.weights,
            X_configs_all,
            layer_weights=self.layer_weights,
            rebuild_model=True,
            save_gram_matrix=False,
        )

        K_s = K_full[: self.n :, self.n :]

        K_ss = K_full[self.n :, self.n :] + self.likelihood * torch.eye(
            len(x_configs),
        )

        mu_s = K_s.t() @ self.K_i @ self.y
        cov_s = K_ss - K_s.t() @ self.K_i @ K_s
        cov_s = torch.clamp(cov_s, self.likelihood, np.inf)
        mu_s = unnormalize_y(mu_s, self.y_mean, self.y_std)
        std_s = torch.sqrt(cov_s)
        std_s = unnormalize_y(std_s, None, self.y_std, True)
        cov_s = std_s ** 2
        if preserve_comp_graph:
            del combined_kernel_copy
        return mu_s, cov_s

    @property
    def x(self):
        return self.x_configs

    def reset_XY(self, train_x: Iterable, train_y: Union[Iterable, torch.Tensor]):
        self.x_configs = train_x
        self.n = len(self.x_configs)
        train_y_tensor = (
            train_y
            if isinstance(train_y, torch.Tensor)
            else torch.Tensor(train_y).float()
        )
        self.y_ = train_y_tensor
        self.y, self.y_mean, self.y_std = normalize_y(train_y_tensor)
        # The Gram matrix of the training data
        self.K_i, self.logDetK = None, None

        # if self.n_vector_kernels > 0:
        #     self.x_features, self.x_features_min, self.x_features_max = \
        #         standardize_x(self._get_vectorial_features(self.x, self.vectorial_feactures))

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
                "Inverse of Gram matrix is not instantiated. Please call the optimize function to "
                "fit on the training data first!"
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
    Average across the samples via a Monte Carlo sampling scheme. Also estimates the empirical variance.
    :param average_occurrences: if True, do a weighted summation based on the frequency distribution of the occurrence
        to compute a gradient *per each feature*. Otherwise, each different occurence (\phi_i = k) will get a different
        gradient estimate.
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
            variance = torch.sum(weight_vector * grad_matrix[:, d] ** 2) - mean ** 2
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
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], "variable")
                print(n[0])
                print("Tensor with grad found:", tensor)
                print(" - gradient:", tensor.grad)
                print()
            except AttributeError:
                getBack(n[0])


def _grid_search_wl_kernel(
    k: WeisfilerLehman,
    subtree_candidates,
    train_x: list,
    train_y: torch.Tensor,
    lik: float,
    subtree_prior=None,  # pylint: disable=unused-argument
    lengthscales=None,
    lengthscales_prior=None,  # pylint: disable=unused-argument
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
        K_i, logDetK = compute_pd_inverse(K, lik)
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
