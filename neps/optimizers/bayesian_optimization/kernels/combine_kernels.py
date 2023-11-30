import logging

import torch

from .utils import extract_configs
from .vectorial_kernels import HammingKernel, Stationary
from .weisfilerlehman import GraphKernels


def _select_dimensions(k):
    if isinstance(k, HammingKernel):
        return "categorical"
    return "continuous"


class CombineKernel:
    def __init__(
        self,
        combined_by="sum",
        *kernels: list,
        **kwargs,  # pylint: disable=unused-argument
    ):
        if combined_by not in ["sum", "product"]:
            raise ValueError(f"Invalid value for combined_by ({combined_by})")

        self.has_graph_kernels = False
        self.has_vector_kernels = False
        self.lengthscale_bounds = (None, None)
        for k in kernels:
            if isinstance(k, GraphKernels):
                self.has_graph_kernels = True
            if not isinstance(k, GraphKernels):
                self.has_vector_kernels = True
                self.lengthscale_bounds = k.lengthscale_bounds
        self.kernels = kernels
        # Store the training graphs and vector features..
        self._gram = None
        self.gr, self.x = None, None
        self.combined_by = combined_by

    def fit_transform(
        self,
        weights: torch.Tensor,
        configs: list,
        normalize: bool = True,
        rebuild_model: bool = True,
        save_gram_matrix: bool = True,
        gp_fit: bool = True,
        feature_lengthscale: list = None,
        **kwargs,
    ):
        N = len(configs)
        K = torch.zeros(N, N) if self.combined_by == "sum" else torch.ones(N, N)

        gr1, x1 = extract_configs(configs)

        for i, k in enumerate(self.kernels):
            if isinstance(k, GraphKernels) and None not in gr1:
                update_val = weights[i] * k.fit_transform(
                    [g[i] for g in gr1] if isinstance(gr1[0], (list, tuple)) else gr1,
                    rebuild_model=rebuild_model,
                    save_gram_matrix=save_gram_matrix,
                    gp_fit=gp_fit,
                    **kwargs,
                )

            elif isinstance(k, Stationary) and None not in x1:
                key = _select_dimensions(k)
                update_val = (
                    weights[i]
                    * k.fit_transform(
                        [x_[key] for x_ in x1],
                        l=feature_lengthscale[key]
                        if isinstance(feature_lengthscale, dict)
                        else None,
                        rebuild_model=rebuild_model,
                        save_gram_matrix=save_gram_matrix,
                    )
                ).double()

            else:
                raise NotImplementedError(
                    "For now, only the Stationary custom built kernel_operators are "
                    "supported! "
                )

            if self.combined_by == "sum":
                K += update_val
            elif self.combined_by == "product":
                K *= update_val

        if normalize:
            K_diag = torch.sqrt(torch.diag(K))
            K /= torch.ger(K_diag, K_diag)
        if save_gram_matrix:
            self._gram = K.clone()

        return K

    def transform(
        self,
        weights: torch.Tensor,
        configs: list,
        x=None,
        feature_lengthscale=None,
    ):
        if self._gram is None:
            raise ValueError(
                "The kernel has not been fitted. Call fit_transform first to generate "
                "the training Gram matrix."
            )
        gr, x = extract_configs(configs)
        # K is in shape of len(Y), len(X)
        size = len(configs)
        K = (
            torch.zeros(size, self._gram.shape[0])
            if self.combined_by == "sum"
            else torch.ones(size, self._gram.shape[0])
        )

        for i, k in enumerate(self.kernels):
            if isinstance(k, GraphKernels) and None not in gr:
                update_val = weights[i] * k.transform(
                    [g[i] for g in gr] if isinstance(gr, list) else gr
                )
            elif isinstance(k, Stationary) and None not in x:
                key = _select_dimensions(k)
                update_val = (
                    weights[i]
                    * k.transform(
                        [x_[key] for x_ in x],
                        l=feature_lengthscale[key]
                        if isinstance(feature_lengthscale, dict)
                        else None,
                    ).double()
                )
            else:
                raise NotImplementedError(
                    "For now, only the Stationary custom built kernel_operators are "
                    "supported! "
                )

            if self.combined_by == "sum":
                K += update_val
            elif self.combined_by == "product":
                K *= update_val

        return K.t()

    def clamp_theta_vector(self, theta_vector):
        if theta_vector is None:
            return None
        # pylint: disable=expression-not-assigned
        [
            t_.clamp_(self.lengthscale_bounds[0], self.lengthscale_bounds[1])
            if t_ is not None and t_.is_leaf
            else None
            for t_ in theta_vector.values()
        ]
        return theta_vector


class SumKernel(CombineKernel):
    def __init__(self, *kernels, **kwargs):
        super().__init__("sum", *kernels, **kwargs)

    def forward_t(
        self,
        weights: torch.Tensor,
        gr2: list,
        x2=None,
        gr1: list = None,
        x1=None,
        feature_lengthscale=None,
    ):
        """
        Compute the kernel gradient w.r.t the feature vector
        Parameters
        ----------
        feature_lengthscale
        x2
        x1
        gr1
        weights
        gr2

        Returns ------- grads: k list of 2-tuple. (K, x2) where K is the weighted Gram
        matrix of that matrix, x2 is the leaf variable on which Jacobian-vector product
        to be computed.

        """
        grads = []
        for i, k in enumerate(self.kernels):
            if isinstance(k, GraphKernels):
                handle = k.forward_t(gr2, gr1=gr1)
                grads.append((weights[i] * handle[0], handle[1], handle[2]))
            elif isinstance(k, Stationary):
                key = _select_dimensions(k)
                handle = k.forward_t(x2=x2[key], x1=x1[key], l=feature_lengthscale[i])
                grads.append((weights[i] * handle[0], handle[1], handle[2]))
            else:
                logging.warning(
                    "Gradient not implemented for kernel type" + str(k.__name__)
                )
                grads.append((None, None))
        assert len(grads) == len(self.kernels)
        return grads


class ProductKernel(CombineKernel):
    def __init__(self, *kernels, **kwargs):
        super().__init__("product", *kernels, **kwargs)

    def dk_dphi(self, weights, gr: list = None, x=None, feature_lengthscale=None):
        raise NotImplementedError
