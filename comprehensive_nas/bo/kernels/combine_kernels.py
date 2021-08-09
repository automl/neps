import logging

try:
    import torch
except ModuleNotFoundError:
    from install_dev_utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)

from ..utils.nasbowl_utils import extract_configs
from .vectorial_kernels import Stationary
from .weisfilerlehman import GraphKernels


class CombineKernel:
    def __init__(
        self,
        combined_by="sum",
        *kernels: list,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.has_graph_kernels = False
        self.has_vector_kernels = False
        for k in kernels:
            if isinstance(k, GraphKernels):
                self.has_graph_kernels = True
            if not isinstance(k, GraphKernels):
                self.has_vector_kernels = True
        self.kernels = kernels
        # Store the training graphs and vector features..
        self._gram = None
        self.gr, self.x = None, None
        assert combined_by in ["sum", "product"]
        self.combined_by = combined_by

    def fit_transform(
        self,
        weights: torch.Tensor,
        configs: list,
        normalize: bool = True,
        rebuild_model: bool = True,
        save_gram_matrix: bool = True,
        gp_fit: bool = True,
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
                update_val = (
                    weights[i]
                    * k.fit_transform(
                        x1, rebuild_model=rebuild_model, save_gram_matrix=save_gram_matrix
                    )
                ).double()

            else:
                raise NotImplementedError(
                    " For now, only the Stationary custom built kernel_operators are supported!"
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
        feature_lengthscale=None,  # pylint: disable=unused-argument
    ):
        if self._gram is None:
            raise ValueError(
                "The kernel has not been fitted. Call fit_transform first to generate the training Gram"
                "matrix."
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
                update_val = weights[i] * k.transform(x).double()
            else:
                raise NotImplementedError(
                    " For now, only the Stationary custom built kernel_operators are supported!"
                )

            if self.combined_by == "sum":
                K += update_val
            elif self.combined_by == "product":
                K *= update_val

        return K.t()


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

        Returns
        -------
        grads: k list of 2-tuple.
        (K, x2) where K is the weighted Gram matrix of that matrix, x2 is the leaf variable on which Jacobian-vector
        product to be computed.

        """
        grads = []
        for i, k in enumerate(self.kernels):
            if isinstance(k, GraphKernels):
                handle = k.forward_t(gr2, gr1=gr1)
                grads.append((weights[i] * handle[0], handle[1], handle[2]))
            elif isinstance(k, Stationary):
                handle = k.forward_t(x2=x2, x1=x1, l=feature_lengthscale)
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
