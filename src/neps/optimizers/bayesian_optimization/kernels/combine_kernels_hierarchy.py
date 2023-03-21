import logging

import numpy as np
import torch

from .utils import extract_configs_hierarchy
from .vectorial_kernels import HammingKernel, Stationary
from .weisfilerlehman import GraphKernels


# normalise weights in front of additive kernels
def transform_weights(weights):
    return torch.exp(weights) / torch.sum(torch.exp(weights))


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
        self.hierarchy_consider = kwargs["hierarchy_consider"]
        self.d_graph_features = kwargs["d_graph_features"]
        # if use global graph features of the final architecture graph, prepare for normalising
        # them based on training data
        if self.d_graph_features > 0:
            self.train_graph_feature_mean = None
            self.train_graph_feature_std = None

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
        weights = transform_weights(weights.clone())
        N = len(configs)
        K = torch.zeros(N, N) if self.combined_by == "sum" else torch.ones(N, N)

        gr1, x1 = extract_configs_hierarchy(
            configs,
            d_graph_features=self.d_graph_features,
            hierarchy_consider=self.hierarchy_consider,
        )

        # normalise the global graph features if we plan to use them
        if self.d_graph_features > 0:
            if gp_fit:
                # compute the mean and std based on training data
                self.train_graph_feature_mean = np.mean(x1, 0)
                self.train_graph_feature_std = np.std(x1, 0)
            x1 = (x1 - self.train_graph_feature_mean) / self.train_graph_feature_std
        # k_values = [] # for debug
        # k_features = [] # for debug
        for i, k in enumerate(self.kernels):
            if isinstance(k, GraphKernels) and None not in gr1:
                if len(gr1) == N and self.hierarchy_consider is None:
                    # only the final graph is used
                    k_i = k.fit_transform(
                        [g[i] for g in gr1] if isinstance(gr1[0], (list, tuple)) else gr1,
                        rebuild_model=rebuild_model,
                        save_gram_matrix=save_gram_matrix,
                        gp_fit=gp_fit,
                        **kwargs,
                    )
                    if normalize:
                        K_i_diag = torch.sqrt(torch.diag(k_i))
                        k_i /= torch.ger(K_i_diag, K_i_diag)
                    update_val = weights[i] * k_i

                else:
                    # graphs in the early hierarchies are also used;
                    # assume the combined kernel list always start with graph kernels i.e. kernels=[graph kernels, hp kernels]
                    gr1_i = gr1[i]
                    k_i = k.fit_transform(
                        [g[i] for g in gr1_i]
                        if isinstance(gr1_i[0], (list, tuple))
                        else gr1_i,
                        rebuild_model=rebuild_model,
                        save_gram_matrix=save_gram_matrix,
                        gp_fit=gp_fit,
                        **kwargs,
                    )
                    if normalize:
                        K_i_diag = torch.sqrt(torch.diag(k_i))
                        k_i /= torch.ger(K_i_diag, K_i_diag)

                    update_val = weights[i] * k_i
                # k_features.append([value.X.shape[1] for key, value in k.kern.X.items()])

            elif isinstance(k, Stationary) and None not in x1:
                k_i = k.fit_transform(
                    x1,
                    rebuild_model=rebuild_model,
                    save_gram_matrix=save_gram_matrix,
                    l=feature_lengthscale,
                )
                update_val = (weights[i] * k_i).double()
            else:
                raise NotImplementedError(
                    " For now, only the Stationary custom built kernel_operators are supported!"
                )

            # k_values.append(k_i) # for debug

            if self.combined_by == "sum":
                K += update_val
            elif self.combined_by == "product":
                K *= update_val

        # self.k_values = k_values # for debug
        # self.k_features = k_features # for debug
        # self.weights_trans = weights # for debug
        # if not normalize:
        #     K_diag = torch.sqrt(torch.diag(K))
        #     K /= torch.ger(K_diag, K_diag)

        if save_gram_matrix:
            self._gram = K.clone()

        return K

    def fit_transform_single_hierarchy(
        self,
        weights: torch.Tensor,
        configs: list,
        hierarchy_id: int,
        normalize: bool = True,
        rebuild_model: bool = True,
        gp_fit: bool = True,
        **kwargs,
    ):
        weights = transform_weights(weights.clone())
        # N = len(configs)
        # K = torch.zeros(N, N) if self.combined_by == "sum" else torch.ones(N, N)

        gr1, _ = extract_configs_hierarchy(
            configs,
            d_graph_features=self.d_graph_features,
            hierarchy_consider=self.hierarchy_consider,
        )
        # get the corresponding graph kernel and hierarchy graph data
        graph_kernel_list = [k for k in self.kernels if isinstance(k, GraphKernels)]
        # first graph kernel is on the final architecture graph
        k_single_hierarchy = graph_kernel_list[int(hierarchy_id + 1)]
        gr1_single_hierarchy = gr1[int(hierarchy_id + 1)]
        weight_single_hierarchy = weights[int(hierarchy_id + 1)]
        k_raw = k_single_hierarchy.fit_transform(
            gr1_single_hierarchy,
            rebuild_model=rebuild_model,
            gp_fit=gp_fit,
            **kwargs,
        )
        k_raw = k_raw.to(torch.float32)
        if normalize:
            K_diag = torch.sqrt(torch.diag(k_raw))
            k_raw /= torch.ger(K_diag, K_diag)

        K = weight_single_hierarchy * k_raw

        return K


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
        weights = transform_weights(weights.clone())
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
