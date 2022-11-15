# type: ignore
import logging
from typing import Tuple

import numpy as np
import torch
from grakel.kernels import ShortestPathAttr
from grakel.utils import graph_from_networkx

from .base_kernel import CustomKernel
from .grakel_replace.edge_histogram import EdgeHistogram
from .grakel_replace.utils import calculate_kernel_matrix_as_tensor
from .grakel_replace.vertex_histogram import VertexHistogram
from .grakel_replace.weisfeiler_lehman import WeisfeilerLehman as _WL
from .graph_kernel import GraphKernel
from .utils import transform_to_undirected
from .vectorial_kernels import BaseNumericalKernel  # TODO: not defined anymore...


class WeisfeilerLehman(GraphKernel, CustomKernel):
    """Weisfeiler Lehman kernel using grakel functions"""

    def __init__(
        self,
        h: int = 0,
        base_type: str = "subtree",
        se_kernel: BaseNumericalKernel = None,
        layer_weights=None,
        node_weights=None,
        oa: bool = False,
        node_label: str = "op_name",
        edge_label: tuple = "op_name",
        n_jobs: int = None,
        return_tensor: bool = True,
        requires_grad: bool = False,
        undirected: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        h: int: The number of Weisfeiler-Lehman iterations
        base_type: str: defines the base kernel of WL iteration. Possible types are 'subtree' (default), 'sp': shortest path
        and 'edge' (The latter two are untested)
        se_kernel: Stationary. defines a stationary vector kernel to be used for successive embedding (i.e. the kernel
            function on which the vector embedding inner products are computed). if None, use the default linear kernel
        node_weights
        oa: whether the optimal assignment variant of the Weisfiler-Lehman kernel should be used
        node_label: the node_label defining the key node attribute.
        edge_label: the edge label defining the key edge attribute. only relevant when base_type == 'edge'
        n_jobs: Parallisation to be used. *current version does not support parallel computing'
        return_tensor: whether return a torch tensor. If False, a numpy array will be returned.
        kwargs
        """
        super().__init__(**kwargs)
        if se_kernel is not None and oa:
            raise ValueError(
                "Only one or none of se (successive embedding) and oa (optimal assignment) may be true!"
            )
        self.h = h
        self.oa = oa
        self.node_label = node_label
        self.edge_label = edge_label
        self.layer_weights = layer_weights
        self.se = se_kernel
        self.requires_grad = requires_grad
        self.undirected = undirected

        if base_type not in ["subtree", "sp", "edge"]:
            raise ValueError(f"Invalid value for base_type ({base_type})")
        if base_type == "subtree":
            base_kernel = VertexHistogram, {
                "sparse": False,
                "requires_ordered_features": requires_grad,
            }
            if oa:
                base_kernel = VertexHistogram, {
                    "oa": True,
                    "sparse": False,
                    "requires_ordered_features": requires_grad,
                }
            elif se_kernel is not None:
                base_kernel = VertexHistogram, {
                    "se_kernel": se_kernel,
                    "sparse": False,
                    "requires_ordered_features": requires_grad,
                }
        elif base_type == "edge":
            base_kernel = EdgeHistogram, {"sparse": False}
            if oa:
                base_kernel = EdgeHistogram, {
                    "oa": True,
                    "sparse": False,
                    "requires_ordered_features": requires_grad,
                }
            elif se_kernel is not None:
                base_kernel = EdgeHistogram, {
                    "se_kernel": se_kernel,
                    "sparse": False,
                    "requires_ordered_features": requires_grad,
                }

        elif base_type == "sp":
            base_kernel = ShortestPathAttr, {}
        else:
            raise NotImplementedError(
                "The selected WL base kernel type"
                + str(base_type)
                + " is not implemented."
            )
        self.base_type = base_type
        self.kern = _WL(
            n_jobs,
            h=h,
            base_graph_kernel=base_kernel,
            normalize=True,
            layer_weights=self.layer_weights,
            node_weights=node_weights,
        )
        self.return_tensor = return_tensor
        self._gram = None
        self._train, self._train_transformed = None, None
        self.__name__ = "WeisfeilerLehman"

    def _optimize_wl_kernel(
        self,
        y: torch.Tensor,
        likelihood: float = 1e-3,
        h_: Tuple[int] = tuple(range(5)),
        lengthscale_: Tuple[float] = tuple(np.e**i for i in range(-2, 3)),
    ):
        _grid_search_wl_kernel(
            self,
            h_,
            self.train_graphs[0],
            y,
            likelihood,
            lengthscales=lengthscale_,
        )
        self._pretrained = True

    def prefit_graph_kernel(
        self, y: torch.Tensor, likelihood: float = 1e-3, **opt_kwargs
    ) -> None:
        self._optimize_wl_kernel(y=y, likelihood=likelihood, **opt_kwargs)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, gpytorch_kernel, **kwargs
    ) -> torch.Tensor:
        if not self._pretrained:
            raise Exception(
                "WL kernel needs to be fitted before (-> call prefit_graph_kernel before)"
            )

        assert "training" in kwargs
        gp_fit = kwargs["training"]
        if gp_fit or len(self.train_graphs[0]) == x1.shape[0] == x2.shape[0]:
            gr = self.train_graphs[0]
        else:
            gr = self.train_graphs[0] + self.eval_graphs[0]

        K = self.fit_transform(
            gr=gr, rebuild_model=True, save_gram_matrix=not gp_fit, gp_fit=gp_fit
        )

        assert K.shape[0] >= x1.shape[0]
        if not gp_fit and K.shape[0] != x1.shape[0]:
            K = K[len(self.train_graphs[0]) :, :]
        assert K.shape[1] >= x2.shape[0]
        if not gp_fit and K.shape[1] != x2.shape[0]:
            K = K[:, len(self.train_graphs[0]) :]
        assert K.shape == (x1.shape[0], x2.shape[0])
        return K

    def change_se_params(self, params: dict):
        """Change the kernel parameter of the successive embedding kernel."""
        if self.se is None:
            logging.warning("SE kernel is None. change_se_params action voided.")
            return
        for k, v in params.items():
            try:
                setattr(self.se, k, v)
            except AttributeError:
                logging.warning(
                    str(k) + " is not a valid attribute name of the SE kernel."
                )
                continue
        self.kern.change_se_kernel(self.se)

    def get_info_se_kernel(self):
        return self.se.lengthscale, self.kern.X[0].X.shape[1]

    def change_kernel_params(self, params: dict):
        for k, v in params.items():
            try:
                getattr(self.kern, k)
                setattr(self.kern, k, v)
            except AttributeError:
                logging.warning(str(k) + " is not a valid attribute name of this kernel.")
                continue
            try:
                setattr(self, k, v)
            except AttributeError:
                pass
        for k in self.kern._initialized.keys():  # pylint: disable=W0212
            self.kern._initialized[k] = False  # pylint: disable=W0212

        self.kern.initialize()

    def fit_transform(
        self,
        gr: list,
        rebuild_model: bool = False,
        save_gram_matrix: bool = True,
        layer_weights=None,
        gp_fit: bool = True,
        **kwargs,
    ):
        # Transform into GraKeL graph format
        if rebuild_model is False and self._gram is not None:
            return self._gram
        if self.undirected:
            gr = transform_to_undirected(gr)
        if self.base_type == "edge":
            if not all([g.graph_type == "edge_attr" for g in gr]):
                raise ValueError(
                    "One or more graphs passed are not edge-attributed graphs. You need all graphs to be"
                    "in edge format to use 'edge' type Weisfiler-Lehman kernel."
                )

            gr_ = list(graph_from_networkx(gr, self.node_label, self.edge_label))
        else:
            gr_ = list(
                graph_from_networkx(
                    gr,
                    self.node_label,
                )
            )

        if rebuild_model or self._gram is None:
            self._train = gr[:]
            self._train_transformed = gr_[:]

        if layer_weights is not None and layer_weights is not self.layer_weights:
            self.change_kernel_params({"layer_weights": layer_weights})
            self.layer_weights = layer_weights

        K = self.kern.fit_transform(gr_, gp_fit=gp_fit)
        if self.return_tensor and not isinstance(K, torch.Tensor):
            K = torch.tensor(K)
        if save_gram_matrix:
            self._gram = K.clone()
            self.layer_weights = self.kern.layer_weights
        return K.type(torch.float32)

    def transform(
        self,
        gr: list,
    ):
        """transpose: by default, the grakel produces output in shape of len(y) * len(x2). Use transpose to
        reshape that to a more conventional shape.."""
        if self.undirected:
            gr = transform_to_undirected(gr)
        if self.base_type == "edge":
            if not all([g.graph_type == "edge_attr" for g in gr]):
                raise ValueError(
                    "One or more graphs passed are not edge-attributed graphs. You need all graphs to be"
                    "in edge format to use 'edge' type Weisfiler-Lehman kernel."
                )
            gr_ = graph_from_networkx(gr, self.node_label, self.edge_label)
        else:
            gr_ = graph_from_networkx(
                gr,
                self.node_label,
            )

        K = self.kern.transform(gr_)
        if self.return_tensor and not isinstance(K, torch.Tensor):
            K = torch.tensor(K)
        return K

    def forward_t(self, gr2, gr1=None):
        """
        Forward pass, but in tensor format.

        Parameters
        ----------
        gr1: single networkx graph

        Returns
        -------
        K: the kernel matrix
        x2 or y: the leaf variable(s) with requires_grad enabled.
        This allows future Jacobian-vector product to be efficiently computed.
        """
        if self.undirected:
            gr2 = transform_to_undirected(gr2)

        # Convert into GraKel compatible graph format
        if self.base_type == "edge":
            gr2 = graph_from_networkx(gr2, self.node_label, self.edge_label)
        else:
            gr2 = graph_from_networkx(gr2, self.node_label)

        if gr1 is None:
            gr1 = self._train_transformed
        else:
            if self.undirected:
                gr1 = transform_to_undirected(gr1)
            if self.base_type == "edge":
                gr1 = graph_from_networkx(gr1, self.node_label, self.edge_label)
            else:
                gr1 = graph_from_networkx(gr1, self.node_label)

        x_ = torch.tensor(
            np.concatenate(self.kern.transform(gr1, return_embedding_only=True), axis=1)
        )
        y_ = torch.tensor(
            np.concatenate(self.kern.transform(gr2, return_embedding_only=True), axis=1)
        )

        # Note that the vector length of the WL procedure is indeterminate, and thus dim(Y) != dim(X) in general.
        # However, since the newly observed features in the test data is always concatenated at the end of the feature
        # matrix, these features will not matter for the inference, and as such we can safely truncate the feature
        # matrix for the test data so that only those appearing in both the training and testing datasets are included.

        x_.requires_grad_()
        y_ = y_[:, : x_.shape[1]].requires_grad_()
        K = calculate_kernel_matrix_as_tensor(x_, y_, oa=self.oa, se_kernel=self.se)
        return K, y_, x_

    def feature_map(self, flatten=True):
        """
        Get the feature map in term of encoding (position in the feature index): the feature string.
        Parameters
        ----------
        flatten: whether flatten the dict (originally, the result is layered in term of h (the number of WL iterations).

        Returns
        -------

        """
        if not self.requires_grad:
            logging.warning(
                "Requires_grad flag is off -- in this case, there is risk that the element order in the "
                "feature map DOES NOT correspond to the order in the feature matrix. To suppress this warning,"
                "when initialising the WL kernel, do WeisfilerLehman(requires_grad=True)"
            )
        if self._gram is None:
            return None
        if not flatten:
            return self.kern._label_node_attr  # pylint: disable=W0212
        else:
            res = {}
            for _, map_ in self.kern._label_node_attr.items():  # pylint: disable=W0212
                for k, v in map_.items():
                    res.update({k: v})
            return res

    def feature_value(self, X_s):
        """Given a list of architectures X_s, compute their WL embedding of size N_s x D, where N_s is the length
        of the list and D is the number of training set features.

        Returns:
            embedding: torch.Tensor of shape N_s x D, described above
            names: list of shape D, which has 1-to-1 correspondence to each element of the embedding matrix above
        """
        if not self.requires_grad:
            logging.warning(
                "Requires_grad flag is off -- in this case, there is risk that the element order in the "
                "feature map DOES NOT correspond to the order in the feature matrix. To suppress this warning,"
                "when initialising the WL kernel, do WeisfilerLehman(requires_grad=True)"
            )
        feat_map = self.feature_map(flatten=False)
        len_feat_map = [len(f) for f in feat_map.values()]
        X_s = graph_from_networkx(
            X_s,
            self.node_label,
        )
        embedding = self.kern.transform(X_s, return_embedding_only=True)
        for j, em in enumerate(embedding):
            # Remove some of the spurious features that pop up sometimes
            embedding[j] = em[:, : len_feat_map[j]]

        # Generate the final embedding
        embedding = torch.tensor(np.concatenate(embedding, axis=1))
        return embedding, list(self.feature_map(flatten=True).values())


def _grid_search_wl_kernel(
    k: WeisfeilerLehman,
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
        # self.logger.debug(K)
        K_i, logDetK = compute_pd_inverse(K, lik)
        # self.logger.debug(train_y)
        nlml = -compute_log_marginal_likelihood(K_i, logDetK, train_y)
        # self.logger.debug(f"{i} {nlml}")
        if nlml < best_nlml:
            best_nlml = nlml
            best_subtree_depth, best_lengthscale = i
            best_K = torch.clone(K)
    # self.logger.debug(f"h: {best_subtree_depth} theta: {best_lengthscale}")
    # self.logger.debug(best_subtree_depth)
    k.change_kernel_params({"h": best_subtree_depth})
    if k.se is not None:
        k.change_se_params({"lengthscale": best_lengthscale})
    k._gram = best_K  # pylint: disable=protected-access


# TODO replace below functions with GPyTorch


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
            jitter_diag = jitter * torch.eye(n, device=K.device) * 10**fail_count
            K_ = K + jitter_diag
            try:
                Kc = torch.linalg.cholesky(K_)
            except AttributeError:  # For torch < 1.8.0
                Kc = torch.cholesky(K_)
            is_successful = True
        except RuntimeError:
            fail_count += 1
    if not is_successful:
        raise RuntimeError(f"Gram matrix not positive definite despite of jitter:\n{K}")
    logDetK = -2 * torch.sum(torch.log(torch.diag(Kc)))
    K_i = torch.cholesky_inverse(Kc)
    return K_i.to(torch.get_default_dtype()), logDetK.to(torch.get_default_dtype())
