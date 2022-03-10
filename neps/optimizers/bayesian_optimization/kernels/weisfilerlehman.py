import logging

import numpy as np
import torch
from grakel.kernels import ShortestPathAttr
from grakel.utils import graph_from_networkx

from .grakel_replace.edge_histogram import EdgeHistogram
from .grakel_replace.utils import calculate_kernel_matrix_as_tensor
from .grakel_replace.vertex_histogram import VertexHistogram
from .grakel_replace.weisfeiler_lehman import WeisfeilerLehman as _WL
from .graph_kernel import GraphKernels
from .utils import transform_to_undirected
from .vectorial_kernels import Stationary


class WeisfilerLehman(GraphKernels):
    """Weisfiler Lehman kernel using grakel functions"""

    def __init__(
        self,
        h: int = 0,
        base_type: str = "subtree",
        se_kernel: Stationary = None,
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
        return K

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
