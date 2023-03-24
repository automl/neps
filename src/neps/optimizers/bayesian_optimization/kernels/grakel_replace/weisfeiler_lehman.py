"""The weisfeiler lehman kernel :cite:`shervashidze2011weisfeiler`."""

import collections
import collections.abc
import logging
import warnings
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
from grakel.graph import Graph
from grakel.kernels import Kernel
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .vertex_histogram import VertexHistogram


class WeisfeilerLehman(Kernel):
    """Compute the Weisfeiler Lehman Kernel.

     See :cite:`shervashidze2011weisfeiler`.

    Parameters
    ----------
    h : int, default=5
        The number of iterations.

    base_graph_kernel : `grakel.kernel_operators.Kernel` or tuple, default=None
        If tuple it must consist of a valid kernel object and a
        dictionary of parameters. General parameters concerning
        normalization, concurrency, .. will be ignored, and the
        ones of given on `__init__` will be passed in case it is needed.
        Default `base_graph_kernel` is `VertexHistogram`.

    node_weights: iterable
        If not None, the nodes will be assigned different weights according
        to this vector. Must be a dictionary with the following format:
        {'node_name1': weight1, 'node_name2': weight2 ... }
        Must be of the same length as the number of different node attributes

    Attributes
    ----------
    X : dict
     Holds a dictionary of fitted subkernel modules for all levels.

    _nx : number
        Holds the number of inputs.

    _h : int
        Holds the number, of iterations.

    _base_graph_kernel : function
        A void function that initializes a base kernel object.

    _inv_labels : dict
        An inverse dictionary, used for relabeling on each iteration.

    """

    _graph_format = "dictionary"

    def __init__(
        self,
        n_jobs=None,
        normalize: bool = False,
        h: int = 5,
        base_graph_kernel=VertexHistogram,
        node_weights=None,
        layer_weights=None,
        as_tensor: bool = True,
    ):
        """Initialise a `weisfeiler_lehman` kernel."""
        super().__init__(n_jobs=n_jobs, normalize=normalize)

        self.h = h
        self.base_graph_kernel = base_graph_kernel
        self._initialized.update(
            {"h": False, "base_graph_kernel": False, "layer_weights": False}
        )
        self._base_graph_kernel = None
        self.weights = None
        self.node_weights = node_weights
        self.as_tensor = as_tensor
        self.layer_weights = layer_weights  # The weights of each layer. If None, each WL iteration has same weight
        self.feature_dims = [
            0,
        ]  # Record the dimensions of the vectors of each WL iteration
        self._params = None
        self._h = None
        self._nx = None
        self._inv_labels = None
        self._inv_label_node_attr = None
        self._label_node_attr = None
        self._feature_weight = None
        self._method_calling = None
        self._is_transformed = None
        self.X = None
        self._X_diag = None

        self.X_fit = dict()
        self.K_precomputed = dict()
        self.base_graph_kernel_precomputed = dict()

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        super().initialize()
        if not self._initialized["base_graph_kernel"]:
            base_graph_kernel = self.base_graph_kernel
            if base_graph_kernel is None:
                base_graph_kernel, params = VertexHistogram, dict()
            # TODO: make sure we're always passing like this
            elif type(base_graph_kernel) is type and issubclass(  # pylint: disable=C0123
                base_graph_kernel, Kernel
            ):
                params = dict()
            else:
                try:
                    base_graph_kernel, params = base_graph_kernel
                except Exception as _error:
                    raise TypeError(
                        "Base kernel was not formulated in "
                        "the correct way. "
                        "Check documentation."
                    ) from _error

                if not (
                    type(base_graph_kernel) is type  # pylint: disable=C0123
                    and issubclass(base_graph_kernel, Kernel)
                ):
                    raise TypeError(
                        "The first argument must be a valid "
                        "grakel.kernel.kernel Object"
                    )
                if not isinstance(params, dict):
                    raise ValueError(
                        "If the second argument of base "
                        "kernel exists, it must be a diction"
                        "ary between parameters names and "
                        "values"
                    )
                params.pop("normalize", None)

            params["normalize"] = False
            params["n_jobs"] = None
            self._base_graph_kernel = base_graph_kernel
            self._params = params
            self._initialized["base_graph_kernel"] = True

        if not self._initialized["h"]:
            if not isinstance(self.h, int) or self.h < 0:
                raise TypeError(
                    "'h' must be a non-negative integer. Got h:" + str(self.h)
                )
            self._h = self.h + 1
            self._initialized["h"] = True

            if self.layer_weights is None or self.layer_weights.shape[0] != self._h:
                self.layer_weights = np.ones((self._h,))
            if self.as_tensor and not isinstance(self.layer_weights, torch.Tensor):
                self.layer_weights = torch.tensor(self.layer_weights)

            self._initialized["h"] = True
            self._initialized["layer_weights"] = True

    def change_se_kernel(self, se_kernel):
        if self._base_graph_kernel is None:
            self.initialize()
        self._params["se_kernel"] = se_kernel
        logging.info("Base kernel changed")

    def parse_input(
        self, X: Iterable, return_embedding_only: bool = False, gp_fit: bool = True
    ):
        """Parse input for weisfeiler lehman.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). A valid input also consists of graph type objects.

        return_embedding_only: bool
            Whether to return the embedding of the graphs only, instead of computing the kernel all
            the way to the end.

        gp_fit: bool
            If False use precomputed vals for first N values, else compute them and save them

        Returns
        -------
        base_graph_kernel : object
        Returns base_graph_kernel.

        if requires_grad is enabled and we call fit_transform or transform, an additional torch tensor
        K_grad is returned as well.

        """
        if self._method_calling not in [1, 2]:
            raise ValueError(
                "method call must be called either from fit " + "or fit-transform"
            )
        elif hasattr(self, "_X_diag"):
            # Clean _X_diag value
            delattr(self, "_X_diag")

        # skip kernel computation if we have already computed the corresponding kernel
        if self._h in self.K_precomputed.keys() and self.X_fit[self._h] == X:
            K = self.K_precomputed[self._h]
            base_graph_kernel = self.base_graph_kernel_precomputed[self._h]
        else:
            # Input validation and parsing
            if not isinstance(X, collections.abc.Iterable):
                raise TypeError("input must be an iterable\n")
            else:
                nx = 0
                Gs_ed, L, distinct_values, extras = dict(), dict(), set(), dict()
                for idx, x in enumerate(iter(X)):
                    is_iter = isinstance(x, collections.abc.Iterable)
                    if is_iter:
                        x = list(x)
                    if is_iter and (len(x) == 0 or len(x) >= 2):
                        if len(x) == 0:
                            warnings.warn("Ignoring empty element on index: " + str(idx))
                            continue
                        else:
                            if len(x) > 2:
                                extra = tuple()
                                if len(x) > 3:
                                    extra = tuple(x[3:])
                                x = Graph(
                                    x[0], x[1], x[2], graph_format=self._graph_format
                                )
                                extra = (
                                    x.get_labels(
                                        purpose=self._graph_format,
                                        label_type="edge",
                                        return_none=True,
                                    ),
                                ) + extra
                            else:
                                x = Graph(x[0], x[1], {}, graph_format=self._graph_format)
                                extra = tuple()

                    elif isinstance(x, Graph):
                        x.desired_format(self._graph_format)
                        el = x.get_labels(
                            purpose=self._graph_format,
                            label_type="edge",
                            return_none=True,
                        )
                        if el is None:
                            extra = tuple()
                        else:
                            extra = (el,)

                    else:
                        raise TypeError(
                            "each element of X must be either a "
                            + "graph object or a list with at least "
                            + "a graph like object and node labels "
                            + "dict \n"
                        )
                    Gs_ed[nx] = x.get_edge_dictionary()
                    L[nx] = x.get_labels(purpose="dictionary")
                    extras[nx] = extra
                    distinct_values |= set(L[nx].values())
                    nx += 1
                if nx == 0:
                    raise ValueError("parsed input is empty")

            # Save the number of "fitted" graphs.
            self._nx = nx
            WL_labels_inverse = OrderedDict()

            # assign a number to each label
            label_count = 0
            for dv in sorted(list(distinct_values)):
                WL_labels_inverse[dv] = label_count
                label_count += 1

            # Initalize an inverse dictionary of labels for all iterations
            self._inv_labels = (
                OrderedDict()
            )  # Inverse dictionary of labels, in term of the *previous layer*
            self._inv_labels[0] = deepcopy(WL_labels_inverse)
            self.feature_dims.append(
                len(WL_labels_inverse)
            )  # Update the zeroth iteration feature dim

            self._inv_label_node_attr = (
                OrderedDict()
            )  # Inverse dictionary of labels, in term of the *node attribute*
            self._label_node_attr = (
                OrderedDict()
            )  # Same as above, but with key and value inverted
            self._label_node_attr[0], self._inv_label_node_attr[0] = self.translate_label(
                WL_labels_inverse, 0
            )

            if self.node_weights is not None:
                self._feature_weight = OrderedDict()
                # Ensure the order is the same
                self._feature_weight[0] = self._compute_feature_weight(
                    self.node_weights, 0, WL_labels_inverse
                )[1]
            else:
                self._feature_weight = None

            def generate_graphs(label_count: int, WL_labels_inverse):
                new_graphs = list()
                for j in range(self._nx):
                    new_labels = dict()
                    for k in L[j].keys():
                        new_labels[k] = WL_labels_inverse[L[j][k]]
                    L[j] = new_labels
                    # add new labels
                    new_graphs.append((Gs_ed[j], new_labels) + extras[j])
                yield new_graphs

                for i in range(1, self._h):
                    label_set, WL_labels_inverse, L_temp = set(), dict(), dict()
                    for j in range(nx):
                        # Find unique labels and sort
                        # them for both graphs
                        # Keep for each node the temporary
                        L_temp[j] = dict()
                        for v in Gs_ed[j].keys():
                            credential = (
                                str(L[j][v])
                                + ","
                                + str(sorted(L[j][n] for n in Gs_ed[j][v].keys()))
                            )
                            L_temp[j][v] = credential
                            label_set.add(credential)

                    label_list = sorted(list(label_set))
                    for dv in label_list:
                        WL_labels_inverse[dv] = label_count
                        label_count += 1

                    # Recalculate labels
                    new_graphs = list()
                    for j in range(nx):
                        new_labels = dict()
                        for k in L_temp[j].keys():
                            new_labels[k] = WL_labels_inverse[L_temp[j][k]]
                        L[j] = new_labels
                        # relabel
                        new_graphs.append((Gs_ed[j], new_labels) + extras[j])
                    self._inv_labels[i] = WL_labels_inverse
                    # Compute the translated inverse node label
                    (
                        self._label_node_attr[i],
                        self._inv_label_node_attr[i],
                    ) = self.translate_label(
                        WL_labels_inverse, i, self._label_node_attr[i - 1]
                    )
                    self.feature_dims.append(
                        self.feature_dims[-1] + len(self._label_node_attr[i])
                    )
                    # Compute the feature weight of the current layer
                    if self.node_weights is not None:
                        self._feature_weight[i] = self._compute_feature_weight(
                            self.node_weights, i, self._inv_label_node_attr[i]
                        )[1]
                    # assert len(self._feature_weight[i] == len(WL_labels_inverse))
                    yield new_graphs

            # Initialise the base graph kernel.
            base_graph_kernel = {}

            K = []
            for i, g in enumerate(generate_graphs(label_count, WL_labels_inverse)):
                param = self._params
                if self._feature_weight is not None:
                    param.update({"mahalanobis_precision": self._feature_weight[i]})
                base_graph_kernel.update({i: self._base_graph_kernel(**param)})
                if return_embedding_only:
                    K.append(
                        base_graph_kernel[i].parse_input(
                            g,
                            label_start_idx=self.feature_dims[i],
                            label_end_idx=self.feature_dims[i + 1],
                        )
                    )
                else:
                    if self._method_calling == 1:
                        base_graph_kernel[i].fit(
                            g,
                            label_start_idx=self.feature_dims[i],
                            label_end_idx=self.feature_dims[i + 1],
                        )
                    else:
                        K.append(
                            self.layer_weights[i]
                            * base_graph_kernel[i].fit_transform(
                                g,
                                label_start_idx=self.feature_dims[i],
                                label_end_idx=self.feature_dims[i + 1],
                            )
                        )

            if gp_fit:
                self.X_fit[self._h] = X
                self.K_precomputed[self._h] = K
                self.base_graph_kernel_precomputed[self._h] = base_graph_kernel

        if return_embedding_only:
            return K
        elif self._method_calling == 1:
            return base_graph_kernel
        elif self._method_calling == 2:
            if self.as_tensor:
                K = torch.stack(K, dim=0).sum(dim=0)
                return K, base_graph_kernel
            return np.sum(K, axis=0), base_graph_kernel

    def fit_transform(
        self, X: Iterable, y=None, gp_fit: bool = True
    ):  # pylint: disable=unused-argument
        """Fit and transform, on the same dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        y : Object, default=None
            Ignored argument, added for the pipeline.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 2
        self._is_transformed = False
        self.initialize()
        self.feature_dims = [
            0,
        ]  # Flush the feature dimensions
        if X is None:
            raise ValueError("transform input cannot be None")
        else:
            km, self.X = self.parse_input(X, gp_fit=gp_fit)

        return km

    def transform(self, X: Iterable, return_embedding_only: bool = True):
        """Calculate the kernel matrix, between given and fitted dataset.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). If None the kernel matrix is calculated upon fit data.
            The test samples.

        return_embedding_only: bool
            Whether to return the embedding of the graphs only, instead of computing the kernel all
            the way to the end.
        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ["X", "_nx", "_inv_labels"])

        # Input validation and parsing
        if X is None:
            raise ValueError("transform input cannot be None")
        else:
            if not isinstance(X, collections.abc.Iterable):
                raise ValueError("input must be an iterable\n")
            else:
                nx = 0
                distinct_values = set()
                Gs_ed, L = dict(), dict()
                for i, x in enumerate(iter(X)):
                    is_iter = isinstance(x, collections.abc.Iterable)
                    if is_iter:
                        x = list(x)
                    if is_iter and len(x) in [0, 2, 3]:
                        if len(x) == 0:
                            warnings.warn("Ignoring empty element on index: " + str(i))
                            continue

                        elif len(x) in [2, 3]:
                            x = Graph(x[0], x[1], {}, self._graph_format)
                    elif isinstance(x, Graph):
                        x.desired_format("dictionary")
                    else:
                        raise ValueError(
                            "each element of X must have at "
                            + "least one and at most 3 elements\n"
                        )
                    Gs_ed[nx] = x.get_edge_dictionary()
                    L[nx] = x.get_labels(purpose="dictionary")

                    # Hold all the distinct values
                    distinct_values |= {
                        v for v in L[nx].values() if v not in self._inv_labels[0]
                    }
                    nx += 1
                if nx == 0:
                    raise ValueError("parsed input is empty")

        nl = len(self._inv_labels[0])
        WL_labels_inverse = {
            dv: idx for (idx, dv) in enumerate(sorted(list(distinct_values)), nl)
        }
        WL_labels_inverse = OrderedDict(WL_labels_inverse)

        def generate_graphs_transform(WL_labels_inverse, nl):
            # calculate the kernel matrix for the 0 iteration
            new_graphs = list()
            for j in range(nx):
                new_labels = dict()
                for k, v in L[j].items():
                    if v in self._inv_labels[0]:
                        new_labels[k] = self._inv_labels[0][v]
                    else:
                        new_labels[k] = WL_labels_inverse[v]
                L[j] = new_labels
                # produce the new graphs
                new_graphs.append([Gs_ed[j], new_labels])
            yield new_graphs

            for i in range(1, self._h):
                new_graphs = list()
                L_temp, label_set = dict(), set()
                nl += len(self._inv_labels[i])
                for j in range(nx):
                    # Find unique labels and sort them for both graphs
                    # Keep for each node the temporary
                    L_temp[j] = dict()
                    for v in Gs_ed[j].keys():
                        credential = (
                            str(L[j][v])
                            + ","
                            + str(sorted(L[j][n] for n in Gs_ed[j][v].keys()))
                        )
                        L_temp[j][v] = credential
                        if credential not in self._inv_labels[i]:
                            label_set.add(credential)

                # Calculate the new label_set
                WL_labels_inverse = dict()
                if len(label_set) > 0:
                    for dv in sorted(list(label_set)):
                        idx = len(WL_labels_inverse) + nl
                        WL_labels_inverse[dv] = idx

                # Recalculate labels
                new_graphs = list()
                for j in range(nx):
                    new_labels = dict()
                    for k, v in L_temp[j].items():
                        if v in self._inv_labels[i]:
                            new_labels[k] = self._inv_labels[i][v]
                        else:
                            new_labels[k] = WL_labels_inverse[v]
                    L[j] = new_labels
                    # Create the new graphs with the new labels.
                    new_graphs.append([Gs_ed[j], new_labels])
                yield new_graphs

        if return_embedding_only:
            K = []
            for i, g in enumerate(generate_graphs_transform(WL_labels_inverse, nl)):
                K.append(
                    self.X[i].transform(
                        g,
                        label_start_idx=self.feature_dims[i],
                        label_end_idx=self.feature_dims[i + 1],
                        return_embedding_only=True,
                    )
                )
            return K

        # Calculate the kernel matrix without parallelization
        if self.as_tensor:
            summand = [
                self.layer_weights[i]
                * self.X[i].transform(
                    g,
                    label_start_idx=self.feature_dims[i],
                    label_end_idx=self.feature_dims[i + 1],
                )
                for i, g in enumerate(generate_graphs_transform(WL_labels_inverse, nl))
            ]
            K = torch.stack(summand, dim=0).sum(dim=0)
        else:
            K = np.sum(
                (
                    self.layer_weights[i]
                    * self.X[i].transform(
                        g,
                        label_start_idx=self.feature_dims[i],
                        label_end_idx=self.feature_dims[i + 1],
                    )
                    for (i, g) in enumerate(
                        generate_graphs_transform(WL_labels_inverse, nl)
                    )
                ),
                axis=0,
            )

        self._is_transformed = True
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            if self.as_tensor:
                div_ = torch.sqrt(torch.ger(Y_diag, X_diag))
                K /= div_
            else:
                old_settings = np.seterr(divide="ignore")
                K = np.nan_to_num(np.divide(K, np.sqrt(np.outer(Y_diag, X_diag))))
                np.seterr(**old_settings)

        return K

    def diagonal(self):
        """Calculate the kernel matrix diagonal for fitted data.

        A funtion called on transform on a seperate dataset to apply
        normalization on the exterior.

        Parameters
        ----------
        None.

        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted data.
            This consists of kernel calculation for each element with itself.

        Y_diag : np.array
            The diagonal of the kernel matrix, of the transformed data.
            This consists of kernel calculation for each element with itself.

        """
        # Check if fit had been called
        check_is_fitted(self, ["X"])
        try:
            check_is_fitted(self, ["_X_diag"])
            if self._is_transformed:
                Y_diag = self.X[0].diagonal()[1]
                for i in range(1, self._h):
                    Y_diag += self.X[i].diagonal()[1]
        except NotFittedError:
            # Calculate diagonal of X
            if self._is_transformed:
                X_diag, Y_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._h):
                    x, y = self.X[i].diagonal()
                    X_diag += x
                    Y_diag += y
                    self._X_diag = X_diag

                # case sub kernel is only fitted
                X_diag = self.X[0].diagonal()
                # X_diag is considered a mutable and should not affect the kernel matrix itself.
                X_diag.flags.writeable = True
                for i in range(1, self._n_iter):
                    x = self.X[i].diagonal()
                    X_diag += x
                self._X_diag = X_diag

        if self.as_tensor:
            self._X_diag = torch.tensor(self._X_diag)
            if Y_diag is not None:
                Y_diag = torch.tensor(Y_diag)
        if self._is_transformed:
            return self._X_diag, Y_diag
        else:
            return self._X_diag

    @staticmethod
    def translate_label(curr_layer: dict, h: int, prev_layer: dict = None):
        """Translate the label to be in terms of the node attributes
        curr_layer: the WL_label_inverse object. A dictionary with element of the format of
        {pattern: encoding}

        return:
           label_in_node_attr: in terms of {encoding: pattern}, but pattern is always in term of the node attribute
           inv_label_in_node_attr: in terms of {pattern: encoding}

        """
        if h == 0:
            return {v: str(k) for k, v in curr_layer.items()}, curr_layer
        else:
            assert prev_layer is not None
            label_in_node_attr, inv_label_in_node_attr = OrderedDict(), OrderedDict()
            for pattern, encoding in curr_layer.items():
                # current pattern is in terms of the encoding previous layer. Find the pattern from the prev_layer
                root, leaf = literal_eval(pattern)
                root_ = prev_layer[root]
                leaf_ = [prev_layer[i] for i in leaf]
                label_in_node_attr.update({encoding: "~".join([root_] + leaf_)})
                inv_label_in_node_attr.update({"~".join([root_] + leaf_): encoding})
            return label_in_node_attr, inv_label_in_node_attr

    @staticmethod
    def _compute_feature_weight(
        node_weight: OrderedDict, h: int, inv_label_node_attr: OrderedDict
    ):
        """
        Compute the feature weight, based on the average weight of the constituent node attributes.
        Return:
            feature_weights: a dictionary with h layers, each of which is a dictionary of the format of
            {tuple1: weight1; tuplr2, weight2 ...} where tuplex is the tuple representation of the learned graph feature

            feature_weight_flattened: same as above, but in a flattened np format.
        """
        feature_weights = OrderedDict()
        feature_weights_flattened = []
        if h == 0:
            feature_weight = OrderedDict(
                {k: (node_weight[k]) ** 2 for k in inv_label_node_attr.keys()}
            )
            feature_weights_flattened = np.array(list(feature_weight.values())).flatten()
        else:
            for k, _ in inv_label_node_attr.items():
                # k is the pattern, v is the encoding
                k_sep = k.split("~")
                average_weight = np.mean([(node_weight[i]) ** 2 for i in k_sep])
                feature_weights.update({k: average_weight})
                feature_weights_flattened.append(average_weight)
        feature_weights_flattened = np.array(feature_weights_flattened).flatten()
        assert len(feature_weights_flattened) == len(inv_label_node_attr)
        return feature_weights, feature_weights_flattened

    def dK_dX(self, X_test: None):
        """
        Do additional forward and backward pass, compute the kernel derivative wrt the testing location.
        If no test locations are provided, the derivatives are evaluated at the training points
        Returns
        -------

        """


def efit(obj, data):
    """Fit an object on data."""
    obj.fit(data)


def efit_transform(obj, data):
    """Fit-Transform an object on data."""
    return obj.fit_transform(data)


def etransform(obj, data):
    """Transform an object on data."""
    return obj.transform(data)
