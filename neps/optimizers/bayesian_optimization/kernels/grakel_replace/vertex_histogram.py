"""The vertex kernel as defined in :cite:`sugiyama2015halting`."""
import logging
from collections import Counter
from collections.abc import Iterable
from warnings import warn

import numpy as np
import torch
from grakel.graph import Graph
from grakel.kernels import Kernel
from numpy import array, einsum, squeeze, zeros
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..vectorial_kernels import Stationary


class VertexHistogram(Kernel):
    """Vertex Histogram kernel as found in :cite:`sugiyama2015halting`.

    Parameters
    ----------
    sparse : bool, or 'auto', default='auto'
        Defines if the data will be stored in a sparse format.
        Sparse format is slower, but less memory consuming and in some cases the only solution.
        If 'auto', uses a sparse matrix when the number of zeros is more than the half of the matrix size.
        In all cases if the dense matrix doesn't fit system memory, I sparse approach will be tried.

    oa: bool: default=True
        Defines whether optimal assignment variant of the kernel should be used.

    se_kernel: default=None
        The standard vectorial kernel to be used for successive embedding (i.e. after the transformation from graph
        to the vector embedding, whether to use an additional kernel to compute the vector similarity.

    se_kernel_params: dict, default=None
        Any parameters to be passed to the se_kernel

    mahalanobis_precision: np.array:
        If supplied, the Malahanobis distance with the precision matrix as supplied will be computed in the dot
        product, instead of the vanilla dot product.

    Attributes
    ----------
    None.

    """

    def __init__(
        self,
        n_jobs=None,
        normalize=False,
        sparse="auto",
        oa=False,
        mahalanobis_precision=None,
        se_kernel: Stationary = None,
        requires_ordered_features: bool = False,
        as_tensor: bool = True,
    ):
        """Initialise a vertex histogram kernel.

        require_ordered_features: bool
            Whether the ordering of the features in the feature matrix matters.
            If True, the features will be parsed in the same order as the WL
            node label.

            Note that if called directly (not from Weisfiler Lehman kernel), turning
            this option on could break the code, as the label in general is non-int.

        """
        super().__init__(n_jobs=n_jobs, normalize=normalize)
        self.as_tensor = as_tensor
        if self.as_tensor:
            self.sparse = False
        else:
            self.sparse = sparse
        self.oa = oa
        self.se_kernel = se_kernel
        self._initialized.update({"sparse": True})
        self.mahalanobis_precision = mahalanobis_precision
        self.require_ordered_features = requires_ordered_features

        self._X_diag = None
        self.X_tensor = None
        self.Y_tensor = None

        self._labels = None
        self.sparse_ = None
        self._method_calling = None
        self._Y = None
        self._is_transformed = None
        self.X = None

    def initialize(self):
        """Initialize all transformer arguments, needing initialization."""
        if not self._initialized["n_jobs"]:
            if self.n_jobs is not None:
                warn("no implemented parallelization for VertexHistogram")
            self._initialized["n_jobs"] = True
        if not self._initialized["sparse"]:
            if self.sparse not in ["auto", False, True]:
                TypeError("sparse could be False, True or auto")
            self._initialized["sparse"] = True

    def parse_input(self, X, label_start_idx=0, label_end_idx=None):
        """Parse and check the given input for VH kernel.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format).



        Returns
        -------
        out : np.array, shape=(len(X), n_labels)
            A np.array for frequency (cols) histograms for all Graphs (rows).

        """
        if self.require_ordered_features:
            if label_start_idx is None or label_end_idx is None:
                raise ValueError(
                    "When requires_ordered_features flag is True, you must supply the start and end"
                    "indices of the feature matrix to have consistent feature dimensions!"
                )
            assert (
                label_end_idx > label_start_idx
            ), "End index must be larger than the start index!"

        if not isinstance(X, Iterable):
            raise TypeError("input must be an iterable\n")
        else:
            rows, cols, data = list(), list(), list()
            if self._method_calling in [0, 1, 2]:
                labels = dict()
                self._labels = labels
            elif self._method_calling == 3:
                labels = dict(self._labels)
            ni = 0
            for i, x in enumerate(iter(X)):
                is_iter = isinstance(x, Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and len(x) in [0, 2, 3]:
                    if len(x) == 0:
                        warn("Ignoring empty element on index: " + str(i))
                        continue
                    else:
                        # Our element is an iterable of at least 2 elements
                        L = x[1]
                elif isinstance(x, Graph):
                    # get labels in any existing format
                    L = x.get_labels(purpose="any")
                else:
                    raise TypeError(
                        "each element of X must be either a "
                        "graph object or a list with at least "
                        "a graph like object and node labels "
                        "dict \n"
                    )

                # construct the data input for the numpy array
                for label, frequency in Counter(L.values()).items():
                    # for the row that corresponds to that graph
                    rows.append(ni)

                    # and to the value that this label is indexed
                    if self.require_ordered_features:
                        try:
                            col_idx = int(label) - label_start_idx  # Offset
                        except ValueError:
                            logging.error(
                                "Failed to convert label to a valid integer. Check whether all labels are"
                                "numeric, and whether you called this kernel directly instead of from the"
                                "Weisfiler-Lehman kernel. Falling back to the default unordered feature"
                                "matrix."
                            )
                            self.require_ordered_features = False
                    if not self.require_ordered_features:
                        col_idx = labels.get(label, None)
                        if col_idx is None:
                            # if not indexed, add the new index (the next)
                            col_idx = len(labels)
                            labels[label] = col_idx

                    # designate the certain column information
                    cols.append(col_idx)

                    # as well as the frequency value to data
                    data.append(frequency)
                ni += 1

            if self.require_ordered_features:
                label_length = max(label_end_idx - label_start_idx, max(cols)) + 1
            else:
                label_length = len(labels)

            if self._method_calling in [0, 1, 2]:
                if self.sparse == "auto":
                    self.sparse_ = len(cols) / float(ni * label_length) <= 0.5
                else:
                    self.sparse_ = bool(self.sparse)

            if self.sparse_:
                features = csr_matrix(
                    (data, (rows, cols)), shape=(ni, label_length), copy=False
                )
            else:
                # Initialise the feature matrix
                try:
                    features = zeros(shape=(ni, label_length))
                    features[rows, cols] = data

                except MemoryError:
                    warn("memory-error: switching to sparse")
                    self.sparse_, features = True, csr_matrix(
                        (data, (rows, cols)), shape=(ni, label_length), copy=False
                    )

            if ni == 0:
                raise ValueError("parsed input is empty")
            return features

    def _calculate_kernel_matrix(self, Y=None):
        """Calculate the kernel matrix given a target_graph and a kernel.

        Each a matrix is calculated between all elements of Y on the rows and
        all elements of X on the columns.

        Parameters
        ----------
        Y : np.array, default=None
            The array between samples and features.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_inputs]
            The kernel matrix: a calculation between all pairs of graphs
            between targets and inputs. If Y is None targets and inputs
            are the taken from self.X. Otherwise Y corresponds to targets
            and self.X to inputs.

        """
        if Y is None:
            if self.oa:
                K = np.zeros((self.X.shape[0], self.X.shape[0]))
                for i in range(self.X.shape[0]):
                    for j in range(i, self.X.shape[0]):
                        K[i, j] = np.sum(np.minimum(self.X[i, :], self.X[j, :]))
                        K[j, i] = K[i, j]
            else:
                if self.se_kernel is not None:
                    K = self.se_kernel.forward(self.X, self.X)
                else:
                    K = self.X @ self.X.T
        else:
            if self.oa:
                K = np.zeros((Y.shape[0], self.X.shape[0]))
                for i in range(Y.shape[0]):
                    for j in range(self.X.shape[0]):
                        K[i, j] = np.sum(
                            np.minimum(self.X[j, :], Y[i, : self.X.shape[1]])
                        )
            else:
                if self.se_kernel is not None:
                    K = self.se_kernel.forward(self.X, Y)
                else:
                    K = Y[:, : self.X.shape[1]] @ self.X.T

        if self.sparse_:
            return K.toarray()
        else:
            return K

    def diagonal(self, use_tensor=False):
        """Calculate the kernel matrix diagonal of the fitted data.

        Parameters
        ----------
        None.

        Returns
        -------
        X_diag : np.array
            The diagonal of the kernel matrix, of the fitted. This consists
            of each element calculated with itself.

        use_tensor: bool:
            The flag to use whether return tensor instead of numpy array. All other operations are the same

        """
        # Check is fit had been called
        check_is_fitted(self, ["X", "sparse_"])
        try:
            check_is_fitted(self, ["_X_diag"])
        except NotFittedError:
            # Calculate diagonal of X
            if use_tensor:
                self._X_diag = torch.einsum("ij,ij->i", [self.X_tensor, self.X_tensor])
            else:
                if self.sparse_:
                    self._X_diag = squeeze(array(self.X.multiply(self.X).sum(axis=1)))
                else:
                    self._X_diag = einsum("ij,ij->i", self.X, self.X)
        try:
            check_is_fitted(self, ["_Y"])
            if use_tensor:
                Y_diag = torch.einsum("ij, ij->i", [self.Y_tensor, self.Y_tensor])
                return self._X_diag, Y_diag
            else:
                if self.sparse_:
                    Y_diag = squeeze(array(self._Y.multiply(self._Y).sum(axis=1)))
                else:
                    Y_diag = einsum("ij,ij->i", self._Y, self._Y)
                return self._X_diag, Y_diag
        except NotFittedError:
            return self._X_diag

    def transform(self, X, return_embedding_only=False, **kwargs):
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
            Whether returns the vector embedding of the kernel only (without actually
            computing the kernel function). This is used when computing the derivative
            of the kernel w.r.t. the test points/

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 3
        # Check is fit had been called
        check_is_fitted(self, ["X"])

        # Input validation and parsing
        if X is None:
            raise ValueError("`transform` input cannot be None")
        else:
            Y = self.parse_input(X, **kwargs)
        if return_embedding_only:
            return Y

        self._Y = Y
        self._is_transformed = True

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_matrix(Y)
        # Self transform must appear before the diagonal call on normilization
        if self.normalize:
            X_diag, Y_diag = self.diagonal()
            km /= np.sqrt(np.outer(Y_diag, X_diag))
        if self.as_tensor:
            km = torch.tensor(km)
        return km

    def fit_transform(self, X, **kwargs):
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

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        K : numpy array, shape = [n_targets, n_input_graphs]
            corresponding to the kernel matrix, a calculation between
            all pairs of graphs between target an features

        """
        self._method_calling = 2
        self.fit(X, **kwargs)

        # Transform - calculate kernel matrix
        km = self._calculate_kernel_matrix()

        self._X_diag = np.diagonal(km)
        if self.normalize:
            km = km / np.sqrt(np.outer(self._X_diag, self._X_diag))
        if self.as_tensor:
            km = torch.tensor(km)
        return km

    def fit(self, X, y=None, **kwargs):  # pylint: disable=unused-argument
        """Fit a dataset, for a transformer.

        Parameters
        ----------
        X : iterable
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that fitting the given graph
            format). The train samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
        Returns self.

        """
        self._is_transformed = False
        self._method_calling = 1

        # Parameter initialization
        self.initialize()

        # Input validation and parsing
        if X is None:
            raise ValueError("`fit` input cannot be None")
        else:
            self.X = self.parse_input(X, **kwargs)

        # Return the transformer
        return self
