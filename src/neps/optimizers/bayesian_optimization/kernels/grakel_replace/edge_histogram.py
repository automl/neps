"""The Edge Histogram kernel as defined in :cite:`sugiyama2015halting`."""
from collections import Counter
from collections.abc import Iterable
from warnings import warn

from grakel.graph import Graph
from numpy import zeros
from scipy.sparse import csr_matrix

from .vertex_histogram import VertexHistogram


class EdgeHistogram(VertexHistogram):
    """Edge Histogram kernel as found in :cite:`sugiyama2015halting`.

    Parameters
    ----------
    sparse : bool, or 'auto', default='auto'
        Defines if the data will be stored in a sparse format.
        Sparse format is slower, but less memory consuming and in some cases the only solution.
        If 'auto', uses a sparse matrix when the number of zeros is more than the half of the matrix size.
        In all cases if the dense matrix doesn't fit system memory, I sparse approach will be tried.

    Attributes
    ----------
    None.

    """

    def parse_input(self, X: Iterable, **kwargs):  # pylint: disable=unused-argument
        """Parse and check the given input for EH kernel.

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
            A np array for frequency (cols) histograms for all Graphs (rows).

        """
        if not isinstance(X, Iterable):
            raise TypeError("input must be an iterable\n")
        else:
            rows, cols, data = list(), list(), list()
            if self._method_calling in [1, 2]:
                labels = dict()
                self._labels = labels  # pylint: disable=W0201
            elif self._method_calling == 3:
                labels = dict(self._labels)
            ni = 0
            for i, x in enumerate(iter(X)):
                is_iter = isinstance(x, Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and len(x) in [0, 3]:
                    if len(x) == 0:
                        warn("Ignoring empty element on index: " + str(i))
                        continue
                    else:
                        # Our element is an iterable of at least 2 elements
                        L = x[2]
                elif isinstance(x, Graph):
                    # get labels in any existing format
                    L = x.get_labels(purpose="any", label_type="edge")
                else:
                    raise TypeError(
                        "each element of X must be either a "
                        + "graph object or a list with at least "
                        + "a graph like object and node labels "
                        + "dict \n"
                    )

                if L is None:
                    raise ValueError("Invalid graph entry at location " + str(i) + "!")
                # construct the data input for the numpy array
                for label, frequency in Counter(L.values()).items():
                    # for the row that corresponds to that graph
                    rows.append(ni)

                    # and to the value that this label is indexed
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

            # Initialise the feature matrix
            if self._method_calling in [1, 2]:
                if self.sparse == "auto":
                    self.sparse_ = (  # pylint: disable=W0201
                        len(cols) / float(ni * len(labels)) <= 0.5
                    )
                else:
                    self.sparse_ = bool(self.sparse)  # pylint: disable=W0201

            if self.sparse_:
                features = csr_matrix(
                    (data, (rows, cols)), shape=(ni, len(labels)), copy=False
                )
            else:
                # Initialise the feature matrix
                try:
                    features = zeros(shape=(ni, len(labels)))
                    features[rows, cols] = data
                except MemoryError:
                    warn("memory-error: switching to sparse")
                    self.sparse_, features = True, csr_matrix(  # pylint: disable=W0201
                        (data, (rows, cols)), shape=(ni, len(labels)), copy=False
                    )

            if ni == 0:
                raise ValueError("parsed input is empty")
            return features
