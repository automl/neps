# Code from https://github.com/xingchenwan/nasbowl

import networkx as nx
import numpy as np
import torch

from .graph_kernel import GraphKernels

INPUT = "input"
OUTPUT = "output"
CONV3X3 = "conv3x3-bn-relu"
CONV1X1 = "conv1x1-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
OPS = [INPUT, CONV3X3, CONV1X1, MAXPOOL3X3, OUTPUT]
OPS_EX = [
    CONV3X3,
    CONV1X1,
    MAXPOOL3X3,
]

OPS_201 = ["avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3", "none", "skip_connect"]

NUM_VERTICES = 7
OP_SPOTS = NUM_VERTICES - 2
MAX_EDGES = 9


def get_op_list(string):
    # given a string, get the list of operations
    tokens = string.split("|")
    ops = [t.split("~")[0] for i, t in enumerate(tokens) if i not in [0, 2, 5, 9]]
    return ops


def edit_distance(g1, g2):
    g1_ops = get_op_list(g1.name)
    g2_ops = get_op_list(g2.name)
    return np.sum([1 for i in range(len(g1_ops)) if g1_ops[i] != g2_ops[i]])


class NASBOTDistance(GraphKernels):  # pylint: disable=abstract-method
    """NASBOT OATMANN distance according to BANANAS paper"""

    def __init__(
        self,
        node_name="op_name",
        include_op_list=None,
        exclude_op_list=None,
        lengthscale=3.0,
        normalize=True,
        max_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_name = node_name
        self.include_op_list = include_op_list if include_op_list is not None else OPS
        self.exclude_op_list = exclude_op_list if exclude_op_list is not None else []
        self.normalize = normalize
        self.lengthscale = lengthscale
        self.max_size = max_size
        self._gram = None

    def _compute_kernel(self, dist, l=None):
        if dist is None:
            return 0.0
        if l is None:
            l = self.lengthscale
        # print(dist)
        return np.exp(-dist / (l**2))

    def _compute_dist(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
    ):
        # if cell-based nasbench201
        if "~" in g1.name:
            g1_ops = get_op_list(g1.name)
            g2_ops = get_op_list(g2.name)

            g1_counts = [g1_ops.count(op) for op in OPS_201]
            g2_counts = [g2_ops.count(op) for op in OPS_201]
            ops_dist = np.sum(np.abs(np.subtract(g1_counts, g2_counts)))
            edit_dist = edit_distance(g1, g2)
            return ops_dist + edit_dist
        else:
            # adjacency matrices
            a1 = nx.to_numpy_array(g1)
            a2 = nx.to_numpy_array(g2)
            row_sums = sorted(np.array(a1).sum(axis=0))
            col_sums = sorted(np.array(a1).sum(axis=1))

            other_row_sums = sorted(np.array(a2).sum(axis=0))
            other_col_sums = sorted(np.array(a2).sum(axis=1))

            row_sums_arr = np.atleast_2d(row_sums)
            col_sums_arr = np.atleast_2d(col_sums)

            other_row_sums_arr = np.atleast_2d(other_row_sums)
            other_col_sums_arr = np.atleast_2d(other_col_sums)
            row_dist = np.sum(
                np.abs(np.diag(np.subtract(row_sums_arr, other_row_sums_arr.T)))
            )
            col_dist = np.sum(
                np.abs(np.diag(np.subtract(col_sums_arr, other_col_sums_arr.T)))
            )
            counts = [0] * len(self.include_op_list)
            other_counts = [0] * len(self.include_op_list)
            for _, attrs in g1.nodes(data=True):
                op_name = attrs[self.node_name]
                if op_name not in self.exclude_op_list:
                    idx = self.include_op_list.index(op_name)
                    counts[idx] += 1
            for _, attrs in g2.nodes(data=True):
                op_name = attrs[self.node_name]
                if op_name not in self.exclude_op_list:
                    idx = self.include_op_list.index(op_name)
                    other_counts[idx] += 1

            ops_dist = np.sum(np.abs(np.subtract(counts, other_counts)))
            return (row_dist + col_dist + ops_dist) + 0.0

    def forward(self, *graphs: nx.Graph, l: float = None):
        n = len(graphs)
        K = torch.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._compute_kernel(
                    self._compute_dist(graphs[i], graphs[j]), l
                )
                K[j, i] = K[i, j]
        if self.normalize:
            K = self.normalize_gram(K)
        return K

    def fit_transform(
        self,
        gr: list,
        l: float = None,
        rebuild_model: bool = False,
        save_gram_matrix: bool = False,
        **kwargs,
    ):
        if (
            not rebuild_model
            and self._gram is not None  # pylint: disable=access-member-before-definition
        ):
            return self._gram  # pylint: disable=access-member-before-definition
        K = self.forward(*gr, l=l)
        if save_gram_matrix:
            self._gram = K.clone()  # pylint: disable=attribute-defined-outside-init
            self._train_x = gr[:]  # pylint: disable=attribute-defined-outside-init
        return K

    def transform(
        self, gr: list, l: float = None, **kwargs
    ):  # pylint: disable=unused-argument
        if self._gram is None:
            raise ValueError("The kernel has not been fitted. Run fit_transform first")
        n = len(gr)
        K = torch.zeros((len(self._train_x), n))
        for i, _ in enumerate(self._train_x):
            for j in range(n):
                K[i, j] = self._compute_kernel(
                    self._compute_dist(self._train_x[i], gr[j]), l
                )
        return K


class AdjacencyDistance(  # pylint: disable=abstract-method
    NASBOTDistance,
):
    def _compute_dist(self, g1: nx.Graph, g2: nx.Graph):
        # adjacency matrices
        a1 = nx.to_numpy_array(g1)
        a2 = nx.to_numpy_array(g2)
        x1 = np.array([attrs[self.node_name] for node, attrs in g1.nodes(data=True)])
        x2 = np.array([attrs[self.node_name] for node, attrs in g2.nodes(data=True)])
        graph_dist = np.sum(a1 != a2)
        ops_dist = np.sum(x1 != x2)
        return (graph_dist + ops_dist) + 0.0


class PathDistance(NASBOTDistance):  # pylint: disable=abstract-method
    def get_paths(self, g: nx.Graph):
        """
        return all paths from input to output
        """
        paths: list = []
        matrix = nx.to_numpy_array(g)
        ops: list = []
        for _, attr in g.nodes(data=True):
            ops.append(attr[self.node_name])
        for j in range(0, NUM_VERTICES):
            if matrix[0][j]:
                paths.append([[]])
            else:
                paths.append([])

        # create paths sequentially
        for i in range(1, NUM_VERTICES - 1):
            for j in range(1, NUM_VERTICES):
                if matrix[i][j]:
                    for path in paths[i]:
                        paths[j].append([*path, ops[i]])
        return paths[-1]

    def get_path_indices(self, g: nx.Graph):
        """
        compute the index of each path
        There are 3^0 + ... + 3^5 paths total.
        (Paths can be length 0 to 5, and for each path, for each node, there
        are three choices for the operation.)
        """
        paths = self.get_paths(g)
        mapping = {CONV3X3: 0, CONV1X1: 1, MAXPOOL3X3: 2}
        path_indices = []

        for path in paths:
            index = 0
            for i in range(NUM_VERTICES - 1):
                if i == len(path):
                    path_indices.append(index)
                    break
                else:
                    index += len(OPS_EX) ** i * (mapping[path[i]] + 1)

        return tuple(path_indices)

    @staticmethod
    def get_paths_201(g: nx.Graph):
        """
        return all paths from input to output
        """
        path_blueprints = [[3], [0, 4], [1, 5], [0, 2, 5]]
        ops = get_op_list(g.name)
        paths = []
        for blueprint in path_blueprints:
            paths.append([ops[node] for node in blueprint])

        return paths

    def get_path_indices_201(self, g: nx.Graph):
        """
        compute the index of each path
        """
        paths = self.get_paths_201(g)
        path_indices = []
        NUM_OPS = len(OPS_201)
        for i, path in enumerate(paths):
            if i == 0:
                index = 0
            elif i in [1, 2]:
                index = NUM_OPS
            else:
                index = NUM_OPS + NUM_OPS**2
            for j, op in enumerate(path):
                index += OPS_201.index(op) * NUM_OPS**j
            path_indices.append(index)

        return tuple(path_indices)

    def encode_paths(self, g: nx.Graph):
        """output one-hot encoding of paths"""
        if "~" in g.name:
            LONGEST_PATH_LENGTH = 3
            num_paths = sum(len(OPS_201) ** i for i in range(1, LONGEST_PATH_LENGTH + 1))
            path_indices = self.get_path_indices_201(g)
        elif "101" in g.name:
            num_paths = sum(len(OPS_EX) ** i for i in range(OP_SPOTS + 1))
            path_indices = self.get_path_indices(g)
        else:
            num_paths = sum(len(self.op_list) ** i for i in range(self.max_size - 1))
            path_indices = self.get_paths(g)
        path_encoding = np.zeros(num_paths)
        for index in path_indices:
            path_encoding[index] = 1
        return path_encoding

    def _compute_dist(self, g1: nx.Graph, g2: nx.Graph):
        encode1 = self.encode_paths(g1)
        encode2 = self.encode_paths(g2)
        return np.sum(np.array(encode1 != np.array(encode2)))
