import copy
import os
import random

import ConfigSpace
import networkx as nx
import numpy as np
from nasbench import api
from nasbench.lib import graph_util

from ..abstract_benchmark import AbstractBenchmark

MAX_EDGES = 9
VERTICES = 7


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


class ModelSpec_Modified(object):
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format="channels_last"):
        """Initialize the module spec.

        Args:
          matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
          ops: V-length list of labels for the base ops used. The first and last
            elements are ignored because they are the input and output vertices
            which have no operations. The elements are retained to keep consistent
            indexing.
          data_format: channels_last or channels_first.

        Raises:
          ValueError: invalid matrix or ops
        """
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("matrix must be square")
        if shape[0] != len(ops):
            raise ValueError("length of ops must match matrix dimensions")
        if not is_upper_triangular(matrix):
            raise ValueError("matrix must be upper triangular")

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self.data_format = data_format

    def hash_spec(self, canonical_ops):
        """Computes the isomorphism-invariant graph hash of this spec.

        Args:
          canonical_ops: list of operations in the canonical ordering which they
            were assigned (i.e. the order provided in the config['available_ops']).

        Returns:
          MD5 hash of this spec which can be used to query the dataset.
        """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
        return graph_util.hash_module(self.matrix, labeling)


class NASBench101(AbstractBenchmark):
    def __init__(
        self, data_dir, multi_fidelity=False, log_scale=True, negative=True, seed=None
    ):
        optim = 0.04944576819737756
        if log_scale:
            optim = np.log(optim)
        if negative:
            optim = -optim
        super().__init__(dim=None, optimum_location=None, optimal_val=optim, bounds=None)

        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.log_scale = log_scale
        if self.multi_fidelity:
            self.dataset = api.NASBench(
                os.path.join(data_dir, "nasbench_full.tfrecord"), seed=0
            )
        else:
            self.dataset = api.NASBench(
                os.path.join(data_dir, "nasbench_only108.tfrecord")
            )
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.model_spec_list = []
        self.negative = negative
        # self.optimal_val = 0.04944576819737756  # lowest mean validation error
        # self.y_star_test = 0.056824247042338016  # lowest mean test error

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative
        self.seed = seed

    def _retrieve(self, G, budget, which="eval"):

        #  set random seed for evaluation
        if which == "eval":
            seed_list = [0, 1, 2]
            if self.seed is None:
                seed = random.choice(seed_list)
            elif self.seed >= 3:
                seed = self.seed
            else:
                seed = seed_list[self.seed]
        else:
            # For testing, there should be no stochasticity
            seed = 3

        # input is a list of graphs [G1,G2, ....]
        if self.multi_fidelity is False:
            assert budget == 108
        # get node label and adjacency matrix
        node_labeling = list(nx.get_node_attributes(G, "op_name").values())
        adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())

        model_spec = ModelSpec_Modified(adjacency_matrix, node_labeling)
        try:
            # data = self.dataset.query(model_spec, epochs=budget)
            fixed_stat, computed_stat = self.dataset.get_metrics_from_spec(model_spec)
            data = {}
            data["module_adjacency"] = fixed_stat["module_adjacency"]
            data["module_operations"] = fixed_stat["module_operations"]
            data["trainable_parameters"] = fixed_stat["trainable_parameters"]
            if seed is not None and seed >= 3:
                compute_data_all = computed_stat[budget]
                data["validation_accuracy"] = np.mean(
                    [cd["final_validation_accuracy"] for cd in compute_data_all]
                )
                data["test_accuracy"] = np.mean(
                    [cd["final_test_accuracy"] for cd in compute_data_all]
                )
                data["training_time"] = np.mean(
                    [cd["final_training_time"] for cd in compute_data_all]
                )
            else:
                compute_data = computed_stat[budget][seed]
                data["validation_accuracy"] = compute_data["final_validation_accuracy"]
                data["test_accuracy"] = compute_data["final_test_accuracy"]
                data["training_time"] = compute_data["final_training_time"]

        except api.OutOfDomainError:
            self.record_invalid(1, 1, 0)

            if self.log_scale:
                y_invalid = np.log(1)
            else:
                y_invalid = 1
            return y_invalid

        self.record_valid(data, model_spec)
        if which == "eval":
            err = 1 - data["validation_accuracy"]
        elif which == "test":
            err = 1 - data["test_accuracy"]
        else:
            raise ValueError("Unknown query parameter: which = " + str(which))
        if self.log_scale:
            y = np.log(err)
        else:
            y = err
        if self.negative:
            y = -y

        cost = {"train_time": data["training_time"]}
        return y, cost

    def eval(self, G, budget=108, n_repeat=1, use_banana=False):
        """"""
        # input is a list of graphs [G1,G2, ....]
        if use_banana:
            return self.banana_retrieve(G, "eval")
        if n_repeat == 1:
            return self._retrieve(G, budget, "eval")
        return (
            np.mean(
                np.array([self._retrieve(G, budget, "eval")[0] for _ in range(n_repeat)])
            ),
            [self._retrieve(G, budget, "eval")[0] for _ in range(n_repeat)],
        )

    def test(self, G, budget=108, n_repeat=1, use_banana=False):
        if use_banana:
            return self.banana_retrieve(G, "test")
        return np.mean(
            np.array([self._retrieve(G, budget, "test")[0] for _ in range(n_repeat)])
        )

    def banana_retrieve(self, G, which="eval"):
        patience = 50
        accs = []
        node_labeling = list(nx.get_node_attributes(G, "op_name").values())
        adjacency_matrix = np.array(nx.adjacency_matrix(G).todense())

        model_spec = api.ModelSpec(adjacency_matrix, node_labeling)
        while len(accs) < 3 and patience > 0:
            patience -= 1
            if which == "eval":
                acc = self.dataset.query(model_spec)["validation_accuracy"]
            else:
                acc = self.dataset.query(model_spec)["test_accuracy"]
            if acc not in accs:
                accs.append(acc)
        err = round((1 - np.mean(accs)), 4)

        if self.log_scale:
            err = np.log(err)
        if self.negative:
            err = -err
        return err

    def record_invalid(self, valid, test, costs):

        self.y_valid.append(valid)
        self.y_test.append(test)
        self.costs.append(costs)
        self.model_spec_list.append({"adjacency": None, "node_labels": None})

    def record_valid(self, data, model_spec):

        # record valid adjacency matrix and node labels
        self.model_spec_list.append(
            {
                "adjacency": model_spec.original_matrix,
                "node_labels": model_spec.original_ops,
            }
        )

        # compute mean test error for the final budget
        _, metrics = self.dataset.get_metrics_from_spec(model_spec)
        mean_test_error = 1 - np.mean(
            [metrics[108][i]["final_test_accuracy"] for i in range(3)]
        )
        self.y_test.append(mean_test_error)

        # compute validation error for the chosen budget
        valid_error = 1 - data["validation_accuracy"]
        self.y_valid.append(valid_error)

        runtime = data["training_time"]
        self.costs.append(runtime)

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.y_test = []
        self.costs = []
        self.model_spec_list = []

    def get_results(self, ignore_invalid_configs=False):

        regret_validation = []
        regret_test = []
        runtime = []
        model_graph_specs = []
        rt = 0

        inc_valid = np.inf
        inc_test = np.inf

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]
                inc_test = self.y_test[i]

            regret_validation.append(float(inc_valid - self.y_star_valid))
            regret_test.append(float(inc_test - self.y_star_test))
            model_graph_specs.append(self.model_spec_list[i])
            rt += self.costs[i]
            runtime.append(float(rt))

        res = dict()
        res["regret_validation"] = regret_validation
        res["regret_test"] = regret_test
        res["runtime"] = runtime
        res["model_graph_specs"] = model_graph_specs

        return res

    @staticmethod
    def get_configuration_space():
        # for unpruned graph
        cs = ConfigSpace.ConfigurationSpace()

        ops_choices = ["conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("op_node_0", ops_choices)
        )
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("op_node_1", ops_choices)
        )
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("op_node_2", ops_choices)
        )
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("op_node_3", ops_choices)
        )
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter("op_node_4", ops_choices)
        )
        for i in range(VERTICES * (VERTICES - 1) // 2):
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter("edge_%d" % i, [0, 1])
            )
        return cs


if __name__ == "__main__":
    output_path = "../data/"
    # with open(os.path.join(output_path, 'valid_arch_samples_pruned'), 'rb') as outfile:
    #     res = pickle.load(outfile)
    #
    # idx = 1
    # A = res['model_graph_specs'][idx]['adjacency']
    # nl = res['model_graph_specs'][idx]['node_labels']
    A = np.array(
        [
            [0, 1, 1, 0, 0, 1, 1],  # input layer
            [0, 0, 0, 0, 0, 1, 0],  # 1x1 conv
            [0, 0, 0, 1, 0, 0, 0],  # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],  # 3x3 max-pool
            [0, 0, 0, 0, 0, 1, 0],  # 3x3 conv
            [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )  # output layer
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    nl = [
        "input",
        "conv1x1-bn-relu",
        "conv3x3-bn-relu",
        "maxpool3x3",
        "conv3x3-bn-relu",
        "conv3x3-bn-relu",
        "output",
    ]
    for i, n in enumerate(nl):
        G.node[i]["op_name"] = n
    nascifar10 = NASBench101(data_dir=output_path, seed=4)
    f = nascifar10.eval
    result = f(G)
