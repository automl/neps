import json
import math
import os
import sys

import ConfigSpace
import networkx as nx
import numpy as np

from ConfigSpace.read_and_write import json as config_space_json_r_w

from ..abstract_benchmark import AbstractBenchmark


# from nasbowl.benchmarks.nas.surrogate_models import utils #TODO check what is imported here later

MAX_EDGES_301 = 13
VERTICES_301 = 6
HPS_301 = 2

edge_to_coord_mapping = {
    0: (0, 2),
    1: (1, 2),
    2: (0, 3),
    3: (1, 3),
    4: (2, 3),
    5: (0, 4),
    6: (1, 4),
    7: (2, 4),
    8: (3, 4),
    9: (0, 5),
    10: (1, 5),
    11: (2, 5),
    12: (3, 5),
    13: (4, 5),
}
coord_to_edge_mapping = {
    (0, 2): 0,
    (1, 2): 1,
    (0, 3): 2,
    (1, 3): 3,
    (2, 3): 4,
    (0, 4): 5,
    (1, 4): 6,
    (2, 4): 7,
    (3, 4): 8,
    (0, 5): 9,
    (1, 5): 10,
    (2, 5): 11,
    (3, 5): 12,
    (4, 5): 13,
}
OPS_301 = [
    "max_pool_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
]


class NASBench301(AbstractBenchmark):
    def __init__(
        self,
        multi_fidelity=False,
        log_scale=False,
        negative=False,
        seed=None,
        surrogate_model_dir=os.path.join(
            os.path.dirname(__file__), "nb_configfiles/gnn_gin"
        ),
        runtime_model_dir=os.path.join(
            os.path.dirname(__file__), "nb_configfiles/xgb_time/"
        ),
    ):
        optim = 0.04944576819737756
        if log_scale:
            optim = np.log(optim)
        if negative:
            optim = -optim
        super().__init__(dim=None, optimum_location=None, optimal_val=optim, bounds=None)

        self.seed = seed
        self.multi_fidelity = multi_fidelity
        self.fidelity_keys = [
            "epochs",
            "NetworkSelectorDatasetInfo:darts:init_channels",
            "NetworkSelectorDatasetInfo:darts:layers",
        ]
        self.fidelity_range_bounds = [1500, 50, 20]  # min values from diag.: 50, 8, 5
        self.log_scale = log_scale

        self.X = []
        self.y_valid = []
        self.costs = []
        self.negative = negative
        self.y_star_valid = 0.04944576819737756

        def _load_model_from_dir(model_dir):
            # Load config
            data_config = json.load(
                open(os.path.join(model_dir, "data_config.json"), "r")
            )
            model_config = json.load(
                open(os.path.join(model_dir, "model_config.json"), "r")
            )

            # Instantiate model
            model = utils.model_dict[model_config["model"]](
                data_root="None",
                log_dir=None,
                seed=data_config["seed"],
                data_config=data_config,
                model_config=model_config,
            )
            # Load the model from checkpoint
            model.load(os.path.join(model_dir, "surrogate_model.model"))
            return model

        self.surrogate_model_dir = surrogate_model_dir
        self.runtime_model_dir = runtime_model_dir

        # Instantiate surrogate model
        self.surrogate_model = _load_model_from_dir(surrogate_model_dir)
        self.runtime_estimator = _load_model_from_dir(runtime_model_dir)
        self.default_config = (
            self.surrogate_model.config_loader.config_space.get_default_configuration().get_dictionary()
        )
        self.default_config[
            "OptimizerSelector:sgd:hyperparam:learning_rate"
        ] = self.default_config.pop("OptimizerSelector:sgd:learning_rate")
        self.default_config[
            "OptimizerSelector:sgd:hyperparam:weight_decay"
        ] = self.default_config.pop("OptimizerSelector:sgd:weight_decay")

    def reinitialize(self, negative=False, seed=None):
        self.negative = negative
        self.seed = seed

    def eval(self, Gs, hps):
        # input is a list of two graphs [G1,G2] representing normal and reduction cell
        return self._retrieve(Gs, hps)

    # Budget should be an array of three fidelities in range (0., 1.] for multi-multi
    def _retrieve(self, Gs, hps, budget=np.array([1.0, 1.0, 1.0])):

        assert budget.all() == 1.0, "Do not support multi-multi fidelity"

        # Map two graphs and hps into a query configuration for NAS301 surrogate
        config_dict = self.tuple_to_config_dict(Gs, hps)

        # Override the hyperparameters
        for param, value in self.default_config.items():

            if "hyperparam" in param:
                if hps is None:
                    config_dict[param] = value
                config_dict[param.replace("hyperparam:", "")] = config_dict.pop(param)
            elif "normal" in param or "reduce" in param:
                if Gs is None:
                    config_dict[param] = value
            else:
                config_dict[param] = value

        # Set requested fidelity
        for i, key in enumerate(self.fidelity_keys):
            config_dict[key] = math.ceil(budget[i] * self.fidelity_range_bounds[i])

        try:
            # Just remove additional ones so the query works
            data = {
                "validation_accuracy": self.surrogate_model.query(
                    config_dict=config_dict
                ),
                "training_time": self.runtime_estimator.query(config_dict=config_dict),
            }

        except ValueError:
            self.record_invalid(config=config_dict)

            if self.log_scale:
                y_invalid = np.log(1)
            else:
                y_invalid = 1
            y_invalid = -y_invalid if self.negative else y_invalid
            return y_invalid, {"train_time": 0}

        self.record_valid(
            config=config_dict,
            validation_accuracy=data["validation_accuracy"],
            cost=data["training_time"],
        )
        err = 100 - data["validation_accuracy"][0]

        if self.log_scale:
            y = np.log(err)
        else:
            y = err
        if self.negative:
            y = -y

        cost = {"train_time": data["training_time"]}
        return y, cost

    # @staticmethod
    # def eval_cost(budget):
    #     return 0.05 + (1 - 0.05) * budget[0] ** 1.76 * budget[1] ** 1.36 * budget[2] ** 1.26

    def record_valid(self, config, validation_accuracy, cost):
        self.X.append(config)
        # compute validation error for the chosen budget
        valid_error = 1 - validation_accuracy / 100
        self.y_valid.append(valid_error)
        self.costs.append(cost)

    def record_invalid(self, config):
        self.record_valid(config, validation_accuracy=1, cost=sys.maxsize)

    def get_results(self, ignore_invalid_configs=False):

        regret_validation = []
        runtime = []
        rt = 0

        inc_valid = np.inf

        for i in range(len(self.X)):

            if ignore_invalid_configs and self.costs[i] == 0:
                continue

            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]

            regret_validation.append(float(inc_valid - self.y_star_valid))
            rt += self.costs[i]
            runtime.append(float(rt))

        res = dict()
        res["regret_validation"] = regret_validation
        res["runtime"] = runtime

        return res

    @staticmethod
    def get_config_space(
        path=os.path.join(
            os.path.dirname(__file__),
            "nb_configfiles/benchmarks/" "config_spaces/configspace.json",
        )
    ):
        with open(os.path.join(path), "r") as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)
        return config_space

    def tuple_to_config_dict(self, Gs, hps):

        config = dict()

        if Gs is not None:
            prefix = "NetworkSelectorDatasetInfo:darts:"
            for g, name in zip(Gs, ["normal", "reduce"]):
                adjacency_matrix = nx.adjacency_matrix(g).todense()
                indices = np.nonzero(adjacency_matrix)
                inputs = dict()
                inputs[3] = list()
                inputs[4] = list()
                inputs[5] = list()
                edges = list()
                ops = list()
                for index in zip(indices[0], indices[1]):
                    edges.append(coord_to_edge_mapping[index])
                    ops.append(adjacency_matrix[index[0], index[1]])
                    if index[1] != 2:
                        inputs[index[1]].append(index[0])

                for key, items in inputs.items():
                    config[
                        prefix + "inputs_node_{}_{}".format(name, key)
                    ] = "{}_{}".format(items[0], items[1])

                for e, o in zip(edges, ops):
                    config[prefix + "edge_{}_{}".format(name, e)] = OPS_301[o - 1]

        if hps is not None:
            i = 0
            for hp in self.get_config_space().get_hyperparameters():
                if "hyperparam" in hp.name:
                    if isinstance(hp, ConfigSpace.OrdinalHyperparameter):
                        ranges = np.arange(start=0, stop=1, step=1 / len(hp.sequence))
                        val = hp.sequence[np.where(not hps[i] < ranges)[0][-1]]
                    elif isinstance(hp, ConfigSpace.CategoricalHyperparameter):
                        ranges = np.arange(start=0, stop=1, step=1 / len(hp.choices))
                        val = hp.choices[np.where(not float(hps[i]) < ranges)[0][-1]]
                    else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                        # rescaling continuous values
                        if hp.log:
                            log_range = np.log(hp.upper) - np.log(hp.lower)
                            val = np.exp(np.log(hp.lower) + hps[i] * log_range)
                        else:
                            val = hp.lower + (hp.upper - hp.lower) * hps[i]
                        if isinstance(hp, ConfigSpace.UniformIntegerHyperparameter):
                            val = int(np.round(val))  # converting to discrete (int)
                        else:
                            val = float(val)
                    config[hp.name] = val
                    i += 1

        return config

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.costs = []


if __name__ == "__main__":
    config = NASBench301.get_config_space().sample_configuration().get_dictionary()

    normal_A = np.array(
        [
            [0, 0, 1, 6, 6, 0],
            [0, 0, 1, 3, 0, 7],
            [0, 0, 0, 0, 6, 0],
            [0, 0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    reduce_A = np.array(
        [
            [0, 0, 3, 4, 5, 5],
            [0, 0, 4, 0, 0, 0],
            [0, 0, 0, 7, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    Gs = list()
    Gs.append(nx.from_numpy_array(normal_A, create_using=nx.DiGraph))
    Gs.append(nx.from_numpy_array(reduce_A, create_using=nx.DiGraph))
    nascifar10 = NASBench301(log_scale=False)
    for i in range(5):
        budget = np.random.rand(3)
        print(budget)
        result = nascifar10.eval(Gs)

    print(result)
    print(nascifar10.get_results())
