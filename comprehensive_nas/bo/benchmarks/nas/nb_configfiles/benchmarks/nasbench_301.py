import copy
import json
import os

import numpy as np

from ConfigSpace.read_and_write import json as config_space_json_r_w
from surrogate_models import utils


class NASBench301(object):
    def __init__(self, surrogate_model_dir, runtime_model_dir):
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

        self.X = []
        self.y_valid = []
        self.costs = []

    def reset_tracker(self):
        # __init__() sans the data loading for multiple runs
        self.X = []
        self.y_valid = []
        self.costs = []

    def objective_function(self, config):
        """
        Call to the objective function, a query to the nasbench-301 surrogate function.
        :param config: Config dictionary from config_instance.get_dictionary()
        :return:
        """
        pass

    def record_invalid(self, config, validation_accuracy, cost):
        self.X.append(config)
        # compute validation error for the chosen budget
        valid_error = 1 - validation_accuracy / 100
        self.y_valid.append(valid_error)
        self.costs.append(cost)

    def record_valid(self, config, validation_accuracy, cost):
        self.X.append(config)

        # compute validation error for the chosen budget
        valid_error = 1 - validation_accuracy / 100
        self.y_valid.append(valid_error)
        self.costs.append(cost)

    @staticmethod
    def get_configuration_space(path):
        with open(os.path.join(path), "r") as fh:
            json_string = fh.read()
            config_space = config_space_json_r_w.read(json_string)
        return config_space

    def sample_configuration(self):
        pass

    def get_results(self, ignore_invalid_configs=False):
        inc_configs = []
        validation_losses = []
        runtimes = []

        rt = 0
        inc_valid = np.inf

        for i in range(len(self.X)):
            rt += self.costs[i]
            if inc_valid > self.y_valid[i]:
                inc_valid = self.y_valid[i]
                inc_config = self.X[i]
                runtimes.append(float(rt))
                inc_configs.append(inc_config)
                validation_losses.append(float(inc_valid))

        res = dict()
        res["inc_config"] = inc_configs
        res["validation_error"] = validation_losses
        res["runtime"] = runtimes

        return res


class NASBench301_arch_only(NASBench301):
    def __init__(self, surrogate_model_dir, runtime_model_dir, fidelity=6):
        super(NASBench301_arch_only, self).__init__(
            surrogate_model_dir, runtime_model_dir
        )
        # Get default config with default hyperparameters
        self.default_config = (
            self.surrogate_model.config_loader.config_space.get_default_configuration().get_dictionary()
        )
        self.fidelity = fidelity
        self.config_space = self.get_configuration_space(
            "nas_benchmark/benchmarks/config_spaces/configspace_arch_only.json"
        )

    def objective_function(self, config, eval_fidelity=None):
        if eval_fidelity is None:
            eval_fidelity = self.fidelity
        query_config_dict = copy.deepcopy(config)

        # Override the hyperparameters
        for param, value in self.default_config.items():
            # Don't override the architectural parameters from the normal and reduction cell
            if not ("normal" in param or "reduce" in param):
                query_config_dict[param] = value

        # Set requested fidelity
        fidelity_multipliers = self.surrogate_model.config_loader.fidelity_multiplier
        fidelity_starts = self.surrogate_model.config_loader.fidelity_starts
        for fidelity, fidelity_start in fidelity_starts.items():
            query_config_dict[fidelity] = int(
                np.round(
                    (fidelity_start * fidelity_multipliers[fidelity] ** eval_fidelity)
                )
            )
        try:
            pred_validation_accuracy = self.surrogate_model.query(
                config_dict=query_config_dict
            )
            pred_runtime = self.runtime_estimator.query(config_dict=query_config_dict)
            self.record_valid(
                query_config_dict, pred_validation_accuracy, cost=pred_runtime
            )
        except ValueError as e:
            # If a non legitimate change has been made to the architecture,
            # meaning that the resulting architecture is no longer in the search space.
            print(e)
            pred_validation_accuracy = 0
            self.record_invalid(query_config_dict, pred_validation_accuracy, cost=0)
        return pred_validation_accuracy


class NASBench301_arch_hyp(NASBench301):
    def __init__(self, surrogate_model_dir, runtime_model_dir, fidelity=6):
        super(NASBench301_arch_hyp, self).__init__(surrogate_model_dir, runtime_model_dir)
        # Get default config with default hyperparameters
        self.default_config = (
            self.surrogate_model.config_loader.config_space.get_default_configuration().get_dictionary()
        )
        self.fidelity = fidelity

        self.config_space = self.get_configuration_space(
            "nas_benchmark/benchmarks/config_spaces/configspace_arch_plus_hyps.json"
        )

    def objective_function(self, config, eval_fidelity=None):
        if eval_fidelity is None:
            eval_fidelity = self.fidelity
        query_config_dict = copy.deepcopy(config)

        # Override the hyperparameters
        for param, value in self.default_config.items():
            # Don't override the architectural parameters from the normal and reduction cell or the hyperparameters
            if not (("normal" in param) or ("reduce" in param)):
                if (
                    param != "OptimizerSelector:sgd:learning_rate"
                    and param != "OptimizerSelector:sgd:weight_decay"
                ):
                    query_config_dict[param] = value

        # Set requested fidelity
        fidelity_multipliers = self.surrogate_model.config_loader.fidelity_multiplier
        fidelity_starts = self.surrogate_model.config_loader.fidelity_starts
        for fidelity, fidelity_start in fidelity_starts.items():
            query_config_dict[fidelity] = int(
                np.round(
                    (fidelity_start * fidelity_multipliers[fidelity] ** eval_fidelity)
                )
            )
        try:
            pred_validation_accuracy = self.surrogate_model.query(
                config_dict=query_config_dict
            )
            pred_runtime = self.runtime_estimator.query(config_dict=query_config_dict)
            self.record_valid(
                query_config_dict, pred_validation_accuracy, cost=pred_runtime
            )
        except ValueError as e:
            # If a non legitimate change has been made to the architecture,
            # meaning that the resulting architecture is no longer in the search space.
            print(e)
            pred_validation_accuracy = 0
            self.record_invalid(query_config_dict, pred_validation_accuracy, cost=0)
        return pred_validation_accuracy


if __name__ == "__main__":
    nasbenchmark = NASBench301_arch_hyp(
        surrogate_model_dir="/home/anonymous/projects/nasbench_301_2/experiments/hpo_2/gnn_diff_pool_sigmoid/0_0_0/gnn_diff_pool/20200129-213104-1751",
        runtime_model_dir="/home/anonymous/projects/nasbench_301_2/experiments/surrogate_models/lgb_time/20200130-104406-6",
        fidelity=5,
    )
    # Get the desired configspace object
    config_space = nasbenchmark.config_space
    for i in range(10):
        # Sample a configuration
        config = config_space.sample_configuration()
        # Evaluate the objective function with the config
        print(nasbenchmark.objective_function(config.get_dictionary()))
