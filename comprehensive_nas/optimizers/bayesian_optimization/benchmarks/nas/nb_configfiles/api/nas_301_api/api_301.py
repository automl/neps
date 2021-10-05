import json
import os

from comprehensive_nas.optimizers.bayesian_optimization.benchmarks.nas.surrogate_models.utils import model_dict


class NASBench301API:
    def __init__(self):
        surrogate_model_dir = "comprehensive_nas/bo/benchmarks/nas/nb_configfiles/gnn_gin"
        runtime_model_dir = "comprehensive_nas/bo/benchmarks/nas/nb_configfiles/xgb_time/"

        def _load_model_from_dir(model_dir):
            # Load config
            data_config = json.load(
                open(os.path.join(model_dir, "data_config.json"), "r")
            )
            model_config = json.load(
                open(os.path.join(model_dir, "model_config.json"), "r")
            )

            # Instantiate model
            model = model_dict[model_config["model"]](
                data_root="None",
                log_dir=None,
                seed=data_config["seed"],
                data_config=data_config,
                model_config=model_config,
            )
            # Load the model from checkpoint
            model.load(os.path.join(model_dir, "surrogate_model.model"))
            return model

        self.surrogate_model = _load_model_from_dir(surrogate_model_dir)
        self.runtime_estimator = _load_model_from_dir(runtime_model_dir)

    def get_default_config(self):
        default_config = (
            self.surrogate_model.config_loader.config_space.get_default_configuration().get_dictionary()
        )
        default_config[
            "OptimizerSelector:sgd:hyperparam:learning_rate"
        ] = default_config.pop("OptimizerSelector:sgd:learning_rate")
        default_config[
            "OptimizerSelector:sgd:hyperparam:weight_decay"
        ] = default_config.pop("OptimizerSelector:sgd:weight_decay")

        return default_config

    def query(self, config_dict):
        return {
            "validation_accuracy": self.surrogate_model.query(config_dict=config_dict),
            "training_time": self.runtime_estimator.query(config_dict=config_dict),
        }
