import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

from .. import utils
from ..surrogate_model import SurrogateModel


class XGBModel(SurrogateModel):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        super(XGBModel, self).__init__(
            data_root, log_dir, seed, model_config, data_config
        )
        self.model = None
        self.model_config["param:objective"] = "reg:squarederror"
        self.model_config["param:eval_metric"] = "rmse"

    def load_results_from_result_paths(self, result_paths):
        """
        Read in the result paths and extract hyperparameters and validation accuracy
        :param result_paths:
        :return:
        """
        # Get the train/test data
        hyps, val_accuracies, test_accuracies = [], [], []

        for result_path in result_paths:
            config_space_instance, val_accuracy, test_accuracy, _ = self.config_loader[
                result_path
            ]
            hyps.append(config_space_instance.get_array())
            val_accuracies.append(val_accuracy)
            test_accuracies.append(test_accuracy)

        X = np.array(hyps)
        y = np.array(val_accuracies)

        # Impute none and nan values
        # Essential to prevent segmentation fault with robo
        idx = np.where(y is None)
        y[idx] = 100

        idx = np.isnan(X)
        X[idx] = -1

        return X, y, test_accuracies

    def parse_param_config(self):
        identifier = "param:"
        param_config = dict()
        for key, val in self.model_config.items():
            if key.startswith(identifier):
                param_config[key.replace(identifier, "")] = val
        return param_config

    def train(self):
        X_train, y_train, _ = self.load_results_from_result_paths(self.train_paths)
        X_val, y_val, _ = self.load_results_from_result_paths(self.val_paths)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        param_config = self.parse_param_config()

        self.model = xgb.train(
            param_config,
            dtrain,
            num_boost_round=self.model_config["param:num_rounds"],
            early_stopping_rounds=self.model_config["early_stopping_rounds"],
            verbose_eval=1,
            evals=[(dval, "val")],
        )

        train_pred, var_train = self.model.predict(dtrain), None
        val_pred, var_val = self.model.predict(dval), None

        # self.save()

        fig_train = utils.scatter_plot(
            np.array(train_pred),
            np.array(y_train),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig_train.savefig(os.path.join(self.log_dir, "pred_vs_true_train.jpg"))
        plt.close()

        fig_val = utils.scatter_plot(
            np.array(val_pred),
            np.array(y_val),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig_val.savefig(os.path.join(self.log_dir, "pred_vs_true_val.jpg"))
        plt.close()

        train_metrics = utils.evaluate_metrics(
            y_train, train_pred, prediction_is_first_arg=False
        )
        valid_metrics = utils.evaluate_metrics(
            y_val, val_pred, prediction_is_first_arg=False
        )

        logging.info("train metrics: %s", train_metrics)
        logging.info("valid metrics: %s", valid_metrics)

        return valid_metrics

    def test(self):
        X_test, y_test, _ = self.load_results_from_result_paths(self.test_paths)
        dtest = xgb.DMatrix(X_test, label=y_test)
        test_pred, var_test = self.model.predict(dtest), None

        fig = utils.scatter_plot(
            np.array(test_pred),
            np.array(y_test),
            xlabel="Predicted",
            ylabel="True",
            title="",
        )
        fig.savefig(os.path.join(self.log_dir, "pred_vs_true_test.jpg"))
        plt.close()

        test_metrics = utils.evaluate_metrics(
            y_test, test_pred, prediction_is_first_arg=False
        )

        logging.info("test metrics %s", test_metrics)

        return test_metrics

    def validate(self):
        X_val, y_val, _ = self.load_results_from_result_paths(self.val_paths)
        dval = xgb.DMatrix(X_val, label=y_val)
        val_pred, var_val = self.model.predict(dval), None

        valid_metrics = utils.evaluate_metrics(
            y_val, val_pred, prediction_is_first_arg=False
        )

        logging.info("validation metrics %s", valid_metrics)

        return valid_metrics

    def save(self):
        pickle.dump(
            self.model, open(os.path.join(self.log_dir, "surrogate_model.model"), "wb")
        )

    def load(self, model_path):
        self.model = pickle.load(open(model_path, "rb"))

    def evaluate(self, result_paths):
        X_test, y_test, _ = self.load_results_from_result_paths(result_paths)
        dtest = xgb.DMatrix(X_test, label=y_test)
        test_pred, var_test = self.model.predict(dtest), None

        test_metrics = utils.evaluate_metrics(
            y_test, test_pred, prediction_is_first_arg=False
        )
        return test_metrics, test_pred, y_test

    def query(self, config_dict):
        config_space_instance = self.config_loader.query_config_dict(config_dict)
        X = config_space_instance.get_array().reshape(1, -1)
        idx = np.isnan(X)
        X[idx] = -1
        dtest = xgb.DMatrix(X)
        pred = self.model.predict(dtest)
        return pred


class XGBModelTime(XGBModel):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        super(XGBModelTime, self).__init__(
            data_root, log_dir, seed, model_config, data_config
        )

    # OVERRIDE
    def load_results_from_result_paths(self, result_paths):
        """
        Read in the result paths and extract hyperparameters and runtime
        :param result_paths:
        :return:
        """
        # Get the train/test data
        hyps, runtimes = [], []

        for result_path in result_paths:
            config_space_instance, runtime = self.config_loader.get_runtime(result_path)
            hyps.append(config_space_instance.get_array())
            runtimes.append(runtime)

        X = np.array(hyps)
        y = np.array(runtimes)

        # Impute none and nan values
        # Essential to prevent segmentation fault with robo
        idx = np.where(y is None)
        y[idx] = 100

        idx = np.isnan(X)
        X[idx] = -1

        # return none to mimic return value of parent class
        return X, y, None
