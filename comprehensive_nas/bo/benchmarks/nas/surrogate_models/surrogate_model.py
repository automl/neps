import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pathvalidate
import torch
import torch.backends.cudnn as cudnn

from . import utils


class SurrogateModel(ABC):
    def __init__(self, data_root, log_dir, seed, model_config, data_config):
        self.data_root = data_root
        self.log_dir = log_dir
        self.model_config = model_config
        self.data_config = data_config
        self.seed = seed

        # Seeding
        np.random.seed(seed)
        cudnn.benchmark = True
        torch.manual_seed(seed)
        cudnn.enabled = True
        torch.cuda.manual_seed(seed)

        # Create config loader
        self.config_loader = utils.ConfigLoader(
            os.path.join(os.path.dirname(__file__), "configspace.json")
        )

        # Load the data
        if log_dir is not None:
            # Add logger
            log_format = "%(asctime)s %(message)s"
            logging.basicConfig(
                stream=sys.stdout,
                level=logging.INFO,
                format=log_format,
                datefmt="%m/%d %I:%M:%S %p",
            )
            fh = logging.FileHandler(os.path.join(log_dir, "log.txt"))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)

            # Dump the config of the run to log_dir
            self.data_config["seed"] = seed

            logging.info("MODEL CONFIG: {}".format(model_config))
            logging.info("DATA CONFIG: {}".format(data_config))
            self._load_data()
            logging.info(
                "DATA: No. train data {}, No. val data {}, No. test data {}".format(
                    len(self.train_paths), len(self.val_paths), len(self.test_paths)
                )
            )
            with open(os.path.join(log_dir, "model_config.json"), "w") as fp:
                json.dump(model_config, fp)

            with open(os.path.join(log_dir, "data_config.json"), "w") as fp:
                json.dump(data_config, fp)

    def _load_data(self):
        # Get the result train/val/test split
        train_paths = []
        val_paths = []
        test_paths = []
        for key, data_config in self.data_config.items():
            if type(data_config) == dict:
                result_loader = utils.ResultLoader(
                    self.data_root,
                    filepath_regex=data_config["filepath_regex"],
                    train_val_test_split=data_config,
                    seed=self.seed,
                )
                train_val_test_split = result_loader.return_train_val_test()
                # Save the paths
                for paths, filename in zip(
                    train_val_test_split, ["train_paths", "val_paths", "test_paths"]
                ):
                    file_path = os.path.join(
                        self.log_dir,
                        pathvalidate.sanitize_filename(
                            "{}_{}.json".format(key, filename)
                        ),
                    )
                    json.dump(paths, open(file_path, "w"))

                train_paths.extend(train_val_test_split[0])
                val_paths.extend(train_val_test_split[1])
                test_paths.extend(train_val_test_split[2])

        # Add extra paths to test
        # Increased ratio of skip-connections.
        matching_files = lambda dir: [
            str(path) for path in Path(os.path.join(self.data_root, dir)).rglob("*.json")
        ]
        test_paths.extend(matching_files("groundtruths/low_parameter/"))

        # Extreme hyperparameter settings
        # Learning rate
        test_paths.extend(matching_files("groundtruths/hyperparameters/learning_rate/"))
        test_paths.extend(matching_files("groundtruths/hyperparameters/weight_decay/"))

        # Load the blacklist to filter out those elements
        if self.model_config["model"].endswith("_time"):
            blacklist = json.load(
                open("surrogate_models/configs/data_configs/blacklist_runtimes.json")
            )
        else:
            blacklist = json.load(
                open("surrogate_models/configs/data_configs/blacklist.json")
            )
        filter_out_black_list = lambda paths: list(
            filter(lambda path: path not in blacklist, paths)
        )
        train_paths, val_paths, test_paths = map(
            filter_out_black_list, [train_paths, val_paths, test_paths]
        )

        # Shuffle the total file paths again
        np.random.shuffle(train_paths), np.random.shuffle(val_paths), np.random.shuffle(
            test_paths
        )

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def validate(self):
        raise NotImplementedError()

    @abstractmethod
    def test(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self):
        raise NotImplementedError()

    @abstractmethod
    def load(self, model_path):
        raise NotImplementedError()

    @abstractmethod
    def query(self, config_dict):
        raise NotImplementedError()
