def extract_paths_for_budget(paths, fidelity_num):
    """
    Find the paths which are matching the budget
    :param paths:
    :param fidelity_num:
    :return:
    """
    fidelity_paths = []
    for path in paths:
        if "results_fidelity_{}".format(fidelity_num) in path:
            fidelity_paths.append(path)
    return fidelity_paths


class ConfigDict:
    """Creates a dict with configs (without fidelity parameters) as keys and a dict of accuracies for different fidelities as values. Would be cleaner using UserDict or whatever the abc for that is..."""

    def __init__(
        self,
        configs,
        accuracies,
        paths=None,
        include_off_diagonal=False,
        allow_repeats=False,
    ):
        self.fidelity_keys = [
            "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max",
            "NetworkSelectorDatasetInfo:darts:layers",
            "NetworkSelectorDatasetInfo:darts:init_channels",
            "epochs",
        ]
        self.epoch_fidelity_dict = {50: 0, 88: 1, 155: 2, 274: 3, 483: 4, 851: 5, 1500: 6}
        self.channel_fidelity_dict = {8: 0, 11: 1, 15: 2, 20: 3, 27: 4, 37: 5, 50: 6}
        self.cells_fidelity_dict = {5: 0, 6: 1, 8: 2, 10: 3, 13: 4, 16: 5, 20: 6}
        self.include_off_diagonal = include_off_diagonal
        self.allow_repeats = allow_repeats
        if paths is not None:
            self.data = self._create_data_dict_with_paths(configs, accuracies, paths)
        else:
            self.data = self._create_data_dict(configs, accuracies)

    def __getattr__(self, config_hash):
        return self.data[config_hash]

    def _create_data_dict(self, configs, accuracies):
        data = dict()
        for config, accuracy in zip(configs, accuracies):
            if not self.include_off_diagonal and not self._assert_diagonal(config):
                continue
            config_hash, fidelity_hash = self._get_config_and_fidelity_hash(config)
            if not config_hash in data.keys():
                data[config_hash] = dict()
            if self.allow_repeats:
                if not fidelity_hash in data[config_hash].keys():
                    data[config_hash][fidelity_hash] = []
                data[config_hash][fidelity_hash].append((config, accuracy))
            else:
                data[config_hash][fidelity_hash] = (config, accuracy)
        return data

    def _create_data_dict_with_paths(self, configs, accuracies, paths):
        data = dict()
        for config, accuracy, path in zip(configs, accuracies, paths):
            if not self.include_off_diagonal and not self._assert_diagonal(config):
                continue
            config_hash, fidelity_hash = self._get_config_and_fidelity_hash(config)
            if not config_hash in data.keys():
                data[config_hash] = dict()
            if self.allow_repeats:
                if not fidelity_hash in data[config_hash].keys():
                    data[config_hash][fidelity_hash] = []
                data[config_hash][fidelity_hash].append((config, accuracy, path))
            else:
                data[config_hash][fidelity_hash] = (config, accuracy, path)
        return data

    def _assert_diagonal(self, config):
        channel_fid = self.channel_fidelity_dict[
            config["NetworkSelectorDatasetInfo:darts:init_channels"]
        ]
        cell_fid = self.cells_fidelity_dict[
            config["NetworkSelectorDatasetInfo:darts:layers"]
        ]
        epoch_fid = self.epoch_fidelity_dict[
            config["SimpleLearningrateSchedulerSelector:cosine_annealing:T_max"]
        ]
        return channel_fid == cell_fid == epoch_fid

    def _get_config_and_fidelity_hash(self, config):
        config_without_fidelities, fidelity_pars = self._separate_config_and_fidelities(
            config
        )
        return self._hash_dict_repr(config_without_fidelities), self._get_fidelity_number(
            fidelity_pars
        )

    def _separate_config_and_fidelities(self, config):
        config_without_fidelities = {
            k: v for k, v in config.items() if k not in self.fidelity_keys
        }
        fidelity_pars = {k: config[k] for k in self.fidelity_keys}
        return config_without_fidelities, fidelity_pars

    def _hash_dict_repr(self, indict):
        return hash(indict.__repr__())

    def _get_fidelity_number(self, config):
        return self.channel_fidelity_dict[
            config["NetworkSelectorDatasetInfo:darts:init_channels"]
        ]
