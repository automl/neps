import json
import os
import random
import time

import numpy as np
import tabulate
import torch
from networkx.readwrite import json_graph


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class StatisticsTracker(object):
    def __init__(self, args):
        self.start_time = time.time()
        self.end_time = np.nan
        self.seed = np.nan

        self.incumbents = []
        self.incumbent_values = []
        self.cum_train_times = []
        self.last_func_eval = np.nan
        self.iteration = 0
        self.max_iters = args.max_iters

        self.n_init = args.n_init
        self.log = args.log
        self.save_path = args.save_path
        self.dataset = args.dataset

        options = vars(args)
        print(options)

        if self.save_path is not None:
            import datetime

            time_string = datetime.datetime.now()
            time_string = time_string.strftime("%Y%m%d_%H%M%S")
            self.save_path = os.path.join(self.save_path, time_string)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            with open(os.path.join(self.save_path, "args.json"), "w") as f:
                json.dump(options, f, indent=6)

    def reset(self, seed):
        self.seed = seed
        set_seed(self.seed)

        self.start_time = time.time()
        self.end_time = np.nan

        self.incumbents = []
        self.incumbent_values = []
        self.cum_train_times = []
        self.last_func_eval = np.nan
        self.iteration = 0

    def calculate_incumbent(self, x, y, next_y):

        best_idx = np.argmax(y[self.n_init :])
        incumbent = x[self.n_init :][best_idx]
        incumbent_value = (
            np.exp(-y[self.n_init :][best_idx]).item()
            if self.log
            else -y[self.n_init :][best_idx].item()
        )
        self.incumbents.append(incumbent)
        self.incumbent_values.append(incumbent_value)
        self.last_func_eval = np.exp(-np.max(next_y)) if self.log else -np.max(next_y)

    def calculate_cost(self, train_details):

        self.end_time = time.time()
        # Compute the cumulative training time.
        try:
            cum_train_time = np.sum([item["train_time"] for item in train_details]).item()
        except TypeError:
            cum_train_time = np.nan
        self.cum_train_times.append(cum_train_time)

    def print(self, x, y, next_y, train_details):

        # Calculate Incumbent
        self.calculate_incumbent(x, y, next_y)
        self.calculate_cost(train_details)

        columns = ["Iteration", "Last func val", "Incumbent Value", "Time", "TrainTime"]

        values = [
            str(self.iteration),
            str(self.last_func_eval),
            str(self.incumbent_values[-1]),
            str(self.end_time - self.start_time),
            str(self.cum_train_times[-1]),
        ]
        table = tabulate.tabulate(
            [values], headers=columns, tablefmt="simple", floatfmt="8.4f"
        )

        if self.iteration % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

    def has_budget(self):
        return bool(self.iteration < self.max_iters)

    def next_iteration(self):
        self.iteration += 1

    def save_results(self):

        results = {
            "incumbents": [inc.parse() for inc in self.incumbents],
            "incumbent_fval": self.incumbent_values,
            "runtime": self.cum_train_times,
        }

        if self.save_path is not None:
            with open(
                os.path.join(
                    self.save_path, "{}:{}.json".format(self.max_iters, self.seed)
                ),
                "w",
            ) as f:
                json.dump(results, f, indent=6)
