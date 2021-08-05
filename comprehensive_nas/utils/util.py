import csv
import json
import os
import pickle
import random
import time
from typing import Iterable

import numpy as np
import tabulate

try:
    import torch
except ModuleNotFoundError:
    from install_dev_utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class Experimentator(object):
    def __init__(self, n_init: int, max_iters: int, seed: int = None) -> None:
        super().__init__()
        self.n_init = n_init
        self.max_iters = max_iters
        self.iteration = 0

        self.seed = seed

    def reset(self, seed):
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.iteration = 0

    def has_budget(self):
        return bool(self.iteration < self.max_iters)

    def next_iteration(self):
        self.iteration += 1


class StatisticsTracker(object):
    def __init__(
        self, args, n_init: int, max_iters: int, save_path: str, log: bool = True
    ):
        self.start_time = time.time()
        self.end_time = np.nan
        self.seed = np.nan

        self.incumbents_eval = []
        self.incumbent_values_eval = []
        self.last_func_evals = []
        self.last_func_tests = []
        self.incumbents_test = []
        self.incumbent_values_test = []
        self.cum_train_times = []
        self.opt_details = []
        self.iteration = 0
        self.max_iters = max_iters

        self.n_init = n_init
        self.log = log
        self.save_path = save_path
        self.iteration = 0

        options = vars(args)

        if self.save_path is not None:
            import datetime

            time_string = datetime.datetime.now()
            time_string = time_string.strftime("%Y%m%d_%H%M%S")
            self.save_path = os.path.join(self.save_path, time_string)
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            with open(os.path.join(self.save_path, "args.json"), "w") as f:
                json.dump(options, f, indent=6)

    def reset(self):
        self.start_time = time.time()
        self.end_time = np.nan

        self.incumbents_eval = []
        self.incumbent_values_eval = []
        self.cum_train_times = []
        self.last_func_evals = []
        self.last_func_tests = []
        self.incumbents_test = []
        self.incumbent_values_test = []
        self.opt_details = []
        self.iteration = 0

    def calculate_incumbent(self, x: Iterable, y):
        best_idx = np.argmax(y[self.n_init :])
        incumbent = x[self.n_init :][best_idx]
        incumbent_value = (
            np.exp(-y[self.n_init :][best_idx]).item()
            if self.log
            else -y[self.n_init :][best_idx].item()
        )
        return incumbent, incumbent_value

    @staticmethod
    def calculate_cum_train_time(train_details):
        # Compute the cumulative training time.
        try:
            cum_train_time = np.sum([item["train_time"] for item in train_details]).item()
        except TypeError:
            cum_train_time = np.nan
        return cum_train_time

    def update(
        self,
        x,
        y_eval,
        y_eval_cur,
        train_details,
        y_test=None,
        y_test_cur=None,
        opt_details=None,
    ):
        # Calculate Incumbent
        incumbent, incumbent_value = self.calculate_incumbent(x, y_eval)
        self.incumbents_eval.append(incumbent)
        self.incumbent_values_eval.append(incumbent_value)

        self.last_func_evals.append(
            np.exp(-np.max(y_eval_cur)) if self.log else -np.max(y_eval_cur)
        )

        if y_test is not None:
            incumbent, incumbent_value = self.calculate_incumbent(x, y_test)
            self.incumbents_test.append(incumbent)
            self.incumbent_values_test.append(incumbent_value)
        if y_test_cur is not None:
            self.last_func_tests.append(
                np.exp(-np.max(y_test_cur)) if self.log else -np.max(y_test_cur)
            )
        if opt_details is not None:
            self.opt_details.extend(opt_details)

        self.end_time = time.time()
        cum_train_time = self.calculate_cum_train_time(train_details)
        self.cum_train_times.append(cum_train_time)
        self.iteration += 1

    def print(self):
        columns = [
            "Iteration",
            "Last func val",
            "Incumbent Value val",
            "Time",
            "Train Time",
        ]

        values = [
            str(self.iteration),
            str(self.last_func_evals[-1]),
            str(self.incumbent_values_eval[-1]),
            str(self.end_time - self.start_time),
            str(self.cum_train_times[-1]),
        ]
        if self.last_func_tests:
            columns.append("Last func test")
            values.extend([self.last_func_tests[-1]])
        if self.incumbent_values_test:
            columns.append("Incumbent Value test")
            values.extend([self.incumbent_values_test[-1]])

        table = tabulate.tabulate(
            [values], headers=columns, tablefmt="simple", floatfmt="8.4f"
        )

        if self.iteration % 40 == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)

    def save_results(self):
        # save all data for later use
        results = {
            "incumbents_eval": [inc.parse() for inc in self.incumbents_eval],
            "incumbent_value_eval": self.incumbent_values_eval,
            "runtime": self.cum_train_times,
            "last_func_evals": self.last_func_evals,
        }

        if self.incumbents_test:
            results["incumbent_test"] = ([inc.parse() for inc in self.incumbents_test],)
        if self.incumbent_values_test:
            results["incumbent_values_test"] = self.incumbent_values_test
        if self.last_func_tests:
            results["last_func_tests"] = self.last_func_tests
        if self.opt_details:
            results["opt_details"] = self.opt_details

        pickle.dump(
            results,
            open(
                os.path.join(self.save_path, "data.p"),
                "wb",
            ),
        )

        # save human-readable results
        columns = ["Iteration", "Last func val", "Incumbent Value val", "Cum Train Time"]
        values = [
            range(1, self.iteration + 1),
            self.last_func_evals,
            self.incumbent_values_eval,
            self.cum_train_times,
        ]
        if self.last_func_tests:
            columns.append("Last func test")
            values.extend([self.last_func_tests])
        if self.incumbent_values_test:
            columns.append("Incumbent Value test")
            values.extend([self.incumbent_values_test])
        zip_iter = zip(*values)
        with open(os.path.join(self.save_path, "results.csv"), "w") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow(columns)
            for i, x in enumerate(zip_iter):
                writer.writerow([i + 1] + list(x))
