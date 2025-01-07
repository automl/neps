# mypy: disable-error-code = union-attr
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import kstest

import neps
from tests.regression_objectives import (
    HartmannObjective,
    JAHSObjective,
    RegressionObjectiveBase,
)
from tests.settings import ITERATIONS, LOSS_FILE, MAX_EVALUATIONS_TOTAL, OPTIMIZERS, TASKS

TASK_OBJECTIVE_MAPPING = {
    "cifar10": JAHSObjective,
    "fashion_mnist": JAHSObjective,
    "hartmann3": HartmannObjective,
    "hartmann6": HartmannObjective,
}

logging.basicConfig(level=logging.INFO)


def incumbent_at(root_directory: str | Path, step: int):
    """
    Return the incumbent of the run at step n

    Args:
        root_directory: root directory of the optimization run
        step: step n at which to return the incumbent
    """
    log_file = Path(root_directory, "all_losses_and_configs.txt")
    losses = [
        float(line[6:])
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if "Loss: " in line
    ]
    incumbent_at_n = min(losses[:step])
    return incumbent_at_n


class RegressionRunner:
    """This class runs the optimization algorithms and stores the results in separate files"""

    def __init__(
        self,
        objective: RegressionObjectiveBase | Callable,
        iterations: int = 100,
        max_evaluations: int = 150,
        max_cost_total: int = 10000,
        experiment_name: str = "",
        **kwargs,
    ):
        """
        Download benchmark, initialize Pipeline space, evaluation function and set paths,

        Args:
            objective: callable that takes a configuration as input and evaluates it
            iterations: number of times to record the whole optimization process
            max_evaluations: maximum number of total evaluations for each optimization process
            max_cost_total: budget for cost aware optimizers
            experiment_name: string to identify different experiments
        """
        self.objective = objective
        if isinstance(objective, RegressionObjectiveBase):
            self.task = self.objective.task
            self.optimizer = self.objective.optimizer
            self.pipeline_space = self.objective.pipeline_space
        else:
            self.task = kwargs.get("task", None)
            if self.task is None:
                raise AttributeError(
                    f"self.task can not be {self.task}, "
                    f"please provide a task argument"
                )

            self.optimizer = kwargs.get("optimizer", None)
            if self.optimizer is None:
                raise AttributeError(
                    f"self.optimizer can not be {self.optimizer}, "
                    f"please provide an optimizer argument"
                )

            self.pipeline_space = kwargs.get("pipeline_space", None)
            if self.pipeline_space is None:
                raise AttributeError(
                    f"self.pipeline_space can not be {self.pipeline_space}, "
                    f"please provide an pipeline_space argument"
                )
        if experiment_name:
            experiment_name += "_"
        self.name = f"{self.optimizer}_{self.task}_{experiment_name}runs"
        self.iterations = iterations
        self.benchmark = None

        # Cost cooling optimizer expects budget but none of the others does
        self.max_cost_total = max_cost_total if "cost" in self.optimizer else None
        self.max_evaluations = max_evaluations

        self.final_losses: list[float] = []

        # Number of samples for testing
        self.sample_size = 10

    @property
    def root_directory(self):
        return f"./{self.name}"

    @property
    def final_losses_path(self):
        return Path(self.root_directory, self.objective_to_minimize_file_name)

    @property
    def objective_to_minimize_file_name(self):
        return f"final_losses_{self.max_evaluations}_.txt"

    def save_losses(self):
        if not self.final_losses_path.parent.exists():
            Path(self.root_directory).mkdir()
        with self.final_losses_path.open(mode="w+", encoding="utf-8") as f:
            f.writelines([str(objective_to_minimize) + "\n" for objective_to_minimize in self.final_losses])
        logging.info(
            f"Saved the results of {len(self.final_losses)} "
            f"runs of {self.max_evaluations} "
            f"max evaluations into the file: {self.final_losses_path}"
        )

    def neps_run(self, working_directory: Path):
        neps.run(
            evaluate_pipeline=self.objective,
            pipeline_space=self.pipeline_space,
            searcher=self.optimizer,
            max_cost_total=self.max_cost_total,
            root_directory=working_directory,
            max_evaluations_total=self.max_evaluations,
        )

        best_error = incumbent_at(working_directory, self.max_evaluations)
        return best_error

    def run_regression(self, save=False):
        """
        Run iterations number of neps runs
        """

        for i in range(self.iterations):
            working_directory = Path(self.root_directory, "results/test_run_" + str(i))

            best_error = self.neps_run(working_directory)

            self.final_losses.append(float(best_error))

        if save:
            self.save_losses()

        return np.array(self.final_losses)

    def read_results(self):
        """
        Read the results of the last run.
        Either returns results of the most recent run, or
        return the values from LOSS_FILE
        """

        if self.final_losses:
            return np.array(self.final_losses)
        elif self.final_losses_path.exists():
            # Read from final_losses_path for each regression run
            self.final_losses = [
                float(objective_to_minimize)
                for objective_to_minimize in self.final_losses_path.read_text(
                    encoding="utf-8"
                ).splitlines()[: self.iterations]
            ]
        else:
            # Read from the results of previous runs if final_losses_path is not saved
            try:
                for i in range(self.iterations):
                    working_directory = Path(
                        self.root_directory, "results/test_run_" + str(i)
                    )
                    best_error = incumbent_at(working_directory, self.max_evaluations)
                    self.final_losses.append(float(best_error))
            except FileNotFoundError as not_found:
                # Try reading from the LOSS_FILE in the worst case
                if LOSS_FILE.exists():
                    with LOSS_FILE.open(mode="r", encoding="utf-8") as f:
                        objective_to_minimize_dict = json.load(f)
                    self.final_losses = objective_to_minimize_dict[self.optimizer][self.task]
                else:
                    raise FileNotFoundError(
                        f"Results from the previous runs are not "
                        f"found, and {LOSS_FILE} does not exist"
                    ) from not_found
        return np.array(self.final_losses)

    def test(self):
        """
        Target run for the regression test, keep all the parameters same.

        Args:
            max_evaluations: Number of evaluations after which to terminate optimization.
        """

        # Sample losses of self.sample_size runs
        samples = []
        for i in range(self.sample_size):
            working_directory = Path(self.root_directory, f"results/test_run_target_{i}")
            best_error = self.neps_run(working_directory)
            samples.append(best_error)

        # Run tests
        target = self.read_results()

        threshold = self.median_threshold(target)

        ks_result = kstest(samples, target)
        median_dist = np.median(samples) - np.median(target)
        ks_test = 0 if ks_result.pvalue < 0.1 else 1
        median_test = 0 if abs(median_dist) > threshold else 1
        median_improvement = 1 if median_dist < 0 else 0

        return ks_test, median_test, median_improvement

    @staticmethod
    def median_threshold(
        target: np.ndarray, percentile: float | int = 92.5, sample_size: int = 10
    ):
        stat_size = 1000
        p_index = int(stat_size * percentile / 100)
        distances = np.zeros(stat_size)
        for i, _ in enumerate(distances):
            _sample = np.random.choice(target, size=sample_size, replace=False)
            median_dist = np.median(_sample) - np.median(target)
            distances[i] = median_dist
        distances.sort()
        return distances[p_index]


if __name__ == "__main__":
    # Collect samples for each optimizer and store the data in the LOSS_FILE
    json_file = Path("losses.json")
    if json_file.exists():
        with json_file.open(mode="r", encoding="utf-8") as f:
            losses_dict = json.load(f)
    else:
        losses_dict = dict()

    print(f"Optimizers the results are already recorded for: {losses_dict.keys()}")
    for optimizer in OPTIMIZERS:
        if optimizer in losses_dict:
            print(f"For {optimizer} recorded tasks are: {losses_dict[optimizer].keys()}")
        for task in TASKS:
            if (
                isinstance(losses_dict.get(optimizer, None), dict)
                and len(losses_dict[optimizer].get(task, [])) == ITERATIONS
            ):
                continue
            else:
                runner = RegressionRunner(
                    objective=TASK_OBJECTIVE_MAPPING[task](
                        optimizer=optimizer, task=task
                    ),
                    max_evaluations=MAX_EVALUATIONS_TOTAL,
                )
                runner.run_regression(save=True)
                best_results = runner.read_results().tolist()
                minv, maxv = min(best_results), max(best_results)
                print(
                    f"For optimizer {optimizer} on {task}:\n "
                    f"\tMin of best results: {minv}\n\tMax of best results: {maxv}"
                )
                if isinstance(losses_dict.get(optimizer, None), dict) and isinstance(
                    losses_dict[optimizer].get(task, None), list
                ):
                    losses_dict[optimizer][task] = best_results
                elif isinstance(losses_dict.get(optimizer, None), dict):
                    update_dict = {task: best_results}
                    losses_dict[optimizer].update(update_dict)
                else:
                    update_dict = {optimizer: {task: best_results}}
                    losses_dict.update(update_dict)

    # print(losses_dict)
    with json_file.open(mode="w", encoding="utf-8") as f:
        json.dump(losses_dict, f)
