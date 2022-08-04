from __future__ import annotations

import logging
import os
from pathlib import Path

import jahs_bench
import numpy as np
import pytest
from jahs_bench.lib.core.configspace import joint_config_space
from typing_extensions import Literal

import neps
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.search_spaces.search_space import pipeline_space_from_configspace

logging.basicConfig(level=logging.INFO)

LOSS_DICT = {
    "mf_bayesian_optimization": [8.445320129394531, 18.25725555419922],
    "bayesian_optimization": [7.922416687011719, 26.533340454101562],
    "random_search": [9.670074462890625, 15.112213134765625],
}


class RegressionTester:
    """Setting standard configurations, running the optimizations,
    and running regression test"""

    def evaluation_func(self):
        """
        If the searcher is cost aware, return the evaluation function with cost
        """

        benchmark = jahs_bench.Benchmark(task=self.task, kind="surrogate", download=True)

        def cost_evaluation(
            pipeline_directory, previous_pipeline_directory, **joint_configuration
        ):  # pylint: disable=unused-argument
            epoch = joint_configuration.pop("epoch")

            results = benchmark(joint_configuration, nepochs=epoch)
            return {
                "loss": 100 - results[epoch]["valid-acc"],
                "cost": results[epoch]["runtime"],
            }

        def loss_evaluation(
            pipeline_directory, previous_pipeline_directory, **joint_configuration
        ):  # pylint: disable=unused-argument
            epoch = joint_configuration.pop("epoch")

            results = benchmark(joint_configuration, nepochs=epoch)
            return 100 - results[epoch]["valid-acc"]

        if "cost" in self.searcher:
            return cost_evaluation
        else:
            return loss_evaluation

    def __init__(
        self,
        searcher: Literal[
            "default",
            "bayesian_optimization",
            "random_search",
            "cost_cooling",
            "mf_bayesian_optimization",
            "grid_search",
            "cost_cooling_bayesian_optimization",  # not implemented yet?
            "regularized_evolution",
        ]
        | BaseOptimizer = "mf_bayesian_optimization",
        iterations: int = 100,
        task: Literal["cifar10", "colorectal_histology", "fashion_mnist"] = "cifar10",
    ):
        """
        Download benchmark, initialize Pipeline space, evaluation function and set paths,

        Args:
            searcher: Choose an optimizer to run, this will also be the name of the run
            iterations: For how many repetitions to run the optimizations
            task: the dataset name for jahs_bench
        """

        self.task = task
        self.searcher = searcher
        self.name = searcher + "_runs"
        self.iterations = iterations
        self.run_pipeline = self.evaluation_func()
        self.root_directory = f"./{self.name}"

        # Cost cooling searcher expects budget but none of the others does
        self.budget = 10000 if "cost" in searcher else None

        if searcher not in LOSS_DICT.keys():
            ValueError(
                f"Regression hasn't been run for {searcher} searcher, "
                f"please update the LOSS_DICT first"
            )

        self.final_losses: list[float] = []
        self.final_losses_path = Path(self.root_directory, "final_losses.txt")
        if not self.final_losses_path.parent.exists():
            Path(self.root_directory).mkdir()

        self.pipeline_space = pipeline_space_from_configspace(joint_config_space)

        # For Regularized evolution sampler ignores fidelity hyperparameters
        # by sampling None for them
        is_fidelity = self.searcher != "regularized_evolution"
        self.pipeline_space["epoch"] = neps.IntegerParameter(
            lower=1, upper=200, is_fidelity=is_fidelity
        )

    def save_losses(self, file_name: str = "final_losses.txt"):
        self.final_losses_path = Path(self.root_directory, file_name)
        with self.final_losses_path.open(mode="w+", encoding="utf-8") as f:
            f.writelines([str(loss) + "\n" for loss in self.final_losses])

    def run(self):
        """
        Run iterations number of neps runs
        """
        for i in range(self.iterations):
            working_directory = Path(self.root_directory, "results/test_run_" + str(i))
            neps.run(
                run_pipeline=self.run_pipeline,
                pipeline_space=self.pipeline_space,
                searcher=self.searcher,
                budget=self.budget,
                root_directory=working_directory,
                max_evaluations_total=150,
            )
            best_error = (
                Path(working_directory, "best_loss_trajectory.txt")
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )

            self.final_losses.append(float(best_error))
        self.save_losses()

        return np.array(self.final_losses)

    def read_results(self):
        """
        Read the results of the last run.
        Either returns results of the most recent run, or
        return minimum and maximum values of the run from LOSS_DICT
        """

        if self.final_losses:
            return np.array(self.final_losses)
        elif self.final_losses_path.exists():
            self.final_losses = [
                float(loss)
                for loss in self.final_losses_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
        else:
            try:
                for i in range(self.iterations):
                    working_directory = Path(
                        self.root_directory, "results/test_run_" + str(i)
                    )
                    best_error = (
                        Path(working_directory, "best_loss_trajectory.txt")
                        .read_text(encoding="utf-8")
                        .splitlines()[-1]
                    )
                    self.final_losses.append(float(best_error))
            except FileNotFoundError:
                self.final_losses = LOSS_DICT[self.searcher]
        return np.array(self.final_losses)

    def test(self, max_evaluations=150):
        """
        Target run for the regression test, keep all the parameters same.

        Args:
            max_evaluations: Number of evaluations after which to terminate optimization.
        """
        working_directory = Path(self.root_directory, "results/test_run_target")
        neps.run(
            run_pipeline=self.run_pipeline,
            pipeline_space=self.pipeline_space,
            searcher=self.searcher,
            budget=self.budget,
            root_directory=working_directory,
            max_evaluations_total=max_evaluations,
        )
        best_error = float(
            Path(working_directory, "best_loss_trajectory.txt")
            .read_text(encoding="utf-8")
            .splitlines()[-1]
        )
        if not self.final_losses:
            results = self.read_results()
        else:
            results = np.array(self.final_losses)

        assert min(results) < best_error < max(results), (
            f"Expected the test error to be between {min(results)} and {max(results)}, "
            f"but the test error was {best_error}"
        )
        logging.info(
            f"Regression test passed: evaluated target for {max_evaluations} "
            f"evaluations, found loss: {best_error} "
            f"between {min(results)} and {max(results)}"
        )


@pytest.fixture(autouse=True)
def use_tmpdir(tmp_path, request):
    os.chdir(tmp_path)
    yield
    os.chdir(request.config.invocation_dir)


# https://stackoverflow.com/a/59745629
# Fail tests if there is a logging.error
@pytest.fixture(autouse=True)
def no_logs_gte_error(caplog):
    yield
    errors = [
        record for record in caplog.get_records("call") if record.levelno >= logging.ERROR
    ]
    assert not errors


@pytest.mark.regression
@pytest.mark.parametrize("searcher", LOSS_DICT.keys(), ids=LOSS_DICT.keys())
def test_regression(searcher):
    n = 100
    tester = RegressionTester(searcher, n, "cifar10")
    tester.test(150)


if __name__ == "__main__":

    n = 1
    for searcher in LOSS_DICT.keys():
        tester = RegressionTester(searcher, n, "cifar10")
        tester.run()
        best_results = tester.read_results()
        minv, maxv = min(best_results), max(best_results)
        print(
            f"For optimizer {searcher}:\n "
            f"\tMin of best results: {minv}\n\tMax of best results: {maxv}"
        )
