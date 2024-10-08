import warnings
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

import neps
from neps.search_spaces.search_space import SearchSpace, pipeline_space_from_configspace


class RegressionObjectiveBase:
    """
    Base class for creating new synthetic or real objectives for the regression tests
    Regression runner uses properties defined here,
    each property should be appropriately defined by the subclasses
    """

    def __init__(self, optimizer: str, task: str):
        self.optimizer = optimizer
        self.task = task
        self.has_fidelity = self.optimizer != "random_search"
        self._run_pipeline: Callable | None = None
        self._pipeline_space: SearchSpace | dict[str, Any] = {}

    @property
    def pipeline_space(self) -> SearchSpace | dict[str, Any]:
        if not self._pipeline_space:
            raise NotImplementedError(
                f"pipeline_space can not be {self._pipeline_space},"
                f" the subclass {type(self)} must implement "
                f"a pipeline_space attribute"
            )
        else:
            return self._pipeline_space

    @pipeline_space.setter
    def pipeline_space(self, value):
        self._pipeline_space = value

    @property
    def run_pipeline(self) -> Callable:
        if self._run_pipeline is None:
            raise NotImplementedError(
                f"run_pipeline can not be None, "
                f"the subclass {type(self)} must "
                f"implement a run_pipeline Callable"
            )
        else:
            return self._run_pipeline

    @run_pipeline.setter
    def run_pipeline(self, value):
        self._run_pipeline = value

    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return self.run_pipeline(*args, **kwargs)


class JAHSObjective(RegressionObjectiveBase):
    def evaluation_func(self):
        """
        If the optimizer is cost aware, return the evaluation function with cost
        """
        import jahs_bench

        self.benchmark = jahs_bench.Benchmark(
            task=self.task, kind="surrogate", download=True, save_dir=self.save_dir
        )

        def cost_evaluation(**joint_configuration):
            epoch = joint_configuration.pop("epoch")
            joint_configuration.update({"N": 5, "W": 16, "Resolution": 1.0})

            results = self.benchmark(joint_configuration, nepochs=epoch)
            return {
                "loss": 100 - results[epoch]["valid-acc"],
                "cost": results[epoch]["runtime"],
            }

        def loss_evaluation(**joint_configuration):
            epoch = joint_configuration.pop("epoch")
            joint_configuration.update({"N": 5, "W": 16, "Resolution": 1.0})

            results = self.benchmark(joint_configuration, nepochs=epoch)
            return 100 - results[epoch]["valid-acc"]

        if "cost" in self.optimizer:
            return cost_evaluation
        else:
            return loss_evaluation

    def __init__(
        self,
        optimizer: str = "mf_bayesian_optimization",
        task: (
            Literal["cifar10", "colorectal_histology", "fashion_mnist"] | str
        ) = "cifar10",
        save_dir: str | Path = "jahs_bench_data",
        **kwargs,
    ):
        """
        Download benchmark, initialize Pipeline space and evaluation function

        Args:
            optimizer: The optimizer that will be run, this is used to determine the
            fidelity parameter of the pipeline space and whether to return the cost value
            in the run_pipeline function
            task: the dataset name for jahs_bench
            save_dir: The (absolute or relative) path to a directory where the data
            required for the benchmark to run will be read from.
        """
        super().__init__(optimizer=optimizer, task=task)
        from tests.joint_config_space import joint_config_space

        self.save_dir = Path(save_dir)
        self.benchmark = None

        self.pipeline_space = pipeline_space_from_configspace(joint_config_space)

        self.pipeline_space["epoch"] = neps.IntegerParameter(
            lower=1, upper=200, is_fidelity=self.has_fidelity
        )
        self.run_pipeline = self.evaluation_func()

        self.surrogate_model = "gp" if self.optimizer != "random_search" else None
        self.surrogate_model_args = kwargs.get("surrogate_model_args", None)


class HartmannObjective(RegressionObjectiveBase):
    z_min = 3
    z_max = 100

    def evaluation_fn(self) -> Callable:
        def hartmann3(**z_nX):
            if self.has_fidelity:
                z = z_nX.get("z")
            else:
                z = self.z_max

            X_0 = z_nX.get("X_0")
            X_1 = z_nX.get("X_1")
            X_2 = z_nX.get("X_2")
            Xs = tuple((X_0, X_1, X_2))

            log_z = np.log(z)
            log_lb, log_ub = np.log(self.z_min), np.log(self.z_max)
            log_z_scaled = (log_z - log_lb) / (log_ub - log_lb)

            # Highest fidelity (1) accounts for the regular Hartmann
            X = np.array([X_0, X_1, X_2]).reshape(1, -1)
            alpha = np.array([1.0, 1.2, 3.0, 3.2])

            alpha_prime = alpha - self.bias * np.power(1 - log_z_scaled, 1)
            A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
            P = np.array(
                [
                    [3689, 1170, 2673],
                    [4699, 4387, 7470],
                    [1091, 8732, 5547],
                    [381, 5743, 8828],
                ]
            )

            inner_sum = np.sum(A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
            H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))

            # and add some noise
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Seed below will overflow
                rng = np.random.default_rng(seed=abs(self.seed * z * hash(Xs)))

            noise = np.abs(rng.normal(size=H.size)) * self.noise * (1 - log_z_scaled)

            loss = float((H + noise)[0])
            cost = 0.05 + (1 - 0.05) * (z / self.z_max) ** 2

            result = {"loss": loss}
            if "cost" in self.optimizer:
                result.update({"cost": cost})

            return result

        def hartmann6(**z_nX):
            if self.has_fidelity:
                z = z_nX.get("z")
            else:
                z = self.z_max

            X_0 = z_nX.get("X_0")
            X_1 = z_nX.get("X_1")
            X_2 = z_nX.get("X_2")
            X_3 = z_nX.get("X_3")
            X_4 = z_nX.get("X_4")
            X_5 = z_nX.get("X_5")
            Xs = tuple((X_0, X_1, X_2, X_3, X_4, X_5))

            # Change by Carl - z now comes in normalized
            log_z = np.log(z)
            log_lb, log_ub = np.log(self.z_min), np.log(self.z_max)
            log_z_scaled = (log_z - log_lb) / (log_ub - log_lb)

            # Highest fidelity (1) accounts for the regular Hartmann
            X = np.array([X_0, X_1, X_2, X_3, X_4, X_5]).reshape(1, -1)
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            alpha_prime = alpha - self.bias * np.power(1 - log_z_scaled, 1)
            A = np.array(
                [
                    [10, 3, 17, 3.5, 1.7, 8],
                    [0.05, 10, 17, 0.1, 8, 14],
                    [3, 3.5, 1.7, 10, 17, 8],
                    [17, 8, 0.05, 10, 0.1, 14],
                ]
            )
            P = np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )

            inner_sum = np.sum(A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
            H = -(np.sum(alpha_prime * np.exp(-inner_sum), axis=-1))

            # and add some noise
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Seed below will overflow
                rng = np.random.default_rng(seed=abs(self.seed * z * hash(Xs)))

            noise = np.abs(rng.normal(size=H.size)) * self.noise * (1 - log_z_scaled)

            loss = float((H + noise)[0])
            cost = 0.05 + (1 - 0.05) * (z / self.z_max) ** 2

            result = {"loss": loss}
            if "cost" in self.optimizer:
                result.update({"cost": cost})

            return result

        if self.dim == 3:
            hartmann_fn = hartmann3
        else:
            hartmann_fn = hartmann6

        return hartmann_fn

    def __init__(
        self,
        optimizer: str,
        task: Literal["hartmann3", "hartmann6"],
        bias: float = 0.5,
        noise: float = 0.1,
        seed: int = 1337,
        **kwargs,
    ):
        """
        Initialize Pipeline space and evaluation function

        Args:
            optimizer: The optimizer that will be run, this is used to determine the
            fidelity parameter of the pipeline space and whether to return the cost value
            in the run_pipeline function
            task: the type of hartmann function used
        """
        super().__init__(optimizer=optimizer, task=task)
        if task == "hartmann3":
            self.dim = 3
        elif self.task == "hartmann6":
            self.dim = 6
        else:
            raise ValueError(
                "Hartmann objective is only defined for 'hartmann3' and 'hartmann6' "
            )

        self.pipeline_space: dict[str, Any] = {
            f"X_{i}": neps.FloatParameter(lower=0.0, upper=1.0) for i in range(self.dim)
        }

        if self.has_fidelity:
            self.pipeline_space["z"] = neps.IntegerParameter(
                lower=self.z_min, upper=self.z_max, is_fidelity=self.has_fidelity
            )

        self.bias = bias
        self.noise = noise
        self.seed = seed
        self.random_state = np.random.default_rng(seed)

        self.surrogate_model = "gp" if self.optimizer != "random_search" else None
        self.surrogate_model_args = kwargs.get("surrogate_model_args", None)

        self.run_pipeline = self.evaluation_fn()
