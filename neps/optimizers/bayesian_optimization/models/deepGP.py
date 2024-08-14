from __future__ import annotations
from dataclasses import dataclass, field

import logging
from copy import deepcopy
from pathlib import Path

import gpytorch
import numpy as np
import torch
import torch.nn as nn
from neps.search_spaces.architecture.graph_grammar import GraphParameter

from neps.exceptions import SurrogateFailedToFit

from neps.search_spaces.search_space import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


def count_non_improvement_steps(root_directory: Path | str) -> int:
    root_directory = Path(root_directory)

    all_losses_file = root_directory / "all_losses_and_configs.txt"
    best_loss_fiel = root_directory / "best_loss_trajectory.txt"

    # Read all losses from the file in the order they are explored
    losses = [
        float(line[6:])
        for line in all_losses_file.read_text(encoding="utf-8").splitlines()
        if "Loss: " in line
    ]
    # Get the best seen loss value
    best_loss = float(best_loss_fiel.read_text(encoding="utf-8").splitlines()[-1].strip())

    # Count the non-improvement
    count = 0
    for loss in reversed(losses):
        if np.greater(loss, best_loss):
            count += 1
        else:
            break

    return count


class NeuralFeatureExtractor(nn.Module):
    """
    Neural network to be used in the DeepGP
    """

    def __init__(self, input_size: int, **kwargs):
        super().__init__()

        # Set number of hyperparameters
        self.input_size = input_size

        self.n_layers = kwargs.get("n_layers", 2)
        self.activation = nn.LeakyReLU()

        layer1_units = kwargs.get("layer1_units", 128)
        self.fc1 = nn.Linear(input_size, layer1_units)
        self.bn1 = nn.BatchNorm1d(layer1_units)

        previous_layer_units = layer1_units
        for i in range(2, self.n_layers):
            next_layer_units = kwargs.get(f"layer{i}_units", 256)
            setattr(
                self,
                f"fc{i}",
                nn.Linear(previous_layer_units, next_layer_units),
            )
            setattr(
                self,
                f"bn{i}",
                nn.BatchNorm1d(next_layer_units),
            )
            previous_layer_units = next_layer_units

        setattr(
            self,
            f"fc{self.n_layers}",
            nn.Linear(
                previous_layer_units + kwargs.get("cnn_nr_channels", 4),
                # accounting for the learning curve features
                kwargs.get(f"layer{self.n_layers}_units", 256),
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                kernel_size=(kwargs.get("cnn_kernel_size", 3),),
                out_channels=4,
            ),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, x, budgets, learning_curves):
        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets = torch.unsqueeze(budgets, dim=1)
        # concatenate budgets with examples
        x = torch.cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.activation(self.bn1(x))

        for i in range(2, self.n_layers):
            x = self.activation(getattr(self, f"bn{i}")(getattr(self, f"fc{i}")(x)))

        # add an extra dimensionality for the learning curve
        # making it nr_rows x 1 x lc_values.
        learning_curves = torch.unsqueeze(learning_curves, 1)
        lc_features = self.cnn(learning_curves)
        # revert the output from the cnn into nr_rows x nr_kernels.
        lc_features = torch.squeeze(lc_features, 2)

        # put learning curve features into the last layer along with the higher level features.
        x = torch.cat((x, lc_features), dim=1)
        x = self.activation(getattr(self, f"fc{self.n_layers}")(x))

        return x


class GPRegressionModel(gpytorch.models.ExactGP):
    """
    A simple GP model.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        """
        Constructor of the GPRegressionModel.

        Args:
            train_x: The initial train examples for the GP.
            train_y: The initial train labels for the GP.
            likelihood: The likelihood to be used.
        """
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@dataclass
class DeepGPDataTransformer:
    # TODO: This class could be used for other models as well
    space: SearchSpace
    fidelity_bounds: tuple[int | float, int | float] | None
    normalize_y: bool
    min_learning_curve_length: int
    learning_curve_pad_value: float
    device: torch.device

    numericals: dict[str, FloatParameter | IntegerParameter] = field(init=False)
    categoricals: dict[str, CategoricalParameter] = field(init=False)
    output_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.numericals = {
            name: h
            for name, h in self.space.items()
            if isinstance(h, (FloatParameter, IntegerParameter)) and not h.is_fidelity
        }
        self.categoricals = {
            name: h
            for name, h in self.space.items()
            if isinstance(h, CategoricalParameter)
        }
        self.output_dim = len(self.numericals) + sum(
            len(c.choices) for c in self.categoricals.values()
        )

    def encode_configs(
        self,
        configs: list[SearchSpace],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_buffer = torch.empty(
            (len(configs), self.output_dim),
            device=self.device,
            dtype=torch.float32,
        )

        # Normals are just fill the columns with the normalized values
        for i, (hp_name, hp) in enumerate(self.numericals.items()):
            budget_tensor = torch.tensor(
                [config[hp_name].value for config in configs],
                device=self.device,
                dtype=torch.float32,
            )

            x_buffer[:, i] = (budget_tensor - hp.lower) / (hp.upper - hp.lower)

        # Categoricals is a bit harder, we create a tensor with all the indices (values)
        # as we did above, but then we sub-select the portion of the buffer for that categorical
        # before inserting the one-hot encoding.
        offset = len(self.numericals)
        for hp_name, hp in self.categoricals.items():
            budget_tensor = torch.tensor(
                [config[hp_name]._value_index for config in configs],  # type: ignore
                device=self.device,
                dtype=torch.float64,
            )

            # .. and insert one-hot encoding (ChatGPT solution, verified locally)
            portion = x_buffer[:, offset : offset + len(hp.choices)]
            portion.scatter_(1, budget_tensor.unsqueeze(1), 1)

            offset += len(hp.choices)

        # Finally, ... budgets
        budgets = [config.fidelity.value for config in configs]  # type: ignore
        budget_tensor = torch.tensor(budgets, device=self.device, dtype=torch.float32)
        if self.fidelity_bounds:
            assert self.space.fidelity is not None
            _min = self.space.fidelity.lower
            _max = self.space.fidelity.upper
            budget_tensor.sub_(_min).div_(_max - _min)

        return x_buffer, budget_tensor

    def encode_learning_curves(self, learning_curves: list[list[float]]) -> torch.Tensor:
        lc_height = len(learning_curves)
        lc_width = max(
            max(len(lc) for lc in learning_curves), self.min_learning_curve_length
        )
        lc_buffer = torch.full(
            (lc_width, lc_height),
            self.learning_curve_pad_value,
            device=self.device,
            dtype=torch.float32,
        )

        for i, lc in enumerate(learning_curves):
            lc_buffer[: len(lc), i] = torch.tensor(
                lc, device=self.device, dtype=torch.float32
            )

        return lc_buffer

    def encode_y(self, y: list[float]) -> torch.Tensor:
        return torch.tensor(y, device=self.device, dtype=torch.float32)


def _train_model(
    x_train: torch.Tensor,
    train_budgets: torch.Tensor,
    learning_curves: torch.Tensor,
    model: GPRegressionModel,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    device: torch.device,
    nn: NeuralFeatureExtractor,
    y_train: torch.Tensor,
    n_epochs: int = 1000,
    batch_size: int = 64,
    optimizer_args: dict | None = None,
    early_stopping: bool = True,
    patience: int = 10,
):
    if optimizer_args is None:
        optimizer_args = {"lr": 0.001}

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)

    # Set to training mode
    mll.train()
    model.train()
    likelihood.train()
    nn.train()

    optimizer = torch.optim.Adam(
        [
            dict({"params": model.parameters()}, **optimizer_args),
            dict({"params": nn.parameters()}, **optimizer_args),
        ]
    )

    count_down = patience
    min_avg_loss_val = np.inf
    average_loss: float = 0.0

    for epoch_nr in range(0, n_epochs):
        if early_stopping and count_down == 0:
            logger.info(
                f"Epoch: {epoch_nr - 1} surrogate training stops due to early "
                f"stopping with the patience: {patience} and "
                f"the minimum average loss of {min_avg_loss_val} and "
                f"the final average loss of {average_loss}"
            )
            break

        n_examples_batch = x_train.size(dim=0)

        # get a random permutation for mini-batches
        permutation = torch.randperm(n_examples_batch)

        # optimize over mini-batches
        total_scaled_loss = 0.0
        for batch_idx, start_index in enumerate(range(0, n_examples_batch, batch_size)):
            end_index = min(start_index + batch_size, n_examples_batch)
            minibatch_size = end_index - start_index + 1

            # if only one example in the batch, skip the batch.
            # Otherwise, the code will fail because of batchnorm
            if minibatch_size <= 1:
                continue

            indices = permutation[start_index:end_index]

            batch_x, batch_budget, batch_lc, batch_y = (
                x_train[indices],
                train_budgets[indices],
                learning_curves[indices],
                y_train[indices],
            )

            # Zero backprop gradients
            optimizer.zero_grad()

            projected_x = nn(batch_x, batch_budget, batch_lc)
            model.set_train_data(projected_x, batch_y, strict=False)
            output = model(projected_x)

            # Calc loss and backprop derivatives
            loss = -mll(output, model.train_targets)  # type: ignore
            episodic_loss_value: float = loss.detach().to("cpu").item()
            # weighted sum over losses in the batch
            total_scaled_loss = total_scaled_loss + episodic_loss_value * minibatch_size

            mse = gpytorch.metrics.mean_squared_error(output, model.train_targets)
            logger.debug(
                f"Epoch {epoch_nr}  Batch {batch_idx} - MSE {mse:.5f}, "
                f"Loss: {episodic_loss_value:.3f}, "
                f"lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}, "
                f"noise: {model.likelihood.noise.item():.3f}, "  # type: ignore
            )

            loss.backward()
            optimizer.step()

        # Get average weighted loss over every batch
        average_loss = total_scaled_loss / n_examples_batch
        if average_loss < min_avg_loss_val:
            min_avg_loss_val = average_loss
            count_down = patience
        elif early_stopping:
            logger.debug(
                f"No improvement over the minimum loss value of {min_avg_loss_val} "
                f"for the past {patience - count_down} epochs "
                f"the training will stop in {count_down} epochs"
            )
            count_down -= 1


@dataclass
class DeepGP:
    """Gaussian process with a deep kernel."""

    # Required
    pipeline_space: SearchSpace

    # Optional
    learning_curve_pad_value: float = 0.0
    root_directory: Path | None = None
    # IMPORTANT: Checkpointing does not use file locking
    # IMPORTANT: hence it is not suitable for multiprocessing settings
    checkpoint_file: Path | str = "surrogate_checkpoint.pth"
    checkpointing: bool = False
    early_stopping: bool = True
    batch_size: int = 64
    n_epochs: int = 1000
    patience: int = 10
    refine_epochs: int = 50
    perf_patience_factor: float = 1.2  # X * max_fidelity
    n_initial_full_trainings: int = 10
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    normalize_budget: bool = True
    normalize_y: bool = True
    neural_network_args: dict = field(default_factory=dict)
    surrogate_model_fit_args: dict = field(default_factory=dict)
    optimizer_args: dict = field(default_factory=dict)

    # Created from the above arguments
    # TODO: Lift this out of DeepGP and let the optimizer worry about pre-processing
    preprocessor: DeepGPDataTransformer = field(init=False)
    max_fidelity: int | float = field(init=False)

    # Post fit parameters, following scikit-learn convention of appending an underscore
    model_: GPRegressionModel | None = field(init=False)
    likelihood_: gpytorch.likelihoods.GaussianLikelihood | None = field(init=False)
    nn_: NeuralFeatureExtractor | None = field(init=False)
    projected_x_train_: torch.Tensor | None = field(init=False)
    y_train_: torch.Tensor | None = field(init=False)

    def __post_init__(self):
        if any(isinstance(h, GraphParameter) for h in self.pipeline_space.values()):
            raise ValueError("Graph parameters are not supported for DeepGP")

        if self.normalize_budget:
            budget_bounds = (pipeline_space.fidelity.lower, pipeline_space.fidelity.upper)  # type: ignore
        else:
            budget_bounds = None

        if self.checkpointing:
            assert (
                self.root_directory is not None
            ), "neps root_directory must be provided for the checkpointing"
            self.checkpoint_path = self.root_directory / self.checkpoint_file

        self.max_fidelity = self.pipeline_space.fidelity.upper  # type: ignore
        self.preprocessor = DeepGPDataTransformer(
            space=self.pipeline_space,
            fidelity_bounds=budget_bounds,
            normalize_y=self.normalize_y,
            min_learning_curve_length=self.neural_network_args.get("cnn_kernel_size", 3),
            learning_curve_pad_value=self.learning_curve_pad_value,
            device=self.device,
        )
        self.model_ = None
        self.likelihood_ = None
        self.nn_ = None
        self.projected_x_train_ = None
        self.y_train_ = None

    def fit(
        self,
        x_train: list[SearchSpace],
        y_train: list[float],
        learning_curves: list[list[float]],
    ):
        x_, train_budget = self.preprocessor.encode_configs(x_train)
        curves = self.preprocessor.encode_learning_curves(learning_curves)
        y_ = self.preprocessor.encode_y(y_train)

        # Required for predictions later
        self.y_train_ = y_

        input_dim = x_.shape[1]

        # Initial state
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPRegressionModel(train_x=x_, train_y=y_, likelihood=likelihood).to(
            self.device
        )
        nn = NeuralFeatureExtractor(input_dim, **self.neural_network_args).to(self.device)

        # If checkpointing and we are improving, load existing model
        if self.checkpointing and self.checkpoint_path.exists():
            assert self.root_directory is not None

            non_improvement_steps = count_non_improvement_steps(self.root_directory)

            patience_steps = self.perf_patience_factor * self.max_fidelity
            if (
                len(y_train) >= self.n_initial_full_trainings
                and non_improvement_steps < patience_steps
            ):
                n_epochs = self.refine_epochs

                checkpoint = torch.load(self.checkpoint_path)
                model.load_state_dict(checkpoint["gp_state_dict"])
                nn.load_state_dict(checkpoint["nn_state_dict"])
                likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
            else:
                n_epochs = self.n_epochs
                logger.debug(f"No improvement for: {non_improvement_steps} evaulations")
        else:
            # Starting from scratch
            n_epochs = self.n_epochs

        logger.debug(f"N Epochs for the full training: {self.n_epochs}")

        try:
            _train_model(
                x_train=x_,
                train_budgets=train_budget,
                learning_curves=curves,
                y_train=y_,
                model=model,
                likelihood=likelihood,
                nn=nn,
                n_epochs=n_epochs,
                device=self.device,
                batch_size=self.batch_size,
                optimizer_args=self.optimizer_args,
                early_stopping=self.early_stopping,
                patience=self.patience,
            )
            self.model_ = model
            self.likelihood_ = likelihood
            self.nn_ = nn

            nn.eval()
            # Cheaper to do this once during fit, rather than on each call to predict
            self.projected_x_train_ = nn(x_, train_budget, curves)

            if self.checkpointing:
                torch.save(
                    {
                        "gp_state_dict": deepcopy(model).state_dict(),
                        "nn_state_dict": deepcopy(nn).state_dict(),
                        "likelihood_state_dict": deepcopy(likelihood.state_dict()),
                    },
                    self.checkpoint_path,
                )
        except gpytorch.utils.errors.NotPSDError as e:
            logger.error(
                "Model training failed loading the untrained model", exc_info=True
            )
            # Delete checkpoint to restart training
            self.checkpoint_path.unlink(missing_ok=True)
            raise SurrogateFailedToFit("DeepGP Failed to fit the training data!") from e

    def predict(
        self,
        x: list[SearchSpace],
        learning_curves: list[list[float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.model_ is not None, "Please fit the model first"
        assert self.nn_ is not None, "Please fit the model first"
        assert self.likelihood_ is not None, "Please fit the model first"
        assert self.projected_x_train_ is not None, "Please fit the model first"
        assert self.y_train_ is not None, "Please fit the model first"

        self.model_.eval()
        self.nn_.eval()
        self.likelihood_.eval()

        x_test, test_budgets = self.preprocessor.encode_configs(x)
        _curves = self.preprocessor.encode_learning_curves(learning_curves)

        with torch.no_grad():
            # Set GP prior
            self.model_.set_train_data(
                inputs=self.projected_x_train_,
                targets=self.y_train_,
                strict=False,
            )

            projected_test_x = self.nn_(x_test, test_budgets, _curves)
            preds = self.likelihood_(self.model_(projected_test_x))

        means = preds.mean.detach().cpu()
        cov = torch.diag(torch.pow(preds.stddev.detach(), 2)).cpu()

        return means, cov
