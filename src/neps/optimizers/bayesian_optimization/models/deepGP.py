from __future__ import annotations

import logging

import gpytorch
import numpy as np
import torch
import torch.nn as nn

from ....search_spaces.search_space import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)


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


class DeepGP:
    """
    Gaussian process with a deep kernel
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        neural_network_args: dict | None = None,
        logger=None,
        surrogate_model_fit_args: dict | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.surrogate_model_fit_args = (
            surrogate_model_fit_args if surrogate_model_fit_args is not None else {}
        )
        super().__init__()
        self.__preprocess_search_space(pipeline_space)
        # set the categories array for the encoder
        self.categories_array = np.array(self.categories)

        if neural_network_args is None:
            neural_network_args = {}

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.device = torch.device("cpu")

        # Save the NN args, necessary for preprocessing
        self.cnn_kernel_size = neural_network_args.get("cnn_kernel_size", 3)
        self.model, self.likelihood, self.mll = self.__initialize_gp_model(
            neural_network_args.get("n_layers", 2)
        )

        # build the neural network
        self.nn = NeuralFeatureExtractor(self.input_size, **neural_network_args)

        self.logger = logger or logging.getLogger("neps")

    def __initialize_gp_model(
        self,
        train_size: int,
    ) -> tuple[
        GPRegressionModel,
        gpytorch.likelihoods.GaussianLikelihood,
        gpytorch.mlls.ExactMarginalLogLikelihood,
    ]:
        """
        Called when the surrogate is first initialized or restarted.

        Args:
            train_size: The size of the current training set.

        Returns:
            model, likelihood, mll - The GP model, the likelihood and
                the marginal likelihood.
        """
        train_x = torch.ones(train_size, train_size).to(self.device)
        train_y = torch.ones(train_size).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = GPRegressionModel(
            train_x=train_x, train_y=train_y, likelihood=likelihood
        ).to(self.device)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.device)

        return model, likelihood, mll

    def __preprocess_search_space(self, pipeline_space: SearchSpace):
        self.categories = []
        self.categorical_hps = []

        parameter_count = 0
        for hp_name, hp in pipeline_space.items():
            # Collect all categories in a list for the encoder
            if isinstance(hp, CategoricalParameter):
                self.categorical_hps.append(hp_name)
                self.categories.extend(hp.choices)
                parameter_count += len(hp.choices)
            else:
                parameter_count += 1

        # add 1 for budget
        self.input_size = parameter_count
        self.continuous_params_size = self.input_size - len(self.categories)

        self.min_fidelity = pipeline_space.fidelity.lower
        self.max_fidelity = pipeline_space.fidelity.upper

    def __encode_config(self, config: SearchSpace):
        categorical_encoding = np.zeros_like(self.categories_array)
        continuous_values = []

        for hp_name, hp in config.items():
            if hp.is_fidelity:
                continue  # Ignore fidelity
            if hp_name in self.categorical_hps:
                label = hp.value
                categorical_encoding[np.argwhere(self.categories_array == label)] = 1
            else:
                continuous_values.append(hp.normalized().value)

        continuous_encoding = np.array(continuous_values)

        encoding = np.concatenate([categorical_encoding, continuous_encoding])
        return encoding

    def __extract_budgets(
        self, x_train: list[SearchSpace], normalized: bool = True
    ) -> np.ndarray:
        budgets = np.array([config.fidelity.value for config in x_train], dtype=np.single)
        if normalized:
            normalized_budgets = (budgets - self.min_fidelity) / (
                self.max_fidelity - self.min_fidelity
            )
            budgets = normalized_budgets
        return budgets

    def __preprocess_learning_curves(
        self, learning_curves: list[list[float]], padding_value: float = 0.0
    ) -> np.ndarray:
        # Add padding to the learning curves to make them the same size

        # Get max learning_curve length
        max_length = 0
        for lc in learning_curves:
            length = len(lc)
            if length > max_length:
                max_length = length

        for lc in learning_curves:
            # add padding to the learning curve to fit the cnn kernel or
            # the max_length depending on which is the largest
            padding_length = max([max_length - len(lc), self.cnn_kernel_size - len(lc)])
            lc.extend([padding_value] * padding_length)

        # TODO: check if the lc values are within bounds [0, 1] (karibbov)
        # TODO: add normalize_lcs option in the future

        return np.array(learning_curves, dtype=np.single)

    def __reset_xy(
        self,
        x_train: list[SearchSpace],
        y_train: list[float],
        learning_curves: list[list[float]],
        normalize_y: bool = False,
        normalize_budget: bool = True,
    ):
        self.normalize_budget = (  # pylint: disable=attribute-defined-outside-init
            normalize_budget
        )
        self.normalize_y = normalize_y  # pylint: disable=attribute-defined-outside-init

        x_train, train_budgets, learning_curves = self._preprocess_input(
            x_train, learning_curves, normalize_budget
        )

        y_train = self._preprocess_y(y_train, normalize_y)

        self.x_train = x_train  # pylint: disable=attribute-defined-outside-init
        self.train_budgets = (  # pylint: disable=attribute-defined-outside-init
            train_budgets
        )
        self.learning_curves = (  # pylint: disable=attribute-defined-outside-init
            learning_curves
        )
        self.y_train = y_train  # pylint: disable=attribute-defined-outside-init

    def _preprocess_input(
        self,
        x: list[SearchSpace],
        learning_curves: list[list[float]],
        normalize_budget: bool = True,
    ):
        budgets = self.__extract_budgets(x, normalize_budget)
        learning_curves = self.__preprocess_learning_curves(learning_curves)

        x = np.array([self.__encode_config(config) for config in x], dtype=np.single)

        x = torch.tensor(x).to(device=self.device)
        budgets = torch.tensor(budgets).to(device=self.device)
        learning_curves = torch.tensor(learning_curves).to(device=self.device)

        return x, budgets, learning_curves

    def _preprocess_y(self, y_train: list[float], normalize_y: bool = False):
        y_train_array = np.array(y_train, dtype=np.single)
        self.min_y = y_train_array.min()  # pylint: disable=attribute-defined-outside-init
        self.max_y = y_train_array.max()  # pylint: disable=attribute-defined-outside-init
        if normalize_y:
            y_train_array = (y_train_array - self.min_y) / (self.max_y - self.min_y)
        y_train_array = torch.tensor(y_train_array).to(device=self.device)
        return y_train_array

    def fit(
        self,
        x_train: list[SearchSpace],
        y_train: list[float],
        learning_curves: list[list[float]],
    ):
        self._fit(x_train, y_train, learning_curves, **self.surrogate_model_fit_args)

    def _fit(
        self,
        x_train: list[SearchSpace],
        y_train: list[float],
        learning_curves: list[list[float]],
        normalize_y: bool = False,
        normalize_budget: bool = True,
        n_epochs: int = 1000,
        batch_size: int = 64,
        optimizer_args: dict | None = None,
        early_stopping: bool = True,
        patience: int = 10,
    ):
        self.__reset_xy(
            x_train,
            y_train,
            learning_curves,
            normalize_y=normalize_y,
            normalize_budget=normalize_budget,
        )

        self.model.to(self.device)
        self.likelihood.to(self.device)
        self.nn.to(self.device)

        self.__train_model(
            self.x_train,
            self.train_budgets,
            self.learning_curves,
            self.y_train,
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_args=optimizer_args,
            early_stopping=early_stopping,
            patience=patience,
        )

    def __train_model(
        self,
        x_train: torch.Tensor,
        train_budgets: torch.Tensor,
        learning_curves: torch.Tensor,
        y_train: torch.Tensor,
        n_epochs: int = 1000,
        batch_size: int = 64,
        optimizer_args: dict | None = None,
        early_stopping: bool = True,
        patience: int = 10,
    ):
        if optimizer_args is None:
            optimizer_args = {"lr": 0.1}

        self.model.train()
        self.likelihood.train()
        self.nn.train()
        self.optimizer = (  # pylint: disable=attribute-defined-outside-init
            torch.optim.Adam(
                [
                    dict({"params": self.model.parameters()}, **optimizer_args),
                    dict({"params": self.nn.parameters()}, **optimizer_args),
                ]
            )
        )

        count_down = patience
        min_avg_loss_val = np.inf
        average_loss: float = 0.0

        for epoch_nr in range(0, n_epochs):
            if early_stopping and count_down == 0:
                self.logger.info(
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
            for batch_idx, start_index in enumerate(
                range(0, n_examples_batch, batch_size)
            ):
                end_index = start_index + batch_size
                if end_index > n_examples_batch:
                    end_index = n_examples_batch
                indices = permutation[start_index:end_index]
                batch_x, batch_budget, batch_lc, batch_y = (
                    x_train[indices],
                    train_budgets[indices],
                    learning_curves[indices],
                    y_train[indices],
                )

                minibatch_size = end_index - start_index
                # if only one example in the batch, skip the batch.
                # Otherwise, the code will fail because of batchnorm
                if minibatch_size <= 1:
                    continue

                # Zero backprop gradients
                self.optimizer.zero_grad()

                projected_x = self.nn(batch_x, batch_budget, batch_lc)
                self.model.set_train_data(projected_x, batch_y, strict=False)
                output = self.model(projected_x)

                # try:
                # Calc loss and backprop derivatives
                loss = -self.mll(output, self.model.train_targets)
                episodic_loss_value: float = loss.detach().to("cpu").item()
                # weighted sum over losses in the batch
                total_scaled_loss = (
                    total_scaled_loss + episodic_loss_value * minibatch_size
                )

                mse = gpytorch.metrics.mean_squared_error(
                    output, self.model.train_targets
                )
                self.logger.debug(
                    f"Epoch {epoch_nr}  Batch {batch_idx} - MSE {mse:.5f}, "
                    f"Loss: {episodic_loss_value:.3f}, "
                    f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f}, "
                    f"noise: {self.model.likelihood.noise.item():.3f}, "
                )

                loss.backward()
                self.optimizer.step()

            # Get average weighted loss over every batch
            average_loss = total_scaled_loss / n_examples_batch
            if average_loss < min_avg_loss_val:
                min_avg_loss_val = average_loss
                count_down = patience
            elif early_stopping:
                self.logger.debug(
                    f"No improvement over the minimum loss value of {min_avg_loss_val} "
                    f"for the past {patience - count_down} epochs "
                    f"the training will stop in {count_down} epochs"
                )
                count_down -= 1
            # except Exception as training_error:
            #     self.logger.error(
            #         f'The following error happened while training: {training_error}')
            #     # An error has happened, trigger the restart of the optimization and restart
            #     # the model with default hyperparameters.
            #     self.restart = True
            #     training_errored = True
            #     break

    def set_prediction_learning_curves(self, learning_curves: list[list[float]]):
        # pylint: disable=attribute-defined-outside-init
        self.prediction_learning_curves = learning_curves
        # pylint: enable=attribute-defined-outside-init

    def predict(
        self, x: list[SearchSpace], learning_curves: list[list[float]] | None = None
    ):
        # Preprocess input
        if learning_curves is None:
            learning_curves = self.prediction_learning_curves
        x_test, test_budgets, learning_curves = self._preprocess_input(
            x, learning_curves, self.normalize_budget
        )

        self.model.eval()
        self.nn.eval()
        self.likelihood.eval()

        with torch.no_grad():
            projected_train_x = self.nn(
                self.x_train, self.train_budgets, self.learning_curves
            )
            self.model.set_train_data(
                inputs=projected_train_x, targets=self.y_train, strict=False
            )

            projected_test_x = self.nn(x_test, test_budgets, learning_curves)

            preds = self.likelihood(self.model(projected_test_x))

        means = preds.mean.detach()

        if self.normalize_y:
            means = (means + self.min_y) * (self.max_y - self.min_y)

        cov = torch.diag(torch.pow(preds.stddev.detach(), 2))

        return means, cov


if __name__ == "__main__":
    print(torch.version.__version__)

    pipe_space = SearchSpace(
        float_=FloatParameter(lower=0.0, upper=5.0),
        e=IntegerParameter(lower=0, upper=10, is_fidelity=True),
    )

    configs = [pipe_space.sample(ignore_fidelity=False) for _ in range(100)]

    y = np.random.random(100).tolist()

    lcs = [
        np.random.random(size=np.random.randint(low=1, high=50)).tolist()
        for _ in range(100)
    ]

    deep_gp = DeepGP(pipe_space, neural_network_args={})

    deep_gp.fit(x_train=configs, learning_curves=lcs, y_train=y)

    means, stds = deep_gp.predict(configs, lcs)

    print(list(zip(means, y)))
    print(stds)
