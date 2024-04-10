from __future__ import annotations

import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

from ....search_spaces.search_space import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    SearchSpace,
)
from ...multi_fidelity.utils import MFObservedData


# TODO: Move to utils
def get_optimizer_losses(root_directory: Path | str) -> list[float]:
    all_losses_file = root_directory / "all_losses_and_configs.txt"

    # Read all losses from the file in the order they are explored
    losses = [
        float(line[6:])
        for line in all_losses_file.read_text(encoding="utf-8").splitlines()
        if "Loss: " in line
    ]
    return losses


def get_best_loss(root_directory: Path | str) -> float:
    root_directory = Path(root_directory)
    best_loss_fiel = root_directory / "best_loss_trajectory.txt"

    # Get the best seen loss value
    best_loss = float(best_loss_fiel.read_text(encoding="utf-8").splitlines()[-1].strip())

    return best_loss


class ConditionedPowerLaw(nn.Module):
    def __init__(
        self,
        nr_initial_features=10,
        nr_units=200,
        nr_layers=3,
        use_learning_curve: bool = True,
        kernel_size: int = 3,
        nr_filters: int = 4,
        nr_cnn_layers: int = 2,
    ):
        """
        Args:
            nr_initial_features: int
                The number of features per example.
            nr_units: int
                The number of units for every layer.
            nr_layers: int
                The number of layers for the neural network.
            use_learning_curve: bool
                If the learning curve should be use in the network.
            kernel_size: int
                The size of the kernel that is applied in the cnn layer.
            nr_filters: int
                The number of filters that are used in the cnn layers.
            nr_cnn_layers: int
                The number of cnn layers to be used.
        """
        super().__init__()

        self.use_learning_curve = use_learning_curve
        self.kernel_size = kernel_size
        self.nr_filters = nr_filters
        self.nr_cnn_layers = nr_cnn_layers

        self.act_func = torch.nn.LeakyReLU()
        self.last_act_func = torch.nn.GLU()
        self.tan_func = torch.nn.Tanh()
        self.batch_norm = torch.nn.BatchNorm1d

        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = nr_initial_features
        if self.use_learning_curve:
            nr_initial_features = nr_initial_features + nr_filters

        layers.append(nn.Linear(nr_initial_features, nr_units))
        layers.append(self.act_func)

        for i in range(2, nr_layers + 1):
            layers.append(nn.Linear(nr_units, nr_units))
            layers.append(self.act_func)

        last_layer = nn.Linear(nr_units, 3)
        layers.append(last_layer)

        self.layers = torch.nn.Sequential(*layers)

        cnn_part = []
        if use_learning_curve:
            cnn_part.append(
                nn.Conv1d(
                    in_channels=2,
                    kernel_size=(self.kernel_size,),
                    out_channels=self.nr_filters,
                ),
            )
            for i in range(1, self.nr_cnn_layers):
                cnn_part.append(self.act_func)
                cnn_part.append(
                    nn.Conv1d(
                        in_channels=self.nr_filters,
                        kernel_size=(self.kernel_size,),
                        out_channels=self.nr_filters,
                    ),
                ),
            cnn_part.append(nn.AdaptiveAvgPool1d(1))

        self.cnn = nn.Sequential(*cnn_part)

    def forward(
        self,
        x: torch.Tensor,
        predict_budgets: torch.Tensor,
        evaluated_budgets: torch.Tensor,
        learning_curves: torch.Tensor,
    ):
        """
        Args:
            x: torch.Tensor
                The examples.
            predict_budgets: torch.Tensor
                The budgets for which the performance will be predicted for the
                hyperparameter configurations.
            evaluated_budgets: torch.Tensor
                The budgets for which the hyperparameter configurations have been
                evaluated so far.
            learning_curves: torch.Tensor
                The learning curves for the hyperparameter configurations.
        """
        # print(x.shape)
        # print(learning_curves.shape)
        # x = torch.cat((x, torch.unsqueeze(evaluated_budgets, 1)), dim=1)
        if self.use_learning_curve:
            lc_features = self.cnn(learning_curves)
            # print(lc_features.shape)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)
            # print(lc_features)
            x = torch.cat((x, lc_features), dim=1)
        # print(x.shape)
        if torch.any(torch.isnan(x)):
            raise ValueError("NaN values in input, the network probably diverged")
        x = self.layers(x)
        alphas = x[:, 0]
        betas = x[:, 1]
        gammas = x[:, 2]
        # print(x)
        output = torch.add(
            alphas,
            torch.mul(
                self.last_act_func(torch.cat((betas, betas))),
                torch.pow(
                    predict_budgets,
                    torch.mul(self.last_act_func(torch.cat((gammas, gammas))), -1),
                ),
            ),
        )

        return output


ModelClass = ConditionedPowerLaw

MODEL_MAPPING: dict[str, type[ModelClass]] = {"power_law": ConditionedPowerLaw}


class PowerLawSurrogate:
    # defaults to be used for functions
    # fit params
    default_lr = 0.001
    default_batch_size = 64
    default_nr_epochs = 250
    default_refine_epochs = 20
    default_early_stopping = False
    default_early_stopping_patience = 10

    # init params
    default_n_initial_full_trainings = 10
    default_n_models = 5
    default_model_config = dict(
        nr_units=128,
        nr_layers=2,
        use_learning_curve=False,
        kernel_size=3,
        nr_filters=4,
        nr_cnn_layers=2,
    )

    # fit+predict params
    default_padding_type = "zero"
    default_budget_normalize = True
    default_use_min_budget = False
    default_y_normalize = False

    # Defined in __init__(...)
    default_no_improvement_patience = ...

    def __init__(
        self,
        pipeline_space: SearchSpace,
        observed_data: MFObservedData | None = None,
        logger=None,
        surrogate_model_fit_args: dict | None = None,
        # IMPORTANT: Checkpointing does not use file locking,
        # IMPORTANT: hence, it is not suitable for multiprocessing settings
        # IMPORTANT: For parallel runs lock the checkpoint file during the whole training
        checkpointing: bool = False,
        root_directory: Path | str | None = None,
        # IMPORTANT: For parallel runs use a different checkpoint_file name for each
        # IMPORTANT: surrogate. This makes sure that parallel runs don't override each
        # IMPORTANT: others saved checkpoint. Although they will still have some conflicts due to
        # IMPORTANT: global optimizer step tracking
        checkpoint_file: Path | str = "surrogate_checkpoint.pth",
        refine_epochs: int = default_refine_epochs,
        n_initial_full_trainings: int = default_n_initial_full_trainings,
        default_model_class: str = "power_law",
        default_model_config: dict[str, Any] = default_model_config,
        n_models: int = default_n_models,
        model_classes: list[str] | None = None,
        model_configs: list[dict[str, Any]] | None = None,
        refine_batch_size: int | None = None,
    ):
        if pipeline_space.has_tabular:
            self.cover_pipeline_space = pipeline_space
            self.real_pipeline_space = pipeline_space.raw_tabular_space
        else:
            self.cover_pipeline_space = pipeline_space
            self.real_pipeline_space = pipeline_space
        # self.pipeline_space = pipeline_space

        self.observed_data = observed_data
        self.__preprocess_search_space(self.real_pipeline_space)
        self.seeds = np.random.choice(100, n_models, replace=False)
        self.model_configs = (
            [dict(nr_initial_features=self.input_size, **default_model_config)] * n_models
            if not model_configs
            else model_configs
        )
        self.model_classes = (
            [MODEL_MAPPING[default_model_class]] * n_models
            if not model_classes
            else [MODEL_MAPPING[m_class] for m_class in model_classes]
        )
        self.device = "cpu"
        self.models: list[ModelClass] = [
            self.__initialize_model(config, self.model_classes[index], self.device)
            for index, config in enumerate(self.model_configs)
        ]

        self.checkpointing = checkpointing
        self.refine_epochs = refine_epochs
        self.refine_batch_size = refine_batch_size
        self.n_initial_full_trainings = n_initial_full_trainings
        self.default_no_improvement_patience = int(
            self.max_fidelity + 0.2 * self.max_fidelity
        )

        if checkpointing:
            assert (
                root_directory is not None
            ), "neps root_directory must be provided for the checkpointing"
            self.root_dir = Path(os.getcwd(), root_directory)
            self.checkpoint_path = Path(os.getcwd(), root_directory, checkpoint_file)

        self.surrogate_model_fit_args = (
            surrogate_model_fit_args if surrogate_model_fit_args is not None else {}
        )

        if self.surrogate_model_fit_args.get("no_improvement_patience", None) is None:
            # To replicate how the original DPL implementation handles the
            # no_improvement_threshold
            self.surrogate_model_fit_args[
                "no_improvement_patience"
            ] = self.default_no_improvement_patience

        self.categories_array = np.array(self.categories)

        self.best_state = None
        self.prediction_learning_curves = None

        self.criterion = torch.nn.L1Loss()

        self.logger = logger or logging.getLogger("neps")

    def __preprocess_search_space(self, pipeline_space: SearchSpace):
        self.categories = []
        self.categorical_hps = []

        parameter_count = 0
        for hp_name, hp in pipeline_space.items():
            # Collect all categories in a list for the encoder
            if hp.is_fidelity:
                continue  # Ignore fidelity
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

    def __encode_config(self, config: SearchSpace) -> np.ndarray:
        categorical_encoding = np.zeros_like(self.categories_array, dtype=np.single)
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

    def __normalize_budgets(
        self, budgets: np.ndarray, use_min_budget: bool
    ) -> np.ndarray:
        min_budget = self.min_fidelity if use_min_budget else 0
        normalized_budgets = (budgets - min_budget) / (self.max_fidelity - min_budget)
        return normalized_budgets

    def __extract_budgets(
        self, x_train: list[SearchSpace], normalized: bool, use_min_budget: bool
    ) -> np.ndarray:
        budgets = np.array([config.fidelity.value for config in x_train], dtype=np.single)

        if normalized:
            budgets = self.__normalize_budgets(budgets, use_min_budget)
        return budgets

    def __preprocess_learning_curves(
        self, learning_curves: list[list[float]], padding_type: str
    ) -> np.ndarray:
        # Add padding to the learning curves to make them the same size
        existing_values_mask = []
        max_length = self.max_fidelity - 1

        if padding_type == "last":
            init_value = self.__get_mean_initial_value()
        else:
            init_value = 0.0

        for lc in learning_curves:
            if len(lc) == 0:
                padding_value = init_value
            elif padding_type == "last":
                padding_value = lc[-1]
            else:
                padding_value = 0.0

            padding_length = int(max_length - len(lc))

            mask = [1] * len(lc) + [0] * padding_length
            existing_values_mask.append(mask)

            lc.extend([padding_value] * padding_length)
        # print(learning_curves)
        learning_curves = np.array(learning_curves, dtype=np.single)
        existing_values_mask = np.array(existing_values_mask, dtype=np.single)

        learning_curves = np.stack((learning_curves, existing_values_mask), axis=1)

        return learning_curves

    def __reset_xy(
        self,
        x_train: list[SearchSpace],
        y_train: list[float],
        learning_curves: list[list[float]],
        normalize_y: bool = default_y_normalize,
        normalize_budget: bool = default_budget_normalize,
        use_min_budget: bool = default_use_min_budget,
        padding_type: str = default_padding_type,
    ):
        self.normalize_budget = (  # pylint: disable=attribute-defined-outside-init
            normalize_budget
        )
        self.use_min_budget = (  # pylint: disable=attribute-defined-outside-init
            use_min_budget
        )
        self.padding_type = padding_type  # pylint: disable=attribute-defined-outside-init
        self.normalize_y = normalize_y  # pylint: disable=attribute-defined-outside-init

        x_train, train_budgets, learning_curves = self._preprocess_input(
            x_train,
            learning_curves,
            self.normalize_budget,
            self.use_min_budget,
            self.padding_type,
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
        normalize_budget: bool,
        use_min_budget: bool,
        padding_type: str,
    ) -> [torch.tensor, torch.tensor, torch.tensor]:
        budgets = self.__extract_budgets(x, normalize_budget, use_min_budget)
        learning_curves = self.__preprocess_learning_curves(learning_curves, padding_type)

        x = np.array([self.__encode_config(config) for config in x], dtype=np.single)

        x = torch.tensor(x).to(device=self.device)
        budgets = torch.tensor(budgets).to(device=self.device)
        learning_curves = torch.tensor(learning_curves).to(device=self.device)

        return x, budgets, learning_curves

    def _preprocess_y(self, y_train: list[float], normalize_y: bool) -> torch.tensor:
        y_train_array = np.array(y_train, dtype=np.single)
        self.min_y = y_train_array.min()  # pylint: disable=attribute-defined-outside-init
        self.max_y = y_train_array.max()  # pylint: disable=attribute-defined-outside-init
        if normalize_y:
            y_train_array = (y_train_array - self.min_y) / (self.max_y - self.min_y)
        y_train_array = torch.tensor(y_train_array).to(device=self.device)
        return y_train_array

    def __is_refine(self, no_improvement_patience: int) -> bool:
        losses = get_optimizer_losses(self.root_dir)

        best_loss = get_best_loss(self.root_dir)

        total_optimizer_steps = len(losses)

        # Count the non-improvement
        non_improvement_steps = 0
        for loss in reversed(losses):
            if np.greater(loss, best_loss):
                non_improvement_steps += 1
            else:
                break

        self.logger.debug(f"No improvement for: {non_improvement_steps} evaulations")

        return (non_improvement_steps < no_improvement_patience) and (
            self.n_initial_full_trainings <= total_optimizer_steps
        )

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
        nr_epochs: int = default_nr_epochs,
        batch_size: int = default_batch_size,
        early_stopping: bool = default_early_stopping,
        early_stopping_patience: int = default_early_stopping_patience,
        no_improvement_patience: int = default_no_improvement_patience,
        optimizer_args: dict[str, Any] | None = None,
        normalize_y: bool = default_y_normalize,
        normalize_budget: bool = default_budget_normalize,
        use_min_budget: bool = default_use_min_budget,
        padding_type: str = default_padding_type,
    ):
        self.__reset_xy(
            x_train,
            y_train,
            learning_curves,
            normalize_y=normalize_y,
            normalize_budget=normalize_budget,
            use_min_budget=use_min_budget,
            padding_type=padding_type,
        )
        # check when to refine
        if (
            self.checkpointing
            and self.__is_refine(no_improvement_patience)
            and self.checkpoint_path.exists()
        ):
            # self.__initialize_model()
            self.load_state()
            weight_new_point = True
            nr_epochs = self.refine_epochs
            batch_size = self.refine_batch_size if self.refine_batch_size else batch_size
        else:
            weight_new_point = False

        if optimizer_args is None:
            optimizer_args = {"lr": self.default_lr}

        for model_index, model in enumerate(self.models):
            self._train_a_model(
                model_index,
                self.x_train,
                self.train_budgets,
                self.y_train,
                self.learning_curves,
                nr_epochs=nr_epochs,
                batch_size=batch_size,
                early_stopping_patience=early_stopping_patience,
                early_stopping=early_stopping,
                weight_new_point=weight_new_point,
                optimizer_args=optimizer_args,
            )

        # save model after training if checkpointing
        if self.checkpointing:
            self.save_state()

    def _train_a_model(
        self,
        model_index: int,
        x_train: torch.tensor,
        train_budgets: torch.tensor,
        y_train: torch.tensor,
        learning_curves: torch.tensor,
        nr_epochs: int,
        batch_size: int,
        early_stopping_patience: int,
        early_stopping: bool,
        weight_new_point: bool,
        optimizer_args: dict[str, Any],
    ):
        # Setting seeds will interfere with SearchSpace random sampling
        if self.cover_pipeline_space.has_tabular:
            seed = self.seeds[model_index]
            torch.manual_seed(seed)
            np.random.seed(seed)

        model = self.models[model_index]

        optimizer = torch.optim.Adam(
            **dict({"params": model.parameters()}, **optimizer_args)
        )

        count_down = early_stopping_patience
        best_loss = np.inf
        best_state = deepcopy(model.state_dict())

        model.train()

        if weight_new_point:
            new_x, new_b, new_lc, new_y = self.prep_new_point()
        else:
            new_x, new_b, new_lc, new_y = [torch.tensor([])] * 4

        for epoch in range(0, nr_epochs):
            if early_stopping and count_down == 0:
                self.logger.info(
                    f"Epoch: {epoch - 1} surrogate training stops due to early "
                    f"stopping with the patience: {early_stopping_patience} and "
                    f"the minimum average loss of {best_loss} and "
                    f"the final average loss of {best_loss}"
                )
                model.load_state_dict(best_state)
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

                if weight_new_point:
                    batch_x = torch.cat((batch_x, new_x))
                    batch_budget = torch.cat((batch_budget, new_b))
                    batch_lc = torch.cat((batch_lc, new_lc))
                    batch_y = torch.cat((batch_y, new_y))

                    # increase the batchsize
                    minibatch_size += new_x.shape[0]

                # if only one example in the batch, skip the batch.
                # Otherwise, the code will fail because of batchnorm
                if minibatch_size <= 1:
                    continue

                # Zero backprop gradients
                optimizer.zero_grad(set_to_none=True)

                outputs = model(batch_x, batch_budget, batch_budget, batch_lc)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_scaled_loss += loss.detach().item() * minibatch_size

            running_loss = total_scaled_loss / n_examples_batch

            if running_loss < best_loss:
                best_loss = running_loss
                count_down = early_stopping_patience
                best_state = deepcopy(model.state_dict())
            elif early_stopping:
                self.logger.debug(
                    f"No improvement over the minimum loss value of {best_loss} "
                    f"for the past {early_stopping_patience - count_down} epochs "
                    f"the training will stop in {count_down} epochs"
                )
                count_down -= 1
        if early_stopping:
            model.load_state_dict(best_state)
        return model

    def set_prediction_learning_curves(self, learning_curves: list[list[float]]):
        # pylint: disable=attribute-defined-outside-init
        self.prediction_learning_curves = learning_curves
        # pylint: enable=attribute-defined-outside-init

    def predict(
        self,
        x: list[SearchSpace],
        learning_curves: list[list[float]] | None = None,
        real_budgets: list[int | float] | None = None,
    ) -> [torch.tensor, torch.tensor]:
        # Preprocess input
        # [print(_x.hp_values()) for _x in x]
        if learning_curves is None:
            learning_curves = self.prediction_learning_curves

        if real_budgets is None:
            # Get the list of budgets the configs are evaluated for
            real_budgets = [len(lc) + 1 for lc in learning_curves]

        x_test, prediction_budgets, learning_curves = self._preprocess_input(
            x,
            learning_curves,
            self.normalize_budget,
            self.use_min_budget,
            self.padding_type,
        )
        # preprocess the list of budgets the configs are evaluated for
        real_budgets = np.array(real_budgets, dtype=np.single)
        real_budgets = self.__normalize_budgets(real_budgets, self.use_min_budget)
        real_budgets = torch.tensor(real_budgets).to(self.device)

        all_predictions = []
        for model in self.models:
            model.eval()

            preds = model(x_test, prediction_budgets, real_budgets, learning_curves)
            all_predictions.append(preds.detach().cpu().numpy())

        means = torch.tensor(np.mean(all_predictions, axis=0)).cpu()
        std_predictions = np.std(all_predictions, axis=0)
        cov = torch.diag(torch.tensor(np.power(std_predictions, 2))).cpu()

        return means, cov

    def load_state(self, state: dict[str, int | str | dict[str, Any]] | None = None):
        # load and save last evaluated config as well
        if state is None:
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = state

        self.last_point = checkpoint["last_point"]

        for model_index in range(checkpoint["n_models"]):
            self.models[model_index].load_state_dict(
                checkpoint[f"model_{model_index}_state_dict"]
            )
            self.models[model_index].to(self.device)

    def get_state(self) -> dict[str, int | str | dict[str, Any]]:
        n_models = len(self.models)
        model_states = {
            f"model_{model_index}_state_dict": deepcopy(
                self.models[model_index].state_dict()
            )
            for model_index in range(n_models)
        }

        # get last point
        last_point = self.get_last_point()
        current_state = dict(n_models=n_models, last_point=last_point, **model_states)

        return current_state

    def __config_ids(self) -> list[str]:
        # Parallelization issues
        all_losses_file = self.root_dir / "all_losses_and_configs.txt"

        if all_losses_file.exists():
            # Read all losses from the file in the order they are explored
            config_ids = [
                str(line[11:])
                for line in all_losses_file.read_text(encoding="utf-8").splitlines()
                if "Config ID: " in line
            ]
        else:
            config_ids = []

        return config_ids

    def save_state(self, state: dict[str, int | str | dict[str, Any]] | None = None):
        # TODO: save last evaluated config as well
        if state is None:
            torch.save(
                self.get_state(),
                self.checkpoint_path,
            )
        else:
            assert (
                "last_point" in state and "n_models" in state
            ), "The state dictionary is not complete"
            torch.save(
                state,
                self.checkpoint_path,
            )

    def get_last_point(self) -> str:
        # Only for single worker case
        last_config_id = self.__config_ids()[-1]
        # For parallel runs
        # get the last config_id that's also in self.observed_configs
        return last_config_id

    def get_new_points(self) -> [list[SearchSpace], list[list[float]], list[float]]:
        # Get points that haven't been trained on before

        config_ids = self.__config_ids()

        if self.last_point:
            index = config_ids.index(self.last_point) + 1
        else:
            index = len(config_ids) - 1

        new_config_indices = [
            tuple(map(int, config_id.split("_"))) for config_id in config_ids[index:]
        ]

        # Only include the points that exist in the observed data already
        # (not a use case for single worker runs)
        existing_index_map = self.observed_data.df.index.isin(new_config_indices)

        new_config_df = self.observed_data.df.loc[existing_index_map, :].copy(deep=True)

        new_configs, new_lcs, new_y = self.observed_data.get_training_data_4DyHPO(
            new_config_df, self.cover_pipeline_space
        )

        return new_configs, new_lcs, new_y

    @staticmethod
    def __initialize_model(
        model_params: dict[str, Any], model_class: type[ModelClass], device: str
    ) -> ModelClass:
        model = model_class(**model_params)
        model.to(device)
        return model

    def prep_new_point(self) -> [torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        new_point, new_lc, new_y = self.get_new_points()

        new_x, new_b, new_lc = self._preprocess_input(
            new_point,
            new_lc,
            self.normalize_budget,
            self.use_min_budget,
            self.padding_type,
        )
        new_y = self._preprocess_y(new_y, self.normalize_y)

        return new_x, new_b, new_lc, new_y

    def __get_mean_initial_value(self):
        mean = self.observed_data.get_trajectories().loc[:, 0].mean()

        return mean


if __name__ == "__main__":
    max_fidelity = 50
    pipe_space = SearchSpace(
        float_=FloatParameter(lower=0.0, upper=5.0),
        e=IntegerParameter(lower=1, upper=max_fidelity, is_fidelity=True),
    )

    configs = [pipe_space.sample(ignore_fidelity=False) for _ in range(100)]

    y = np.random.random(100).tolist()

    lcs = [
        np.random.random(size=np.random.randint(low=1, high=max_fidelity)).tolist()
        for _ in range(100)
    ]

    surrogate = PowerLawSurrogate(pipe_space)

    surrogate.fit(x_train=configs, learning_curves=lcs, y_train=y)

    means, stds = surrogate.predict(configs, lcs)

    print(list(zip(means, y)))
    print(stds)
