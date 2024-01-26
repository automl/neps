"""
ONLY A LOCAL COPY ON THE CLUSTER WORKSPACE
"""

from __future__ import annotations

import logging
import psutil

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

import pfns4hpo  # DO NOT PUSH TO MASTER WITHOUT RESOLVING THIS DEPENDENCY


class PFN_SURROGATE:
    """
    PFN model
    """

    def __init__(
        self,
        pipeline_space: SearchSpace,
        logger=None,
        surrogate_model_fit_args: dict = None,
        model_name: str = None,
        minimize: bool = True,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.minimize = minimize
        self.model_name = model_name
        self.__preprocess_search_space(pipeline_space)
        # set the categories array for the encoder
        self.categories_array = np.array(self.categories)

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # self.device = torch.device("cpu")

        # build the neural network
        self.nn = pfns4hpo.PFN_MODEL(model_name).to(self.device)
        self.logger = logger or logging.getLogger("neps")

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
        self.input_size = parameter_count + 1
        self.continuous_params_size = self.input_size - len(self.categories)

        self.min_fidelity = pipeline_space.fidelity.lower
        self.max_fidelity = pipeline_space.fidelity.upper

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor):
        # setting train_x, train_y, does no actual fitting
        self.train_x = train_x
        self.train_y = (1 - train_y) if self.minimize else train_y

    def predict(self, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor):
        self.nn.eval()

        with torch.no_grad():
            means, variances = self.nn.predict_mean_variance(
                x_train=train_x, 
                y_train=(1 - train_y) if self.minimize else train_y,  # PFN model trained on accuracy, NePS minimizes
                x_test=test_x,
            )
        cov = torch.diag(variances.flatten()).detach()
        mean = means.flatten().detach()
        if self.minimize:
            mean = 1 - mean

        return mean, cov

    @torch.no_grad()
    def get_ucb(self, x_test, beta: float=(1-.682)/2, x_train: torch.Tensor=None, y_train: torch.Tensor=None):
        # y values are always transformed for maximizing
        x_train = self.train_x if x_train is None else x_train
        y_train = self.train_y if y_train is None else ((1 - y_train) if self.minimize else y_train)
        logits = self.nn(x_train=x_train, y_train=y_train, x_test=x_test)
        ucb = self.nn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=True
        )
        return ucb
    
    @torch.no_grad()
    def get_lcb(self, x_test, beta: float=(1-.682)/2, x_train: torch.Tensor=None, y_train: torch.Tensor=None):
        # y values are always transformed for maximizing
        x_train = self.train_x if x_train is None else x_train
        y_train = self.train_y if y_train is None else ((1 - y_train) if self.minimize else y_train)
        logits = self.nn(x_train=x_train, y_train=y_train, x_test=x_test)
        lcb = self.nn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=False  # IMPORTANT to be False, should calculate the LCB using the lower-bound ICDF as per beta
        )
        return lcb

    @torch.no_grad()
    def get_ei(self, x_test, inc, x_train=None, y_train=None):
        inc = inc.unsqueeze(1).to(self.device)
        x_train = self.train_x if x_train is None else x_train
        y_train = self.train_y if y_train is None else ((1 - y_train) if self.minimize else y_train)
        logits = self.nn(x_train=x_train, y_train=y_train, x_test=x_test)  # torch.Size([x_train.shape[0], 1, 10000])
        scores = self.nn.model.criterion.ei(logits, best_f=((1 - inc) if self.minimize else inc))
        return scores

    @torch.no_grad()
    def get_pi(self, x_test, inc, x_train=None, y_train=None):
        inc = inc.unsqueeze(1).to(self.device)
        x_train = self.train_x if x_train is None else x_train
        y_train = self.train_y if y_train is None else ((1 - y_train) if self.minimize else y_train)
        logits = self.nn(x_train=x_train, y_train=y_train, x_test=x_test)  # torch.Size([x_train.shape[0], 1, 10000])
        scores = self.nn.model.criterion.pi(logits.squeeze(), best_f=((1 - inc) if self.minimize else inc))
        return scores

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

    pfn = PFN_SURROGATE(pipe_space)

    pfn.fit(x_train=configs, learning_curves=lcs, y_train=y)

    means, vars = pfn.predict(configs, lcs)

    print(list(zip(means, y)))
    print(vars)
