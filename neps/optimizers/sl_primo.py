from __future__ import annotations

import copy
from collections.abc import (
    Mapping,
    Sequence,
    Sequence as TypeSequence, Callable
)
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from gpytorch.utils.warnings import NumericalWarning

from neps.optimizers.models.gp import (
    encode_trials_for_gp,
    fit_and_acquire_from_gp,
    make_default_single_obj_gp,
)
from neps.optimizers.optimizer import ImportedConfig, SampledConfig
from neps.utils.common import disable_warnings
from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.optimizers.random_search import RandomSearch
    from neps.space import SearchSpace
    from neps.space.encoding import ConfigEncoder
    from neps.state import BudgetInfo, Trial

class ScalingLawSurrogate(nn.Module):
    """
    Models L(x, C) = A(x) * C^(-alpha)
    In Log space: log(L) = log(A(x)) - alpha * log(C)
    
    A(x) is modeled by an MLP. alpha is a learnable parameter.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # MLP to predict log(A(x)) - the 'quality' of the config
        self.quality_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        # Learnable scaling exponent alpha, initialized near 0.5 (inverse square root scaling)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_configs: torch.Tensor, log_flops: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_configs: Tensor of shape (batch, input_dim) - Encoded hyperparameters
            log_flops: Tensor of shape (batch, 1) - Natural log of FLOPs
        Returns:
            Predicted log_loss: Tensor of shape (batch, 1)
        """
        # Predict the quality intercept term log(A(x))
        log_A = self.quality_net(x_configs)
        
        # Apply scaling law: log(L) = log(A) - alpha * log(C)
        # We constrain alpha to be positive usually, but here we let it float
        # or we can force positivity via softplus if needed.
        log_loss = log_A - self.alpha * log_flops
        return log_loss

@dataclass
class SL_PriMO:
    """The Multi objective algorithm for search space including architectural choices."""

    space: SearchSpace
    """The search space to use, without the fidelity."""

    encoder: ConfigEncoder
    """The encoder to use for the search space."""

    initial_design_size: int
    """The number of initial designs to use."""

    # fid_max: int | float
    # """The maximum fidelity value in the BracketOptimizer's search space."""

    # fid_name: str
    # """The name of the fidelity in the BracketOptimizer's search space."""

    get_number_of_parameters: Callable[..., int] # gets the hyperparams selected and spits the num of params
    get_total_flops: Callable[..., int] # gets the hyperparams selected and spits the num of flops for feed forward
    # parameters that have effect on the number of params, then when 
    # we change these we should make sure that the N/D stays in the given range.

    scalarization_weights: dict[str, float] | None = None
    """The scalarization weights to use for the objectives for BO."""

    device: torch.device | None = None
    """The device to use for the GP optimization."""

    priors: Mapping[str, Prior] | None = None
    """The priors to use for this optimizer."""

    n_init_used: int = field(default=0, init=False)
    """The effective number of initial seed configurations used
    for the Bayesian optimization. This refers to the number of
    configurations that were evaluated at the maximum fidelity.
    """

    epsilon: float = 0.25
    """The epsilon value to use for the epsilon-greedy decaying prior-weighted
    acquisition function. This is the probability of not using the prior
    acquisition function.
    """
    scaling_model: ScalingLawSurrogate | None = field(default=None, init=False)
    _is_scaling_model_fitted: bool = field(default=False, init=False)

    def __call__(  # noqa: C901, PLR0912
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        assert n is None, "not supported yet"
        
        # so a simple search in lower budget => max_
        k = 30
        initial_budget = 10
        n = 10000
        
        # ask 100 different configurations randomly find those with less that buget_info/k return value of get_total_flops
        if len(trials) < initial_budget and budget_info is not None and budget_info.cost_to_spend is not None:
            parameters = self.space.searchables # later support fidelity for scaling params
            
            sampler = Prior.from_parameters(parameters)

            configs = sampler.sample(n, to=self.encoder.domains)
            configs_list = self.encoder.decode(configs)
            assert configs.shape[0] == n
            for i in range(n):
                # n_params = self.get_number_of_parameters(configs_list[i])
                total_flops = self.get_total_flops(**configs_list[i])
                if total_flops <= budget_info.cost_to_spend / k:
                    print(f"Conf {configs_list[i]}")
                    conf = configs_list[i]
                    conf.update(self.space.constants)
                    conf.update(
                        {
                            key: value.upper
                            for key, value in self.space.fidelities.items()
                            if key not in conf
                        }
                    )
                    return SampledConfig(id=1, config=conf)
        
        if not self._is_scaling_model_fitted and len(trials) >= self.initial_design_size:
            self.fit_scaling_law(trials)

             # 2. Update the Prior!
            self.create_prior_from_scaling_law(budget_info.cost_to_spend)
        
        evaluated_trial = [trial for trial in trials.values() if trial.report is not None and trial.report.objective_to_minimize is not None]  
        assert len(evaluated_trial) > 0, "There should be some randomly evaluated trials."
        num_objectives = len(evaluated_trial[0].report.objective_to_minimize)


        # Set scalarization weights if not set
        if self.scalarization_weights is None:
            self.scalarization_weights = np.random.uniform(size=num_objectives)
            self.scalarization_weights /= np.sum(self.scalarization_weights)

        # Scalarize trials.report.objective_to_minimize and remove fidelity
        # from the trial configs
        nxt_id = 1
        _trials = {}
        for trial_id, trial in trials.items():
            _trial = copy.deepcopy(trial)
            # Skip trials that are not evaluated at the maximum fidelity
            if _trial.config.get(self.fid_name) != self.fid_max:
                continue
            # Skip trials that are still pending
            if _trial.report is None:
                continue
            assert _trial.report.objective_to_minimize is not None, (
                f"Trial {trial_id} has no objective to minimize."
            )
            assert isinstance(_trial.report.objective_to_minimize, Sequence), (
                "Trial objectives must be a sequence for PriMO, "
                f"got {type(_trial.report.objective_to_minimize)}"
            )

            # Convert the objective to an ndarray and scalarize it
            if not isinstance(_trial.report.objective_to_minimize, np.ndarray):
                _trial.report.objective_to_minimize = np.array(
                    _trial.report.objective_to_minimize
                )
            _trial.report.objective_to_minimize = np.dot(
                _trial.report.objective_to_minimize, self.scalarization_weights
            )

            # Remove the fidelity from the trial config
            # Cannot do simple pop since Config type is Mapping in most places in Neps
            _trial.config = {k: v for k, v in _trial.config.items() if k != self.fid_name}

            _trials[trial_id] = _trial

            # Get the next ID for the sampled configuration
            if "_" in trial_id:
                config_id_str, _, _ = trial_id.split("_")
            else:
                config_id_str = trial_id

            nxt_id = max(nxt_id, int(config_id_str) + 1)

        assert len(_trials) > 0, (
            "No trials found with the maximum fidelity. "
            "Consider increasing the initial design size to run MOASHA longer."
        )

        if self.n_init_used == 0:
            self.n_init_used = len(_trials)

        # Sample new configurations using the Bayesian optimization
        sampled_config = self.sample_using_bo(
            trials=_trials,
            budget_info=budget_info,
            n=n,
        )
        sampled_config.update(
            {
                self.fid_name: self.fid_max,
                **self.space.constants,
            }
        )
        return SampledConfig(id=str(nxt_id), config=sampled_config)

    def fit_scaling_law(self, trials: Mapping[str, Trial]):
        """
        Fits the scaling law model using available trial data.
        """
        print("Fitting Scaling Law Surrogate Model...")
        
        # 1. Prepare Data
        X_list = []
        log_C_list = []
        log_y_list = []

        for trial in trials.values():
            if trial.report is None or trial.report.objective_to_minimize is None:
                continue
            
            # Extract Config (x)
            # encoding expects a list of configs
            # We assume single objective for the scaling fit for simplicity, 
            # or the first objective if multi-obj. 
            loss = trial.report.objective_to_minimize
            if isinstance(loss, Sequence) or isinstance(loss, np.ndarray):
                loss = loss[0] # Take first objective as primary loss for scaling
            
            # Extract Cost (C) - FLOPs
            flops = self.get_total_flops(**trial.config)
            
            # Prepare for Tensor conversion
            X_list.append(trial.config)
            log_C_list.append(np.log(max(flops, 1e-5))) # Avoid log(0)
            log_y_list.append(np.log(max(loss, 1e-5)))

        if len(X_list) < 10:
            print("Not enough data to fit scaling law.")
            return

        # Encode Configs
        X_tensor = self.encoder.encode(X_list).to(self.device).float()
        log_C_tensor = torch.tensor(log_C_list, device=self.device).float().unsqueeze(-1)
        log_y_tensor = torch.tensor(log_y_list, device=self.device).float().unsqueeze(-1)

        # 2. Initialize Model
        input_dim = X_tensor.shape[1]
        if self.scaling_model is None:
            self.scaling_model = ScalingLawSurrogate(input_dim).to(self.device)

        # 3. Training Loop
        optimizer = optim.Adam(self.scaling_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        self.scaling_model.train()
        epochs = 500
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.scaling_model(X_tensor, log_C_tensor)
            loss = loss_fn(preds, log_y_tensor)
            loss.backward()
            optimizer.step()
        
        self.scaling_model.eval()
        self._is_scaling_model_fitted = True
        print(f"Scaling Law Fitted. Scaling Exponent (alpha): {self.scaling_model.alpha.item():.4f}")

    def predict_full_budget_loss(self, config_tensor: torch.Tensor, budget_flops: float) -> torch.Tensor:
        """
        Predicts loss at full budget using the fitted scaling law.
        """
        if not self._is_scaling_model_fitted:
            raise RuntimeError("Scaling model not fitted yet.")
        
        with torch.no_grad():
            log_budget = torch.tensor(np.log(budget_flops), device=self.device).float()
            # Broadcast log_budget to match batch size
            log_budget = log_budget.expand(config_tensor.shape[0], 1)
            
            log_pred = self.scaling_model(config_tensor, log_budget)
            return torch.exp(log_pred)

    def create_prior_from_scaling_law(self, budget_flops: float):
        
        print("Searching Scaling Law for best predicted region")
        
        n_candidates = 5000
        parameters = self.space.searchables
        # Temporary uniform sampler just for scanning
        temp_sampler = Prior.from_parameters(parameters) 
        
        candidates_enc = temp_sampler.sample(n_candidates, to=self.encoder.domains)
        
        # 2. Predict their performance at FULL BUDGET using Scaling Law
        predictions = self.predict_full_budget_loss(candidates_enc, budget_flops)
        
        # 3. Find the best candidate (min loss)
        best_idx = torch.argmin(predictions)
        best_config_enc = candidates_enc[best_idx]
        
        # Decode back to dictionary to get values for the Prior
        # (Since encoder works in batches, we keep it as list of length 1)
        # Note: We need a way to decode a single tensor row. 
        # Assuming encoder.decode accepts tensor (1, D)
        best_config_dict = self.encoder.decode(best_config_enc.unsqueeze(0))[0]
        
        print(f"🌟 Scaling Law identifies promising center: {best_config_dict}")
        
        # 4. Create a CenteredPrior
        # We define a "High" confidence because the scaling law aggregates data 
        # from many points, giving us a strong belief.
        centers = {}
        confidences = {}
        
        for name, value in best_config_dict.items():
            centers[name] = value
            confidences[name] = 0.75 # High confidence
            
        # Use the class method from your first prompt
        new_prior = Prior.from_parameters(
            parameters,
            center_values=centers,
            confidence_values=confidences
        )
        
        self.priors = new_prior
        print("Bayesian Optimization Prior updated to focus on this region.")

    def sample_using_initial_design(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> SampledConfig | list[SampledConfig]:
        """Use the initial design to sample new configurations."""
        assert n is None, "TODO"

        # Sample new configurations using the initial design
        return self.bracket_optimizer(
            trials=trials,
            budget_info=budget_info,
            n=n,
        )

    def sample_using_bo(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None = None,
        n: int | None = None,
    ) -> dict[str, Any]:
        """Use Bayesian optimization to sample new configurations."""
        assert n is None, "TODO"
        n_sampled = len(trials)

        data, encoder = encode_trials_for_gp(
            trials,
            self.space.searchables,
            device=self.device,
            encoder=self.encoder,
        )

        selected_prior = None
        if self.priors is not None:
            selected_prior = np.random.choice(
                list(self.priors.values()),
            )

        selected_prior = np.random.choice(
            [selected_prior, None],
            p=[1 - self.epsilon, self.epsilon],
        )

        # If we should use the prior, weight the acquisition function by
        # the probability of it being sampled from the prior.
        pibo_exp_term = None
        prior = None
        if selected_prior:
            pibo_exp_term = _pibo_exp_term(n_sampled, encoder.ndim, self.n_init_used)
            # If the exp term is insignificant, skip prior acq. weighting
            prior = None if pibo_exp_term < 1e-4 else selected_prior

        n_to_acquire = 1

        gp = make_default_single_obj_gp(x=data.x, y=data.y, encoder=encoder)
        with disable_warnings(NumericalWarning):
            acquisition = qLogNoisyExpectedImprovement(
                model=gp,
                X_baseline=data.x,
                # Unfortunatly, there's no option to indicate that we minimize
                # the AcqFunction so we need to do some kind of transformation.
                # https://github.com/pytorch/botorch/issues/2316#issuecomment-2085964607
                objective=LinearMCObjective(
                    weights=torch.tensor([-1.0], device=self.device)
                ),
                X_pending=data.x_pending,
                prune_baseline=True,
            )
        candidates = fit_and_acquire_from_gp(
            gp=gp,
            x_train=data.x,
            encoder=encoder,
            acquisition=acquisition,
            prior=prior,
            n_candidates_required=n_to_acquire,
            pibo_exp_term=pibo_exp_term,
            hide_warnings=True,
        )

        return encoder.decode_one(candidates)

    def threshold_reached(
        self,
        trials: Mapping[str, Trial],
        threshold: int | float,
    ) -> bool:
        used_fidelity = [
            t.config[self.fid_name] for t in trials.values() if t.report is not None
        ]
        fidelity_units_used = sum(used_fidelity) / self.fid_max
        return fidelity_units_used >= threshold

    def plot_flops_per_objective(self, trials):
        flops = []
        
    
        for trial in trials.values():
            if trial.report is not None:
                config = trial.config
                objectives = trial.report.objective_to_minimize
                flops = self.get_total_flops(**config)

                print(f"Flops: {flops}, Objectives: {objectives}")

    def plot_flop_param_ratio(self, trials):
        # plot
        for trial in trials.values():
            if trial.report is not None:
                config = trial.config
                objectives = trial.report.objective_to_minimize
                flops = self.get_total_flops(**config)
                n_params = self.get_number_of_parameters(**config)
                ratio = flops / n_params
                # plot the points
                
                
                print(f"Flop/Param Ratio: {ratio}, Objectives: {objectives}")

    def calc_elasticity(self):
        elasticity: dict[str, float] = {}
        for param_name, value in self.space.arch_params.items():
            # calculate elasticity of num Flops for each architecture param
            values = []
            flops = []
            for v in range(value.lower, value.upper + 1):
                config = {param_name: v}
                values.append(v)
                flops.append(self.get_total_flops(**config))
            values = np.array(values)
            flops = np.array(flops)
            # compute elasticity using finite differences
            d_flops = np.gradient(flops, values)
            elasticity = (d_flops * values) / flops
            max_elasticity = np.max(np.abs(elasticity))
            if not (0.5 <= max_elasticity <= 2.0):
                raise ValueError(
                    f"Elasticity of number of FLOPs with respect to "
                    f"architecture parameter '{param_name}' is {max_elasticity:.2f}, "
                    f"which is outside the recommended range [0.5, 2.0]. "
                    f"Please adjust the search space to ensure stable optimization."
                )
            elasticity[param_name] = max_elasticity
        return elasticity

def _pibo_exp_term(
    n_sampled_already: int,
    ndims: int,
    initial_design_size: int,
) -> float:
    import math

    n_bo_samples = n_sampled_already - initial_design_size
    return math.exp(-(n_bo_samples**2) / ndims)


