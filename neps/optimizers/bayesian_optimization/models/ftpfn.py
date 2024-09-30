from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from ifbo import FTPFN

from neps.sampling.samplers import Sampler
from neps.search_spaces.domain import Domain
from neps.search_spaces.encoding import CategoricalToUnitNorm, ConfigEncoder
from neps.search_spaces.search_space import SearchSpace
from neps.state.trial import Trial


def _keep_highest_budget_evaluation(
    x: torch.Tensor,
    id_col: int = 0,
    budget_col: int = 1,
) -> torch.Tensor:
    # Does a lexsort, same as if we sorted by (config_id, budget), where
    # theyre are sorted according to increasing config_id and then increasing budget.
    # x[i2] -> sorted by config id and budget
    i1 = torch.argsort(x[:, budget_col])
    i2 = i1[torch.argsort(x[i1][:, id_col], stable=True)]
    sorted_x = x[i2]

    # Now that it's sorted, we essentially want to count the occurence of each id into counts
    _, counts = torch.unique_consecutive(sorted_x[:, id_col], return_counts=True)

    # Now we can use these counts to get to the last occurence of each id
    # The -1 is because we want to index from 0 but sum starts at 1.
    ii = counts.cumsum(0) - 1
    return sorted_x[ii]


def _download_workaround_for_ifbo_issue_10(path: Path | None, version: str) -> Path:
    # TODO: https://github.com/automl/ifBO/issues/10
    import requests
    from ifbo.download import FILE_URL, FILENAME

    target_path = Path(path) if path is not None else Path.cwd().resolve() / ".model"
    target_path.mkdir(parents=True, exist_ok=True)

    _target_zip_path = target_path / FILENAME(version)

    # Just a heuristic check to determine if the model already exists.
    # Kind of hard to know what the name of the extracted file will be
    # Basically we just check if the tar.gz file is there and unpacked.
    # If there is a new version, then it wont exist and we will download it.
    if _target_zip_path.exists() and any(
        p.name.endswith(".pt") for p in target_path.iterdir()
    ):
        return target_path

    _file_url = FILE_URL(version)

    # Download the tar.gz file and decompress it
    response = requests.get(_file_url, allow_redirects=True)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the surrogate model from {_file_url}."
            f" Got status code: {response.status_code}"
        )

    with open(_target_zip_path, "wb") as f:
        try:
            f.write(response.content)
        except Exception as e:
            raise ValueError(
                f"Failed to write the surrogate model to {_target_zip_path}."
            ) from e

    # Decompress the .tar.gz file using tarfile
    import tarfile

    try:
        with tarfile.open(_target_zip_path, "r:gz") as tar:
            tar.extractall(path=target_path)
    except Exception as e:
        raise ValueError(
            f"Failed to decompress the surrogate model at {_target_zip_path}."
        ) from e

    return target_path


def _cast_tensor_shapes(x: torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 3 and x.shape[1] == 1:
        return x
    if len(x.shape) == 2:
        return x.reshape(x.shape[0], 1, x.shape[1])
    if len(x.shape) == 1:
        return x.reshape(x.shape[0], 1)
    raise ValueError(f"Shape not recognized: {x.shape}")


# NOTE: Ifbo was trained using 32 bit
FTPFN_DTYPE = torch.float32


def encode_trials_for_ftpfn(
    trials: Mapping[str, Trial],
    space: SearchSpace,
    budget_domain: Domain,
    encoder: ConfigEncoder,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = FTPFN_DTYPE,
    error_value: float = 0.0,
) -> FTPFNData:
    """Encode the trials into a format that the FTPFN model can understand.

    !!! warning "Pending trials"

        For trials which do not have a loss reported yet, they are considered pending
        and will have `torch.nan` as their score inside the returned y values.
        If using
        [`acquire_next_from_ftpfn()`][neps.optimizers.bayesian_optimization.models.ftpfn.acquire_next_from_ftpfn],
        the result of these configurations will be fantasized.

    !!! warning "Error values"

        The FTPFN model requires that all loss values lie in the interval [0, 1].
        By default, using the value of `error_value=0.0`, we encode crashed configurations as
        having an error value of 0.

    Args:
        trials: The trials to encode
        encoder: The encoder to use
        space: The search space
        budget_domain: The domain to use for the budgets of the FTPFN
        device: The device to use
        dtype: The dtype to use

    Returns:
        The encoded trials and their corresponding **scores**
    """
    # Select all trials which have something we can actually use for modelling
    # The absence of a report signifies pending
    selected = {trial_id: trial for trial_id, trial in trials.items()}
    assert space.fidelity_name is not None
    assert space.fidelity is not None
    assert 0 <= error_value <= 1
    train_configs = encoder.encode([t.config for t in selected.values()], device=device)
    ids = torch.tensor(
        [int(config_id.split("_", maxsplit=1)[0]) for config_id in selected.keys()],
        device=device,
        dtype=dtype,
    )
    # PFN uses `0` id for test configurations
    ids = ids + 1

    train_fidelities = torch.tensor(
        [t.config[space.fidelity_name] for t in selected.values()],
        device=device,
        dtype=dtype,
    )
    train_budgets = budget_domain.cast(train_fidelities, frm=space.fidelity.domain)

    # TODO: Document that it's on the user to ensure these are already all bounded
    # We could possibly include some bounded transform to assert this.
    minimize_ys = torch.tensor(
        [
            torch.nan
            if trial.report is None
            else (error_value if trial.report.loss is None else trial.report.loss)
            for trial in trials.values()
        ],
        device=device,
        dtype=dtype,
    )
    if minimize_ys.max() > 1 or minimize_ys.min() < 0:
        raise RuntimeError(
            "ifBO requires that all loss values reported lie in the interval [0, 1]"
            " but recieved loss value outside of that range!"
            f"\n{minimize_ys}"
        )
    maximize_ys = 1 - minimize_ys
    return FTPFNData(
        ids=ids,
        x=train_configs,
        y=maximize_ys,
        budgets=train_budgets,
        pending_mask=minimize_ys.isnan(),
    )


@dataclass
class FTPFNData:
    """Dataclass to hold the data for the FTPFN model.

    The layout of the data is as follows:

    * `ids`: The configuration ids. These will have +1 added to them as FTPFN uses `0`
    for test configurations, but NePS starts ids at `0`.
    * `x`: The encoded configurations, includes everything that was encoded by the encoder
        passed to
        [`encode_trials_for_ftpfn()`][neps.optimizers.bayesian_optimization.models.ftpfn.encode_trials_for_ftpfn]
    * `y`: The scores of the configurations, these are inverted such they are to be maximized, where 1 is the maximum
        score obtainable and 0 is the minimum. Any configuration which did not have a loss gets a score of `nan`.
    * `budgets`: The budgets of the configurations, normalized to the range [0, 1].
        These are normalized such that the lower bound of the fidelity domain maps to `1/max_fid`
        while the upper bound maps to `1`.
    * `pending_mask`: A mask to indicate which configurations are pending, i.e. have not been evaluated yet.
        If there are no pending configurations, this should be `None`.
    """

    ids: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    budgets: torch.Tensor
    pending_mask: torch.Tensor | None = None


def create_border_configs(
    ndims: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    max_samples: int = 2**9,
) -> torch.Tensor:
    n_samples = 2**ndims
    _arange = torch.arange(n_samples, device=device, dtype=torch.int32)
    # 2**9 is only 512 samples, so we can afford to exhaustively generate them
    # We likely won't have this many hyperparameters anywho
    if n_samples <= max_samples:
        configs = _arange
    else:
        # Otherwise, we take a random sample of the 2**n possible border configs
        rand_uniq_indices = torch.randperm(n_samples, device=device)[:max_samples]
        configs = _arange[rand_uniq_indices]

    # https://stackoverflow.com/a/63546308/5332072
    bit_masks = 2 ** _arange[ndims]
    return configs.unsqueeze(1).bitwise_and(bit_masks).ne(0).to(dtype)


def acquire_next_from_ftpfn(
    *,
    ftpfn: FTPFNSurrogate,
    data: FTPFNData,
    encoder: ConfigEncoder,
    budget_domain: Domain,
    fidelity_domain: Domain,
    seed: int | None = None,
    acq_strategy: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, FTPFNSurrogate], torch.Tensor
    ],
    dtype: torch.dtype | None = FTPFN_DTYPE,
    extra_acq_samples: torch.Tensor | None = None,
) -> tuple[int | None, int | float | None, dict[str, Any]]:
    X = torch.cat([data.ids.unsqueeze(1), data.budgets.unsqueeze(1), data.x], dim=1).to(
        dtype
    )
    ys = data.y.clone().detach()

    # In-fill pending with predicted performance
    if data.pending_mask is not None:
        not_pending = ~data.pending_mask
        pending_ys = ftpfn.get_mean_performance(
            train_x=X[not_pending],
            train_y=ys[not_pending],
            test_x=X[data.pending_mask],
        )
        ys[data.pending_mask] = pending_ys

    # We also need to append existing configurations that are in training data, but bump up their
    # budget by one step.
    # 1. Exclude all configurations which are currently pending
    acq_existing = X
    if data.pending_mask is not None:
        acq_existing = X[~data.pending_mask]

    # 2. Remove duplicate configurations from x train, keeping only the most recent eval
    acq_existing = _keep_highest_budget_evaluation(acq_existing, id_col=0, budget_col=1)

    # 3. Remove configs that have been fully evaluated
    acq_existing = acq_existing[acq_existing[:, 1] < budget_domain.upper]

    # 4. Include the extra acquisition samples
    if extra_acq_samples is None:
        samples = [acq_existing]
    else:
        _shape = (len(extra_acq_samples), 1)
        acq_extra = torch.cat(
            [
                torch.zeros(_shape, dtype=dtype, device=ftpfn.device),
                torch.full(_shape, budget_domain.lower, dtype=dtype, device=ftpfn.device),
                extra_acq_samples,
            ],
            dim=1,
        )
        samples = [acq_existing, acq_extra]

    # 5. Now we can fuse them together
    acq_samples = torch.cat(samples, dim=0).to(dtype=dtype)

    # We keep a copy of the original budgets incase they get modified
    # so we can return the fidelity of the sample that had the best acquisition score
    budgets_prior_to_acq = acq_samples[:, 1].clone().detach()

    # Now we offload acquisition to the caller
    acq_scores = acq_strategy(X, ys, acq_samples, ftpfn)

    # Extract out the row which had the best PI
    best_ix = acq_scores.argmax()

    best_id = int(acq_samples[best_ix, 0].round().item())
    if best_id == 0:  # It was a new acq. sample
        best_real_id = None
        best_fid = None
    else:  # It was a sample to continue, decrement the 1 added earlier
        best_real_id = best_id - 1
        best_fid = fidelity_domain.cast_one(
            budgets_prior_to_acq[best_ix].item(), frm=budget_domain
        )

    best_vector = acq_samples[best_ix, 2:].unsqueeze(0)
    best_config = encoder.decode(best_vector)[0]

    return best_real_id, best_fid, best_config


_CACHED_FTPFN_MODEL: dict[tuple[str, str], FTPFN] = {}


class FTPFNSurrogate:
    """Wrapper around the IfBO model."""

    def __init__(
        self,
        target_path: Path | None = None,
        version: str = "0.0.1",
        device: torch.device | None = None,
    ):
        if target_path is None:
            # TODO: We also probably want to link this to the actual root directory
            # or some shared directory between runs as relying on the path of the initial
            # python invocation is likely to lead to issues somewhere.
            # TODO: ifbo support for windows has issues with decompression
            # We basically just do the same thing they do but manually
            target_path = _download_workaround_for_ifbo_issue_10(target_path, version)

        key = (str(target_path), version)
        ftpfn = _CACHED_FTPFN_MODEL.get(key)
        if ftpfn is None:
            ftpfn = FTPFN(target_path=target_path, version=version, device=device)
            _CACHED_FTPFN_MODEL[key] = ftpfn

        self.ftpfn = ftpfn
        self.device = self.ftpfn.device

    def _get_logits(
        self, train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor
    ) -> torch.Tensor:
        return self.ftpfn.model(
            _cast_tensor_shapes(train_x),
            _cast_tensor_shapes(train_y),
            _cast_tensor_shapes(test_x),
        )

    @torch.no_grad()
    def get_mean_performance(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x).squeeze()
        return self.ftpfn.model.criterion.mean(logits)

    @torch.no_grad()
    def get_pi(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        y_best: torch.Tensor | float,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.pi(logits.squeeze(), best_f=y_best)

    @torch.no_grad()
    def get_ei(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        y_best: torch.Tensor | float,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.ei(logits.squeeze(), best_f=y_best)

    @torch.no_grad()
    def get_lcb(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        beta: float = (1 - 0.682) / 2,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=False,  # IMPORTANT to be False, should calculate the LCB using the lower-bound ICDF as per beta
        )

    @torch.no_grad()
    def get_ucb(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        beta: float = (1 - 0.682) / 2,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=True,  # IMPORTANT to be True, should calculate the UCB using the upper-bound ICDF as per beta
        )
