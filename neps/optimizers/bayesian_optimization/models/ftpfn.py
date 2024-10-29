from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from ifbo import FTPFN

from neps.sampling.priors import Prior

if TYPE_CHECKING:
    from neps.sampling.samplers import Sampler
    from neps.search_spaces.domain import Domain
    from neps.search_spaces.encoding import ConfigEncoder
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

    # Now that it's sorted, we want to count the occurence of each id into counts
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
    response = requests.get(_file_url, allow_redirects=True, timeout=10)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to download the surrogate model from {_file_url}."
            f" Got status code: {response.status_code}"
        )

    with Path(_target_zip_path).open("wb") as f:
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
            # NOTE: There is a filter available from 3.12,
            # Ideally this should be fixed upstream in ifbo.
            # Essentially we'd like to only extract the .pt files
            # and not allow absolute paths
            tar.extractall(path=target_path)  # noqa: S202
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


def encode_ftpfn(
    trials: Mapping[str, Trial],
    space: SearchSpace,
    budget_domain: Domain,
    encoder: ConfigEncoder,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = FTPFN_DTYPE,
    error_value: float = 0.0,
    pending_value: float = torch.nan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the trials into a format that the FTPFN model can understand.

    !!! warning "Pending trials"

        For trials which do not have a loss reported yet, they are considered pending.
        By default this is torch.nan and we recommend fantasizing these values.

    !!! warning "Error values"

        The FTPFN model requires that all loss values lie in the interval [0, 1].
        By default, using the value of `error_value=0.0`, we encode crashed configurations
        as having an error value of 0.

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
    selected = dict(trials.items())
    assert space.fidelity_name is not None
    assert space.fidelity is not None
    assert 0 <= error_value <= 1
    train_configs = encoder.encode(
        [t.config for t in selected.values()], device=device, dtype=dtype
    )
    ids = torch.tensor(
        [int(config_id.split("_", maxsplit=1)[0]) for config_id in selected],
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
    train_budgets = budget_domain.cast(
        train_fidelities, frm=space.fidelity.domain, dtype=dtype
    )

    # TODO: Document that it's on the user to ensure these are already all bounded
    # We could possibly include some bounded transform to assert this.
    minimize_ys = torch.tensor(
        [
            pending_value
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
    x_train = torch.cat(
        [ids.unsqueeze(1), train_budgets.unsqueeze(1), train_configs], dim=1
    )
    return x_train, maximize_ys


def decode_ftpfn_data(
    x: torch.Tensor,
    encoder: ConfigEncoder,
    budget_domain: Domain,
    fidelity_domain: Domain,
) -> list[tuple[int | None, int | float, dict[str, Any]]]:
    if x.ndim == 1:
        x = x.unsqueeze(0)

    _raw_ids = x[:, 0].tolist()
    # Subtract 1 to get the real id, otherwise if it was a test ID, we say it had None
    real_ids = [None if _id == 0 else int(_id) - 1 for _id in _raw_ids]
    fidelities = fidelity_domain.cast(x[:, 1], frm=budget_domain).tolist()
    configs = encoder.decode(x[:, 2:])
    return list(zip(real_ids, fidelities, configs, strict=False))


def acquire_next_from_ftpfn(
    *,
    ftpfn: FTPFNSurrogate,
    continuation_samples: torch.Tensor,
    encoder: ConfigEncoder,
    budget_domain: Domain,
    initial_samplers: list[tuple[Sampler, int]],
    local_search_sample_size: int = 128,
    local_search_confidence: float = 0.95,  # [0, 1]
    acq_function: Callable[[torch.Tensor], torch.Tensor],
    seed: torch.Generator | None = None,
    dtype: torch.dtype | None = FTPFN_DTYPE,
) -> torch.Tensor:
    # 1. Remove duplicate configurations from continuation_samples,
    # keeping only the most recent eval
    acq_existing = _keep_highest_budget_evaluation(
        continuation_samples, id_col=0, budget_col=1
    )

    # 2. Remove configs that have been fully evaluated
    acq_existing = acq_existing[acq_existing[:, 1] < budget_domain.upper]
    if len(acq_existing) != 0:
        # Get the best configuration for continuation
        acq_scores = acq_function(acq_existing)
        best_ix = acq_scores.argmax()

        best_score = acq_scores[best_ix].item()
        best_row = acq_existing[best_ix].clone().detach()
        del acq_existing
        del acq_scores
    else:
        best_score = -float("inf")
        best_row = torch.tensor([])

    # We'll be re-using 0 id and min budget alot, just create them once and re-use
    _N = max(*(s[1] for s in initial_samplers), local_search_sample_size)
    ids = torch.zeros((_N, 1), dtype=dtype, device=ftpfn.device)
    min_budget = torch.full(
        size=(_N, 1), fill_value=budget_domain.lower, dtype=dtype, device=ftpfn.device
    )

    # Acquisition maximization by sampling from samplers and performing an additional
    # round of local sampling around the best point
    local_sample_confidence = [local_search_confidence] * len(encoder.domains)
    for sampler, size in initial_samplers:
        # 1. Use provided sampler and eval samples with acq
        samples = sampler.sample(
            size, to=encoder.domains, seed=seed, device=ftpfn.device, dtype=dtype
        )
        _N = len(samples)
        X_test = torch.cat([ids[:_N], min_budget[:_N], samples], dim=1)
        acq_scores = acq_function(X_test)

        # ... update best if needed
        sample_best_ix = acq_scores.argmax()
        sample_best_score = acq_scores[sample_best_ix]
        sample_best_row = X_test[sample_best_ix].clone().detach()
        if sample_best_score > best_score:
            best_score = sample_best_score
            best_row = sample_best_row

        # 2. Sample around best point from above samples and eval acq.
        _mode = sample_best_row[2:]
        local_sampler = Prior.from_domains_and_centers(
            centers=list(zip(_mode.tolist(), local_sample_confidence, strict=False)),
            domains=encoder.domains,
        )
        samples = local_sampler.sample(
            local_search_sample_size,
            to=encoder.domains,
            seed=seed,
            device=ftpfn.device,
            dtype=dtype,
        )
        _N = len(samples)
        X_test = torch.cat([ids[:_N], min_budget[:_N], samples], dim=1)
        acq_scores = acq_function(X_test)

        local_best_ix = acq_scores.argmax()
        local_best_score = acq_scores[local_best_ix].clone().detach()
        if local_best_score > best_score:
            best_score = local_best_score
            best_row = X_test[local_best_ix].clone().detach()

    # Finally, if the best
    return best_row


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

    @torch.no_grad()  # type: ignore
    def get_mean_performance(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x).squeeze()
        return self.ftpfn.model.criterion.mean(logits)

    @torch.no_grad()  # type: ignore
    def get_pi(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        y_best: torch.Tensor | float,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.pi(logits.squeeze(), best_f=y_best)

    @torch.no_grad()  # type: ignore
    def get_ei(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        y_best: torch.Tensor | float,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.ei(logits.squeeze(), best_f=y_best)

    @torch.no_grad()  # type: ignore
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
            # IMPORTANT to be False, calculate the LCB using lower-bound ICDF as per beta
            maximize=False,
        )

    @torch.no_grad()  # type: ignore
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
            # IMPORTANT to be True, calculate the UCB using upper-bound ICDF as per beta
            maximize=True,
        )
