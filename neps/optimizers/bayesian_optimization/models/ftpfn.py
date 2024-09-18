from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ifbo import FTPFN


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


_CACHED_FTPFN_MODEL: dict[tuple[str, str], FTPFN] = {}


class FTPFNModel:
    """Wrapper around the IfBO model."""

    def __init__(
        self,
        target_path: Path | None = None,
        version: str = "0.0.1",
        **kwargs: Any,
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
            ftpfn = FTPFN(target_path=target_path, version=version)
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
        # TODO: just calculate from train_y?
        y_best: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.pi(
            logits.squeeze(),
            best_f=(1 - y_best).unsqueeze(1),
        )

    @torch.no_grad()
    def get_ei(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        y_best: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._get_logits(train_x, train_y, test_x)
        return self.ftpfn.model.criterion.ei(
            logits.squeeze(), best_f=(1 - y_best).unsqueeze(1)
        )

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
