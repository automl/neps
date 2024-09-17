from __future__ import annotations

from typing import Any
from pathlib import Path
import torch

from ifbo import FTPFN


def _download_workaround_for_ifbo_issue_10(path: Path | None, version: str) -> Path:
    # TODO: https://github.com/automl/ifBO/issues/10
    import requests
    from ifbo.download import FILE_URL, FILENAME
    from ifbo.surrogate import _resolve_model_path

    target_path = Path(path) if path is not None else Path.cwd().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    _target_zip_path = target_path / FILENAME(version)
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


class FTPFNSurrogate:
    """Special class to deal with PFN surrogate model and freeze-thaw acquisition."""

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

        self.ftpfn = FTPFN(target_path=target_path, version=version)
        self.target_path = self.ftpfn.target_path
        self.version = self.ftpfn.version
        self.train_x: torch.Tensor | None = None
        self.train_y: torch.Tensor | None = None

    @property
    def device(self):
        return self.ftpfn.device

    def _get_logits(self, test_x: torch.Tensor) -> torch.Tensor:
        assert self.train_x is not None, "Train data is not set."
        assert self.train_y is not None, "Train data is not set."
        return self.ftpfn.model(
            self._cast_tensor_shapes(self.train_x),
            self._cast_tensor_shapes(self.train_y),
            self._cast_tensor_shapes(test_x),
        )

    def _cast_tensor_shapes(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3 and x.shape[1] == 1:
            return x
        if len(x.shape) == 2:
            return x.reshape(x.shape[0], 1, x.shape[1])
        if len(x.shape) == 1:
            return x.reshape(x.shape[0], 1)
        raise ValueError(f"Shape not recognized: {x.shape}")

    @torch.no_grad()
    def get_mean_performance(self, test_x: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits(test_x).squeeze()
        return self.ftpfn.model.criterion.mean(logits)

    @torch.no_grad()
    def get_pi(self, test_x: torch.Tensor, y_best: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits(test_x)
        return self.ftpfn.model.criterion.pi(
            logits.squeeze(), best_f=(1 - y_best).unsqueeze(1)
        )

    @torch.no_grad()
    def get_ei(self, test_x: torch.Tensor, y_best: torch.Tensor) -> torch.Tensor:
        logits = self._get_logits(test_x)
        return self.ftpfn.model.criterion.ei(
            logits.squeeze(), best_f=(1 - y_best).unsqueeze(1)
        )

    @torch.no_grad()
    def get_lcb(
        self, test_x: torch.Tensor, beta: float = (1 - 0.682) / 2
    ) -> torch.Tensor:
        logits = self._get_logits(test_x)
        lcb = self.ftpfn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=False,  # IMPORTANT to be False, should calculate the LCB using the lower-bound ICDF as per beta
        )
        return lcb

    @torch.no_grad()
    def get_ucb(
        self, test_x: torch.Tensor, beta: float = (1 - 0.682) / 2
    ) -> torch.Tensor:
        logits = self._get_logits(test_x)
        lcb = self.ftpfn.model.criterion.ucb(
            logits=logits,
            best_f=None,
            rest_prob=beta,
            maximize=True,  # IMPORTANT to be True, should calculate the UCB using the upper-bound ICDF as per beta
        )
        return lcb
