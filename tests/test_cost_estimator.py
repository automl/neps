from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoscan.utils.cost_estimator import CostEstimator

RECORDS = [
    {"config": {"width": 256, "layers": 6}, "cost": 1.2e9},
    {"config": {"width": 256, "layers": 12}, "cost": 2.4e9},
    {"config": {"width": 512, "layers": 6}, "cost": 4.7e9},
]


def _write_table(path: Path, records: list) -> Path:
    path.write_text(json.dumps(records))
    return path


def test_callable_is_only_passed_the_parameters_it_names() -> None:
    def flops(width: int, layers: int) -> float:
        return width * layers

    estimator = CostEstimator.from_user(flops)
    assert estimator(width=4, layers=3, lr=0.1, dataset="mnist") == 12.0


def test_callable_with_kwargs_is_passed_everything() -> None:
    estimator = CostEstimator.from_user(lambda **config: len(config))
    assert estimator(width=4, layers=3, lr=0.1) == 3.0


def test_table_lookup_ignores_other_parameters(tmp_path: Path) -> None:
    estimator = CostEstimator.from_user(_write_table(tmp_path / "costs.json", RECORDS))
    assert estimator.table_keys == ("layers", "width")
    assert estimator(width=256, layers=12, lr=0.1) == 2.4e9
    # Integer valued floats, as decoded by some optimizers, match as well
    assert estimator(width=512.0, layers=6.0) == 4.7e9


def test_table_miss_names_the_missing_combination(tmp_path: Path) -> None:
    estimator = CostEstimator.from_json(_write_table(tmp_path / "costs.json", RECORDS))
    with pytest.raises(ValueError, match=r"no entry for \{'layers': 12, 'width': 512\}"):
        estimator(width=512, layers=12)
    with pytest.raises(ValueError, match="no value for 'layers'"):
        estimator(width=512)


@pytest.mark.parametrize(
    "records",
    [
        [],
        {"width": 1},
        [{"config": {"width": 1}}],
        [*RECORDS, {"config": {"width": 1024}, "cost": 1.0}],
    ],
)
def test_malformed_table_is_rejected(tmp_path: Path, records: object) -> None:
    with pytest.raises(ValueError, match=r"(?i)cost table"):
        CostEstimator.from_json(_write_table(tmp_path / "costs.json", records))  # type: ignore[arg-type]


def test_validate_table_against_space(tmp_path: Path) -> None:
    estimator = CostEstimator.from_json(_write_table(tmp_path / "costs.json", RECORDS))
    estimator.validate({"width": ..., "layers": ..., "lr": ...}, {})
    with pytest.raises(ValueError, match=r"\['layers'\], which are not in"):
        estimator.validate({"width": ..., "lr": ...}, {})
