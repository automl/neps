from __future__ import annotations

from typing import Any

import pytest

from scalefm.optimizers import load_optimizer
from scalefm.optimizers.algorithms import determine_optimizer_automatically
from scalefm.optimizers.ask_and_tell import AskAndTell
from scalefm.space import HPOFloat, HPOInteger, SearchSpace
from scalefm.space.scalefm_spaces.parameters import (
    Float,
    IntegerFidelity,
    PipelineSpace,
)


def _space(*, prior: bool, fidelity: bool) -> SearchSpace:
    parameters: dict[str, Any] = {
        "x": HPOFloat(0, 1, prior=0.5 if prior else None),
    }
    if fidelity:
        parameters["epochs"] = HPOInteger(1, 10, is_fidelity=True)
    return SearchSpace(parameters)


@pytest.mark.parametrize(
    ("prior", "fidelity", "expected"),
    [
        (False, False, ("bayesian_optimization", {})),
        (True, False, ("random_search", {"use_priors": True})),
        (
            False,
            True,
            (
                "random_search",
                {"use_priors": False, "ignore_fidelity": "highest_fidelity"},
            ),
        ),
        (
            True,
            True,
            (
                "random_search",
                {"use_priors": True, "ignore_fidelity": "highest_fidelity"},
            ),
        ),
    ],
)
def test_auto_choice(
    prior: bool,
    fidelity: bool,
    expected: tuple[str, dict[str, Any]],
) -> None:
    space = _space(prior=prior, fidelity=fidelity)
    assert determine_optimizer_automatically(space) == expected

    optimizer, info = load_optimizer("auto", space)
    assert info["name"] == expected[0]

    ask_and_tell = AskAndTell(optimizer)
    for _ in range(3):
        trial = ask_and_tell.ask()
        if fidelity:
            assert trial.config["epochs"] == 10
        ask_and_tell.tell(trial, {"objective_to_minimize": 1.0, "cost": 1.0})


def test_auto_choice_for_pipeline_space() -> None:
    class Space(PipelineSpace):
        x = Float(0, 1)
        epochs = IntegerFidelity(1, 10)

    assert determine_optimizer_automatically(Space()) == (
        "complex_random_search",
        {"ignore_fidelity": "highest_fidelity"},
    )
