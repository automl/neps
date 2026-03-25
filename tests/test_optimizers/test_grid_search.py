from __future__ import annotations

import pytest

from neps import Categorical, Fidelity, Float, Integer, PipelineSpace
from neps.optimizers.algorithms import grid_search, neps_grid_search
from neps.space import (
    HPOCategorical,
    HPOConstant,
    HPOFloat,
    HPOInteger,
    SearchSpace,
)


class TestGridSearchBasic:
    def test_grid_search_basic_float(self) -> None:
        space = SearchSpace(
            {
                "x": HPOFloat(lower=0.0, upper=1.0),
                "y": HPOFloat(lower=-10.0, upper=10.0),
            }
        )
        result = grid_search(space, size_per_numerical_dimension=5)
        configs = result.configs_list
        assert len(configs) == 25
        x_values = [c["x"] for c in configs]
        y_values = [c["y"] for c in configs]
        assert x_values[0] == pytest.approx(0.0)
        assert x_values[-1] == pytest.approx(1.0)
        assert y_values[0] == pytest.approx(-10.0)
        assert y_values[-1] == pytest.approx(10.0)

    def test_grid_search_basic_integer(self) -> None:
        space = SearchSpace(
            {
                "x": HPOInteger(lower=0, upper=10),
                "y": HPOInteger(lower=0, upper=10),
            }
        )
        result = grid_search(space, size_per_numerical_dimension=5)
        configs = result.configs_list
        assert len(configs) == 25
        x_values = [c["x"] for c in configs]
        y_values = [c["y"] for c in configs]
        assert x_values[0] == 0
        assert x_values[-1] == 10
        assert y_values[0] == 0
        assert y_values[-1] == 10

    def test_grid_search_categorical(self) -> None:
        space = SearchSpace(
            {
                "x": HPOCategorical(choices=["a", "b", "c", "d"]),
                "y": HPOInteger(lower=0, upper=10),
            }
        )
        result = grid_search(space, size_per_numerical_dimension=5)
        configs = result.configs_list
        assert len(configs) == 20
        choices = {c["x"] for c in configs}
        assert choices == {"a", "b", "c", "d"}

    def test_grid_search_constant(self) -> None:
        space = SearchSpace(
            {
                "const": HPOConstant(value=42),
                "x": HPOFloat(lower=0.0, upper=1.0),
            }
        )
        result = grid_search(space, size_per_numerical_dimension=5)
        configs = result.configs_list
        assert len(configs) == 5
        assert configs[0]["const"] == 42


class TestGridSearchSizePerNumericalHP:
    def test_size_per_numerical_hp_single_integer(self) -> None:
        space = SearchSpace(
            {
                "x": HPOFloat(lower=0.0, upper=1.0),
                "y": HPOFloat(lower=0.0, upper=1.0),
            }
        )
        for size in [3, 5, 10]:
            result = grid_search(space, size_per_numerical_dimension=size)
            configs = result.configs_list
            assert len(configs) == size * size

    def test_size_per_numerical_hp_dict(self) -> None:
        space = SearchSpace(
            {
                "x": HPOFloat(lower=0.0, upper=1.0),
                "y": HPOFloat(lower=0.0, upper=1.0),
                "z": HPOInteger(lower=0, upper=10),
            }
        )
        config = {
            "x": 3,
            "y": 5,
            "z": 4,
        }
        result = grid_search(space, size_per_numerical_dimension=config)
        configs = result.configs_list
        assert len(configs) == 60

    def test_size_per_numerical_hp_dict_partial(self) -> None:
        space = SearchSpace(
            {
                "x": HPOFloat(lower=0.0, upper=1.0),
                "y": HPOFloat(lower=0.0, upper=1.0),
                "z": HPOInteger(lower=0, upper=10),
            }
        )
        config = {"x": 3}
        result = grid_search(space, size_per_numerical_dimension=config)
        configs = result.configs_list
        assert len(configs) == 75

    def test_size_per_numerical_hp_dict_invalid_key(self) -> None:
        space = SearchSpace(
            {
                "x": HPOFloat(0.0, 1.0),
            }
        )
        config = {"invalid_param": 5}
        with pytest.raises(ValueError, match="invalid keys"):
            grid_search(space, size_per_numerical_dimension=config)


class TestGridSearchLogScale:
    def test_grid_search_log_float(self) -> None:
        space = SearchSpace(
            {
                "x": HPOFloat(lower=1e-3, upper=1e3, log=True),
            }
        )
        result = grid_search(space, size_per_numerical_dimension=5)
        configs = result.configs_list
        assert len(configs) == 5
        values = sorted([c["x"] for c in configs])
        assert values[0] == pytest.approx(1e-3)
        assert values[-1] == pytest.approx(1e3)

    def test_grid_search_log_integer(self) -> None:
        space = SearchSpace(
            {
                "x": HPOInteger(lower=2, upper=1024, log=True, log_base=2),
            }
        )
        result = grid_search(space, size_per_numerical_dimension=100)
        configs = result.configs_list
        values = sorted([c["x"] for c in configs])
        assert len(values) == 10
        for v in values:
            assert v >= 2
            assert v <= 1024
            import math

            if v > 0:
                log_val = math.log2(v)
                assert log_val == int(log_val)


class TestNePSGridSearch:
    def test_neps_grid_search_basic(self) -> None:
        class Space(PipelineSpace):
            x = Float(lower=0.0, upper=1.0)
            y = Categorical(choices=("a", "b", "c"))

        space = Space()
        result = neps_grid_search(space, size_per_numerical_dimension=3)
        configs = result.configs_list
        assert len(configs) == 9

    def test_neps_grid_search_with_fidelity(self) -> None:
        class Space(PipelineSpace):
            x = Float(lower=0.0, upper=1.0)
            y = Fidelity(Integer(lower=1, upper=10))

        space = Space()
        with pytest.raises(ValueError, match="fidelity"):
            neps_grid_search(space, ignore_fidelity=False, size_per_numerical_dimension=3)

    def test_neps_grid_search_ignore_fidelity_highest(self) -> None:
        class Space(PipelineSpace):
            x = Float(lower=0.0, upper=1.0)
            y = Fidelity(Integer(lower=1, upper=10))

        space = Space()
        result = neps_grid_search(
            space,
            ignore_fidelity="highest_fidelity",
            size_per_numerical_dimension=3,
        )
        configs = result.configs_list
        assert len(configs) == 3
        assert all(c["ENVIRONMENT__y"] == 10 for c in configs)

    def test_neps_grid_search_ignore_fidelity_true(self) -> None:
        class Space(PipelineSpace):
            x = Float(lower=0.0, upper=1.0)
            y = Fidelity(Integer(lower=1, upper=10))

        space = Space()
        result = neps_grid_search(
            space, ignore_fidelity=True, size_per_numerical_dimension=3
        )
        configs = result.configs_list
        assert len(configs) == 6
