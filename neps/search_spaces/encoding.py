from __future__ import annotations

from dataclasses import dataclass, field
from grakel.utils import graph_from_networkx

from typing import Any, TypeAlias, TypeVar, Generic
from typing_extensions import Self, override, Self
from itertools import chain
import torch
from neps.search_spaces import (
    CategoricalParameter,
    IntegerParameter,
    FloatParameter,
)

from neps.search_spaces.search_space import SearchSpace, Parameter

WLInput: TypeAlias = tuple[dict, dict | None, dict | None]


@dataclass
class GraphEncoder:
    hps: tuple[str]

    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
    ) -> dict[str, list[WLInput]]:
        return {hp: [config[hp].value for config in x] for hp in self.hps}


T = TypeVar("T")


@dataclass
class Transformer(Generic[T]):
    hps: tuple[str]

    def encode(self, x: list[dict[str, Any]], space: SearchSpace) -> T: ...

    def value_decode(self, x: T, space: SearchSpace) -> dict[str, list[Any]]: ...

    def decode(self, x: T, space: SearchSpace) -> list[dict[str, Any]]:
        values = self.value_decode(x, space)
        return [(dict(zip(values, t))) for t in zip(*values.values())]


@dataclass
class WLInputTransformer(Transformer[WLInput]):
    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
    ) -> dict[str, list[WLInput]]:
        _graphs: dict[str, list[WLInput]] = {}
        for hp_name in space.graphs.keys():
            gs = [conf[hp_name].value for conf in x]
            _graphs[hp_name] = graph_from_networkx(gs)  # type: ignore

        return _graphs

    def value_decode(
        self,
        x: dict[str, list[WLInput]],
        space: SearchSpace,
    ) -> dict[str, list[Any]]:
        raise NotImplementedError("Cannot decode WLInput to values.")


@dataclass
class TensorTransformer(Transformer[torch.Tensor]):
    def output_cols(self, space: SearchSpace) -> int: ...

    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        width = len(self.hps)
        buffer = torch.empty(size=(len(x), width), dtype=dtype, device=device)

        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, CategoricalParameter)
            values = torch.tensor(
                [config[name]._value_index for config in x], dtype=dtype, device=device
            )

        return buffer


@dataclass
class IntegerCategoricalTransformer(TensorTransformer):
    def output_cols(self, space: SearchSpace) -> int:
        return len(self.hps)

    @override
    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.int

        buffer = torch.empty(size=(len(x), len(self.hps)), dtype=dtype, device=device)
        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, CategoricalParameter)
            values = torch.tensor(
                [config[name].value for config in x], dtype=dtype, device=device
            )
            buffer[:, i] = values

        return buffer

    @override
    def value_decode(self, x: torch.Tensor, space: SearchSpace) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}
        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, CategoricalParameter)
            enc = x[:, i]
            values[name] = [hp.choices[i] for i in enc.tolist()]

        return values


@dataclass
class MinMaxNormalizer(TensorTransformer):
    def output_cols(self, space: SearchSpace) -> int:
        return len(self.hps)

    @override
    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float64

        width = len(self.hps)
        buffer = torch.empty(size=(len(x), width), dtype=dtype, device=device)

        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, (FloatParameter, IntegerParameter))
            values = torch.tensor(
                [config[name].value for config in x], dtype=dtype, device=device
            )
            if hp.log_bounds:
                lower, upper = hp.log_bounds
                buffer[:, i] = (torch.log(values) - lower) / (upper - lower)
            else:
                lower, upper = hp.lower, hp.upper
                buffer[:, i] = (values - lower) / (upper - lower)

        return buffer

    @override
    def value_decode(
        self,
        x: torch.Tensor,
        space: SearchSpace,
    ) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}

        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, (FloatParameter, IntegerParameter))
            enc = x[:, i]
            if hp.log_bounds:
                lower, upper = hp.log_bounds
                enc = torch.exp(enc * (upper - lower) + lower)
            else:
                lower, upper = hp.lower, hp.upper
                enc = enc * (upper - lower) + lower

            if isinstance(hp, IntegerParameter):
                enc = torch.round(enc).to(torch.int)

            values[name] = enc.tolist()

        return values


@dataclass
class StandardNormalizer(TensorTransformer):
    std_means: dict[str, tuple[float, float]] = field(default_factory=dict)

    def output_cols(self, space: SearchSpace) -> int:
        return len(self.hps)

    @override
    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.float64

        width = len(self.hps)
        buffer = torch.empty(size=(len(x), width), dtype=dtype, device=device)
        std_means: dict[str, tuple[float, float]] = {}

        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, (FloatParameter, IntegerParameter))
            values = torch.tensor(
                [config[name].value for config in x], dtype=dtype, device=device
            )
            if hp.log_bounds:
                values = torch.log(values)

            mean, std = values.mean(), values.std()
            std_means[name] = (mean.item(), std.item())

            buffer[:, i] = (values - mean) / std

        self.std_means = std_means
        return buffer

    @override
    def value_decode(self, x: torch.Tensor, space: SearchSpace) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}

        for i, name in enumerate(self.hps):
            hp = space[name]
            assert isinstance(hp, Parameter)
            enc = x[:, i]
            if isinstance(hp, (FloatParameter, IntegerParameter)):
                std, mean = self.std_means[name]
                if hp.log_bounds:
                    enc = torch.exp(enc * std + mean)
                else:
                    enc = enc * std + mean

                if isinstance(hp, IntegerParameter):
                    enc = torch.round(enc).to(torch.int)

                values[name] = enc.tolist()
            else:
                raise ValueError(f"Invalid hyperparameter type: {type(hp)}")

        return values


@dataclass
class OneHotEncoder(TensorTransformer):
    def output_cols(self, space: SearchSpace) -> int:
        return sum(len(hp.choices) for hp in (space[name] for name in self.hps))  # type: ignore

    @override
    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.bool

        categoricals: dict[str, CategoricalParameter] = {}
        for name in self.hps:
            hp = space[name]
            assert isinstance(hp, CategoricalParameter)
            categoricals[name] = hp

        width = sum(len(hp.choices) for hp in categoricals.values())
        buffer = torch.zeros(size=(len(x), width), dtype=dtype, device=device)

        offset = 0
        for name, hp in categoricals.items():
            n_choices = len(hp.choices)
            _xs = [config[name]._value_index for config in x]
            cat_tensor = torch.tensor(_xs, dtype=torch.int64, device=device).unsqueeze(1)
            buffer[:, offset : offset + n_choices].scatter_(1, cat_tensor, 1)
            offset += n_choices

        return buffer

    @override
    def value_decode(
        self,
        x: torch.Tensor,
        space: SearchSpace,
    ) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}

        offset = 0
        for name in self.hps:
            hp = space[name]
            assert isinstance(hp, CategoricalParameter)
            n_choices = len(hp.choices)
            enc = x[:, offset : offset + n_choices].argmax(dim=1)

            values[name] = [hp.choices[i] for i in enc]
            offset += n_choices

        return values


@dataclass
class JointTransformer(TensorTransformer):
    transforms: tuple[TensorTransformer, ...]

    def output_cols(self, space: SearchSpace) -> int:
        return sum(t.output_cols(space) for t in self.transforms)

    @classmethod
    def join(cls, *transforms: TensorTransformer) -> Self:
        hps = tuple(chain.from_iterable(t.hps for t in transforms))
        return cls(hps, transforms)

    @override
    def encode(
        self,
        x: list[dict[str, Any]],
        space: SearchSpace,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return torch.cat(
            [t.encode(x, space, dtype=dtype, device=device) for t in self.transforms],
            dim=1,
        )

    @override
    def value_decode(
        self,
        x: torch.Tensor,
        space: SearchSpace,
    ) -> dict[str, list[Any]]:
        values: dict[str, list[Any]] = {}
        offset = 0
        for t in self.transforms:
            width = t.output_cols(space)
            t_values = t.value_decode(x[:, offset : offset + width], space)
            values.update(t_values)
            offset += width

        return values
