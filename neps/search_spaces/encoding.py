from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
)
from typing_extensions import Protocol, override

import numpy as np
import numpy.typing as npt
import torch
from grakel.utils import graph_from_networkx

from neps.search_spaces.domain import (
    UNIT_FLOAT_DOMAIN,
    Domain,
)
from neps.search_spaces.hyperparameters.categorical import CategoricalParameter
from neps.search_spaces.hyperparameters.float import FloatParameter
from neps.search_spaces.hyperparameters.integer import IntegerParameter

if TYPE_CHECKING:
    import networkx as nx

    from neps.search_spaces.parameter import Parameter
    from neps.search_spaces.search_space import SearchSpace

WLInput: TypeAlias = tuple[dict, dict | None, dict | None]
V = TypeVar("V", int, float)
T = TypeVar("T")


class Transformer(Protocol[T]):
    def encode(self, x: Sequence[Any]) -> T: ...

    def decode(self, x: T) -> list[Any]: ...


class TensorTransformer(Transformer[torch.Tensor], Protocol):
    domain: Domain

    def encode(
        self,
        x: Sequence[Any],
        *,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor: ...


@dataclass
class CategoricalToIntegerTransformer(TensorTransformer):
    choices: Sequence[Any]

    domain: Domain = field(init=False)
    _lookup: dict[Any, int] | None = field(init=False)

    def __post_init__(self):
        assert len(self.choices) > 0

        self.domain = Domain.indices(len(self.choices))
        self._lookup = None
        if len(self.choices) > 3:
            try:
                self._lookup = {c: i for i, c in enumerate(self.choices)}
            except TypeError:
                self._lookup = None

    @override
    def encode(
        self,
        x: Sequence[Any],
        *,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = torch.int if out is None else out.dtype

        values = (
            [self._lookup[c] for c in x]
            if self._lookup
            else [self.choices.index(c) for c in x]
        )

        tensor = torch.tensor(values, dtype=torch.int64, device=device)
        if out is None:
            return tensor.to(dtype)

        out.copy_(tensor.to(out.dtype)).round_()
        return out

    @override
    def decode(self, x: torch.Tensor) -> list[Any]:
        return [self.choices[int(i)] for i in torch.round(x).tolist()]


@dataclass
class CategoricalToUnitNorm(TensorTransformer):
    choices: Sequence[Any]

    domain: Domain = field(init=False)
    _integer_transformer: CategoricalToIntegerTransformer = field(init=False)

    def __post_init__(self):
        self._integer_transformer = CategoricalToIntegerTransformer(self.choices)

    @override
    def encode(
        self,
        x: Sequence[Any],
        *,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        integers = self._integer_transformer.encode(
            x,
            dtype=dtype if dtype is not None else torch.float64,
            device=device,
            out=out,
        )
        if out is not None:
            return integers.div_(len(self.choices) - 1)

        return integers / (len(self.choices) - 1)

    @override
    def decode(self, x: torch.Tensor) -> list[Any]:
        x = torch.round(x * (len(self.choices) - 1)).type(torch.int64)
        return self._integer_transformer.decode(x)


# TODO: Maybe add a shift argument, could be useful to have `0` as midpoint
# and `-0.5` as lower bound with `0.5` as upper bound.
@dataclass
class MinMaxNormalizer(TensorTransformer, Generic[V]):
    original_domain: Domain[V]
    bins: int | None = None

    domain: Domain[float] = field(init=False)

    def __post_init__(self):
        if self.bins is None:
            self.domain = UNIT_FLOAT_DOMAIN
        else:
            self.domain = Domain.float(0.0, 1.0, bins=self.bins)

    @override
    def encode(
        self,
        x: list[V],
        *,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if out is not None:
            dtype = out.dtype
            device = out.device
        else:
            dtype = torch.float64 if dtype is None else dtype

        values = torch.tensor(x, dtype=dtype, device=device)
        values = self.domain.cast(values, frm=self.original_domain)
        if out is None:
            return values

        out.copy_(values)
        return out

    @override
    def decode(self, x: torch.Tensor) -> list[V]:
        values = self.original_domain.from_unit(x)
        return values.tolist()


@dataclass
class WLInputTransformer(Transformer[WLInput]):
    hp: str

    def encode(self, x: Sequence[nx.Graph]) -> list[WLInput]:
        return [graph_from_networkx(g) for g in x]  # type: ignore

    def decode(self, x: Mapping[str, Sequence[WLInput]]) -> dict[str, list[Any]]:
        raise NotImplementedError("Cannot decode WLInput to values.")


@dataclass
class GraphEncoder:
    transformers: dict[str, WLInputTransformer]
    column_lookup: dict[str, int] = field(init=False)

    def __post_init__(self):
        transformers = sorted(self.transformers.items(), key=lambda t: t[0])
        self.transformers = dict(transformers)
        self.column_lookup: dict[str, int] = {
            name: i for i, (name, _) in enumerate(self.transformers.items())
        }

    def select(
        self, x: npt.NDArray[np.object_], hp: str | Sequence[str]
    ) -> npt.NDArray[np.object_]:
        # Kind of a redundant function but made to be compatible with TensorPack
        if isinstance(hp, str):
            return x[:, self.column_lookup[hp]]

        return x[:, [self.column_lookup[h] for h in hp]]

    def encode(self, x: Sequence[Any]) -> npt.NDArray[np.object_]:
        buffer = np.empty((len(x), len(self.transformers)), dtype=np.object_)
        for hp, transformer in self.transformers.items():
            values = [conf[hp] for conf in x]
            buffer[:, self.column_lookup[hp]] = transformer.encode(values)  # type: ignore
        return buffer

    def decode_dicts(self, x: npt.NDArray[np.object_]) -> list[dict[str, Any]]:
        raise NotImplementedError("Cannot decode graph embeddings.")


@dataclass
class TensorEncoder:
    transformers: dict[str, TensorTransformer]
    index_of: dict[str, int] = field(init=False)
    domain_of: dict[str, Domain] = field(init=False)
    n_numerical: int = field(init=False)
    n_categorical: int = field(init=False)

    def __post_init__(self):
        transformers = sorted(self.transformers.items(), key=lambda t: t[0])
        self.transformers = dict(transformers)

        n_numerical = 0
        n_categorical = 0
        for _, transformer in transformers:
            if isinstance(transformer, CategoricalToIntegerTransformer):
                n_categorical += 1
            else:
                n_numerical += 1

        self.index_of = {name: i for i, name in enumerate(self.transformers.keys())}
        self.domain_of = {name: t.domain for name, t in self.transformers.items()}
        self.n_numerical = n_numerical
        self.n_categorical = n_categorical

    @property
    def ncols(self) -> int:
        return len(self.transformers)

    @property
    def domains(self) -> list[Domain]:
        return list(self.domain_of.values())

    def names(self) -> list[str]:
        return list(self.transformers.keys())

    def select(self, x: torch.Tensor, hp: str | Sequence[str]) -> torch.Tensor:
        if isinstance(hp, str):
            return x[:, self.index_of[hp]]

        return x[:, [self.index_of[h] for h in hp]]

    def encode(
        self,
        x: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        width = len(self.transformers)
        buffer = torch.empty((len(x), width), dtype=torch.float64, device=device)

        for hp_name, transformer in self.transformers.items():
            values = [conf[hp_name] for conf in x]
            lookup = self.index_of[hp_name]

            # Encode directly into buffer
            transformer.encode(
                values,
                out=buffer[:, lookup],
                dtype=torch.float64,
                device=device,
            )

        return buffer

    def pack(
        self,
        x: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | None = None,
    ) -> TensorPack:
        return TensorPack(self.encode(x, device=device), self)

    def unpack(self, x: torch.Tensor) -> list[dict[str, Any]]:
        values: dict[str, list[Any]] = {}
        for hp_name, transformer in self.transformers.items():
            lookup = self.index_of[hp_name]
            tensor = x[:, lookup]
            values[hp_name] = transformer.decode(tensor)

        keys = list(values.keys())
        return [
            dict(zip(keys, vals, strict=False))
            for vals in zip(*values.values(), strict=False)
        ]

    @classmethod
    def default(
        cls,
        parameters: Mapping[str, Parameter],
        *,
        custom_transformers: dict[str, TensorTransformer] | None = None,
    ) -> TensorEncoder:
        custom = custom_transformers or {}
        sorted_params = sorted(parameters.items())
        transformers: dict[str, TensorTransformer] = {}
        for name, hp in sorted_params:
            if name in custom:
                transformers[name] = custom[name]
                continue

            match hp:
                case FloatParameter() | IntegerParameter():
                    transformers[name] = MinMaxNormalizer(hp.domain)
                case CategoricalParameter():
                    transformers[name] = CategoricalToIntegerTransformer(hp.choices)
                case _:
                    raise ValueError(f"Unsupported parameter type: {type(hp)}")

        return TensorEncoder(transformers)


@dataclass
class TensorPack:
    tensor: torch.Tensor
    encoder: TensorEncoder

    def __len__(self) -> int:
        return len(self.tensor)

    @property
    def n_numerical(self) -> int:
        return self.encoder.n_numerical

    @property
    def n_categorical(self) -> int:
        return self.encoder.n_categorical

    @property
    def ncols(self) -> int:
        return self.encoder.ncols

    @property
    def domains(self) -> dict[str, Domain]:
        return self.encoder.domains

    def select(self, hp: str | Sequence[str]) -> torch.Tensor | npt.NDArray[np.object_]:
        return self.encoder.select(self.tensor, hp)

    def names(self) -> list[str]:
        return self.encoder.names()

    def to_dicts(self) -> list[dict[str, Any]]:
        return self.encoder.unpack(self.tensor)

    def split(self, index: int) -> tuple[TensorPack, TensorPack]:
        left = TensorPack(self.encoder, tensor=self.tensor[:index])
        right = TensorPack(self.encoder, tensor=self.tensor[index:])
        return left, right

    def join(self, *other: TensorPack) -> TensorPack:
        assert all(o.encoder == self.encoder for o in other)

        numerical = torch.cat([self.tensor, *[o.tensor for o in other]], dim=0)
        return TensorPack(self.encoder, tensor=numerical)

    @classmethod
    def default_encoding(
        cls,
        x: Sequence[Mapping[str, Any]],
        space: SearchSpace,
    ) -> TensorPack:
        default_encoder = TensorEncoder.default(space)
        tensor = default_encoder.encode(x)
        return TensorPack(default_encoder, tensor)
