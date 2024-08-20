from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Sequence,
    Sized,
    TypeAlias,
    TypeVar,
    overload,
)
from typing_extensions import Protocol, override

import numpy as np
import numpy.typing as npt
import torch
from grakel.utils import graph_from_networkx

from neps.search_spaces.domain import (
    UNIT_FLOAT_DOMAIN,
    Domain,
    NumberDomain,
    OneHotDomain,
)

if TYPE_CHECKING:
    import networkx as nx

    from neps.search_spaces.search_space import SearchSpace

WLInput: TypeAlias = tuple[dict, dict | None, dict | None]
V = TypeVar("V", int, float)
T = TypeVar("T")


class Transformer(Protocol[T]):
    def encode(self, x: Sequence[Any]) -> T: ...

    def decode(self, x: T) -> list[Any]: ...


class TensorTransformer(Transformer[torch.Tensor], Protocol):
    domain: Domain
    output_cols: int

    def encode(
        self,
        x: list[Any],
        *,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor: ...


@dataclass
class CategoricalToIntegerTransformer(TensorTransformer):
    choices: list[Any]

    domain: NumberDomain = field(init=False)
    output_cols: int = field(init=False)
    _lookup: dict[Any, int] | None = field(init=False)

    def __post_init__(self):
        assert len(self.choices) > 0

        self.domain = NumberDomain.indices(len(self.choices))
        self.output_cols = 1
        if len(self.choices) > 3:
            try:
                self._lookup = {c: i for i, c in enumerate(self.choices)}
            except TypeError:
                self._lookup = None

    @override
    def encode(
        self,
        x: list[Any],
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

        if out is None:
            return torch.tensor(values, dtype=dtype, device=device)

        assert out.shape == (len(x),), f"{out.shape} != {(len(x),)}"
        out[:] = torch.tensor(values, dtype=out.dtype, device=out.device)
        return out

    @override
    def decode(self, x: torch.Tensor) -> list[Any]:
        return [self.choices[i] for i in x]


# TODO: Maybe add a shift argument, could be useful to have `0` as midpoint
# and `-0.5` as lower bound with `0.5` as upper bound.
@dataclass
class MinMaxNormalizer(TensorTransformer, Generic[V]):
    original_domain: NumberDomain[V]

    domain: NumberDomain[float] = field(init=False)
    output_cols: int = field(init=False)

    def __post_init__(self):
        self.domain = UNIT_FLOAT_DOMAIN
        self.output_cols = 1

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

        values = torch.tensor(list(x), dtype=dtype, device=device)
        values = self.domain.cast(values, frm=self.original_domain)
        if out is None:
            return values

        assert out.shape == (len(x),), f"{out.shape} != {(len(x),)}"
        out[:] = values
        return out

    @override
    def decode(self, x: torch.Tensor) -> list[V]:
        values = self.original_domain.from_unit(x)
        return values.tolist()


@dataclass
class OneHotEncoder(TensorTransformer):
    choices: list[Any]

    domain: OneHotDomain = field(init=False)
    output_cols: int = field(init=False)
    categorical_to_integer: CategoricalToIntegerTransformer = field(init=False)

    def __post_init__(self):
        self.categorical_to_integer = CategoricalToIntegerTransformer(self.choices)
        self.output_cols = len(self.choices)

    @override
    def encode(
        self,
        x: list[Any],
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

        ints = self.categorical_to_integer.encode(x, dtype=torch.int64, device=device)
        shape = (len(x), self.output_cols)
        if out is None:
            buffer = torch.zeros(size=shape, dtype=dtype, device=device)
        else:
            assert out.shape == shape, f"{out.shape} != {shape}"
            buffer = out

        cat_tensor = torch.tensor(ints, dtype=torch.int64, device=device).unsqueeze(1)
        buffer.scatter_(1, cat_tensor, 1)
        return buffer

    @override
    def decode(self, x: torch.Tensor) -> list[Any]:
        ints = torch.argmax(x, dim=1)
        return self.categorical_to_integer.decode(ints)


@dataclass
class WLInputTransformer(Transformer[WLInput]):
    hp: str

    def encode(self, x: Sequence[nx.Graph]) -> list[WLInput]:
        return [graph_from_networkx(g) for g in x]  # type: ignore

    def decode(self, x: dict[str, list[WLInput]]) -> dict[str, list[Any]]:
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

    def encode(self, x: list[SearchSpace]) -> npt.NDArray[np.object_]:
        buffer = np.empty((len(x), len(self.transformers)), dtype=np.object_)
        for hp, transformer in self.transformers.items():
            values = [conf[hp].value for conf in x]
            buffer[:, self.column_lookup[hp]] = transformer.encode(values)  # type: ignore
        return buffer

    def decode_dicts(self, x: npt.NDArray[np.object_]) -> list[dict[str, Any]]:
        raise NotImplementedError("Cannot decode graph embeddings.")


@dataclass
class TensorEncoder:
    transformers: dict[str, TensorTransformer]
    column_lookup: dict[str, tuple[int, int]] = field(init=False)

    def __post_init__(self):
        transformers = sorted(
            self.transformers.items(), key=lambda t: (t[1].output_cols, t[0])
        )
        self.transformers = dict(transformers)
        self.column_lookup: dict[str, tuple[int, int]] = {}
        offset = 0
        for name, transformer in self.transformers.items():
            self.column_lookup[name] = (offset, offset + transformer.output_cols)
            offset += transformer.output_cols

    def select(self, x: torch.Tensor, hp: str | Sequence[str]) -> torch.Tensor:
        if isinstance(hp, str):
            return x[:, slice(*self.column_lookup[hp])]
        cols = torch.concatenate([torch.arange(*self.column_lookup[h]) for h in hp])
        return x[:, cols]

    def encode(
        self,
        x: list[SearchSpace],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        width = sum(t.output_cols for t in self.transformers.values())
        buffer = torch.empty((len(x), width), dtype=torch.float64, device=device)

        for hp_name, transformer in self.transformers.items():
            values = [conf[hp_name] for conf in x]
            lookup = self.column_lookup[hp_name]

            # Encode directly into buffer
            transformer.encode(
                values,
                out=buffer[:, slice(*lookup)],
                dtype=torch.float64,
                device=device,
            )

        return buffer

    def decode_dicts(self, x: torch.Tensor) -> list[dict[str, Any]]:
        values: dict[str, list[Any]] = {}
        for hp_name, transformer in self.transformers.items():
            lookup = self.column_lookup[hp_name]
            values[hp_name] = transformer.decode(x[:, slice(*lookup)])

        keys = list(values.keys())
        return [dict(zip(keys, vals)) for vals in zip(*values.values())]


@dataclass
class DataEncoder:
    tensors: TensorEncoder | None = None
    graphs: GraphEncoder | None = None

    def encode(
        self,
        x: list[SearchSpace],
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor | None, npt.NDArray[np.object_] | None]:
        tensor = self.tensors.encode(x, device=device) if self.tensors else None
        graphs = self.graphs.encode(x) if self.graphs else None
        return tensor, graphs

    @overload
    def select(self, x: torch.Tensor, hp: str | Sequence[str]) -> torch.Tensor: ...

    @overload
    def select(
        self, x: npt.NDArray[np.object_], hp: str | Sequence[str]
    ) -> npt.NDArray[np.object_]: ...

    def select(
        self,
        x: torch.Tensor | npt.NDArray[np.object_],
        hp: str | Sequence[str],
    ) -> torch.Tensor | npt.NDArray[np.object_]:
        if isinstance(x, torch.Tensor):
            assert self.tensors is not None
            return self.tensors.select(x, hp)

        assert self.graphs is not None
        return self.graphs.select(x, hp)

    def decode_dicts(
        self,
        x: torch.Tensor
        | npt.NDArray[np.object_]
        | tuple[torch.Tensor | None, npt.NDArray[np.object_] | None],
    ) -> list[dict[str, Any]]:
        if isinstance(x, tuple):
            tensors, graphs = x
        elif isinstance(x, torch.Tensor):
            tensors, graphs = x, None
        else:
            tensors, graphs = None, x

        tensor_values: list[dict[str, Any]] | None = None
        if tensors is not None:
            assert self.tensors is not None
            tensor_values = self.tensors.decode_dicts(tensors)

        graph_values: list[dict[str, Any]] | None = None
        if graphs is not None:
            assert self.graphs is not None
            graph_values = self.graphs.decode_dicts(graphs)

        if tensor_values is not None and graph_values is not None:
            assert len(tensor_values) == len(graph_values)
            return [{**t, **g} for t, g in zip(tensor_values, graph_values)]

        if tensor_values is not None:
            return tensor_values

        assert graph_values is not None
        return graph_values


@dataclass
class DataPack(Sized):
    space: SearchSpace
    encoder: DataEncoder
    numerical: torch.Tensor | None = None
    graphs: npt.NDArray[np.object_] | None = None
    _len: int = field(init=False)

    def __post_init__(self):
        if self.numerical is not None and self.graphs is not None:
            assert len(self.numerical) == len(self.graphs)
            self._len = len(self.numerical)
        elif self.numerical is not None:
            self._len = len(self.numerical)
        elif self.graphs is not None:
            self._len = len(self.graphs)
        else:
            raise ValueError("At least one of numerical or graphs must be provided")

    def __len__(self) -> int:
        return self._len

    def select(self, hp: str | Sequence[str]) -> torch.Tensor | npt.NDArray[np.object_]:
        if isinstance(hp, str):
            if self.encoder.tensors and hp in self.encoder.tensors.transformers:
                assert self.numerical is not None
                return self.encoder.tensors.select(self.numerical, hp)

            if self.encoder.graphs and hp in self.encoder.graphs.transformers:
                assert self.graphs is not None
                return self.encoder.graphs.select(self.graphs, hp)

            tkeys = (
                None
                if self.encoder.tensors is None
                else self.encoder.tensors.transformers.keys()
            )
            gkeys = (
                None
                if self.encoder.graphs is None
                else self.encoder.graphs.transformers.keys()
            )
            raise KeyError(
                f"Unknown hyperparameter {hp}. Not in either tensors or graphs"
                f"\nTensors: {tkeys}"
                f"\nGraphs: {gkeys}"
            )

        all_in_tensors = False
        all_in_graphs = False
        tkeys = None
        gkeys = None
        if self.encoder.tensors:
            all_in_tensors = all(h in self.encoder.tensors.transformers for h in hp)

        if self.encoder.graphs:
            all_in_graphs = all(h in self.encoder.graphs.transformers for h in hp)
            gkeys = self.encoder.graphs.transformers.keys()

        if not all_in_tensors and not all_in_graphs:
            raise ValueError(
                "Cannot select from both tensors and graphs!"
                f"Got keys: {hp}"
                f"\nTensors: {tkeys}"
                f"\nGraphs: {gkeys}"
            )

        if all_in_tensors:
            assert self.numerical is not None
            assert self.encoder.tensors is not None
            return self.encoder.tensors.select(self.numerical, hp)

        assert self.graphs is not None
        assert self.encoder.graphs is not None
        return self.encoder.graphs.select(self.graphs, hp)

    def decode(self) -> list[SearchSpace]:
        return [
            self.space.from_dict(d)
            for d in self.encoder.decode_dicts((self.numerical, self.graphs))
        ]
