from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Mapping,
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
from torch._dynamo.utils import product

from neps.search_spaces.architecture.graph_grammar import GraphParameter
from neps.search_spaces.domain import (
    UNIT_FLOAT_DOMAIN,
    Domain,
)
from neps.search_spaces.hyperparameters.float import FloatParameter
from neps.search_spaces.hyperparameters.integer import IntegerParameter

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


# TODO: Maybe add a shift argument, could be useful to have `0` as midpoint
# and `-0.5` as lower bound with `0.5` as upper bound.
@dataclass
class MinMaxNormalizer(TensorTransformer, Generic[V]):
    original_domain: Domain[V]

    domain: Domain[float] = field(init=False)

    def __post_init__(self):
        self.domain = UNIT_FLOAT_DOMAIN

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
    column_lookup: dict[str, int] = field(init=False)
    n_numerical: int = field(init=False)
    n_categorical: int = field(init=False)

    def __post_init__(self):
        transformers = sorted(self.transformers.items(), key=lambda t: t[0])
        self.transformers = dict(transformers)
        self.column_lookup: dict[str, int] = {}
        n_numerical = 0
        n_categorical = 0
        for i, (name, transformer) in enumerate(self.transformers.items()):
            self.column_lookup[name] = i
            if isinstance(transformer, CategoricalToIntegerTransformer):
                n_categorical += 1
            else:
                n_numerical += 1

        self.n_numerical = n_numerical
        self.n_categorical = n_categorical

    def domains(self) -> dict[str, Domain]:
        return {
            name: transformer.domain for name, transformer in self.transformers.items()
        }

    def names(self) -> list[str]:
        return list(self.transformers.keys())

    def select(self, x: torch.Tensor, hp: str | Sequence[str]) -> torch.Tensor:
        if isinstance(hp, str):
            return x[:, self.column_lookup[hp]]

        cols = torch.concatenate([torch.arange(*self.column_lookup[h]) for h in hp])
        return x[:, cols]

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
            lookup = self.column_lookup[hp_name]

            # Encode directly into buffer
            transformer.encode(
                values,
                out=buffer[:, lookup],
                dtype=torch.float64,
                device=device,
            )

        return buffer

    def decode_dicts(self, x: torch.Tensor) -> list[dict[str, Any]]:
        values: dict[str, list[Any]] = {}
        for hp_name, transformer in self.transformers.items():
            lookup = self.column_lookup[hp_name]
            tensor = x[:, lookup]
            values[hp_name] = transformer.decode(tensor)

        keys = list(values.keys())
        return [dict(zip(keys, vals)) for vals in zip(*values.values())]


@dataclass
class DataEncoder:
    tensors: TensorEncoder | None = None
    graphs: GraphEncoder | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    n_numerical: int = field(init=False)
    n_categorical: int = field(init=False)
    n_graphs: int = field(init=False)

    def __post_init__(self):
        self.n_numerical = 0 if self.tensors is None else self.tensors.n_numerical
        self.n_categorical = 0 if self.tensors is None else self.tensors.n_categorical
        self.n_graphs = 0 if self.graphs is None else len(self.graphs.transformers)

    def encode(
        self,
        x: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | None = None,
    ) -> DataPack:
        tensor = self.tensors.encode(x, device=device) if self.tensors else None
        graphs = self.graphs.encode(x) if self.graphs else None
        return DataPack(encoder=self, tensor=tensor, graphs=graphs)

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

    def indices(self, hp: str | Sequence[str]) -> tuple[int, ...]:
        if isinstance(hp, str):
            if self.tensors and hp in self.tensors.transformers:
                lower, upper = self.tensors.column_lookup[hp]
                return tuple(torch.arange(lower, upper).tolist())

            if self.graphs and hp in self.graphs.transformers:
                raise ValueError("Cannot select indices from graphs.")

            tkeys = None if self.tensors is None else self.tensors.transformers.keys()
            gkeys = None if self.graphs is None else self.graphs.transformers.keys()
            raise KeyError(
                f"Unknown hyperparameter {hp}. Not in either tensors or graphs"
                f"\nTensors: {tkeys}"
                f"\nGraphs: {gkeys}"
            )

        return tuple(sorted(chain.from_iterable(self.indices(h) for h in hp)))

    @classmethod
    def default_encoder(
        cls,
        space: SearchSpace,
        *,
        include_fidelities: bool | list[str] = False,
    ) -> DataEncoder:
        tensor_transformers: dict[str, TensorTransformer] = {}
        graph_transformers: dict[str, WLInputTransformer] = {}

        for hp_name, hp in space.categoricals.items():
            tensor_transformers[hp_name] = CategoricalToIntegerTransformer(hp.choices)

        for hp_name, hp in space.numerical.items():
            assert isinstance(hp, (FloatParameter, IntegerParameter))
            tensor_transformers[hp_name] = MinMaxNormalizer(hp.domain)

        for hp_name, hp in space.graphs.items():
            assert isinstance(hp, GraphParameter)
            graph_transformers[hp_name] = WLInputTransformer(hp_name)

        if include_fidelities is True:
            include_fidelities = list(space.fidelities.keys())

        if include_fidelities:
            for fid_name in include_fidelities:
                hp = space.fidelities[fid_name]
                assert isinstance(hp, (FloatParameter, IntegerParameter))
                tensor_transformers[fid_name] = MinMaxNormalizer(hp.domain)

        tensor_encoder = (
            TensorEncoder(tensor_transformers) if any(tensor_transformers) else None
        )
        graph_encoder = (
            GraphEncoder(graph_transformers) if any(graph_transformers) else None
        )
        return DataEncoder(tensors=tensor_encoder, graphs=graph_encoder)

    def has_categoricals(self) -> bool:
        return self.tensors is not None and any(
            isinstance(t, CategoricalToIntegerTransformer)
            for t in self.tensors.transformers.values()
        )

    def has_graphs(self) -> bool:
        return self.graphs is not None

    def has_numericals(self) -> bool:
        return self.tensors is not None and any(
            not isinstance(t, CategoricalToIntegerTransformer)
            for t in self.tensors.transformers.values()
        )

    def categorical_product_indices(self) -> list[dict[int, int]]:
        cats: dict[int, list[int]] = {}
        if self.tensors is None:
            return []

        for i, (_hp_name, transformer) in enumerate(self.tensors.transformers.items()):
            if isinstance(transformer, CategoricalToIntegerTransformer):
                cats[i] = list(range(len(transformer.choices)))

        if len(cats) == 0:
            return []

        if len(cats) == 1:
            key, values = cats.popitem()
            return [{key: v} for v in values]

        return [dict(zip(cats.keys(), vs)) for vs in product(*cats.values())]


@dataclass
class DataPack(Sized):
    encoder: DataEncoder
    tensor: torch.Tensor | None = None
    graphs: npt.NDArray[np.object_] | None = None
    _len: int = field(init=False)

    def __post_init__(self):
        if self.tensor is not None and self.graphs is not None:
            assert len(self.tensor) == len(self.graphs)
            self._len = len(self.tensor)
        elif self.tensor is not None:
            self._len = len(self.tensor)
        elif self.graphs is not None:
            self._len = len(self.graphs)
        else:
            raise ValueError("At least one of numerical or graphs must be provided")

    def __len__(self) -> int:
        return self._len

    def select(self, hp: str | Sequence[str]) -> torch.Tensor | npt.NDArray[np.object_]:
        if isinstance(hp, str):
            if self.encoder.tensors and hp in self.encoder.tensors.transformers:
                assert self.tensor is not None
                return self.encoder.tensors.select(self.tensor, hp)

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
            assert self.tensor is not None
            assert self.encoder.tensors is not None
            return self.encoder.tensors.select(self.tensor, hp)

        assert self.graphs is not None
        assert self.encoder.graphs is not None
        return self.encoder.graphs.select(self.graphs, hp)

    def decode(self, space: SearchSpace) -> list[SearchSpace]:
        return [
            space.from_dict(d)
            for d in self.encoder.decode_dicts((self.tensor, self.graphs))
        ]

    def split(self, index: int) -> tuple[DataPack, DataPack]:
        if self.tensor is not None:
            numerical_left = self.tensor[:index]
            numerical_right = self.tensor[index:]
        else:
            numerical_left = None
            numerical_right = None

        if self.graphs is not None:
            graphs_left = self.graphs[:index]
            graphs_right = self.graphs[:index]
        else:
            graphs_left = None
            graphs_right = None

        return (
            DataPack(
                self.encoder,
                tensor=numerical_left,
                graphs=graphs_left,
            ),
            DataPack(
                self.encoder,
                tensor=numerical_right,
                graphs=graphs_right,
            ),
        )

    def join(self, *other: DataPack) -> DataPack:
        assert all(o.encoder == self.encoder for o in other)

        if self.tensor is not None:
            other_numericals = []
            for o in other:
                assert o.tensor is not None
                other_numericals.append(o.tensor)
            numerical = torch.cat([self.tensor, *other_numericals], dim=0)
        else:
            numerical = None

        if self.graphs is not None:
            other_graphs = []
            for o in other:
                assert o.graphs is not None
                other_graphs.append(o.graphs)
            graphs = np.concatenate([self.graphs, *other_graphs], axis=0)
        else:
            graphs = None

        return DataPack(self.encoder, tensor=numerical, graphs=graphs)
