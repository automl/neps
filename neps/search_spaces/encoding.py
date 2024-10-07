"""Encoding of hyperparameter configurations into tensors.

For the most part, you can just use
[`ConfigEncoder.default()`][neps.search_spaces.encoding.ConfigEncoder.default]
to create an encoder over a list of hyperparameters, along with any constants you
want to include when decoding configurations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar
from typing_extensions import Protocol, override

import torch

from neps.search_spaces.domain import UNIT_FLOAT_DOMAIN, Domain
from neps.search_spaces.hyperparameters.categorical import CategoricalParameter
from neps.search_spaces.hyperparameters.float import FloatParameter
from neps.search_spaces.hyperparameters.integer import IntegerParameter

if TYPE_CHECKING:
    from neps.search_spaces.parameter import Parameter

WLInput: TypeAlias = tuple[dict, dict | None, dict | None]
V = TypeVar("V", int, float)


class TensorTransformer(Protocol):
    """A protocol for encoding and decoding hyperparameter values into tensors."""

    domain: Domain

    def encode(
        self,
        x: Sequence[Any],
        *,
        out: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode a sequence of hyperparameter values into a tensor.

        Args:
            x: A sequence of hyperparameter values.
            out: An optional tensor to write the encoded values to.
            dtype: The dtype of the tensor.
            device: The device of the tensor.

        Returns:
            The encoded tensor.
        """
        ...

    def decode(self, x: torch.Tensor) -> list[Any]:
        """Decode a tensor of hyperparameter values into a sequence of values.

        Args:
            x: A tensor of hyperparameter values.

        Returns:
            A sequence of hyperparameter values.
        """
        ...


@dataclass
class CategoricalToIntegerTransformer(TensorTransformer):
    """A transformer that encodes categorical values into integers."""

    choices: Sequence[Any]

    domain: Domain = field(init=False)
    _lookup: dict[Any, int] | None = field(init=False)

    def __post_init__(self) -> None:
        assert len(self.choices) > 0

        self.domain = Domain.indices(len(self.choices), is_categorical=True)
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
    """A transformer that encodes categorical values into a unit normalized tensor.

    If there are `n` choices, the tensor will have `n` bins between `0` and `1`.
    """

    choices: Sequence[Any]

    domain: Domain = field(init=False)
    _integer_transformer: CategoricalToIntegerTransformer = field(init=False)

    def __post_init__(self) -> None:
        self.domain = Domain.floating(
            0.0,
            1.0,
            bins=len(self.choices),
            is_categorical=True,
        )
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
        )
        binned_floats = self.domain.cast(
            integers, frm=self._integer_transformer.domain, dtype=dtype
        )
        if out is not None:
            return out.copy_(binned_floats)

        return binned_floats

    @override
    def decode(self, x: torch.Tensor) -> list[Any]:
        x = torch.round(x * (len(self.choices) - 1)).type(torch.int64)
        return self._integer_transformer.decode(x)


# TODO: Maybe add a shift argument, could be useful to have `0` as midpoint
# and `-0.5` as lower bound with `0.5` as upper bound.
@dataclass
class MinMaxNormalizer(TensorTransformer, Generic[V]):
    """A transformer that normalizes values to the unit interval."""

    original_domain: Domain[V]
    bins: int | None = None

    domain: Domain[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.bins is None:
            self.domain = UNIT_FLOAT_DOMAIN
        else:
            self.domain = Domain.floating(0.0, 1.0, bins=self.bins)

    @override
    def encode(
        self,
        x: Sequence[V],
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
class ConfigEncoder:
    """An encoder for hyperparameter configurations.

    This class is used to encode and decode hyperparameter configurations into tensors
    and back. It's main uses currently are to support surrogate models that require
    tensors.

    The primary methods/properties to be aware of are:
    * [`default()`](neps.search_spaces.encoding.ConfigEncoder.default]: Create a default
        encoder over a list of hyperparameters. Please see the method docs for more
        details on how it encodes different types of hyperparameters.
    * [`encode()`]]neps.search_spaces.encoding.ConfigEncoder.encode]: Encode a list of
        configurations into a single tensor using the transforms of the encoder.
    * [`decode()`][neps.search_spaces.encoding.ConfigEncoder.decode]: Decode a 2d tensor
        of length `N` into a list of `N` configurations.
    * [`domains`][neps.search_spaces.encoding.ConfigEncoder.domains): The
        [`Domain`][neps.search_spaces.domain.Domain] that each hyperparameter is encoded
        into. This is useful in combination with classes like
        [`Sampler`][neps.sampling.samplers.Sampler],
        [`Prior`][neps.sampling.priors.Prior], and
        [`TorchDistributionWithDomain`][neps.sampling.distributions.TorchDistributionWithDomain],
        which require knowledge of the
        domains of each column for the tensor, for example, to sample values directly
        into the encoded space, getting log probabilities of the encoded values.
    * [`ncols`][neps.search_spaces.encoding.ConfigEncoder.ncols]: The number of columns
        in the encoded tensor, useful for initializing some `Sampler`s.
    """

    transformers: dict[str, TensorTransformer]
    constants: Mapping[str, Any] = field(default_factory=dict)

    index_of: dict[str, int] = field(init=False)
    domain_of: dict[str, Domain] = field(init=False)
    n_numerical: int = field(init=False)
    n_categorical: int = field(init=False)

    def __post_init__(self) -> None:
        transformers = sorted(self.transformers.items(), key=lambda t: t[0])
        self.transformers = dict(transformers)

        n_numerical = 0
        n_categorical = 0
        for _, transformer in transformers:
            if transformer.domain.is_categorical:
                n_categorical += 1
            else:
                n_numerical += 1

        self.index_of = {name: i for i, name in enumerate(self.transformers.keys())}
        self.domain_of = {name: t.domain for name, t in self.transformers.items()}
        self.n_numerical = n_numerical
        self.n_categorical = n_categorical

    @property
    def ncols(self) -> int:
        """The number of columns in the encoded tensor."""
        return len(self.transformers)

    @property
    def domains(self) -> list[Domain]:
        """The domains of the encoded hyperparameters."""
        return list(self.domain_of.values())

    def encode(
        self,
        x: Sequence[Mapping[str, Any]],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Encode a list of hyperparameter configurations into a tensor.

        !!! warning "Constants"

            Constants included in configurations will not be encoded into the tensor,
            but are included when decoding.

        !!! warning "Parameters with no transformers"

            Any parameters in the configurations, whos key is not in
            `self.transformers`, will be ignored.

        Args:
            x: A list of hyperparameter configurations.
            device: The device of the tensor.
            dtype: The dtype of the tensor.

        Returns:
            A tensor of shape `(len(x), ncols)` containing the encoded configurations.
        """
        dtype = torch.float64 if dtype is None else dtype
        width = len(self.transformers)
        buffer = torch.empty((len(x), width), dtype=dtype, device=device)

        for hp_name, transformer in self.transformers.items():
            values = [conf[hp_name] for conf in x]
            lookup = self.index_of[hp_name]

            # Encode directly into buffer
            transformer.encode(
                values,
                out=buffer[:, lookup],
                dtype=dtype,
                device=device,
            )

        return buffer

    def decode(self, x: torch.Tensor) -> list[dict[str, Any]]:
        """Decode a tensor of hyperparameter configurations into a list of configurations.

        Args:
            x: A tensor of shape `(N, ncols)` containing the encoded configurations.

        Returns:
            A list of `N` configurations, including any constants that were included
            when creating the encoder.
        """
        values: dict[str, list[Any]] = {}
        N = len(x)
        for hp_name, transformer in self.transformers.items():
            lookup = self.index_of[hp_name]
            tensor = x[:, lookup]
            values[hp_name] = transformer.decode(tensor)

        constants = {name: [v] * N for name, v in self.constants.items()}
        values.update(constants)

        keys = list(values.keys())
        return [
            dict(zip(keys, vals, strict=False))
            for vals in zip(*values.values(), strict=False)
        ]

    @classmethod
    def default(
        cls,
        parameters: Mapping[str, Parameter],
        constants: Mapping[str, Any] | None = None,
        *,
        custom_transformers: dict[str, TensorTransformer] | None = None,
    ) -> ConfigEncoder:
        """Create a default encoder over a list of hyperparameters.

        This method creates a default encoder over a list of hyperparameters. It
        automatically creates transformers for each hyperparameter based on its type.
        The transformers are as follows:

        * `FloatParameter` and `IntegerParameter` are normalized to the unit interval.
        * `CategoricalParameter` is transformed into an integer.

        Args:
            parameters: A mapping of hyperparameter names to hyperparameters.
            constants: A mapping of constant hyperparameters to include when decoding.
            custom_transformers: A mapping of hyperparameter names to custom transformers.

        Returns:
            A `ConfigEncoder` instance
        """
        if constants is not None:
            overlap = set(parameters) & set(constants)
            if any(overlap):
                raise ValueError(
                    "`constants=` and `parameters=` cannot have overlapping"
                    f" keys: {overlap=}"
                )
            if custom_transformers is not None:
                overlap = set(custom_transformers) & set(constants)
                if any(overlap):
                    raise ValueError(
                        f"Can not apply `custom_transformers=`"
                        f" to `constants=`: {overlap=}"
                    )
        else:
            constants = {}

        custom = custom_transformers or {}
        sorted_params = sorted(parameters.items())
        transformers: dict[str, TensorTransformer] = {}
        for name, hp in sorted_params:
            if name in custom:
                transformers[name] = custom[name]
                continue

            match hp:
                case FloatParameter() | IntegerParameter():
                    transformers[name] = MinMaxNormalizer(hp.domain)  # type: ignore
                case CategoricalParameter():
                    transformers[name] = CategoricalToIntegerTransformer(hp.choices)
                case _:
                    raise ValueError(
                        f"Unsupported parameter type: {type(hp)}. If hp is a constant, "
                        " please provide it as `constants=`."
                    )

        return ConfigEncoder(transformers, constants=constants)