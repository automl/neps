"""Encoding of hyperparameter configurations into tensors.

For the most part, you can just use
[`ConfigEncoder.from_parameters()`][neps.space.encoding.ConfigEncoder.from_parameters]
to create an encoder over a list of hyperparameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from typing_extensions import Protocol, override

import torch

from neps.space.domain import Domain
from neps.space.parameters import HPOCategorical, HPOFloat, HPOInteger, Parameter

V = TypeVar("V", int, float)


class TensorTransformer(Protocol[V]):
    """A protocol for encoding and decoding hyperparameter values into tensors."""

    domain: Domain[V]

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

    def encode_one(
        self,
        x: Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> V:
        """Encode a single hyperparameter value into a tensor.

        Args:
            x: A single hyperparameter value.
            dtype: The dtype of the tensor.
            device: The device of the tensor.

        Returns:
            The encoded tensor.
        """
        return self.encode([x], dtype=dtype, device=device).item()  # type: ignore


@dataclass
class CategoricalToIntegerTransformer(TensorTransformer[int]):
    """A transformer that encodes categorical values into integers."""

    choices: Sequence[Any]

    domain: Domain[int] = field(init=False)
    _lookup: dict[Any, int] | None = field(init=False)

    def __post_init__(self) -> None:
        assert len(self.choices) > 0

        self.domain = Domain.indices(len(self.choices), is_categorical=True)
        self._lookup = None
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
class CategoricalToUnitNorm(TensorTransformer[float]):
    """A transformer that encodes categorical values into a unit normalized tensor.

    If there are `n` choices, the tensor will have `n` bins between `0` and `1`.
    """

    choices: Sequence[Any]

    domain: Domain[float] = field(init=False)
    _cat_int_domain: Domain[int] = field(init=False)
    _lookup: dict[Any, int] | None = field(init=False)

    def __post_init__(self) -> None:
        self.domain = Domain.floating(
            0.0,
            1.0,
            bins=len(self.choices),
            is_categorical=True,
        )
        self._cat_int_domain = Domain.indices(len(self.choices), is_categorical=True)
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
        integers = torch.tensor(values, dtype=torch.int64, device=device)
        binned_floats = self.domain.cast(
            integers,
            frm=self._cat_int_domain,
            dtype=dtype,
        )
        if out is not None:
            return out.copy_(binned_floats)

        return binned_floats

    @override
    def decode(self, x: torch.Tensor) -> list[Any]:
        x = self._cat_int_domain.cast(x, frm=self.domain)
        return [self.choices[int(i)] for i in torch.round(x).tolist()]


# TODO: Maybe add a shift argument, could be useful to have `0` as midpoint
# and `-0.5` as lower bound with `0.5` as upper bound.
@dataclass
class MinMaxNormalizer(TensorTransformer[float], Generic[V]):
    """A transformer that normalizes values to the unit interval."""

    original_domain: Domain[V]
    bins: int | None = None

    domain: Domain[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.bins is None:
            self.domain = Domain.unit_float()
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
    * [`from_parameters()`][neps.space.encoding.ConfigEncoder.from_parameters]: Create a
        default encoder over a list of hyperparameters. Please see the method docs for
        more details on how it encodes different types of hyperparameters.
    * [`encode()`]]neps.space.encoding.ConfigEncoder.encode]: Encode a list of
        configurations into a single tensor using the transforms of the encoder.
    * [`decode()`][neps.space.encoding.ConfigEncoder.decode]: Decode a 2d tensor
        of length `N` into a list of `N` configurations.
    * [`domains`][neps.space.encoding.ConfigEncoder.domains]: The
        [`Domain`][neps.space.domain.Domain] that each hyperparameter is encoded
        into. This is useful in combination with classes like
        [`Sampler`][neps.sampling.samplers.Sampler],
        [`Prior`][neps.sampling.Prior], and
        [`TorchDistributionWithDomain`][neps.sampling.distributions.TorchDistributionWithDomain],
        which require knowledge of the
        domains of each column for the tensor, for example, to sample values directly
        into the encoded space, getting log probabilities of the encoded values.
    * [`ndim`][neps.space.encoding.ConfigEncoder.ndim]: The number of columns
        in the encoded tensor, useful for initializing some `Sampler`s.
    """

    transformers: dict[str, TensorTransformer]
    constants: Mapping[str, Any] = field(default_factory=dict)

    # These are all just computed properties for easier logic
    index_of: dict[str, int] = field(init=False)
    domain_of: dict[str, Domain] = field(init=False)
    n_numerical: int = field(init=False)
    n_categorical: int = field(init=False)
    categorical_slice: slice | None = field(init=False)
    numerical_slice: slice | None = field(init=False)
    numerical_domains: list[Domain] = field(init=False)
    categorical_domains: list[Domain] = field(init=False)

    def __post_init__(self) -> None:
        # Sort such that numericals are sorted first and categoricals after,
        # with sorting within each group being done by name
        transformers = sorted(
            self.transformers.items(),
            key=lambda t: (t[1].domain.is_categorical, t[0]),
        )
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
        self.numerical_domains = [
            t.domain for t in self.transformers.values() if not t.domain.is_categorical
        ]
        self.categorical_domains = [
            t.domain for t in self.transformers.values() if t.domain.is_categorical
        ]
        self.numerical_slice = slice(0, n_numerical) if n_numerical > 0 else None
        self.categorical_slice = (
            slice(n_numerical, n_numerical + n_categorical) if n_categorical > 0 else None
        )

    def pdist(
        self,
        x: torch.Tensor,
        *,
        numerical_ord: int = 2,
        categorical_ord: int = 0,
        dtype: torch.dtype = torch.float64,
        square_form: bool = False,
    ) -> torch.Tensor:
        """Compute the pairwise distance between rows of a tensor.

        Will sum the results of the numerical and categorical distances.
        The encoding will be normalized such that all numericals lie within the unit
        cube, and categoricals will by default, have a `p=0` norm, which is equivalent
        to the Hamming distance.

        Args:
            x: A tensor of shape `(N, ncols)`.
            numerical_ord: The order of the norm to use for the numerical columns.
            categorical_ord: The order of the norm to use for the categorical columns.
            dtype: The dtype of the output tensor.
            square_form: If `True`, the output will be a square matrix of shape
                `(N, N)`. If `False`, the output will be a single dim tensor of shape
                `1/2 * N * (N - 1)`.

        Returns:
            The distances, shaped according to `square_form`.
        """
        dists: torch.Tensor | None = None
        if self.numerical_slice is not None:
            # Ensure they are all within the unit cube
            numericals = Domain.translate(
                x[..., self.numerical_slice],
                frm=self.numerical_domains,
                to=Domain.unit_float(),
            )

            dists = torch.nn.functional.pdist(numericals, p=numerical_ord)

        if self.categorical_slice is not None:
            cat_dists = torch.nn.functional.pdist(
                x[..., self.categorical_slice],
                p=categorical_ord,
            )
            dists = cat_dists if dists is None else (dists + cat_dists)

        if dists is None:
            raise ValueError("No columns to compute distances on.")

        if not square_form:
            return dists

        # Turn the single dimensional vector into a square matrix
        N = len(x)
        sq = torch.zeros((N, N), dtype=dtype)
        row_ix, col_ix = torch.triu_indices(N, N, offset=1)
        sq[row_ix, col_ix] = dists
        sq[col_ix, row_ix] = dists
        return sq

    @property
    def ndim(self) -> int:
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

    def decode_one(self, x: torch.Tensor) -> dict[str, Any]:
        """Decode a tensor representing one configuration into a dict."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.decode(x)[0]

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

    def decode_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a tensor of hyperparameter configurations into the original tensor.

        Args:
            x: A tensor of shape `(N, ncols)` containing the encoded configurations.

        Returns:
            A list of `N` configurations, including any constants that were included
            when creating the encoder.
        """
        decoded_x = torch.empty_like(x)
        N = len(x)
        for hp_name, transformer in self.transformers.items():
            lookup = self.index_of[hp_name]
            t = x[..., lookup]
            decoded_x[..., lookup] = torch.tensor(transformer.original_domain.from_unit(t))
        return decoded_x


    @classmethod
    def from_parameters(
        cls,
        parameters: Mapping[str, Parameter],
        *,
        custom_transformers: Mapping[str, TensorTransformer] | None = None,
    ) -> ConfigEncoder:
        """Create a default encoder over a list of hyperparameters.

        This method creates a default encoder over a list of hyperparameters. It
        automatically creates transformers for each hyperparameter based on its type.

        The transformers are as follows:

        * `Float` and `Integer` are normalized to the unit interval.
        * `Categorical` is transformed into an integer.

        Args:
            parameters: The parameters to build an encoder for
            custom_transformers: A mapping of hyperparameter names
                to custom transformers to use

        Returns:
            A `ConfigEncoder` instance
        """
        custom = custom_transformers or {}
        transformers: dict[str, TensorTransformer] = {}
        for name, hp in parameters.items():
            if name in custom:
                transformers[name] = custom[name]
                continue

            match hp:
                case HPOFloat() | HPOInteger():
                    transformers[name] = MinMaxNormalizer(hp.domain)  # type: ignore
                case HPOCategorical():
                    transformers[name] = CategoricalToIntegerTransformer(hp.choices)
                case _:
                    raise ValueError(f"Unsupported parameter type: {type(hp)}.")

        return cls(transformers)
