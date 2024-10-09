"""The base [`Parameter`][neps.search_spaces.Parameter] class.

The `Parameter` refers to both the hyperparameter definition but also
holds a [`.value`][neps.search_spaces.Parameter.value] which can be
set or empty, in which case it is `None`.

!!! tip

    A `Parameter` which allows for mutations and crossovers should implement
    the [`MutatableParameter`][neps.search_spaces.MutatableParameter] protocol.

!!! tip

    A `Parameter` which allows for defining a
    [`.default`][neps.search_spaces.Parameter.default] and some prior,
    i.e. some default value along with a confidence that this is a good setting,
    should implement the [`ParameterWithPrior`][neps.search_spaces.ParameterWithPrior]
    class.

    This is utilized by certain optimization routines to inform the search process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Generic, Protocol, TypeVar, runtime_checkable
from typing_extensions import Self

ValueT = TypeVar("ValueT")
SerializedT = TypeVar("SerializedT")


class Parameter(ABC, Generic[ValueT, SerializedT]):
    """A base class for hyperparameters.

    Attributes:
        default: default value for the hyperparameter. This value
            is used as a prior to inform algorithms about a decent
            default value for the hyperparameter, as well as use
            attributes from [`ParameterWithPrior`][neps.search_spaces.ParameterWithPrior],
            to aid in optimization.
        is_fidelity: whether the hyperparameter is fidelity.
        value: value for the hyperparameter, if any.
        normalized_value: normalized value for the hyperparameter.
    """

    def __init__(
        self,
        *,
        value: ValueT | None,
        default: ValueT | None,
        is_fidelity: bool,
    ):
        """Create a new `Parameter`.

        Args:
            value: value for the hyperparameter.
            default: default value for the hyperparameter.
            is_fidelity: whether the hyperparameter is fidelity.
        """
        self.default = default
        self.is_fidelity = is_fidelity

        # TODO(eddiebergman): The reason to have this not as a straight alone
        # attribute is that the graph parameters currently expose there own
        # way of calculating a value on demand.
        # To fix this would mean to essentially decouple GraphParameter entirely
        # from Parameter as it's less of a heirarchy and more of just a small overlap
        # of functionality.
        self._value = value
        self.normalized_value = (
            self.value_to_normalized(value) if value is not None else None
        )

        # TODO: Pass in through subclasses
        self.default_confidence_score: float

    # TODO(eddiebergman): All this does is just check values which highly unlikely
    # what we want. However this needs to be tackled in a seperate PR.
    #
    # > The Princess is in another castle.
    #
    def __eq__(self, other: Any) -> bool:
        # Assuming that two different classes should represent two different parameters
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.value is not None and other.value is not None:
            return self.serialize_value(self.value) == self.serialize_value(other.value)

        return False

    @abstractmethod
    def clone(self) -> Self:
        """Create a copy of the `Parameter`."""

    @property
    def value(self) -> ValueT | None:
        """Get the value of the hyperparameter, or `None` if not set."""
        return self._value

    def sample(self) -> Self:
        """Sample a new version of this `Parameter` with a random value.

        Will set the [`.value`][neps.search_spaces.Parameter.value] to the
        sampled value.

        Returns:
            A new `Parameter` with a sampled value.
        """
        value = self.sample_value()
        copy_self = self.clone()
        copy_self.set_value(value)
        return copy_self

    @abstractmethod
    def sample_value(self) -> ValueT:
        """Sample a new value."""

    @abstractmethod
    def set_default(self, default: ValueT | None) -> None:
        """Set the default value for the hyperparameter.

        The `default=` is used as a prior and used to inform
        algorithms about a decent default value for the hyperparameter.

        Args:
            default: default value for the hyperparameter.
        """

    @abstractmethod
    def set_value(self, value: ValueT | None) -> None:
        """Set the value for the hyperparameter.

        Args:
            value: value for the hyperparameter.
        """

    @abstractmethod
    def value_to_normalized(self, value: ValueT) -> float:
        """Convert a value to a normalized value.

        Normalization is different per hyperparameter type,
        but roughly refers to numeric values.

        * `(0, 1)` scaling in the case of
            a [`NumericalParameter`][neps.search_spaces.NumericalParameter],
        * `{0.0, 1.0}` for a [`ConstantParameter`][neps.search_spaces.ConstantParameter],
        * `[0, 1, ..., n]` for a
            [`Categorical`][neps.search_spaces.CategoricalParameter].

        Args:
            value: value to convert.

        Returns:
            The normalized value.
        """

    @abstractmethod
    def normalized_to_value(self, normalized_value: float) -> ValueT:
        """Convert a normalized value back to value in the defined hyperparameter range.

        Args:
            normalized_value: normalized value to convert.

        Returns:
            The value.
        """

    @abstractmethod
    def _get_non_unique_neighbors(
        self,
        num_neighbours: int,
        *,
        std: float = 0.2,
    ) -> list[Self]: ...

    @classmethod
    @abstractmethod
    def serialize_value(cls, value: ValueT) -> SerializedT:
        """Ensure the hyperparameter value is in a serializable format.


        Returns:
            A serializable version of the hyperparameter value
        """

    @classmethod
    @abstractmethod
    def deserialize_value(cls, value: SerializedT) -> ValueT:
        """Deserialize a serialized value into the hyperparameter's value.

        Args:
            value: value to deserialize.
        """

    def load_from(self, value: SerializedT) -> None:
        """Load a serialized value into the hyperparameter's value.

        Args:
            value: value to load.
        """
        deserialized_value = self.deserialize_value(value)
        self.set_value(deserialized_value)


class ParameterWithPrior(Parameter[ValueT, SerializedT]):
    """A base class for hyperparameters with priors.

    Attributes:
        default_confidence_choice: The choice of how confident any algorithm should
            be in the default value being a good value.
        default_confidence_score: A score used by algorithms to utilize the default value.
        has_prior: whether the hyperparameter has a prior that can be used by an
            algorithm. In many cases, this refers to having a default value.
    """

    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, float]]
    default_confidence_choice: str
    default_confidence_score: float
    has_prior: bool

    @abstractmethod
    def compute_prior(self, *, log: bool = True) -> float:
        """Compute the likelihood of the currently set value from
        the sampling distribution of the hyperparameter.

        Args:
            log: whether to return the log likelihood.
        """

    # NOTE(eddiebergman): Like the normal `Parameter.sample` but with `user_priors`.
    @abstractmethod
    def sample_value(self, *, user_priors: bool = False) -> ValueT:
        """Sample a new value.

        Similar to
        [`Parameter.sample_value()`][neps.search_spaces.Parameter.sample_value],
        but a `ParameterWithPrior` can use the confidence score by setting
        `user_priors=True`.

        Args:
            user_priors: whether to use the confidence score
                when sampling a value.

        Returns:
            The sampled value.
        """

    def set_default_confidence_score(self, default_confidence: str) -> None:
        """Set the default confidence score for the hyperparameter.

        Args:
            default_confidence: the choice of how confident any algorithm should
                be in the default value being a good value.

        Raises:
            ValueError: if the confidence score is not a valid choice.
        """
        if default_confidence not in self.DEFAULT_CONFIDENCE_SCORES:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Invalid default confidence score: {default_confidence}"
                f" for {cls_name}. Expected one of:"
                f" {list(self.DEFAULT_CONFIDENCE_SCORES.keys())}"
            )

        self.default_confidence_score = self.DEFAULT_CONFIDENCE_SCORES[default_confidence]

    def sample(self, *, user_priors: bool = False) -> Self:
        """Sample a new version of this `Parameter` with a random value.

        Similar to
        [`Parameter.sample()`][neps.search_spaces.Parameter.sample],
        but a `ParameterWithPrior` can use the confidence score by setting
        `user_priors=True`.

        Args:
            user_priors: whether to use the confidence score
                when sampling a value.

        Returns:
            A new `Parameter` with a sampled value.
        """
        value = self.sample_value(user_priors=user_priors)
        copy_self = self.clone()
        copy_self.set_value(value)
        return copy_self


@runtime_checkable
class MutatableParameter(Protocol):
    """A protocol for hyperparameters that can be mutated.

    Particpating parameters must implement the
    [`mutate()`][neps.search_spaces.MutatableParameter.mutate] method
    and the [`crossover()`][neps.search_spaces.MutatableParameter.crossover]
    method.
    """

    def mutate(self, parent: Self | None = None) -> Self:
        """Mutate the hyperparameter.

        Args:
            parent: the parent hyperparameter to mutate from.

        Returns:
            The mutated hyperparameter.
        """
        ...

    def crossover(self, parent1: Self, parent2: Self | None = None) -> tuple[Self, Self]:
        """Crossover the hyperparameter with another hyperparameter.

        Args:
            parent1: the first parent hyperparameter.
            parent2: the second parent hyperparameter.
                If left as `None`, this hyperparameter will be used as the second parent
                to crossover with.

        Returns:
            A tuple of the two crossovered hyperparameters.
        """
        ...
