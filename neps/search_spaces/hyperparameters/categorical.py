from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import Literal, TypeAlias

from neps.env import ALLOW_LARGE_CATEGORIES
from neps.search_spaces.domain import Domain

if TYPE_CHECKING:
    from neps.utils.types import Arr, i64

# Should probably replace this with an enum
Confidence: TypeAlias = Literal["low", "medium", "high"]


@dataclass
class CategoricalParameter:
    """A list of **unordered** choices for a parameter.

    ```python
    import neps

    optimizer_choice = neps.CategoricalParameter(
        ["adam", "sgd", "rmsprop"],
        default="adam"
    )
    ```

    Args:
        choices: choices for the hyperparameter.
        default: default value for the hyperparameter, must be in `choices=`
            if provided.
        default_confidence: confidence score for the default value, used when
            condsider prior based optimization.
    """

    CATEGORY_SIZE_TO_PREFER_HASH: ClassVar[int] = 10
    LARGE_CATEGORY_LIMIT: ClassVar[int] = 20

    choices: list[float | int | str]
    default: Any | None = None
    default_confidence: Confidence = "low"

    domain: Domain[i64] = field(init=False, repr=False)
    _lookup: dict[Any, int] | None = field(init=False, repr=False)
    default_index: int | None = field(init=False, repr=False)
    size: int = field(init=False, repr=False)

    def __post_init__(self):
        self.choices = list(self.choices)
        if not ALLOW_LARGE_CATEGORIES and len(self.choices) > self.LARGE_CATEGORY_LIMIT:
            raise ValueError(
                f"NePS was not designed to handle more than {self.LARGE_CATEGORY_LIMIT}"
                " categories. Many operations will be slow or fail. Please let us know"
                " your use case! To remove this restriction, set the env var"
                " `NEPS_ALLOW_LARGE_CATEGORIES=1` or set"
                " `CategoricalParameter.LARGE_CATEGORY_LIMIT` to at"
                f" least {len(self.choices)} to acommodate this parameter."
            )
        if self.default:
            try:
                self.default_index = self.choices.index(self.default)
            except ValueError as e:
                raise ValueError(
                    f"Default value of `{self.default}` must be in choices."
                    f" Choices gotten are: {self.choices}"
                ) from e
        else:
            self.default_index = False

        self.domain = Domain.indices(len(self.choices))
        self.size = len(self.choices)

        self._lookup: dict[Any, int] | None = None
        # Prefer hash lookup for large categories, otherwise an interated
        # index lookup should be faster
        if len(self.choices) > self.CATEGORY_SIZE_TO_PREFER_HASH:
            try:
                self._lookup = {v: i for i, v in enumerate(self.choices)}
            except TypeError:
                self._lookup = None
        else:
            self._lookup = None

    def lookup(self, indices: list[int] | Arr[i64]) -> list[float | int | str]:
        return [self.choices[i] for i in indices]

    def index(self, value: Any) -> int:
        if self._lookup is not None:
            return self._lookup[value]
        return self.choices.index(value)
