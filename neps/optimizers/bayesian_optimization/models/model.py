from abc import ABC
from typing import Callable, Generic, Sequence, TypeVar

from neps.search_spaces.config import Config
from neps.search_spaces.search_space import SearchSpace


E = TypeVar("E")


class SurrogateModel(ABC, Generic[E]):
    config_encoder: Callable[[Sequence[Config], SearchSpace], E]
