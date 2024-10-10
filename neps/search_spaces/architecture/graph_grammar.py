from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Any, ClassVar, Mapping
from typing_extensions import override, Self
from neps.utils.types import NotSet
from typing import TYPE_CHECKING, Any, ClassVar, Mapping
from typing_extensions import Self, override

import networkx as nx

from neps.search_spaces.parameter import ParameterWithPrior
from neps.utils.types import NotSet

from .core_graph_grammar import CoreGraphGrammar
from .mutations import bananas_mutate, simple_mutate

if TYPE_CHECKING:
    from .cfg import Grammar


# TODO(eddiebergman): This is a halfway solution, but essentially a lot
# of things `Parameter` does, does not fit nicely with a Graph based
# parameters, in the future we probably better just have these as two seperate
# classes. For now, this class sort of captures the overlap between
# `Parameter` and Graph based parameters.
# The problem here is that the `Parameter` expects the `load_from`
# and the `.value` to be the same type, which is not the case for
# graph based parameters.
class GraphParameter(  # noqa: D101
    ParameterWithPrior[nx.DiGraph, str]
):
    # NOTE(eddiebergman): What I've managed to learn so far is that
    # these hyperparameters work mostly with strings externally,
    # i.e. setting the value through `load_from` or `set_value` should be a string.
    # At that point, the actual `.value` is a graph object created from said
    # string. This would most likely break with a few things in odd places
    # and I'm surprised it's lasted this long.
    # At serialization time, it doesn't actually serialize the .value but instead
    # relies on the string it was passed initially, I'm not actually sure if there's
    # a way to go from the graph object to the string in this code...
    # Essentially on the outside, we need to ensure we don't pass ih the graph object
    # itself
    DEFAULT_CONFIDENCE_SCORES: ClassVar[Mapping[str, float]] = {"not_in_use": 1.0}
    default_confidence_choice = "not_in_use"
    has_prior: bool
    input_kwargs: dict[str, Any]

    @property
    @abstractmethod
    def id(self) -> str: ...

    # NOTE(eddiebergman): Unlike traditional parameters, it seems
    @property
    @abstractmethod
    def value(self) -> nx.DiGraph: ...

    # NOTE(eddiebergman): This is a function common to the three graph
    # parameters that is used for `load_from`
    @abstractmethod
    def create_from_id(self, value: str) -> None: ...

    # NOTE(eddiebergman): Function shared between graph parameters.
    # Used to `set_value()`
    @abstractmethod
    def reset(self) -> None: ...

    @override
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GraphGrammar):
            return NotImplemented

        return self.id == other.id

    @abstractmethod
    def compute_prior(self, normalized_value: float) -> float: ...

    @override
    def set_value(self, value: str | None) -> None:
        # NOTE(eddiebergman): Not entirely sure how this should be done
        # as previously this would have just overwritten a property method
        # `self.value = None`
        if not isinstance(value, str):
            raise ValueError(
                "Expected a string for setting value a `GraphParameter`",
                f" got {type(value)}",
            )
        self.reset()
        self.normalized_value = value

        if value is None:
            return

        self.create_from_id(value)

    @override
    def set_default(self, default: str | None) -> None:
        # TODO(eddiebergman): Could find no mention of the word 'default' in the
        # GraphGrammers' hence... well this is all I got
        self.default = default

    @override
    def sample_value(self, *, user_priors: bool = False) -> nx.DiGraph:
        # TODO(eddiebergman): This could definitely be optimized
        # Right now it copies the entire object just to get a value out
        # of it.
        return self.sample(user_priors=user_priors).value

    @classmethod
    def serialize_value(cls, value: nx.DiGraph) -> str:
        """Functionality relying on this for GraphParameters should
        special case and use `self.id`.

        !!! warning

            Graph parameters don't directly support serialization.
            Instead they rely on holding on to the original string value
            from which they were created from.
        """
        raise NotImplementedError

    @classmethod
    def deserialize_value(cls, value: str) -> nx.DiGraph:
        """Functionality relying on this for GraphParameters should
        special case for whever this is needed...

        !!! warning

            Graph parameters don't directly support serialization.
            Instead they rely on holding on to the original string value
            from which they were created from.
        """
        raise NotImplementedError

    @override
    def load_from(self, value: str | Self) -> None:
        if isinstance(value, GraphParameter):
            value = value.id
        self.create_from_id(value)

    @abstractmethod
    def mutate(  # noqa: D102
        self, parent: Self | None = None, *, mutation_strategy: str = "bananas"
    ) -> Self: ...

    def _get_non_unique_neighbors(self, num_neighbours: int) -> list[Self]:
        raise NotImplementedError

    def value_to_normalized(self, value: nx.DiGraph) -> float:  # noqa: D102
        raise NotImplementedError

    def normalized_to_value(self, normalized_value: float) -> nx.DiGraph:  # noqa: D102
        raise NotImplementedError

    @override
    def clone(self) -> Self:
        new_self = self.__class__(**self.input_kwargs)

        # HACK(eddiebergman): It seems the subclasses all have these and
        # so we just copy over those attributes, deepcloning anything that is mutable
        if self._value is not None:
            _attrs_that_subclasses_use_to_reoresent_a_value = (
                ("_value", True),
                ("string_tree", False),
                ("string_tree_list", False),
                ("_function_id", False),
            )
            for _attr, is_mutable in _attrs_that_subclasses_use_to_reoresent_a_value:
                retrieved_attr = getattr(self, _attr, NotSet)
                if retrieved_attr is NotSet:
                    continue

                if is_mutable:
                    setattr(new_self, _attr, deepcopy(retrieved_attr))
                else:
                    setattr(new_self, _attr, retrieved_attr)

        return new_self


class GraphGrammar(GraphParameter, CoreGraphGrammar):
    hp_name = "graph_grammar"

    def __init__(  # noqa: D107, PLR0913
        self,
        grammar: Grammar,
        terminal_to_op_names: dict,
        prior: dict | None = None,
        terminal_to_graph_edges: dict | None = None,
        edge_attr: bool = True,  # noqa: FBT001, FBT002
        edge_label: str = "op_name",
        zero_op: list | None = None,
        identity_op: list | None = None,
        new_graph_repr_func: bool = False,  # noqa: FBT001, FBT002
        name: str | None = None,
        scope: str | None = None,
        **kwargs,
    ):
        if identity_op is None:
            identity_op = ["Identity", "id"]
        if zero_op is None:
            zero_op = ["Zero", "zero"]
        if isinstance(grammar, list) and len(grammar) != 1:
            raise NotImplementedError("Does not support multiple grammars")

        CoreGraphGrammar.__init__(
            self,
            grammars=grammar,
            terminal_to_op_names=terminal_to_op_names,
            terminal_to_graph_edges=terminal_to_graph_edges,
            edge_attr=edge_attr,
            edge_label=edge_label,
            zero_op=zero_op,
            identity_op=identity_op,
            name=name,
            scope=scope,
            **kwargs,
        )
        GraphParameter.__init__(self, value=None, default=None, is_fidelity=False)

        self.string_tree: str = ""
        self._function_id: str = ""
        self.new_graph_repr_func = new_graph_repr_func

        if prior is not None:
            self.grammars[0].prior = prior
        self.has_prior = prior is not None

    @override
    def sample(self, *, user_priors: bool = False) -> Self:
        copy_self = self.clone()
        copy_self.reset()
        copy_self.string_tree = copy_self.grammars[0].sampler(1, user_priors=user_priors)[
            0
        ]
        _ = copy_self.value  # required for checking if graph is valid!
        return copy_self

    @property
    @override
    def value(self) -> nx.DiGraph:
        if self._value is None:
            if self.new_graph_repr_func:
                self._value = self.get_graph_representation(
                    self.id,
                    self.grammars[0],
                    edge_attr=self.edge_attr,
                )
                assert isinstance(self._value, nx.DiGraph)
            else:
                _value = self.from_stringTree_to_graph_repr(
                    self.string_tree,
                    self.grammars[0],
                    valid_terminals=self.terminal_to_op_names.keys(),
                    edge_attr=self.edge_attr,
                )
                # NOTE: This asumption was not true but I don't really know
                # how to handle it otherwise, will just leave it as is for now
                #  -x- assert isinstance(_value, nx.DiGraph), _value
                self._value = _value
        return self._value

    @override
    def mutate(
        self,
        parent: GraphGrammar | None = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ) -> Self:
        if parent is None:
            parent = self
        parent_string_tree = parent.string_tree

        if mutation_strategy == "bananas":
            child_string_tree, is_same = bananas_mutate(
                parent_string_tree=parent_string_tree,
                grammar=self.grammars[0],
                mutation_rate=mutation_rate,
            )
        else:
            child_string_tree, is_same = simple_mutate(
                parent_string_tree=parent_string_tree,
                grammar=self.grammars[0],
            )

        if is_same:
            raise Exception("Parent is the same as child!")

        return parent.create_new_instance_from_id(
            self.string_tree_to_id(child_string_tree)
        )

    @override
    def compute_prior(self, *, log: bool = True) -> float:
        return self.grammars[0].compute_prior(self.string_tree, log=log)

    @property
    def id(self) -> str:  # noqa: D102
        if self._function_id is None or self._function_id == "":
            if self.string_tree == "":
                raise ValueError("Cannot infer identifier!")
            self._function_id = self.string_tree_to_id(self.string_tree)
        return self._function_id

    @id.setter
    def id(self, value: str) -> None:
        self._function_id = value

    def create_from_id(self, identifier: str) -> None:  # noqa: D102
        self.reset()
        self._function_id = identifier
        self.id = identifier
        self.string_tree = self.id_to_string_tree(self.id)
        _ = self.value  # required for checking if graph is valid!

    @staticmethod
    def id_to_string_tree(identifier: str) -> str:  # noqa: D102
        return identifier

    @staticmethod
    def string_tree_to_id(string_tree: str) -> str:  # noqa: D102
        return string_tree

    @abstractmethod
    def create_new_instance_from_id(self, identifier: str):  # noqa: D102
        raise NotImplementedError

    def reset(self) -> None:  # noqa: D102
        self.clear_graph()
        self.string_tree = ""
        self._value = None
        self._function_id = ""

    def compose_functions(  # noqa: D102
        self,
        flatten_graph: bool = True,  # noqa: FBT001, FBT002
    ) -> nx.DiGraph:
        return self._compose_functions(self.id, self.grammars[0], flatten_graph)
