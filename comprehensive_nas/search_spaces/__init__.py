from .graph_dense.graph_dense import GraphDenseParameter
from .graph_grammar.graph_grammar import (
    GraphGrammar,
    GraphGrammarCell,
    GraphGrammarRepetitive,
)
from .numerical.categorical import CategoricalParameter
from .numerical.constant import ConstantParameter
from .numerical.float import FloatParameter
from .numerical.integer import IntegerParameter
from .numerical.numerical import NumericalParameter

HyperparameterMapping = {
    "categorical": CategoricalParameter,
    "constant": ConstantParameter,
    "float": FloatParameter,
    "integer": IntegerParameter,
    "graph_dense": GraphDenseParameter,
    "graph_grammar": GraphGrammar,
    "graph_grammar_cell": GraphGrammarCell,
    "graph_grammar_repetitive": GraphGrammarRepetitive,
}
