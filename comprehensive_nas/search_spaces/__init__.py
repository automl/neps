from .graph_dense.graph_dense import GraphDenseHyperparameter
from .graph_grammar.graph_grammar import (
    GraphGrammar,
    GraphGrammarCell,
    GraphGrammarRepetitive,
)
from .numerical.categorical import CategoricalHyperparameter
from .numerical.constant import ConstantHyperparameter
from .numerical.float import FloatHyperparameter
from .numerical.integer import IntegerHyperparameter

HyperparameterMapping = {
    "categorical": CategoricalHyperparameter,
    "constant": ConstantHyperparameter,
    "float": FloatHyperparameter,
    "integer": IntegerHyperparameter,
    "graph_dense": GraphDenseHyperparameter,
    "graph_grammar": GraphGrammar,
    "graph_grammar_cell": GraphGrammarCell,
    "graph_grammar_repetitive": GraphGrammarRepetitive,
}
