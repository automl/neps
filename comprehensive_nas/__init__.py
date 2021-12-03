from .api import run_comprehensive_nas
from .search_spaces import (
    CategoricalParameter,
    ConstantParameter,
    FloatParameter,
    GraphDenseParameter,
    GraphGrammar,
    GraphGrammarCell,
    GraphGrammarRepetitive,
    IntegerParameter,
)
from .search_spaces.search_space import SearchSpace

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
