from .api import run_comprehensive_nas
from .search_spaces import (
    CategoricalHyperparameter,
    ConstantHyperparameter,
    FloatHyperparameter,
    GraphDenseHyperparameter,
    GraphGrammar,
    GraphGrammarCell,
    GraphGrammarRepetitive,
    IntegerHyperparameter,
)
from .search_spaces.search_space import SearchSpace

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
