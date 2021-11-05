from .api import run_comprehensive_nas
from .search_spaces import (
    CategoricalHyperparameter,
    ConstantHyperparameter,
    FloatHyperparameter,
    GraphHyperparameter,
    IntegerHyperparameter,
)
from .search_spaces.search_space import SearchSpace

HyperparameterMapping = {
    "categorical": CategoricalHyperparameter,
    "constant": ConstantHyperparameter,
    "float": FloatHyperparameter,
    "integer": IntegerHyperparameter,
    "graph_dense": GraphHyperparameter,
    "graph_grammar": None,
}
