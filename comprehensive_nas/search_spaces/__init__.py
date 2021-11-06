from .graph_dense.graph_dense import GraphHyperparameter
from .numerical.categorical import CategoricalHyperparameter
from .numerical.constant import ConstantHyperparameter
from .numerical.float import FloatHyperparameter
from .numerical.integer import IntegerHyperparameter

HyperparameterMapping = {
    "categorical": CategoricalHyperparameter,
    "constant": ConstantHyperparameter,
    "float": FloatHyperparameter,
    "integer": IntegerHyperparameter,
    "graph_dense": GraphHyperparameter,
    "graph_grammar": None,
}