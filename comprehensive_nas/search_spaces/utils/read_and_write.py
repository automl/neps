import json

from ..graph_dense.graph_dense import GraphHyperparameter
from ..numerical.categorical import CategoricalHyperparameter
from ..numerical.constant import ConstantHyperparameter
from ..numerical.float import FloatHyperparameter
from ..numerical.integer import IntegerHyperparameter
from ..search_space import SearchSpace


def read(json):

    pass


def write(search_space):
    if not isinstance(search_space, SearchSpace):
        raise TypeError("Cannot write a non-search_space object")