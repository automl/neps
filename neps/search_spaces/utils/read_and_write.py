# import json

# from ..graph_dense.graph_dense import GraphHyperparameter
# from ..numerical.categorical import CategoricalParameter
# from ..numerical.constant import ConstantParameter
# from ..numerical.float import FloatParameter
# from ..numerical.integer import IntegerParameter
from ..search_space import SearchSpace


def read():

    pass


def write(search_space):
    if not isinstance(search_space, SearchSpace):
        raise TypeError("Cannot write a non-search_space object")
