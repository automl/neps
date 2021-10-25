import math
from abc import abstractmethod

import numpy as np


class Hyperparameter:
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Hyperparameter's name should be a string.")
        self.name = name

    @abstractmethod
    def sample(self, random_state):
        raise NotImplementedError

    @abstractmethod
    def mutate(self, parent=None):
        raise NotImplementedError

    @abstractmethod
    def crossover(self, parent1, parent2=None):
        raise NotImplementedError


if __name__ == "__main__":

    # from numerical.categorical import CategoricalHyperparameter
    # from numerical.integer import IntegerHyperparameter
    # from numerical.float import FloatHyperparameter
    # from numerical.constant import ConstantHyperparameter
    from graph_dense.graph_dense import GraphHyperparameter

    # hp = CategoricalHyperparameter(name="h", choices=["sgd", "adam", "xd"])
    # hp2 = FloatHyperparameter(name="h2", lower=1, upper=1000, log=False)
    # hp3 = FloatHyperparameter(name="h3", lower=1, upper=1000, log=False)
    # hp4 = FloatHyperparameter(name="h4", lower=1, upper=1000, log=True)
    # hp5 = ConstantHyperparameter(name="h5", value="stojak_na_kwiatki")

    nb201_choices = [
        "nor_conv_3x3",
        "nor_conv_1x1",
        "avg_pool_3x3",
        "skip_connect",
        "none",
    ]

    hp6 = GraphHyperparameter(name="nb2", num_nodes=6, edge_choices=nb201_choices)

    rs = np.random.RandomState(5)
    x = []
    # for _ in range(10000):
    #     x.append(hp.sample(rs))
    #
    # # print(x)
    # import matplotlib.pyplot as plt
    #
    # plt.hist(x)
    # plt.show()
