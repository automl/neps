import random
import numpy as np

from networkx.readwrite import json_graph
from .graph_dense.graph_dense import GraphHyperparameter

class SearchSpace:
    def __init__(self, *hyperparameters):
        self.hyperparameters = hyperparameters

        self.hps = None
        self.graphs = None
        self.name = ""

    def sample_config(self, random_state):
        hps = []
        graphs = []
        for hyperparameter in self.hyperparameters:
            if isinstance(hyperparameter, GraphHyperparameter):
                graphs.append(hyperparameter.sample(random_state))
            else:
                hps.append(hyperparameter.sample(random_state))

        self.hps = hps if len(hps) > 0 else None
        self.graphs = graphs if len(graphs) > 0 else None
        self.name = self.parse()

    def mutate(self, config, mutate_probability_per_hyperparameter=0.5):
        for hyperparameter in config:
            if random.random() < mutate_probability_per_hyperparameter:
                pass
                hyperparameter.mutate()

        self.name = self.parse()

    def parse(self):
        config = []
        if self.graphs is not None:
            if not isinstance(self.graphs, list):
                # Convert to a singleton list
                gl = [self.graphs]
            else:
                gl = self.graphs
            config.append([json_graph.node_link_data(g)['graph']['name'] for g in gl])
        if self.hps is not None:
            config.append(self.hps)
        return config

    @property
    def id(self):
        return self.name

    def get_graph(self):
        return self.graphs

    def get_hps(self):
        return self.hps


if __name__ == "__main__":
    from numerical.categorical import CategoricalHyperparameter
    from numerical.integer import IntegerHyperparameter
    from numerical.float import FloatHyperparameter
    from numerical.constant import ConstantHyperparameter
    from graph_dense.graph_dense import GraphHyperparameter

    search_space = SearchSpace(
            CategoricalHyperparameter(name='operation', choices=["multiply", "add"]),
            IntegerHyperparameter(name='operant_a', lower=1, upper=100),
            FloatHyperparameter(name='operant_b', lower=1, upper=100, log=True),

            # architecture=cnas.DenseGraph(num_nodes=3, edge_choices={"identity", "3x3_conv"}),
    )
    rs = np.random.RandomState(5)
    search_space.sample_config(rs)
