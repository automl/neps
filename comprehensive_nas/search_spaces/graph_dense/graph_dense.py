from typing import List

from ..hyperparameter import Hyperparameter
from ..numerical.categorical import CategoricalHyperparameter
from .utils import create_nasbench201_graph


class GraphHyperparameter(Hyperparameter):
    def __init__(self, name: str, num_nodes: int, edge_choices: List[str]):
        super(GraphHyperparameter, self).__init__(name)

        self.num_nodes = num_nodes
        self.edge_choices = edge_choices
        self.graph = []
        for edge_id in range(self.num_nodes):
            self.graph.append(
                CategoricalHyperparameter(
                    name="edge_%d" % edge_id, choices=self.edge_choices
                )
            )

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.num_nodes == other.num_nodes
            and self.edge_choices == other.edge_choices
        )

    def __hash__(self):
        return hash((self.name, self.num_nodes, self.edge_choices))

    def __repr__(self):
        return "Graph {}, num_nodes: {}, edge_choices: {}".format(
            self.name, self.num_nodes, self.edge_choices
        )

    def __copy__(self):
        return self.__class__(
            name=self.name, num_nodes=self.num_nodes, edge_choices=self.edge_choices
        )

    def sample(self, random_state):
        graph = []
        for edge in self.graph:
            graph.append(edge.sample(random_state))
        # TODO
        graph = create_nasbench201_graph(graph)
        return graph

    def mutate(self, parent=None):
        pass

    def crossover(self, parent1, parent2=None):
        pass
