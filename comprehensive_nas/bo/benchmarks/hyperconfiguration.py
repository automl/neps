from networkx.readwrite import json_graph


class HyperConfiguration:
    def __init__(self, graph=None, hps=None):
        self.graph = graph
        self.hps = hps

    def parse(self):
        config = []
        if self.graph is not None:
            if not isinstance(self.graph, list):
                # Convert a single input X_s to a singleton list
                self.graph = [self.graph]
            config.append([json_graph.node_link_data(g) for g in self.graph])
        if self.hps is not None:
            config.append(self.hps)
        return config
