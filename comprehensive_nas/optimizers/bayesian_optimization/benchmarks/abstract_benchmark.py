from networkx.readwrite import json_graph
import random
import numpy as np

class AbstractBenchmark:
    """An abstract class specifying a prototype objective function"""

    def __init__(self, seed, optimize_arch, optimize_hps):
        # self.dim = len(self.get_meta_information()["bounds"])

        self.seed = seed
        self.optimize_arch = optimize_arch
        self.optimize_hps = optimize_hps

        self.graph = None
        self.hps = None

        self.name = ""

    def __call__(self, *args):
        return self.query(*args)

    # def query(self, mode='eval', *args, **kwargs):
    #     raise NotImplementedError()

    def reinitialize(self, seed=1):
        self.seed = seed  # pylint: disable=attribute-defined-outside-init
        np.random.seed(seed)
        random.seed(seed)

    def sample(self, **kwargs):
        raise NotImplementedError

    def parse(self):
        config = []
        if self.graph is not None:
            if not isinstance(self.graph, list):
                # Convert a single input X_s to a singleton list
                gl = [self.graph]
            else:
                gl = self.graph
            config.append([json_graph.node_link_data(g) for g in gl])
        if self.hps is not None:
            config.append(self.hps)
        return config

    @property
    def id(self):
        return self.name

    def get_graph(self):
        return self.graph

    def get_hps(self):
        return self.hps
