class AbstractBenchmark:
    """An abstract class specifying a prototype objective function"""

    def __init__(self, dim=None, optimum_location=None, optimal_val=None, bounds=None,):
        self.dim = dim
        self.optimum_location = optimum_location
        self.optimal_val = optimal_val
        self.bounds = bounds

    def __call__(self, Gs, hps, *args):
        return self.eval(Gs, hps, *args)

    def eval(self, G, hps, *args):
        raise NotImplementedError()

    def reinitialize(self, *args, **kwargs):
        raise NotImplementedError
