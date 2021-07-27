import networkx as nx
import numpy as np
import torch

from nasbowl.acqusition_functions.ei import ComprehensiveExpectedImprovement


class ComprehensiveUpperConfidentBound(ComprehensiveExpectedImprovement):
    """
    Graph version of the upper confidence bound acquisition function
    """

    def __init__(self, surrogate_model, beta=None, strategy=None,
                 iters=0):
        """Same as graphEI with the difference that a beta coefficient is asked for, as per standard GP-UCB acquisition
        """
        super(ComprehensiveUpperConfidentBound, self).__init__(surrogate_model=surrogate_model,
                                                               strategy=strategy, iters=iters)

        self.beta = beta

    def eval(self, x_graphs, x_hps, asscalar=False):
        # TODO: predict on graph/hps/both/..
        mu, cov = self.surrogate_model.predict(x_graphs, x_hps)
        std = torch.sqrt(torch.diag(cov))
        if self.beta is None:
            self.beta = 3. * torch.sqrt(0.5 * torch.log(torch.tensor(2. * self.iters + 1.)))
        acq = mu + self.beta * std
        if asscalar:
            acq = acq.detach().numpy().item()
        return acq.mean()
