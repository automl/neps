import torch

from .ei import ComprehensiveExpectedImprovement


class ComprehensiveUpperConfidentBound(ComprehensiveExpectedImprovement):
    """
    Graph version of the upper confidence bound acquisition function
    """

    def __init__(self, surrogate_model, beta=None, strategy=None, iters=0):
        """Same as graphEI with the difference that a beta coefficient is asked for, as per standard GP-UCB acquisition"""
        super().__init__(surrogate_model=surrogate_model, strategy=strategy, iters=iters)

        self.beta = beta

    def eval(self, x, asscalar=False):
        # TODO: predict on graph/hps/both/..
        mu, cov = self.surrogate_model.predict(x)
        std = torch.sqrt(torch.diag(cov))
        if self.beta is None:
            self.beta = 3.0 * torch.sqrt(
                0.5 * torch.log(torch.tensor(2.0 * self.iters + 1.0))
            )
        acq = mu + self.beta * std
        if asscalar:
            acq = acq.detach().numpy().item()
        return acq.mean().detach().numpy().item()
