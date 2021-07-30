import torch


class GraphKernels:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_hyperparameters = 0
        self.rbf_lengthscale = False
        self.kern = None
        self.__name__ = "GraphKernelBase"

    @staticmethod
    def normalize_gram(K: torch.Tensor):
        K_diag = torch.sqrt(torch.diag(K))
        K_diag_outer = torch.ger(K_diag, K_diag)
        return K / K_diag_outer

    def fit_transform(
        self, gr: list, rebuild_model=False, save_gram_matrix=False, **kwargs
    ):
        raise NotImplementedError

    def transform(
        self,
        gr: list,
    ):
        raise NotImplementedError

    def forward_t(self, gr2, gr1: list = None):
        """
        Compute the derivative of the kernel function k(phi, phi*) with respect to phi* (the training point)
        """
        raise NotImplementedError(
            "The kernel gradient is not implemented for the graph kernel called!"
        )
