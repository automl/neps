import torch

from ....search_spaces.graph_grammar.graph import Graph
from .base_kernel import Kernel


class GraphKernel(Kernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_hyperparameters = 0
        self.rbf_lengthscale = False
        self.kern = None
        self.__name__ = "GraphKernel"
        self._pretrained = False

        self._train_graphs: dict = {}
        self._eval_graphs: dict = {}

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

    def does_apply_on(self, graph):
        return isinstance(graph, Graph)

    @property
    def train_graphs(self):
        return self._train_graphs

    @train_graphs.setter
    def train_graphs(self, new_train_graphs: dict) -> None:
        self._train_graphs = new_train_graphs

    def build(self, hp_shapes: dict):
        self.train_graphs = [v.hp_instances for v in hp_shapes.values()]
        self._pretrained = False
        return super().build(hp_shapes)

    @property
    def eval_graphs(self):
        return self._eval_graphs

    @eval_graphs.setter
    def eval_graphs(self, new_eval_graphss: dict) -> None:
        self._eval_graphs = new_eval_graphss

    def set_eval_graphs(self, new_graphs):
        self.eval_graphs = new_graphs
