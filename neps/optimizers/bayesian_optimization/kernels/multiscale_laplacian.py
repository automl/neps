import torch
from grakel.utils import graph_from_networkx

from .grakel_replace.multiscale_laplacian import MultiscaleLaplacian as ML
from .grakel_replace.multiscale_laplacian import MultiscaleLaplacianFast as MLF
from .utils import transform_to_undirected
from .weisfilerlehman import GraphKernels


class MultiscaleLaplacian(GraphKernels):
    def __init__(
        self,
        n: int = 1,
        n_jobs: int = 1,
        random_state=None,
        gamma: float = 0.01,
        heta: float = 0.01,
        max_n_eigs: int = 3,
        n_vertex_samples: int = 5,
        fast: bool = True,
        node_label: str = "op_name",
        edge_label: tuple = None,
        return_tensor: bool = True,
        reindex_node_label: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.random_state = random_state
        self.gamma = gamma
        self.heta = heta
        self.max_n_eigs = max_n_eigs
        self.n_vertex_samples = n_vertex_samples
        if fast:
            self.kern = MLF(
                n_jobs,
                normalize=True,
                random_state=random_state,
                L=n,
                P=max_n_eigs,
                gamma=gamma,
                heta=heta,
                n_samples=n_vertex_samples,
            )
        else:
            self.kern = ML(n_jobs, True, False, L=n, gamma=gamma, heta=heta)
        self.node_label = node_label
        self.edge_label = edge_label
        self.return_tensor = return_tensor
        self._gram = None
        self._train = None
        self.reindex_node_label = reindex_node_label
        self.check_dict = {}
        self.__name__ = "MultiscaleLaplacian"

    def _reindex_node_label(self, gr: list):
        """It seems that MLK needs numeric node features. Reindex the feature"""
        gr_copy = gr[:]
        idx = 0
        for i, g in enumerate(gr_copy):
            for node, attr in g.nodes(data=True):
                if attr[self.node_label] not in self.check_dict.keys():
                    self.check_dict.update({attr[self.node_label]: idx})
                    # Assign the index
                    gr_copy[i].nodes[node][self.node_label] = idx
                    idx += 1
                else:
                    gr_copy[i].nodes[node][self.node_label] = self.check_dict[
                        attr[self.node_label]
                    ]
        return gr_copy

    def fit_transform(
        self, gr: list, rebuild_model=False, save_gram_matrix=False, **kwargs
    ):
        if rebuild_model is False and self._gram is not None:
            return self._gram
        gr = transform_to_undirected(gr)
        if self.reindex_node_label:
            gr = self._reindex_node_label(gr)
        gr_ = graph_from_networkx(gr, self.node_label, self.edge_label)
        K = self.kern.fit_transform(gr_)
        if self.return_tensor:
            K = torch.tensor(K)
        if save_gram_matrix:
            self._gram = K.clone()
            self._train = gr[:]
        return K

    def transform(
        self,
        gr: list,
    ):
        gr = transform_to_undirected(gr)
        if self.reindex_node_label:
            gr = self._reindex_node_label(gr)
        gr_ = graph_from_networkx(gr, self.node_label, self.edge_label)
        K = self.kern.transform(gr_)
        if self.return_tensor:
            K = torch.tensor(K)
        return K

    def forward_t(self, gr2, gr1: list = None):
        return super().forward_t(gr2, gr1=gr1)
