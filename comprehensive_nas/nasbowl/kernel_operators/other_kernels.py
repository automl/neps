from grakel.kernels import RandomWalkLabeled as _RWL, ShortestPath as _SPA
from .graph_kernel import GraphKernels
import logging
from grakel.utils import graph_from_networkx
import torch


class RandomWalk(GraphKernels):
    """
    Random walk kernel using Grakel interface
    (A thin wrapper around the GraKel interface)
    """
    def __init__(self, lambda_=0.001,  fast: bool = True,
                 node_label: str = 'op_name',
                 **kwargs):
        super(RandomWalk, self).__init__(**kwargs)
        self.fast = fast
        self.node_label = node_label
        if not self.fast:
            logging.warning(".fast flag has been turned off, and O(n^6) complexity is incurred in computing the exact"
                            "random walk kernel!")
        self.kern = _RWL(lamda=lambda_, method_type='fast' if fast else 'baseline',
                                      normalize=True)   # For use in Gaussian Process, normalize is required
        self.__name__ = 'RandomWalk'
        self._gram, self._train = None, None

    def fit_transform(self, gr: list, rebuild_model=False, save_gram_matrix=False, **kwargs):
        if rebuild_model is False and self._gram is not None:
            return self._gram
        gr_ = list(graph_from_networkx(gr, self.node_label, ))
        if rebuild_model or self._gram is None:
            self._train = gr[:]
            self._train_transformed = gr_[:]
        K = self.kern.fit_transform(gr_)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K)
        if save_gram_matrix:
            self._gram = K.clone()
        return K

    def transform(self, gr: list, ):
        gr_ = graph_from_networkx(gr, self.node_label, )
        K = self.kern.transform(gr_)
        if not isinstance(K, torch.Tensor):
            K = torch.tensor(K)
        return K

    def transform_t(self, *args):
        raise NotImplementedError


class ShortestPath(RandomWalk):
    def __init__(self, sp_algo='auto', node_label='op_name', **kwargs):
        super(ShortestPath, self).__init__(**kwargs)
        self.node_label = node_label
        self.sp_algo = sp_algo
        self.kern = _SPA(algorithm_type=self.sp_algo,
                         normalize=True,
                         with_labels=True)  # For use in Gaussian Process, normalize is required
        self.__name__ = 'RandomWalk'
        self._gram, self._train = None, None
