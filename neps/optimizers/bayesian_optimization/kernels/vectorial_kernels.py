from copy import deepcopy
from math import sqrt

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    from neps.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)


class Stationary:
    """Here we follow the structure of GPy to build a sub class of stationary kernel. All the classes (i.e. the class
    of stationary kernel_operators) derived from this class use the scaled distance to compute the Gram matrix."""

    def __init__(self, lengthscale=1.0, outputscale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lengthscale = lengthscale
        self._gram = None
        self._train = None
        self.outputscale = outputscale

    def forward(self, x1, x2=None, l=None, **params):
        if l is not None:
            return _scaled_distance(l, x1, x2)
        return _scaled_distance(self.lengthscale, x1, x2)

    def fit_transform(
        self,
        x1,
        l=None,
        rebuild_model=True,
        save_gram_matrix=True,
    ):
        if isinstance(self, HammingKernel):
            x1 = [[value for value in hp if isinstance(value, str)] for hp in x1]
        else:
            x1 = [[value for value in hp if not isinstance(value, str)] for hp in x1]
            x1 = torch.tensor(x1, dtype=torch.float64)

        if not rebuild_model and self._gram is not None:
            return self._gram
        K = self.forward(x1, l=l)
        if save_gram_matrix:
            self._train = deepcopy(x1)
            assert isinstance(K, torch.Tensor), "it doesnt work with np arrays.."
            self._gram = K.clone()
        return K

    def transform(
        self,
        x1,
        l=None,
    ):
        if self._gram is None:
            raise ValueError("The kernel has not been fitted. Run fit_transform first")
        return self.forward(self._train, x1, l=l)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward_t(self, x2, x1=None, l=None):
        if x1 is None:
            x1 = torch.tensor(self._train)
        x2 = torch.tensor(x2).requires_grad_(True)
        K = self.forward(x1, x2, l)
        return K, x2


class RBFKernel(Stationary):
    def forward(self, x1, x2=None, l=None, **kwargs):
        if l is None:
            dist_sq = _scaled_distance(self.lengthscale, x1, x2, sq_dist=True)
        else:
            dist_sq = _scaled_distance(l, x1, x2, sq_dist=True)
        if isinstance(dist_sq, torch.Tensor):
            return self.outputscale * torch.exp(-0.5 * dist_sq)
        return self.outputscale * np.exp(-0.5 * dist_sq)


class LayeredRBFKernel(RBFKernel):
    """
    Same as the conventional RBF kernel, but adapted in a way as a midway between spherical RBF and ARD RBF. In this
    case, one weight is assigned to each Weisfiler-Lehman iteration only (e.g. one weight for h=0, another for h=1 and
    etc.)
    """

    def forward(self, ard_dims, x1, x2=None, l=None, **kwargs):
        l = l if l is not None else self.lengthscale
        assert l.shape[0] == ard_dims.shape[0], (
            "LayeredRBF expects the lengthscale vector to have the same "
            "dimensionality as the "
            "number of WL iterations, but got lengthscale vector of shape"
            + str(l.shape[0])
            + "and WL iteration of shape "
            + str(ard_dims.shape[0])
        )
        if not isinstance(ard_dims, torch.Tensor):
            ard_dims = torch.tensor(ard_dims)
        M = torch.cat(
            [torch.ones(int(ard_dims[i])) * l[i] for i in range(ard_dims.shape[0])]
        )
        return super().forward(x1, x2, M, **kwargs)


class Matern32Kernel(Stationary):
    def forward(self, x1, x2=None, l=None, **kwargs):
        if l is None:
            dist = _scaled_distance(
                self.lengthscale,
                x1,
                x2,
            )
        else:
            dist = _scaled_distance(
                l,
                x1,
                x2,
            )
        if isinstance(dist, torch.Tensor):
            return (
                self.outputscale * (1 + sqrt(3.0) * dist) * torch.exp(-sqrt(3.0) * dist)
            )
        return self.outputscale * (1 + sqrt(3.0) * dist) * np.exp(-sqrt(3.0) * dist)


class Matern52Kernel(Stationary):
    def forward(self, x1, x2=None, l=None, **kwargs):
        if l is None:
            dist = _scaled_distance(
                self.lengthscale,
                x1,
                x2,
            )
        else:
            dist = _scaled_distance(
                l,
                x1,
                x2,
            )
        sq_dist = dist ** 2
        if isinstance(dist, torch.Tensor):
            return (
                self.outputscale
                * (1 + sqrt(5.0) * dist + 5.0 / 3.0 * sq_dist)
                * torch.exp(-sqrt(5.0) * dist)
            )
        return (
            self.outputscale
            * (1 + sqrt(5.0) * dist + 5.0 / 3.0 * sq_dist)
            * np.exp(-sqrt(5.0) * dist)
        )


class HammingKernel(Stationary):
    def forward(self, x1, x2=None, l=None, **kwargs):
        if l is None:
            dist = _hamming_distance(
                self.lengthscale,
                x1,
                x2,
            )
        else:
            dist = _hamming_distance(
                l,
                x1,
                x2,
            )
        return dist
        # # TODO
        # sq_dist = dist ** 2
        # if isinstance(dist, torch.Tensor):
        #     return self.outputscale * (1 + sqrt(5.) * dist + 5. / 3. * sq_dist) * torch.exp(-sqrt(5.) * dist)
        # return self.outputscale * (1 + sqrt(5.) * dist + 5. / 3. * sq_dist) * np.exp(-sqrt(5.) * dist)


class RationalQuadraticKernel(Stationary):
    def __init__(self, lengthscale, outputscale=1.0, power=2.0, **kwargs):
        super().__init__(lengthscale, outputscale, **kwargs)
        self.power = power

    def forward(self, x1, x2=None, **kwargs):
        dist_sq = _scaled_distance(self.lengthscale, x1, x2, sq_dist=True)
        return self.outputscale * (1 + dist_sq / 2.0) ** (-self.power)


def _unscaled_distance(X, X2=None, sq_dist=False):
    """The unscaled distance between X and X2. if x2 is not supplied, then the squared Euclidean distance is
    computed within X"""
    if isinstance(X, torch.Tensor):
        assert X.ndimension() == 2
        if X2 is not None:
            assert isinstance(X2, torch.Tensor)
            assert X2.ndimension() == 2
        if X2 is None:
            Xsq = torch.sum(X ** 2, 1)
            r2 = -2 * X @ X.t() + Xsq[:, None] + Xsq[None, :]
        else:
            X1sq = torch.sum(X ** 2, 1)
            X2sq = torch.sum(X2 ** 2, 1)
            r2 = -2 * X @ X2.t() + X1sq[:, None] + X2sq[None, :]
        r2 += 1e-8
        if not sq_dist:
            r2 = torch.sqrt(r2)
    else:
        assert X.ndim == 2
        if X2 is not None:
            assert X2.ndim == 2
        if X2 is None:
            Xsq = np.sum(X ** 2, 1)
            r2 = -2 * X @ X.transpose() + Xsq[:, None] + Xsq[None, :]
        else:
            X1sq = np.sum(X ** 2, 1)
            X2sq = np.sum(X2 ** 2, 1)
            r2 = -2 * X @ X2.transpose() + X1sq[:, None] + X2sq[None, :]
        if not sq_dist:
            r2 = np.sqrt(r2)
    return r2


def _scaled_distance(lengthscale, X, X2=None, sq_dist=False):
    """Compute the *scaled* distance between X and x2 (or, if X2 is not supplied, the distance between X and itself)
    by the lengthscale.
    if a scalar (float) or a dim=1 lengthscale vector is supplied, then it is assumed that we use one lengthscale for
    all dimensions. Otherwise we have an ARD kernel and in which case the length of the lengthscale vector must be
    the same as the dimensionality of the problem."""
    if X2 is None:
        X2 = X
    if isinstance(lengthscale, float) or len(lengthscale) == 1:
        return (
            _unscaled_distance(X, X2) / lengthscale
            if sq_dist is False
            else _unscaled_distance(X, X2, sq_dist=True) / (lengthscale ** 2)
        )
    else:
        # ARD kernel - one lengthscale per dimension
        assert (
            len(lengthscale) == X.shape[1]
        ), "For a non-scaler theta, it needs to be of the same length as the dim" " of the input data, but got input dim of " + str(
            X.shape[1]
        ) + " and lengthscale dimension of " + str(
            lengthscale.shape[0]
        )
        dist = _unscaled_distance(X / lengthscale, X2 / lengthscale)
        return dist if not sq_dist else dist ** 2


def _hamming_distance(lengthscale, X, X2=None):
    if X2 is None:
        X2 = X

    def _distance(X, X2, lengthscale):
        indicator = np.expand_dims(X, axis=1) != X2
        K = (-1 / (2 * lengthscale ** 2) * indicator).sum(axis=2)
        K = np.exp(K)
        return torch.from_numpy(K)

    if isinstance(lengthscale, float) or len(lengthscale) == 1:
        return _distance(X, X2, lengthscale) / lengthscale
    else:
        # return _distance(X / lengthscale, X2 / lengthscale, lengthscale)
        return _distance(X, X2, lengthscale=1)
