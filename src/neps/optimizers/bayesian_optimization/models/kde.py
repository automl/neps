from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from statsmodels.nonparametric import kernels
from statsmodels.nonparametric._kernel_base import (
    EstimatorSettings,
    GenericKDE,
    _adjust_shape,
)

from metahyper.api import ConfigResult

KERNEL_MAP = dict(
    wangryzin=kernels.wang_ryzin,
    aitchisonaitken=kernels.aitchison_aitken,
    gaussian=kernels.gaussian,
)


def weighted_generalized_kernel_prod(
    bw: npt.ArrayLike,
    data: npt.ArrayLike,
    data_predict: npt.ArrayLike,
    var_types: list[str],
    num_options: list[int],
    data_weights: npt.ArrayLike = None,
    float_kerneltype: str = "gaussian",
    ordered_kerneltype: str = "wangryzin",
    unordered_kerneltype: str = "aitchisonaitken",
    to_sum: bool = True,
) -> np.ndarray:
    """
    Ripped straight from Statsmodels, with generalization to datapoint weights.
    Returns the non-normalized Generalized Product Kernel Estimator
    Parameters
    ----------
    bw : 1-D ndarray
        The user-specified bandwidth parameters.
    data : 1D or 2-D ndarray
        The training data.
    data_predict : 1-D ndarray
        The evaluation points at which the kernel estimation is performed.
    var_type : str, optional
        The variable type (continuous, ordered, unordered).
    num_options: list[int]:
        The number of options for integer and categorical variables
    float_kerneltype : str, optional
        The kernel used for the continuous variables.
    ordered_kerneltype : str, optional
        The kernel used for the ordered discrete variables.
    unordered_kerneltype : str, optional
        The kernel used for the unordered discrete variables.
    to_sum : bool, optional
        Whether or not to sum the calculated array of densities.  Default is
        True.
    Returns
    -------
    dens : array_like
        The generalized product kernel density estimator.
    Notes
    -----
    The formula for the multivariate kernel estimator for the pdf is:
    .. math:: f(x)=\frac{1}{nh_{1}...h_{q}}\\sum_{i=1}^
                        {n}K\\left(\frac{X_{i}-x}{h}\right)
    where
    .. math:: K\\left(\frac{X_{i}-x}{h}\right) =
                k\\left( \frac{X_{i1}-x_{1}}{h_{1}}\right)\times
                k\\left( \frac{X_{i2}-x_{2}}{h_{2}}\right)\times...\times
                k\\left(\frac{X_{iq}-x_{q}}{h_{q}}\right)
    """
    # TODO clean up this mess of a function - I blame statsmodels even though
    # I basically stole their code / Carl
    kernel_types = dict(f=float_kerneltype, o=ordered_kerneltype, u=unordered_kerneltype)

    K_val = np.empty(data.shape)
    for dim, var_type in enumerate(var_types):
        if var_type == "c":
            # constants do not affect the density at all
            K_val[:, dim] = 1
            continue

        # Go from the variable type (cont.) --> kernel type (gaussian)
        kernel_type = kernel_types[var_type]
        func = KERNEL_MAP[kernel_type]
        if var_type == "u":
            K_val[:, dim] = (
                func(
                    bw[dim], data[:, dim], data_predict[dim], num_levels=num_options[dim]
                )
                * data_weights
            )
        else:
            test_data = K_val[:, dim] = func(bw[dim], data[:, dim], data_predict[dim])
            if np.isnan(test_data).sum() > 0:
                raise SystemError("Test data is NaN")
            if np.isnan(data_weights).sum() > 0:
                raise SystemError("data_weights is NaN", data_weights)
            K_val[:, dim] = func(bw[dim], data[:, dim], data_predict[dim]) * data_weights

    isfloat = np.array([f == "f" for f in var_types])

    # pseudo-normalization of the density, seems to work so-so
    dens = K_val.prod(axis=1) / np.prod(bw[isfloat])
    if to_sum:
        return dens.sum(axis=0)
    else:
        return dens


class KernelDensityEstimator(GenericKDE):
    def __init__(
        self,
        param_types,
        num_options: list | None = None,
        logged_params: list | None = None,
        is_fidelity: list | None = None,
        fixed_bw: list | None = None,
        bandwidth_factor: float = 0.5,
        min_bw: float = 1e-3,
        prior=None,
        prior_weight: float = 0.0,
        prior_as_samples: bool | list = False,
        min_density: float = 1e-8,
        **estimator_kwargs: dict,
    ):
        """
        Implementation of a Kernel Density Estimator with multivariate weighting functionality and
        possible prior incorporation.

        Args:
            var_type ([type]): [description]
            num_options (list optional): The number of options for integer and categorical variables
            bw ([type], optional): [description]. Defaults to None.
            defaults ([type], optional): [description]. Defaults to None.
            data_weights ([type], optional): [description]. Defaults to None.
            prior ([type], optional): [description]. Defaults to None.
            prior_as_samples (bool, optional): [description]. Defaults to False.
        """
        # filter away the fitelity
        self.fixed_bw = fixed_bw
        self.is_fidelity = is_fidelity or [False] * len(param_types)
        self.fid_array = np.array(is_fidelity)
        self.param_types = np.array(param_types)[~self.fid_array]
        if num_options is None:
            num_options = [np.inf] * len(param_types)

        self.num_options = np.array(num_options)[~self.fid_array]
        if fixed_bw is not None:
            self.fixed_bw = np.array(fixed_bw)[~self.fid_array]
        self.bandwidth_factor = bandwidth_factor
        self.min_bw = min_bw
        self.prior = prior
        self.prior_weight = prior_weight
        self.prior_as_samples = prior_as_samples
        self.min_density = min_density
        self.estimator_kwargs = estimator_kwargs or {}
        # TODO consider the logged parameters when fitting the KDE
        self.logged_params = logged_params
        self.num_constant_dims = np.sum(np.array(param_types) == "c")

    def fit(
        self,
        configs: list[ConfigResult],
        config_weights: list[float] = None,
        fixed_bw: list[float] = None,
    ) -> None:
        if config_weights is None:
            data_weights = np.ones(len(configs))
        else:
            data_weights = np.asarray(config_weights)

        self.data = self._convert_configs_to_numpy(configs)
        self.data_weights = np.sqrt(data_weights) / np.sum(np.sqrt(data_weights))
        self.nobs, self.k_vars = np.shape(self.data)

        # remove the dimensions from the data that are just constant
        # which means that the KDE bandwith is decided without considering said dimension
        self.k_vars -= self.num_constant_dims
        if fixed_bw is not None:
            self.bw = np.array(fixed_bw)
            return

        # These are attributes to fit within the statsmodels KDE framework
        # https://github.com/statsmodels/statsmodels/blob/b79d71862dd9ca30ed173c9ad9b96a18e48d8dbb/statsmodels/nonparametric/_kernel_base.py#L99
        # TODO build own KDE if we see the need

        # the defaults have to be set with each fit call, since we get new data for each one
        defaults = EstimatorSettings(**self.estimator_kwargs)
        self._set_defaults(defaults)

        if not self.efficient:
            self.bw = self._compute_bw(self.fixed_bw)
        else:
            self.bw = self._compute_efficient(self.fixed_bw)
        self.bw = np.clip(self.bw, a_min=self.min_bw, a_max=np.inf)
        self.bw = self.bw * self.bandwidth_factor

    def _convert_configs_to_numpy(self, configs, drop_fidelity=True):
        """Creates a N x D normalized numpy array for N configurations of dimensionality D. Does not
        support graphs.

        Args:
            configs ([type]):

        Returns:
            np.ndarray: N x D normalized numpy array of configs
        """
        configs_np = np.array(
            [[x_.normalized().value for x_ in list(x.values())] for x in configs]
        )
        if drop_fidelity:
            return configs_np[:, ~self.fid_array]
        return configs_np

    def pdf(self, configs):
        """Evaluates the combined probability density function of the KDE and the prior. If the prior
        is not specified as samples, the linear combination of the prior and KDE is queried. Otherwise,
        only the KDE is queried.
        """
        # TODO incorporate querying the prior?
        return self._pdf(self._convert_configs_to_numpy(configs))

    def _pdf(self, X):
        """
        Evaluate the probability density function. Operates on a numpy array
        Parameters
        ----------
        X : array_like, optional
            Points to evaluate at.
        Returns
        -------
        pdf_est : array_like
            Probability density function evaluated at `X`.
        Notes
        -----
        The probability density is given by the generalized product kernel
        estimator:
        .. math:: K_{h}(X_{i},X_{j}) =
            \\prod_{s=1}^{q}h_{s}^{-1}k\\left(\frac{X_{is}-X_{js}}{h_{s}}\right)
        """
        X = _adjust_shape(X, self.k_vars + self.num_constant_dims)

        pdf_est = []
        for i in range(np.shape(X)[0]):
            # TODO add in the number of values for each of the non-continous variables
            pdf_est.append(
                weighted_generalized_kernel_prod(
                    self.bw,
                    data=self.data,
                    data_predict=X[i, :],
                    data_weights=self.data_weights,
                    var_types=self.param_types,
                    num_options=self.num_options,
                )
                / self.nobs
            )

        pdf_est = np.squeeze(pdf_est)
        return np.maximum(pdf_est, self.min_density)

    def visualize_2d(self, ax, grid_points: int = 101, color: str = "k"):
        assert len(self.param_types) == 2
        X1 = np.linspace(0, 1, grid_points)
        X2 = np.linspace(0, 1, grid_points)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.append(X1.reshape(-1, 1), X2.reshape(-1, 1), axis=1)
        Z = self._pdf(X)
        Z_min, Z_max = -np.abs(Z).max(), np.abs(Z).max()

        Z = Z.reshape(grid_points, grid_points)

        c = ax.pcolormesh(X1, X2, Z, cmap=color, vmin=Z_min, vmax=Z_max)
        ax.set_title("pcolormesh")
        ax.axis([0, 1, 0, 1])
        return ax
