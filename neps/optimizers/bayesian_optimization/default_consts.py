import math

DEFAULT_KERNELS = ["m52", "hm"]  # TODO : add 'wl'
DEFAULT_MEAN = "constant"
DEFAULT_COMBINE = "sum"
EPSILON = 1e-9
MIN_INFERRED_NOISE_LEVEL = 1e-4
LENGTHSCALE_MIN = math.exp(-6.754111155189306)
LENGTHSCALE_MAX = math.exp(0.0858637988771976)
LENGTHSCALE_SAFE_MARGIN = 1e-4
