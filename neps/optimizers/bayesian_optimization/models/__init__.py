from neps.optimizers.bayesian_optimization.models.gp import ComprehensiveGP
from neps.utils.common import MissingDependencyError

from .ftpfn import FTPFNSurrogate

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "ftpfn": FTPFNSurrogate,
}
