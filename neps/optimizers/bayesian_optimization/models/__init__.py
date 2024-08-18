from neps.utils.common import MissingDependencyError

from neps.optimizers.bayesian_optimization.models.gp import ComprehensiveGP

try:
    from neps.optimizers.models.deepGP import DeepGP
except ImportError as e:
    DeepGP = MissingDependencyError("gpytorch", e)

try:
    from .pfn import PFN_SURROGATE  # only if available locally
except Exception as e:
    PFN_SURROGATE = MissingDependencyError("pfn", e)

SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "pfn": PFN_SURROGATE,
}
