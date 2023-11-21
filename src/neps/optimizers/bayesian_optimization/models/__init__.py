from ....utils.common import MissingDependencyError
from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy

try:
    from .deepGP import DeepGP
except ImportError as e:
    DeepGP = MissingDependencyError("gpytorch", e)

try:
    from .pfn import PFN_SURROGATE  # only if available locally
except Exception as e:
    PFN_SURROGATE = MissingDependencyError("pfn", e)

SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "pfn": PFN_SURROGATE
}


