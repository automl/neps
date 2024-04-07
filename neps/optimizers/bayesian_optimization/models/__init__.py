from ....metahyper.utils import MissingDependencyError
from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .DPL import PowerLawSurrogate

try:
    from .deepGP import DeepGP
except ImportError as e:
    DeepGP = MissingDependencyError("gpytorch", e)

try:
    from .pfn import PFN_SURROGATE  # only if available locally
except ImportError as e:
    PFN_SURROGATE = MissingDependencyError("pfn", e)

SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "dpl": PowerLawSurrogate,
    "pfn": PFN_SURROGATE,
}
