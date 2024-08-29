from neps.utils.common import MissingDependencyError

from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy

try:
    from .deepGP import DeepGP
except ImportError as e:
    DeepGP = MissingDependencyError("gpytorch", e)

from .pfn import IFBOSurrogate


SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "ftpfn": IFBOSurrogate,
}
