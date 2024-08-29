from neps.utils.common import MissingDependencyError

from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy

from .pfn import IFBOSurrogate


SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "ftpfn": IFBOSurrogate,
}
