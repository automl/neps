from neps.utils.common import MissingDependencyError

from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy

from .ftpfn import FTPFNSurrogate


SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "ftpfn": FTPFNSurrogate,
}
