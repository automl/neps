from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .kde import KernelDensityEstimator

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "kde": KernelDensityEstimator,
}
