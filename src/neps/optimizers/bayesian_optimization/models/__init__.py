from .deepGP import DeepGP
from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .pfn import PFN_SURROGATE

SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "pfn": PFN_SURROGATE
}
