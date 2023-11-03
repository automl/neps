from .deepGP import DeepGP
from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy


SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
}

try:
    from .pfn import PFN_SURROGATE  # only if available locally
    SurrogateModelMapping.update({"pfn": PFN_SURROGATE})
except:
    pass
