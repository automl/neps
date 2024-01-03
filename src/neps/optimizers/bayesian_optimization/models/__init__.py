from .deepGP import DeepGP
from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .DPL import PowerLawSurrogate


SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "dpl": PowerLawSurrogate
}

try:
    from .pfn import PFN_SURROGATE  # only if available locally
    SurrogateModelMapping.update({"pfn": PFN_SURROGATE})
except:
    pass
