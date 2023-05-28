from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .string_hierarchy import GPStringHierarchy, NASK

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "gp_string_hierarchy": GPStringHierarchy,
}

StringKernelModelMapping = {
    "nask": NASK,
}
