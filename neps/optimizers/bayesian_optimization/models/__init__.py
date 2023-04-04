from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .string_hierarchy import GPStringHierarchy, StringKernelV1

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "gp_string_hierarchy": GPStringHierarchy,
}

StringKernelModelMapping = {
    "string_v1": StringKernelV1,
}
