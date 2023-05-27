from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .string_hierarchy import GPStringHierarchy, ASK

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "gp_string_hierarchy": GPStringHierarchy,
}

StringKernelModelMapping = {
    "ASK": ASK,
}
