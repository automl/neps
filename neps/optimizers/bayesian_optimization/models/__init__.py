from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .gp_string_hierarchy import GPStringHierarchy

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "gp_string_hierarchy": GPStringHierarchy,
}
