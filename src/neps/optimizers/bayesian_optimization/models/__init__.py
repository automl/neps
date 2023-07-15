from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy
}
