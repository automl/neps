from .deepGP import DeepGP
from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy

SurrogateModelMapping = {
    "deep_gp": DeepGP,
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
}
