from .gp import ComprehensiveGP
from .gp_hierarchy import ComprehensiveGPHierarchy
from .kde import MultiFidelityPriorWeightedKDE

SurrogateModelMapping = {
    "gp": ComprehensiveGP,
    "gp_hierarchy": ComprehensiveGPHierarchy,
    "kde": MultiFidelityPriorWeightedKDE,
}
