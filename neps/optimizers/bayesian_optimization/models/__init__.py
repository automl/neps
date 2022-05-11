from .gp import GPModel
from .gp_hierarchy import ComprehensiveGPHierarchy

SurrogateModelMapping = {
    "gp": GPModel,
    "gp_hierarchy": ComprehensiveGPHierarchy,
}
