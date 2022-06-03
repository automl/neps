from .gp import GPModel

# from .gp_hierarchy import ComprehensiveGPHierarchy
# TODO: make gp_hierarchy work (?)

SurrogateModelMapping = {
    "gp": GPModel,
    # "gp_hierarchy": ComprehensiveGPHierarchy,
}
