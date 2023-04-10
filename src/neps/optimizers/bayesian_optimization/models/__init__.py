from .gp import GPModel  # type: ignore[attr-defined]

# from .gp_hierarchy import ComprehensiveGPHierarchy
# TODO: make gp_hierarchy work (?)
# from .kde import KernelDensityEstimator

SurrogateModelMapping = {
    "gp": GPModel,
    "gp_hierarchy": GPModel,
    # "kde": KernelDensityEstimator,
}
