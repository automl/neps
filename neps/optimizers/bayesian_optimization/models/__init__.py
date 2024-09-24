from neps.optimizers.bayesian_optimization.models.ftpfn import FTPFNSurrogate

# TODO: Need the GP back here
#  * What actually uses the GP
SurrogateModelMapping = {
    "ftpfn": FTPFNSurrogate,
}

__all__ = ["FTPFNSurrogate", "SurrogateModelMapping"]
