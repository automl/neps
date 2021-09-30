import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import json as cs_json

cs = CS.ConfigurationSpace()

kernel = CSH.CategoricalHyperparameter(
    "kernel", choices=["linear", "rbf", "poly", "sigmoid"], default_value="rbf"
)
degree = CSH.UniformIntegerHyperparameter(
    "degree", lower=1, upper=128, log=True, default_value=18
)
coef0 = CSH.UniformFloatHyperparameter(
    "coef0", lower=-0.5, upper=0.5, log=False, default_value=0.27073849350462764
)
tol = CSH.UniformFloatHyperparameter(
    "tol", lower=1e-4, upper=1e-2, log=True, default_value=0.00022086199204762887
)
gamma = CSH.CategoricalHyperparameter(
    "gamma", choices=["scale", "auto"], default_value="scale"
)
C = CSH.UniformFloatHyperparameter(
    "C", lower=1.0, upper=20, log=True, default_value=3.0079917803781395
)
nu = CSH.UniformFloatHyperparameter(
    "nu", lower=0.01, upper=1, log=False, default_value=0.40518108684088283
)
shrinking = CSH.CategoricalHyperparameter(
    "shrinking", choices=["True", "False"], default_value="False"
)

cs.add_hyperparameters([kernel, tol, gamma, C, nu, shrinking, degree, coef0])

with open("svr_nu_configspace.json", "w") as f:
    f.write(cs_json.write(cs))
