print(
    "In order to run neps with multiple processes or multiple machines, simply call "
    "`neps.run` multiple times. All calls to `neps.run` need to use the same"
    " `root_directory` on the same filesystem, otherwise there is no synchronization "
    "between the `neps.run`'s.\n"
)
print(
    "For example, start the HPO example in two shells from the same directory: \n\n"
    "(in shell 1) \n python -m neps_examples.basic_usage.hyperparameters\n\n"
    "(in shell 2) \n python -m neps_examples.basic_usage.hyperparameters"
)
