# Neural Pipeline Search

##  Installation

1. Create a Python/Conda environment (we used Anaconda with Python 3.7).
2. Install other dependencies via ```poetry install```.
3. Then install ```torch``` via ```python -m comprehensive_nas.utils.install_torch```.

## Examples
To run our exemplary experiments, see below

1. Simple hyperparameter optimization
    ```bash
    python -m neps_examples.hyperparameters.optimize
    ```
