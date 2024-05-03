from __future__ import annotations

from neps.search_spaces.hyperparameters.numerical import NumericalParameter


class ConstantParameter(NumericalParameter):
    """A constant value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with values that should not change during
    optimization. For example, the `batch_size` hyperparameter in a neural
    network search space can be a `ConstantParameter` with a value of `32`.

    ```python
    import neps

    batch_size = neps.ConstantParameter(32)
    ```
    """

    def __init__(self, value: int | float | str, *, is_fidelity: bool = False):
        """Create a new `ConstantParameter`.

        Args:
            value: value for the hyperparameter.
            is_fidelity: whether the hyperparameter is fidelity.
        """
        super().__init__()
        self.value = value
        self.is_fidelity = is_fidelity
        self.default = value
        self.lower = value
        self.upper = value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (self.value == other.value
                and self.is_fidelity == other.is_fidelity)

    def __repr__(self):
        return f"<Constant, value: {self.id}>"

    def sample(self, user_priors: bool = False):
        pass

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
        **kwargs,
    ):
        return self

    def crossover(self, parent1, parent2=None):
        raise NotImplementedError

    def _get_neighbours(self, **kwargs):
        pass
