from __future__ import annotations

from copy import deepcopy

from typing_extensions import Literal

from neps.search_spaces.hyperparameters.float import FloatParameter


class IntegerParameter(FloatParameter):
    """An integer value for a parameter.

    This kind of [`Parameter`][neps.search_spaces.parameter] is used
    to represent hyperparameters with continuous integer values, optionally specifying if it exists
    on a log scale.
    For example, `batch_size` could be a value in `(32, 128)`, while the `num_layers`
    hyperparameter in a neural network search space can be a `IntegerParameter`
    with a range of `(1, 1000)` but on a log scale.

    ```python
    import neps

    batch_size = neps.IntegerParameter(32, 128)
    num_layers = neps.IntegerParameter(1, 1000, log=True)
    ```
    """

    def __init__(
        self,
        lower: float | int,
        upper: float | int,
        *,
        log: bool = False,
        is_fidelity: bool = False,
        default: None | float | int = None,
        default_confidence: Literal["low", "medium", "high"] = "low",
    ):
        """Create a new `IntegerParameter`.

        Args:
            lower: lower bound for the hyperparameter.
            upper: upper bound for the hyperparameter.
            log: whether the hyperparameter is on a log scale.
            is_fidelity: whether the hyperparameter is fidelity.
            default: default value for the hyperparameter.
            default_confidence: confidence score for the default value, used when
                condsider prior based optimization.
        """

        super().__init__(
            lower,
            upper,
            log=log,
            is_fidelity=is_fidelity,
            default=default,
            default_confidence=default_confidence,
        )
        # We subtract/add 0.499999 from lower/upper bounds respectively, such that
        # sampling in the float space gives equal probability for all integer values,
        # i.e. [x - 0.499999, x + 0.499999]
        self.float_hp = FloatParameter(
            lower=self.lower - 0.499999,
            upper=self.upper + 0.499999,
            log=self.log,
            is_fidelity=is_fidelity,
            default=default,
            default_confidence=default_confidence,
        )
        self.value: None | int = None

    def __repr__(self):
        return f"<Integer, range: [{self.lower}, {self.upper}], value: {self.value}>"

    def load_from(self, value):
        super().load_from(int(value))

    def _set_float_hp_val(self):
        # IMPORTANT function to call wherever `self.float_hp` is used in this class
        self.float_hp.value = None if self.value is None else float(self.value)
        self.float_hp.default = None if self.default is None else float(self.default)
        if self.log:
            self.float_hp._set_log_values()

    def sample(self, user_priors: bool = False):
        self.float_hp.sample(user_priors=user_priors)
        self.value = int(round(self.float_hp.value))  # type: ignore[arg-type]

    def mutate(
        self,
        parent=None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "local_search",
        **kwargs,
    ):
        if self.is_fidelity:
            raise ValueError("Trying to mutate fidelity param!")
        self._set_float_hp_val()
        mutant = self.float_hp.mutate(
            parent=parent,
            mutation_rate=mutation_rate,
            mutation_strategy=mutation_strategy,
            **kwargs,
        )
        child = float_to_integer(mutant)
        return child

    def crossover(self, parent1, parent2=None):
        if self.is_fidelity:
            raise ValueError("Trying to crossover fidelity param!")
        if parent2 is None:
            parent2 = self

        proxy_self = deepcopy(self)
        proxy_self.value = round((parent1.value + parent2.value) / 1)
        children = proxy_self._get_neighbours(std=0.1, num_neighbours=2)

        if all(not c for c in children):
            raise Exception("Cannot create crossover")
        # expected len(children) == num_neighbours
        return children

    def _get_neighbours(self, std: float = 0.2, num_neighbours: int = 1):
        self._set_float_hp_val()
        neighbours = self.float_hp._get_neighbours(std, num_neighbours)
        for idx, neighbour in enumerate(neighbours):
            neighbours[idx] = float_to_integer(neighbour)
        return neighbours

    def normalized(self):
        hp = FloatParameter(
            lower=self.lower,
            upper=self.upper,
            log=self.log,
            is_fidelity=self.is_fidelity,
            default=self.default,
        )
        hp.value = self.value
        return hp.normalized()

    def set_default_confidence_score(self, default_confidence):
        self._set_float_hp_val()
        self.float_hp.set_default_confidence_score(default_confidence)
        super().set_default_confidence_score(default_confidence)


def float_to_integer(float_hp):
    int_hp = IntegerParameter(
        lower=int(round(float_hp.lower)),
        upper=int(round(float_hp.upper)),
        log=float_hp.log,
    )
    int_hp.value = None if float_hp.value is None else int(round(float_hp.value))

    return int_hp
